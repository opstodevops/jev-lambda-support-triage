// Confidence-aware support triage: an ordinary Express server, run unchanged on
// AWS Lambda Web Functions. Locally: AWS_LAMBDA_HTTP_ENDPOINT=127.0.0.1:3000 node index.js
import express from 'express';
import helmet from 'helmet';
import { randomUUID } from 'node:crypto';
import { decide, getClient, QUEUES, QUEUE_IDS, MODEL } from './src/jev.js';
import { route, AUTO_ROUTE_THRESHOLD } from './src/routing.js';
import { emitMetrics } from './src/telemetry.js';

const [HOST, PORT] = (process.env.AWS_LAMBDA_HTTP_ENDPOINT || '0.0.0.0:3000').split(':');
const ENV_ID = randomUUID().slice(0, 8);
const ENV_STARTED = Date.now();
let servedRequests = 0; // per-environment counter, used only to flag the first (cold) request

const MAX_MESSAGE_CHARS = 4000;
const MAX_EVAL_ITEMS = 500;

getClient().catch(() => {}); // warm the secret fetch; errors surface on the first request

const app = express();
app.use(helmet());
app.use(express.json({ limit: '1mb' }));

app.get('/health', (req, res) => res.json({ status: 'ok', env: ENV_ID, model: MODEL }));

function badRequest(res, msg) {
  return res.status(400).json({ error: msg });
}

function validMessage(m) {
  return typeof m === 'string' && m.trim().length > 0 && m.length <= MAX_MESSAGE_CHARS;
}

// Optional overrides for the robustness experiments: reworded instructions,
// reworded queue descriptions, or a shuffled option order. Queue ids can't change.
function questionOptions(body) {
  const opts = {};
  if (body.instructions !== undefined) {
    if (typeof body.instructions !== 'string' || body.instructions.length > 500) throw new Error('instructions must be a string of at most 500 chars');
    opts.instructions = body.instructions;
  }
  let criteria;
  if (body.criteria !== undefined) {
    const keys = Object.keys(body.criteria ?? {});
    const valid = keys.length === QUEUE_IDS.length && QUEUE_IDS.every((q) => typeof body.criteria[q] === 'string' && body.criteria[q].length <= 500);
    if (!valid) throw new Error(`criteria must map exactly these queue ids to strings: ${QUEUE_IDS.join(', ')}`);
    criteria = body.criteria;
  }
  if (body.shuffleSeed !== undefined) {
    if (!Number.isInteger(body.shuffleSeed)) throw new Error('shuffleSeed must be an integer');
    criteria = shuffleKeys(criteria ?? QUEUES, body.shuffleSeed);
  }
  if (criteria) opts.criteria = criteria;
  return opts;
}

// Same options, different order (seeded Fisher-Yates with mulberry32), to test order sensitivity.
function shuffleKeys(criteria, seed) {
  const keys = Object.keys(criteria);
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = keys.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [keys[i], keys[j]] = [keys[j], keys[i]];
  }
  return Object.fromEntries(keys.map((k) => [k, criteria[k]]));
}

app.post('/triage', async (req, res) => {
  const started = performance.now();
  const coldStart = servedRequests++ === 0;
  const { message } = req.body ?? {};
  if (!validMessage(message)) return badRequest(res, `message must be a non-empty string of at most ${MAX_MESSAGE_CHARS} chars`);
  const threshold = req.body.threshold ?? AUTO_ROUTE_THRESHOLD;
  if (typeof threshold !== 'number' || threshold < 0 || threshold > 1) return badRequest(res, 'threshold must be a number between 0 and 1');
  let opts;
  try { opts = questionOptions(req.body); } catch (err) { return badRequest(res, err.message); }

  try {
    const decision = await decide(message, opts);
    const action = route(decision, threshold);
    const totalMs = Math.round((performance.now() - started) * 10) / 10;
    res.json({ decision, action, timing: { jevMs: decision.jevMs, totalMs }, env: { id: ENV_ID, coldStart, ageMs: Date.now() - ENV_STARTED } });
    emitMetrics({ jevMs: decision.jevMs, totalMs, confidence: decision.confidence, action: action.action, source: 'triage' });
  } catch (err) {
    console.error(JSON.stringify({ level: 'ERROR', msg: 'triage failed', error: err.name, status: err.status }));
    res.status(502).json({ error: 'decision model call failed', type: err.name });
  }
});

// Runs a labelled batch through Jev and streams each decision back as Server-Sent Events,
// so the caller can draw accuracy-versus-threshold while the run is still going.
app.post('/eval', async (req, res) => {
  const { items, concurrency = 8 } = req.body ?? {};
  if (!Array.isArray(items) || items.length === 0 || items.length > MAX_EVAL_ITEMS) return badRequest(res, `items must be an array of 1-${MAX_EVAL_ITEMS} {id, text}`);
  if (!items.every((it) => it && (typeof it.id === 'string' || typeof it.id === 'number') && validMessage(it.text))) return badRequest(res, 'each item needs an id and a valid text');
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 16) return badRequest(res, 'concurrency must be an integer 1-16');
  let opts;
  try { opts = questionOptions(req.body); } catch (err) { return badRequest(res, err.message); }
  const runId = typeof req.body.runId === 'string' ? req.body.runId.slice(0, 64) : randomUUID();

  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  res.flushHeaders();
  const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  const abort = new AbortController();
  res.on('close', () => abort.abort());
  const heartbeat = setInterval(() => res.write(': keep-alive\n\n'), 15_000);

  const started = performance.now();
  send('start', { runId, count: items.length, concurrency, env: ENV_ID, model: MODEL });
  let next = 0, errors = 0, inputTokens = 0;
  const worker = async () => {
    while (next < items.length && !abort.signal.aborted) {
      const item = items[next++];
      try {
        const decision = await decide(item.text, opts, { signal: abort.signal });
        inputTokens += decision.usage?.input_tokens ?? 0;
        send('result', { id: item.id, ...decision, tOffsetMs: Math.round(performance.now() - started) });
        emitMetrics({ jevMs: decision.jevMs, totalMs: decision.jevMs, confidence: decision.confidence, source: 'eval' });
      } catch (err) {
        if (abort.signal.aborted) return;
        errors++;
        send('item_error', { id: item.id, error: err.name, status: err.status });
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  clearInterval(heartbeat);
  if (!abort.signal.aborted) {
    send('done', { runId, count: items.length, errors, inputTokens, elapsedMs: Math.round(performance.now() - started) });
    res.end();
  }
});

app.use((err, req, res, next) => {
  if (err.type === 'entity.too.large' || err instanceof SyntaxError) return badRequest(res, 'invalid or too-large JSON body');
  next(err);
});

app.listen(Number(PORT), HOST, () => console.log(JSON.stringify({ level: 'INFO', msg: 'listening', host: HOST, port: PORT, env: ENV_ID })));
