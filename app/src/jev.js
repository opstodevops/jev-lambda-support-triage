// Jev client and the three typed questions used for triage.
import { readFileSync } from 'node:fs';
import { TypeSafeClient } from '@typesafe-ai/sdk';
import { SecretsManagerClient, GetSecretValueCommand } from '@aws-sdk/client-secrets-manager';

export const QUEUES = JSON.parse(readFileSync(new URL('./queues.json', import.meta.url), 'utf8'));
export const QUEUE_IDS = Object.keys(QUEUES);
// OpenRouter by default. A TypeSafe key alone switches to TypeSafe's own API and model id.
const direct = !process.env.OPENROUTER_API_KEY && Boolean(process.env.TYPESAFE_API_KEY);
export const MODEL = process.env.JEV_MODEL || (direct ? 'jev-1.13.0' : 'jev-1.13');
export const BASE_URL = process.env.JEV_BASE_URL || (direct ? 'https://api.typesafe.ai' : 'https://openrouter.ai/api');
export const DEFAULT_INSTRUCTIONS = 'Which support queue should handle this customer message?';

// Local runs can pass the key as an environment variable. Deployed revisions read it from
// Secrets Manager, because revision environment variables are readable in plaintext.
async function apiKey() {
  const fromEnv = process.env.OPENROUTER_API_KEY || process.env.TYPESAFE_API_KEY;
  if (fromEnv) return fromEnv;
  const secrets = new SecretsManagerClient({});
  const { SecretString } = await secrets.send(new GetSecretValueCommand({ SecretId: process.env.JEV_SECRET_ID }));
  return SecretString.trim();
}

// Start fetching the API key at module load, but don't block the HTTP listener on it:
// the runtime gives init 10 seconds, and the key is only needed by the first Jev call.
let clientPromise;
export function getClient() {
  clientPromise ??= apiKey()
    .then((key) => new TypeSafeClient({
      apiKey: key,
      baseURL: BASE_URL,
      defaultModel: MODEL,
      timeout: 10_000,
      retry: { maxRetries: 2 },
    }))
    .catch((err) => { clientPromise = undefined; throw err; });
  return clientPromise;
}

const URGENCY_LEVELS = [
  'Routine: a general question with no time pressure.',
  'Soon: the customer is blocked or inconvenienced and needs help today.',
  'Urgent: money is at risk, the account may be compromised, or the customer is locked out.',
];

// Three independent questions about the same message, sent in one request.
export function buildQuestions({ instructions = DEFAULT_INSTRUCTIONS, criteria = QUEUES } = {}) {
  return {
    queue: { type: 'choice', instructions, criteria },
    urgency: { type: 'score', instructions: 'How urgent is this message for the support team?', criteria: URGENCY_LEVELS },
    self_serve: { type: 'noul', instructions: 'Could a standard help-centre article fully resolve this, with no staff member needing to look at or act on the customer\'s account?' },
  };
}

// One Jev round trip for one message. Returns the answers plus timing and usage.
export async function decide(message, questionOpts, { signal } = {}) {
  const client = await getClient();
  const started = performance.now();
  const { data } = await client
    .systemOne({ state: message, questions: buildQuestions(questionOpts) }, { signal })
    .withResponse();
  const jevMs = performance.now() - started;
  const { queue, urgency, self_serve: selfServe } = data.answers;
  return {
    queue: queue.choice,
    confidence: queue.confidence,
    probabilities: queue.probabilities,
    urgency: urgency.score,
    urgencyConfidence: urgency.confidence,
    selfServeP: selfServe.noul,
    model: data.model,
    usage: data.usage,
    jevMs: Math.round(jevMs * 10) / 10,
  };
}
