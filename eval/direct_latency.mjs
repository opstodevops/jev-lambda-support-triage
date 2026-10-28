// Calls Jev directly (no Web Function) with the same three questions the app asks, to compare
// network placement. Run from the app directory so packages resolve:
//   cd app && node ../eval/direct_latency.mjs ../data/eval.jsonl ../results/latency_direct.jsonl 100
// The key is read from Secrets Manager into memory and never printed.
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const appRequire = createRequire(`${process.cwd()}/package.json`);
const load = (pkg) => import(pathToFileURL(appRequire.resolve(pkg)).href);
const { SecretsManagerClient, GetSecretValueCommand } = await load('@aws-sdk/client-secrets-manager');
const { TypeSafeClient } = await load('@typesafe-ai/sdk');
const { buildQuestions } = await import(pathToFileURL(`${process.cwd()}/src/jev.js`).href);

const [input, output, n = '100'] = process.argv.slice(2);
// Region and secret name follow the same variables as scripts/deploy.sh.
const sm = new SecretsManagerClient({ region: process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION });
const secretId = process.env.JEV_SECRET_NAME || 'jev-triage/openrouter-api-key';
const { SecretString } = await sm.send(new GetSecretValueCommand({ SecretId: secretId }));
const client = new TypeSafeClient({ apiKey: SecretString.trim(), baseURL: 'https://openrouter.ai/api', defaultModel: 'jev-1.13' });
const rows = readFileSync(input, 'utf8').trim().split('\n').slice(0, Number(n)).map(JSON.parse);
const out = [];
for (const r of rows) {
  const t0 = performance.now();
  await client.systemOne({ state: r.text, questions: buildQuestions() });
  out.push(JSON.stringify({ id: r.id, ms: Math.round((performance.now() - t0) * 10) / 10 }));
}
writeFileSync(output, out.join('\n') + '\n');
