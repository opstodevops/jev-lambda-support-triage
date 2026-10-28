# Confidence-aware support triage with Jev on AWS Lambda Web Functions

Sample code for routing support tickets only when the model is sure, using Jev on AWS Lambda Web Functions.

An Express app asks [Jev](https://docs.typesafe.ai), TypeSafe's decision model, three typed questions about a customer message: which queue, how urgent, and whether a help article could resolve it. The app routes the message automatically only when the queue answer is confident and sends the rest to human review. It runs unchanged on AWS Lambda Web Functions behind an IAM-authenticated endpoint.

```
Client --SigV4--> Web Function endpoint (IamAuth) --> Express app (Node.js 24, Arm64)
                                                        |-- Secrets Manager (OpenRouter API key)
                                                        |-- Jev via OpenRouter System One API (third party)
                                                        '-- CloudWatch Logs and EMF metrics
```

The architecture diagram source is [`docs/architecture.drawio`](docs/architecture.drawio).

## Repository layout

| Path | Purpose |
|---|---|
| `app/` | The Web Function: `index.js` (Express server), `src/jev.js` (Jev client and questions), `src/queues.json` (queue descriptions), `src/routing.js` (thresholds), `src/telemetry.js` (CloudWatch EMF metrics) |
| `scripts/deploy.sh` | Stores the API key in Secrets Manager, deploys with `aws lambda-web deploy`, and grants the execution role read access to the secret |
| `scripts/cleanup.sh` | Deletes everything `deploy.sh` created |
| `eval/` | `wf_client.py` (SigV4 client and SSE reader), `metrics.py`, `baselines.py`, `robustness.py`, `prepare_data.py`, `direct_latency.mjs` |
| `data/` | Hand-authored queue and intent files. Run the prepare_data script to download the public datasets and build the labelled evaluation files. See [`data/README.md`](data/README.md) |
| `docs/` | Architecture diagram and figures |

## Prerequisites

- AWS CLI 2.38 or later (adds `aws lambda-web`), with credentials that can create Lambda Web Functions, IAM roles and inline policies, and Secrets Manager secrets.
- Node.js 20 or later and npm. The app has no native dependencies.
- `curl` 7.75 or later for `--aws-sigv4`. Python 3.10 or later for the evaluation scripts.
- An [OpenRouter](https://openrouter.ai) API key. OpenRouter bills Jev usage separately from AWS.
- A Region where [Lambda Web Functions](https://docs.aws.amazon.com/lambda/latest/dg/web-functions.html) is available.

## Configuration

All scripts read the same environment variables:

| Variable | Default | Used by |
|---|---|---|
| `AWS_REGION` | `AWS_DEFAULT_REGION`, then your AWS CLI profile | all scripts and `eval/wf_client.py` |
| `FUNCTION_NAME` | `jev-triage` | `deploy.sh`, `cleanup.sh` |
| `JEV_SECRET_NAME` | `jev-triage/openrouter-api-key` | `deploy.sh`, `cleanup.sh`, `eval/direct_latency.mjs` |

## Deploy

```bash
export AWS_REGION=us-west-2   # or ap-southeast-2, or another Region with Web Functions
./scripts/deploy.sh
```

On the first run, the script prompts for the OpenRouter key (input is hidden) and stores it in Secrets Manager. Run it again to deploy a code change. It prints the endpoint domain when it finishes. Measured in `us-west-2`: 34 seconds for the first deploy, including `npm ci`.

The script creates the Web Function `jev-triage` with an `IamAuth` endpoint (`--max-environments 5`, `--rate-limit 100`, `--timeout-seconds 900`), the execution role `awscli-lambdaweb-jev-triage`, and the secret. The function, role, and secret are tagged `project=jev-triage`. `aws lambda-web deploy` also uploads the code to the shared bucket `awscli-lambdaweb-<account-id>-<region>-an`, which it creates if needed.

## Test

```bash
DOMAIN=<endpoint domain printed by deploy.sh>
python3 -m venv .venv && .venv/bin/pip install boto3==1.43.113 requests==2.34.2
.venv/bin/python -I eval/wf_client.py --domain "$DOMAIN" health
.venv/bin/python -I eval/wf_client.py --domain "$DOMAIN" triage "My card hasn't arrived"
```

The blog post shows the same calls with `curl --aws-sigv4`. An unsigned request, or a signed request without `x-amz-content-sha256`, returns HTTP 403.

## Run locally

```bash
cd app && npm ci --omit=dev
read -rs -p "OpenRouter key: " OPENROUTER_API_KEY && export OPENROUTER_API_KEY
AWS_LAMBDA_HTTP_ENDPOINT=127.0.0.1:3000 node index.js
```

Set `TYPESAFE_API_KEY` instead to call TypeSafe's API directly (model `jev-1.13.0`). Environment variables are fine for local runs. Deployed revisions read the key from Secrets Manager, because revision environment variables are readable in plaintext.

## Reproduce the evaluation

The public datasets are not in this repository. Run the prepare_data script to download the public datasets and build the labelled evaluation files:

```bash
python -I eval/prepare_data.py
```

```bash
.venv/bin/pip install scikit-learn numpy pandas matplotlib
.venv/bin/python -I eval/wf_client.py --domain "$DOMAIN" eval data/eval.jsonl results/raw_jev_base.jsonl
.venv/bin/python -I eval/jev_to_pred.py results/raw_jev_base.jsonl jev
.venv/bin/python -I eval/baselines.py
.venv/bin/python -I eval/metrics.py results/pred_jev_maxprob.jsonl=Jev results/pred_tfidf_lr_full.jsonl="TF-IDF" --charts results/charts
```

The full run makes about 870 Jev calls, about $0.03 at the OpenRouter price on 2026-10-14.

## Clean up

```bash
./scripts/cleanup.sh            # add --keep-secret to keep the API key
```

It deletes the Web Function, the execution role and its inline policies, the code packages under `lambda-web/jev-triage/`, both log groups, and the secret (without a recovery window). It leaves the shared CLI bucket in place; delete it if nothing else uses it. Measured in `us-west-2`: 16 seconds.

## Security considerations

This is sample code for evaluation, not a production configuration:

- The endpoint uses `IamAuth`, but it is reachable over the internet. Any principal in the account with `lambda:InvokeWebFunctionEndpoint` can call it.
- `aws lambda-web deploy` creates the execution role and a shared S3 bucket. For production, define the role, bucket, and Web Function resources in AWS CloudFormation.
- The log groups have no retention policy and use the default CloudWatch Logs encryption.
- Each message is sent to OpenRouter and TypeSafe. Check their data handling before you send real customer data.

## Cost

The deployment creates billed resources: Web Function requests, active CPU, and memory; a Secrets Manager secret; and CloudWatch Logs and custom metrics. Jev usage is billed by OpenRouter. See [AWS Lambda pricing](https://aws.amazon.com/lambda/pricing/), and run `cleanup.sh` when you finish.

## License

To be added before publishing.
