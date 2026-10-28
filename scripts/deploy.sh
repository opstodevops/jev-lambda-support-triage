#!/usr/bin/env bash
# Deploys the triage service as a Lambda Web Function with the AWS CLI.
#
#   AWS_REGION=us-west-2 ./scripts/deploy.sh
#
# Run it again to ship a code change. The Region comes from AWS_REGION, then
# AWS_DEFAULT_REGION, then your AWS CLI profile. FUNCTION_NAME and JEV_SECRET_NAME
# override the default names.
set -euo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region || true)}}"
[[ -n "$REGION" ]] || { echo "Set AWS_REGION, for example: export AWS_REGION=us-west-2" >&2; exit 1; }
NAME="${FUNCTION_NAME:-jev-triage}"
SECRET="${JEV_SECRET_NAME:-jev-triage/openrouter-api-key}"
ROLE="awscli-lambdaweb-$NAME"
TAG_KEY="project"
TAG_VALUE="jev-triage"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AWS_REGION="$REGION"

echo "==> Region $REGION, function $NAME"

# 1. API key. Created once; the prompt hides input and the key never appears on a command line.
if ! SECRET_ARN=$(aws secretsmanager describe-secret --secret-id "$SECRET" --query ARN --output text 2>/dev/null); then
  read -rs -p "OpenRouter API key: " KEY; echo
  SECRET_ARN=$(printf '%s' "$KEY" | aws secretsmanager create-secret --name "$SECRET" \
    --secret-string file:///dev/stdin --tags "Key=$TAG_KEY,Value=$TAG_VALUE" --query ARN --output text)
  unset KEY
fi
echo "==> Secret $SECRET_ARN"

# 2. Dependencies. Pure JavaScript, so installing on x86 for Arm64 is fine.
(cd "$ROOT/app" && npm ci --omit=dev --no-audit --no-fund)

# 3. Create the function on the first run, otherwise deploy a new revision.
CREATE_ARGS=()
if ! aws lambda-web get-web-function --function-name "$NAME" >/dev/null 2>&1; then
  CREATE_ARGS=(--create --auth-type IamAuth --max-environments 5 --rate-limit 100)
fi
aws lambda-web deploy --name "$NAME" --code "$ROOT/app" "${CREATE_ARGS[@]}" \
  --timeout-seconds 900 --env "JEV_SECRET_ID=$SECRET" --env NODE_ENV=production \
  --tags "$TAG_KEY=$TAG_VALUE" --output json >/dev/null

# 4. Let the CLI-created execution role read this one secret, and tag the role.
aws iam put-role-policy --role-name "$ROLE" --policy-name read-jev-secret --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"$SECRET_ARN\"}]}"
aws iam tag-role --role-name "$ROLE" --tags "Key=$TAG_KEY,Value=$TAG_VALUE"

DOMAIN=$(aws lambda-web get-web-function-endpoint --function-name "$NAME" --endpoint-name dev \
  --query domainName --output text)
echo
echo "Deployed. Endpoint (IamAuth): https://$DOMAIN"
echo "Try:  DOMAIN=$DOMAIN python3 eval/wf_client.py --domain \$DOMAIN health"
