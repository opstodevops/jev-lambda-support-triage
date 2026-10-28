#!/usr/bin/env bash
# Deletes everything scripts/deploy.sh created. Pass --keep-secret to keep the API key.
#
#   AWS_REGION=us-west-2 ./scripts/cleanup.sh [--keep-secret]
#
# Uses the same Region and name variables as deploy.sh. Resources that are already gone are skipped.
set -uo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region || true)}}"
[[ -n "$REGION" ]] || { echo "Set AWS_REGION, for example: export AWS_REGION=us-west-2" >&2; exit 1; }
export AWS_REGION="$REGION"
NAME="${FUNCTION_NAME:-jev-triage}"
SECRET="${JEV_SECRET_NAME:-jev-triage/openrouter-api-key}"
ROLE="awscli-lambdaweb-$NAME"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="awscli-lambdaweb-$ACCOUNT-$REGION-an"

echo "==> Region $REGION"
echo "==> Web Function $NAME"
if aws lambda-web get-web-function --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda-web delete-web-function --function-name "$NAME"
else echo "    not found"; fi

echo "==> Execution role $ROLE"
if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  for p in $(aws iam list-role-policies --role-name "$ROLE" --query 'PolicyNames' --output text); do
    aws iam delete-role-policy --role-name "$ROLE" --policy-name "$p"; done
  for p in $(aws iam list-attached-role-policies --role-name "$ROLE" --query 'AttachedPolicies[].PolicyArn' --output text); do
    aws iam detach-role-policy --role-name "$ROLE" --policy-arn "$p"; done
  aws iam delete-role --role-name "$ROLE"
else echo "    not found"; fi

echo "==> Code packages in s3://$BUCKET/lambda-web/$NAME/ (all versions)"
if aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  aws s3api list-object-versions --bucket "$BUCKET" --prefix "lambda-web/$NAME/" \
    --query '[Versions,DeleteMarkers][][].[Key,VersionId]' --output text |
  while read -r key version; do
    [[ -n "$key" && "$key" != "None" ]] || continue
    aws s3api delete-object --bucket "$BUCKET" --key "$key" --version-id "$version" >/dev/null
    echo "    deleted $key ($version)"
  done
else echo "    bucket not found"; fi

echo "==> Log groups"
for g in "/aws/lambda/web/$NAME" "/aws/lambda/web/$NAME.local"; do
  aws logs delete-log-group --log-group-name "$g" 2>/dev/null && echo "    deleted $g" || echo "    $g not found"
done

if [[ "${1:-}" != "--keep-secret" ]]; then
  echo "==> Secret $SECRET"
  aws secretsmanager delete-secret --secret-id "$SECRET" --force-delete-without-recovery >/dev/null 2>&1 \
    && echo "    deleted" || echo "    not found"
fi
echo "Done. The shared CLI bucket $BUCKET is left in place; delete it if nothing else uses it."
