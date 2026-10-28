"""SigV4-signed client for the IAM-authenticated Web Function endpoint.

  python -I eval/wf_client.py --domain <domain> health
  python -I eval/wf_client.py --domain <domain> triage "My card hasn't arrived yet"
  python -I eval/wf_client.py --domain <domain> eval data/eval.jsonl results/raw_jev_base.jsonl
  python -I eval/wf_client.py --domain <domain> latency data/eval.jsonl results/latency_e2e.jsonl --n 200
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


class WebFunctionClient:
    def __init__(self, domain, region, service="lambda"):
        self.base = f"https://{domain}"
        self.region = region
        self.service = service
        self.session = boto3.Session(region_name=region)
        self.http = requests.Session()

    def _signed(self, method, path, body=None):
        data = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        # Web Functions require the payload hash header; plain SigV4Auth doesn't add it.
        headers["X-Amz-Content-SHA256"] = hashlib.sha256((data or "").encode()).hexdigest()
        req = AWSRequest(method=method, url=self.base + path, data=data, headers=headers)
        creds = self.session.get_credentials().get_frozen_credentials()
        SigV4Auth(creds, self.service, self.region).add_auth(req)
        return dict(req.headers.items()), data

    def get(self, path):
        headers, _ = self._signed("GET", path)
        r = self.http.get(self.base + path, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def post(self, path, body):
        headers, data = self._signed("POST", path, body)
        r = self.http.post(self.base + path, headers=headers, data=data, timeout=60)
        r.raise_for_status()
        return r.json()

    def stream(self, path, body, timeout=900):
        """POST and yield (event, data) pairs from a Server-Sent Events response."""
        headers, data = self._signed("POST", path, body)
        with self.http.post(self.base + path, headers=headers, data=data, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            event, lines = "message", []
            for raw in r.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                if raw == "":
                    if lines:
                        yield event, json.loads("\n".join(lines))
                    event, lines = "message", []
                elif raw.startswith(":"):
                    continue
                elif raw.startswith("event:"):
                    event = raw[6:].strip()
                elif raw.startswith("data:"):
                    lines.append(raw[5:].strip())


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_eval(client, args):
    rows = load_jsonl(args.input)
    out = Path(args.output)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    extra = json.loads(args.extra) if args.extra else {}
    print(f"{len(todo)} to run ({len(done)} already in {out})", file=sys.stderr)
    with out.open("a") as f:
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i:i + args.chunk]
            body = {"items": [{"id": r["id"], "text": r["text"]} for r in chunk],
                    "concurrency": args.concurrency, "runId": args.run_id, **extra}
            t0 = time.perf_counter()
            n = 0
            for event, data in client.stream("/eval", body):
                if event == "result":
                    f.write(json.dumps(data) + "\n")
                    n += 1
                elif event == "item_error":
                    print(f"  error {data}", file=sys.stderr)
                elif event == "done":
                    print(f"  chunk {i // args.chunk + 1}: {n} results, {data['errors']} errors, "
                          f"{data['elapsedMs']} ms server, {time.perf_counter() - t0:.1f} s client", file=sys.stderr)
            f.flush()


def run_latency(client, args):
    """Sequential /triage calls: client-side round trip vs. server-side Jev time."""
    rows = load_jsonl(args.input)[: args.n]
    with Path(args.output).open("w") as f:
        for r in rows:
            t0 = time.perf_counter()
            resp = client.post("/triage", {"message": r["text"]})
            e2e = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"id": r["id"], "e2e_ms": round(e2e, 1), **resp["timing"], **resp["env"]}) + "\n")
            if args.sleep:
                time.sleep(args.sleep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
                   help="Region of the endpoint (default: AWS_REGION)")
    p.add_argument("--service", default="lambda", help="SigV4 signing name")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health")
    t = sub.add_parser("triage"); t.add_argument("message")
    e = sub.add_parser("eval")
    e.add_argument("input"); e.add_argument("output")
    e.add_argument("--chunk", type=int, default=200)
    e.add_argument("--concurrency", type=int, default=8)
    e.add_argument("--run-id", default="eval")
    e.add_argument("--extra", help="JSON merged into the request body (instructions, criteria, shuffleSeed)")
    l = sub.add_parser("latency")
    l.add_argument("input"); l.add_argument("output")
    l.add_argument("--n", type=int, default=200)
    l.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()
    if not args.region:
        p.error("set --region or AWS_REGION to the endpoint's Region")

    client = WebFunctionClient(args.domain, args.region, args.service)
    if args.cmd == "health":
        print(json.dumps(client.get("/health")))
    elif args.cmd == "triage":
        print(json.dumps(client.post("/triage", {"message": args.message}), indent=2))
    elif args.cmd == "eval":
        run_eval(client, args)
    elif args.cmd == "latency":
        run_latency(client, args)


if __name__ == "__main__":
    main()
