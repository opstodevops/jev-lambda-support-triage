"""Convert raw /eval SSE results into the prediction format used by metrics.py.

Writes two files, because Jev reports both per-option probabilities and a separate
`confidence`, and the calibration question is which of the two to threshold on:
  results/pred_<name>.jsonl          confidence = Jev's reported confidence
  results/pred_<name>_maxprob.jsonl  confidence = probability of the chosen option

  python -I eval/jev_to_pred.py results/raw_jev_base.jsonl jev
"""
import json
import sys
from pathlib import Path

PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000  # USD, jev-1.13 list price; output tokens are free


def main(raw_path, name):
    rows = [json.loads(l) for l in Path(raw_path).read_text().splitlines() if l.strip()]
    out_conf = Path(f"results/pred_{name}.jsonl").open("w")
    out_max = Path(f"results/pred_{name}_maxprob.jsonl").open("w")
    for r in rows:
        usage = r.get("usage") or {}
        cost = usage.get("cost")
        base = {
            "id": r["id"],
            "pred": r["queue"],
            "probs": r["probabilities"],
            "self_serve_p": r["selfServeP"],
            "urgency": r["urgency"],
            "latency_ms": r["jevMs"],
            "input_tokens": usage.get("input_tokens"),
            "cost_usd": cost if cost is not None else usage.get("input_tokens", 0) * PRICE_PER_INPUT_TOKEN,
            "cost_source": "openrouter_usage" if cost is not None else "list_price",
        }
        out_conf.write(json.dumps({**base, "confidence": r["confidence"]}) + "\n")
        out_max.write(json.dumps({**base, "confidence": r["probabilities"][r["queue"]]}) + "\n")
    print(f"{len(rows)} rows -> results/pred_{name}.jsonl, results/pred_{name}_maxprob.jsonl")


if __name__ == "__main__":
    main(*sys.argv[1:3])
