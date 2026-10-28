"""Decision flip rate between a base run and perturbed runs on the same messages.

  python -I eval/robustness.py results/raw_small_base.jsonl results/raw_small_shuffle1.jsonl ...
"""
import json
import statistics
import sys
from pathlib import Path


def load(path):
    return {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines()) if r}


def compare(base, other):
    ids = sorted(set(base) & set(other))
    flips = [i for i in ids if base[i]["queue"] != other[i]["queue"]]
    dconf = [abs(base[i]["confidence"] - other[i]["confidence"]) for i in ids]
    dss = [abs(base[i]["selfServeP"] - other[i]["selfServeP"]) for i in ids]
    # Flips that happen on confident base decisions are the ones that matter for automation.
    confident = [i for i in ids if base[i]["confidence"] >= 0.8]
    confident_flips = [i for i in confident if base[i]["queue"] != other[i]["queue"]]
    return {
        "n": len(ids),
        "flip_rate": len(flips) / len(ids),
        "flip_rate_confident_base": len(confident_flips) / len(confident) if confident else None,
        "n_confident_base": len(confident),
        "mean_abs_conf_delta": statistics.fmean(dconf),
        "p95_abs_conf_delta": sorted(dconf)[int(0.95 * (len(dconf) - 1))],
        "mean_abs_self_serve_delta": statistics.fmean(dss),
    }


if __name__ == "__main__":
    base = load(sys.argv[1])
    report = {Path(p).stem: compare(base, load(p)) for p in sys.argv[2:]}
    print(json.dumps(report, indent=2))
