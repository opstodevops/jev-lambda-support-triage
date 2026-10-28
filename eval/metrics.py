"""Metrics for every triage method (baselines and Jev).

Prediction file format (JSONL, one row per example):
  {"id", "pred", "probs": {queue: p}, "confidence", "latency_ms"?, "self_serve_p"?, "cost_usd"?}
Ground truth comes from data/eval.jsonl (override with --eval). A prediction file may
cover a subset of the eval ids (for example eval_small); metrics use the ids it contains.

  python -I eval/metrics.py results/pred_a.jsonl [results/pred_b.jsonl=Label ...] \
      --out results/metrics_<name>.json --charts results/charts

Append "=Label" to a path to set the label shown in the charts and tables.

Definitions
  confidence      the row's "confidence" field (for most methods: probability of the
                  predicted class). Rows without it fall back to probs[pred].
  ECE             15 equal-width bins on confidence, bin b = (b/15, (b+1)/15] (0 goes in
                  bin 0); ECE = sum_b n_b/N * |acc_b - conf_b|.
  Brier           multiclass: mean over rows of sum_q (p_q - 1[q == truth])^2, queues missing
                  from probs count as 0. Range 0 (perfect) .. 2.
  selective       for t in 0.00..0.99: coverage = share of rows with confidence >= t, and
                  accuracy on those rows.
  AURC            area under the risk-coverage curve. Rows are grouped by distinct confidence
                  value (ties enter together), sorted high to low; AURC = sum_k risk_k * dcov_k.
  coverage@acc    the largest coverage reachable by any confidence threshold (all distinct
                  confidence values are tried) with accuracy on covered rows >= target.
  OOS             rows with source == "clinc_oos": share predicted "other", and mean
                  confidence on OOS rows that were NOT predicted "other" (overconfident misses).
  self_serve      only if rows carry "self_serve_p": AUROC and accuracy at 0.5 on rows where
                  the label self_serve is not null.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUEUES = list(json.loads((ROOT / "data" / "queues.json").read_text()).keys())
N_BINS = 15
THRESHOLDS = [round(i / 100, 2) for i in range(100)]

LABELS = {
    "keyword": "Keyword rules",
    "tfidf_lr_20": "TF-IDF + LR (20 per queue)",
    "tfidf_lr_full": "TF-IDF + LR (10k labels)",
}

# Categorical palette (validated CVD-safe order, light surface), assigned in fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Colour follows the method, not its position in the command line.
FIXED_SLOT = [("jev", 0), ("keyword", 1), ("tfidf_lr_20", 2), ("tfidf_lr_full", 3),
              ("bedrock", 4), ("claude", 4), ("haiku", 4)]


def assign_colours(names):
    used, out = set(), {}
    for n in names:
        slot = next((s for key, s in FIXED_SLOT if n == key or n.startswith(key)), None)
        if slot is None or slot in used:
            slot = next(i for i in range(len(PALETTE)) if i not in used) if len(used) < len(PALETTE) else 0
        used.add(slot)
        out[n] = PALETTE[slot]
    return out


SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


# ---------------------------------------------------------------------------
# core metric functions (array in, number out) - reusable on their own
# ---------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def per_class_f1(y_true, y_pred, labels=QUEUES):
    out = {}
    for q in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == q and p == q)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != q and p == q)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == q and p != q)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[q] = {"precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": tp + fn}
    return out


def macro_f1(y_true, y_pred, labels=QUEUES):
    f = per_class_f1(y_true, y_pred, labels)
    return float(np.mean([f[q]["f1"] for q in labels]))


def confusion(y_true, y_pred, labels=QUEUES):
    idx = {q: i for i, q in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def reliability(conf, correct, n_bins=N_BINS):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    bins = []
    for i in range(n_bins):
        m = b == i
        n = int(m.sum())
        bins.append({"lo": i / n_bins, "hi": (i + 1) / n_bins, "count": n,
                     "mean_confidence": float(conf[m].mean()) if n else None,
                     "accuracy": float(correct[m].mean()) if n else None})
    return bins


def ece(conf, correct, n_bins=N_BINS):
    N = len(conf)
    return float(sum(x["count"] / N * abs(x["accuracy"] - x["mean_confidence"])
                     for x in reliability(conf, correct, n_bins) if x["count"]))


def brier(prob_rows, y_true, labels=QUEUES):
    tot = 0.0
    for probs, t in zip(prob_rows, y_true):
        tot += sum((float(probs.get(q, 0.0)) - (1.0 if q == t else 0.0)) ** 2 for q in labels)
    return tot / len(y_true)


def risk_coverage(conf, correct):
    """Points (coverage, accuracy, threshold) for every distinct confidence value, high to low."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    N = len(conf)
    pts = []
    for u in np.unique(conf)[::-1]:
        m = conf >= u
        pts.append((m.sum() / N, float(correct[m].mean()), float(u)))
    return pts


def aurc(conf, correct):
    area, prev = 0.0, 0.0
    for cov, acc, _ in risk_coverage(conf, correct):
        area += (1 - acc) * (cov - prev)
        prev = cov
    return float(area)


def coverage_at_accuracy(conf, correct, target):
    best = (0.0, None)
    for cov, acc, u in risk_coverage(conf, correct):
        if acc >= target - 1e-12 and cov > best[0]:
            best = (float(cov), u)
    return {"coverage": best[0], "threshold": best[1]}


def selective(conf, correct, thresholds=THRESHOLDS):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    out = []
    for t in thresholds:
        m = conf >= t - 1e-12
        out.append({"t": t, "coverage": float(m.mean()),
                    "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ---------------------------------------------------------------------------
def evaluate(pred_rows, gold_by_id):
    rows = [(p, gold_by_id[p["id"]]) for p in pred_rows if p["id"] in gold_by_id]
    unknown_ids = sum(1 for p in pred_rows if p["id"] not in gold_by_id)
    n = len(rows)
    y = [g["queue"] for _, g in rows]
    yhat = [p["pred"] for p, _ in rows]
    conf = [float(p["confidence"]) if p.get("confidence") is not None
            else float(p.get("probs", {}).get(p["pred"], 0.0)) for p, _ in rows]
    correct = [a == b for a, b in zip(y, yhat)]
    probs = [p.get("probs") or {p["pred"]: 1.0} for p, _ in rows]

    res = {"n": n, "n_ids_not_in_eval": unknown_ids,
           "n_invalid_pred": sum(1 for q in yhat if q not in QUEUES),
           "accuracy": accuracy(y, yhat), "macro_f1": macro_f1(y, yhat)}
    unamb = [i for i, (_, g) in enumerate(rows) if not g.get("ambiguous")]
    res["accuracy_excl_ambiguous"] = accuracy([y[i] for i in unamb], [yhat[i] for i in unamb])
    res["n_excl_ambiguous"] = len(unamb)
    amb = [i for i, (_, g) in enumerate(rows) if g.get("ambiguous")]
    res["accuracy_ambiguous_only"] = accuracy([y[i] for i in amb], [yhat[i] for i in amb]) if amb else None
    res["per_queue"] = {q: {k: _r(v) if k != "support" else v for k, v in d.items()}
                        for q, d in per_class_f1(y, yhat).items()}
    res["confusion"] = {"labels": QUEUES, "rows_true_cols_pred": confusion(y, yhat).tolist()}

    res["mean_confidence"] = float(np.mean(conf))
    res["ece"] = ece(conf, correct)
    res["brier"] = brier(probs, y)
    res["mean_prob_sum"] = float(np.mean([sum(float(v) for v in pr.values()) for pr in probs]))
    res["reliability"] = [{k: _r(v) if isinstance(v, float) else v for k, v in b.items()}
                          for b in reliability(conf, correct)]

    res["aurc"] = aurc(conf, correct)
    res["coverage_at_95_acc"] = coverage_at_accuracy(conf, correct, 0.95)
    res["coverage_at_98_acc"] = coverage_at_accuracy(conf, correct, 0.98)
    res["selective"] = [{k: _r(v) if isinstance(v, float) else v for k, v in s.items()}
                        for s in selective(conf, correct)]
    res["risk_coverage_curve"] = [[_r(c), _r(a), _r(u, 6)] for c, a, u in risk_coverage(conf, correct)]

    oos = [i for i, (_, g) in enumerate(rows) if g.get("source") == "clinc_oos"]
    if oos:
        missed = [i for i in oos if yhat[i] != "other"]
        res["oos"] = {"n": len(oos),
                      "frac_predicted_other": sum(1 for i in oos if yhat[i] == "other") / len(oos),
                      "n_missed": len(missed),
                      "mean_confidence_missed": float(np.mean([conf[i] for i in missed])) if missed else None}
        inscope = [i for i, (_, g) in enumerate(rows) if g.get("source") != "clinc_oos"]
        res["oos"]["false_other_rate_inscope"] = (sum(1 for i in inscope if yhat[i] == "other") / len(inscope)
                                                  if inscope else None)

    ss = [(float(p["self_serve_p"]), bool(g["self_serve"])) for p, g in rows
          if p.get("self_serve_p") is not None and g.get("self_serve") is not None]
    if ss:
        s, lab = zip(*ss)
        res["self_serve"] = {"n": len(ss), "auroc": auroc(s, lab),
                             "accuracy_at_0.5": float(np.mean([(a >= 0.5) == b for a, b in ss])),
                             "base_rate": float(np.mean(lab))}

    lat = [float(p["latency_ms"]) for p, _ in rows if p.get("latency_ms") is not None]
    if lat:
        res["latency_ms"] = {"n": len(lat), "p50": float(np.percentile(lat, 50)),
                             "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)),
                             "mean": float(np.mean(lat))}
    cost = [float(p["cost_usd"]) for p, _ in rows if p.get("cost_usd") is not None]
    if cost:
        res["cost_usd"] = {"n": len(cost), "total": sum(cost), "per_1k": 1000 * float(np.mean(cost))}
    return _round_tree(res)


def _round_tree(x):
    if isinstance(x, dict):
        return {k: _round_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_tree(v) for v in x]
    if isinstance(x, float):
        return _r(x, 6) if abs(x) < 1e-3 else _r(x, 4)
    return x


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def chart_reliability(name, label, m, out_dir, colour):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(8, 5), dpi=150, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    bins = m["reliability"]
    w = 1 / N_BINS
    xs = [b["lo"] + w / 2 for b in bins if b["count"]]
    accs = [b["accuracy"] for b in bins if b["count"]]
    confs = [b["mean_confidence"] for b in bins if b["count"]]
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=INK2, label="Perfect calibration")
    ax.bar(xs, accs, width=w - 0.006, color=colour, edgecolor=SURFACE, linewidth=1, label="Accuracy in bin")
    ax.scatter(confs, accs, s=22, color=INK, zorder=3, label="Mean confidence in bin")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{label}: confidence vs accuracy", pad=26)
    ax.text(0, 1.04, f"ECE {m['ece']:.3f} · Brier {m['brier']:.3f} · accuracy {m['accuracy']:.1%} · n={m['n']}",
            transform=ax.transAxes, color=INK2, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    share = [b["count"] / m["n"] for b in bins]
    axh.bar([b["lo"] + w / 2 for b in bins], share, width=w - 0.006, color=INK2, edgecolor=SURFACE)
    axh.set_ylabel("Share of\nexamples", fontsize=9)
    axh.set_xlabel("Confidence (15 equal-width bins)")
    axh.set_xlim(0, 1)
    axh.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12)
    p = Path(out_dir) / f"reliability_{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_coverage(entries, out_dir, colours, fname="accuracy_vs_coverage.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    lo = 1.0
    for i, (name, label, m) in enumerate(entries):
        c = colours[name]
        curve = m["risk_coverage_curve"]
        cov = [0.0] + [pt[0] for pt in curve]
        acc = [curve[0][1]] + [pt[1] for pt in curve]
        lo = min(lo, min(acc))
        ax.step(cov, acc, where="pre", lw=2, color=c, label=f"{label} (AURC {m['aurc']:.3f})")
    for tgt in (0.95, 0.98):
        ax.axhline(tgt, color=INK2, lw=0.9, ls=":")
        ax.text(0.005, tgt + 0.004, f"{tgt:.0%} accuracy", color=INK2, fontsize=9, va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(max(0, math.floor((lo - 0.03) * 20) / 20), 1.005)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Coverage (share of messages auto-routed, most confident first)")
    ax.set_ylabel("Accuracy on auto-routed messages")
    ax.set_title("Accuracy vs coverage when routing only confident predictions")
    ax.legend(loc="lower left", fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)
    p = Path(out_dir) / fname
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
def summary_line(name, m):
    oos = m.get("oos", {})
    lat = m.get("latency_ms", {})
    return (f"{name:22s} n={m['n']:4d} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
            f"acc_unamb={m['accuracy_excl_ambiguous']:.3f} ECE={m['ece']:.3f} Brier={m['brier']:.3f} "
            f"AURC={m['aurc']:.3f} cov@95={m['coverage_at_95_acc']['coverage']:.3f} "
            f"cov@98={m['coverage_at_98_acc']['coverage']:.3f} "
            f"oos_other={oos.get('frac_predicted_other', float('nan')):.2f} "
            f"p50={lat.get('p50', float('nan')):.2f}ms")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+", help="pred_<name>.jsonl[=Label]")
    ap.add_argument("--eval", default=str(ROOT / "data" / "eval.jsonl"))
    ap.add_argument("--out", help="write {name: metrics} JSON here")
    ap.add_argument("--charts", help="directory for PNG charts")
    ap.add_argument("--coverage-chart", default="accuracy_vs_coverage.png",
                    help="file name of the combined chart inside --charts")
    args = ap.parse_args(argv)

    gold = {r["id"]: r for r in read_jsonl(args.eval)}
    entries = []
    for spec in args.preds:
        path, _, label = spec.partition("=")
        name = Path(path).stem.removeprefix("pred_")
        m = evaluate(read_jsonl(path), gold)
        m["file"] = path
        m["label"] = label or LABELS.get(name, name)
        entries.append((name, m["label"], m))
        print(summary_line(name, m))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({n: m for n, _, m in entries}, indent=1) + "\n")
        print(f"wrote {args.out}")
    if args.charts:
        Path(args.charts).mkdir(parents=True, exist_ok=True)
        colours = assign_colours([n for n, _, _ in entries])
        for n, label, m in entries:
            print("wrote", chart_reliability(n, label, m, args.charts, colours[n]))
        print("wrote", chart_coverage(entries, args.charts, colours, args.coverage_chart))


if __name__ == "__main__":
    sys.exit(main())
