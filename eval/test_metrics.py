"""Sanity checks for eval/metrics.py.  python -I eval/test_metrics.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import metrics as M  # noqa: E402

Q = M.QUEUES
rng = np.random.default_rng(0)
gold = {}
for i in range(700):
    q = Q[i % len(Q)]
    gold[f"x{i}"] = {"id": f"x{i}", "queue": q, "source": "clinc_oos" if q == "other" else "banking77",
                     "ambiguous": i % 5 == 0, "self_serve": None if q == "other" else bool(i % 2)}


def onehot(q):
    return {k: float(k == q) for k in Q}


# 1. perfect, fully confident predictor
perfect = [{"id": k, "pred": g["queue"], "probs": onehot(g["queue"]), "confidence": 1.0,
            "latency_ms": 1.0, "self_serve_p": None if g["self_serve"] is None else float(g["self_serve"])}
           for k, g in gold.items()]
m = M.evaluate(perfect, gold)
assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["accuracy_excl_ambiguous"] == 1
assert m["ece"] == 0 and m["brier"] == 0 and m["aurc"] == 0
assert m["coverage_at_95_acc"]["coverage"] == 1 and m["coverage_at_98_acc"]["coverage"] == 1
assert m["oos"]["frac_predicted_other"] == 1 and m["oos"]["mean_confidence_missed"] is None
assert m["self_serve"]["auroc"] == 1 and m["self_serve"]["accuracy_at_0.5"] == 1
assert all(s["coverage"] == 1 and s["accuracy"] == 1 for s in m["selective"])
assert np.trace(np.array(m["confusion"]["rows_true_cols_pred"])) == 700
print("perfect predictor: OK")

# 2. always wrong at confidence 1: ECE 1, Brier 2, accuracy 0, AURC 1
wrong = [{"id": k, "pred": Q[(Q.index(g["queue"]) + 1) % 7], "probs": onehot(Q[(Q.index(g["queue"]) + 1) % 7]),
          "confidence": 1.0} for k, g in gold.items()]
m = M.evaluate(wrong, gold)
assert m["accuracy"] == 0 and m["ece"] == 1 and m["brier"] == 2 and m["aurc"] == 1
assert m["coverage_at_95_acc"]["coverage"] == 0 and m["oos"]["frac_predicted_other"] == 0
assert m["oos"]["mean_confidence_missed"] == 1
print("always-wrong predictor: OK")

# 3. uniform probs, 1/7 confidence, random preds: ECE ~ |acc - 1/7| small; Brier = 6/7
uni = [{"id": k, "pred": Q[rng.integers(7)], "probs": {q: 1 / 7 for q in Q}, "confidence": 1 / 7}
       for k in gold]
m = M.evaluate(uni, gold)
assert abs(m["brier"] - 6 / 7) < 1e-3, m["brier"]
assert abs(m["ece"] - abs(m["accuracy"] - 1 / 7)) < 1e-3
print("uniform predictor: OK")

# 4. calibrated synthetic predictor: confidence c, correct w.p. c -> ECE near 0
ids = list(gold)
conf = rng.uniform(0.3, 1.0, 20000)
corr = rng.uniform(size=20000) < conf
assert M.ece(conf, corr) < 0.02, M.ece(conf, corr)
# perfectly ranked confidence beats reversed ranking on AURC
c2 = np.r_[np.full(50, 0.9), np.full(50, 0.4)]
k2 = np.r_[np.ones(50, bool), np.zeros(50, bool)]
assert M.aurc(c2, k2) < M.aurc(c2, ~k2)
assert M.coverage_at_accuracy(c2, k2, 0.95) == {"coverage": 0.5, "threshold": 0.9}
print("calibrated synthetic / AURC ordering: OK")

# 5. AUROC matches sklearn, with ties
from sklearn.metrics import roc_auc_score, f1_score  # noqa: E402
s = rng.integers(0, 5, 500) / 4
y = rng.uniform(size=500) < 0.3 + 0.4 * s
assert math.isclose(M.auroc(s, y), roc_auc_score(y, s), abs_tol=1e-12)
yt = [Q[i] for i in rng.integers(0, 7, 400)]
yp = [Q[i] for i in rng.integers(0, 7, 400)]
assert math.isclose(M.macro_f1(yt, yp), f1_score(yt, yp, labels=Q, average="macro"), abs_tol=1e-12)
print("AUROC / macro-F1 vs sklearn: OK")
print("ALL OK")
