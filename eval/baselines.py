"""Classical baselines for the support-triage experiment.

Writes results/pred_<method>.jsonl for each method, one line per eval example:
  {"id", "pred", "probs": {queue: p}, "confidence", "latency_ms"}
and prints accuracy and macro-F1.

  python -I eval/baselines.py                      # all three methods
  python -I eval/baselines.py --methods keyword    # just one

Methods
  keyword        Hand-written regex rules per queue (see KEYWORD_RULES). Written from the
                 queue briefs in queues.json, general banking-support knowledge and a look at
                 train_fewshot20.jsonl only; eval.jsonl texts were not read while writing them.
  tfidf_lr_20    TF-IDF (word 1-2 grams + char_wb 3-5 grams, sublinear tf) + logistic
                 regression, C chosen by 5-fold stratified CV (log loss) on train_fewshot20.
  tfidf_lr_full  The same pipeline trained on train_full (Banking77 train + oos_train).

Keyword confidence (a rules engine has no real probability, so we define one):
  * each rule that fires adds its weight to its queue's score;
  * probs = score / total score over queues, so when exactly one queue matches it gets
    probability 1.0 and confidence 1.0; when several match, confidence is the winning
    queue's share of the matched weight;
  * when no rule fires the message falls back to "other" with probs {"other": 1.0}
    (confidence 1.0) - the rules cannot tell "no evidence" from "certain".
  Ties are broken by TIE_BREAK order, which mirrors the routing rules in the queue briefs
  (fees first, then unrecognised activity / security, then ATM, top-up, payments, cards).
"""
import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
QUEUES = list(json.loads((DATA / "queues.json").read_text()).keys())
SEED = 7

# ---------------------------------------------------------------------------
# (a) keyword rules: (queue, weight, regex). Weight 3 = phrase that names the
# queue's topic directly; weight 1-2 = generic word that only hints at it;
# weight 4-5 = the brief says this signal wins over others ("any fee or charge",
# "anything the customer does not recognise", "a declined cash withdrawal").
# ---------------------------------------------------------------------------
NOT_RECOGNISED = (
    r"(did ?n[o']?t|do ?n[o']?t|never|not) (make|made|do|did|authori[sz]e|recogni[sz]e|set up|setup|request)"
    r"|(not|un)recogni[sz]ed?|unauthori[sz]ed|fraud\w*|someone (else )?(has )?(used|using|use)"
    r"|(used|using) by (someone|somebody)|(may|might) be (using|used)|without (my )?(permission|consent)|somebody (else )?(has )?(used|using)"
)
KEYWORD_RULES = [
    # fees_exchange - any fee or charge, exchange rates, currencies
    ("fees_exchange", 5, r"\bfees?\b|\bcharges?\b(?! ?back)|\bcharged\b(?! twice| two times| double| again)|surcharge|commission"),
    ("fees_exchange", 3, r"\bcosts?\b|how much (is|does|do|will)|\bfree\b|\bprice\b"),
    ("fees_exchange", 5, r"exchange rate|\brates?\b|exchang\w*|convert\w*|conversion"),
    ("fees_exchange", 2, r"currenc(y|ies)|\bfiat\b|\b(gbp|usd|eur|aud|euros?|dollars?|pounds?)\b|swift"),
    # account_security - identity, personal details, account closure, lost/stolen, unrecognised activity
    ("account_security", 5, NOT_RECOGNISED),
    ("account_security", 3, r"\blost\b|\bstolen\b|\bstole\b|\bmugged\b|\brobbed\b|\btheft\b|compromis\w*|\bhack\w*"),
    ("account_security", 3, r"identity|\bid\b|identification|passport|driv\w* licen[cs]e|source of (my )?funds?|where my (funds|money) come"),
    ("account_security", 2, r"\bverif\w*"),
    ("account_security", 3, r"passcode|password|log ?in|locked out|\bsign in"),
    ("account_security", 3, r"(personal|my) (details|information|info|address|name)|change my (details|address|name)|update my"),
    ("account_security", 3, r"\bage\b|how old|years old|under ?18|\bminimum age"),
    ("account_security", 3, r"(close|delete|cancel|terminate|remove) (my |the |this )?account|account (closure|deletion)"),
    ("account_security", 1, r"\bphone\b|\bsecur\w*|\bblock (my|the) card"),
    # cash_atm - ATM withdrawals
    ("cash_atm", 3, r"\batms?\b|cash ?machine|cashpoint|cash point"),
    ("cash_atm", 4, r"withdr[ae]w\w*|withdrawl\w*|\bwd\b|take out (cash|money)|took out (cash|money)|get cash|got cash"),
    ("cash_atm", 2, r"\bcash\b|dispens\w*"),
    ("cash_atm", 3, r"(ate|eaten|swallow\w*|kept|keep|took|taken|retain\w*|stuck)\b.{0,20}\bcard|card.{0,20}\b(ate|eaten|swallow\w*|kept|stuck|retained)\b"),
    # top_up_balance - adding money to your own account
    ("top_up_balance", 3, r"top.?ups?|topping.?up|topped.?up|auto.?top"),
    ("top_up_balance", 3, r"(add|adding|added|put|putting|load|loading|deposit\w*|get) (more |some )?(money|funds|cash)( in(to)?| to)?"),
    ("top_up_balance", 3, r"apple ?pay|google ?pay|\bcheques?\b|\bchecks?\b(?= (deposit|in|into))|deposit\w*"),
    ("top_up_balance", 3, r"balance.{0,30}(not|n't|hasn'?t|has not|didn'?t|did not)\s*(been )?(update|change|show|increase|go up)|(not|n't) (been )?(updated|reflected) .{0,20}balance"),
    ("top_up_balance", 3, r"bank transfer"),
    ("top_up_balance", 1, r"\bbalance\b|into my account|to my account"),
    # payments_transfers - sending/receiving money, card payment status, refunds
    ("payments_transfers", 3, r"transfer\w*|send(ing)? money|sent money|\bwire\b"),
    ("payments_transfers", 3, r"beneficiar\w*|payee|recipient|\bfriend\b|someone (sent|send)|receiv\w* (a |the |my )?(money|payment|transfer|funds)"),
    ("payments_transfers", 3, r"refund\w*|money back|reverse\w*|revert\w*|returned"),
    ("payments_transfers", 3, r"charged (twice|two times|double|again)|duplicate\w*|twice|double (charge|payment)"),
    ("payments_transfers", 2, r"pending|still processing|not (yet )?(gone|went|go) through|hasn'?t (arrived|gone through)|cancel\w*|\bfailed\b"),
    ("payments_transfers", 1, r"\bpayments?\b|transactions?|\bpaid\b|\bpay\b"),
    # card_services - getting and using a card
    ("card_services", 3, r"\bpin\b|activat\w*|reactivat\w*"),
    ("card_services", 3, r"virtual card|disposable|physical card|new card|replacement card|replace (my |the )?card|order (a |another |my )?(new )?card|extra card|another card|additional card"),
    ("card_services", 3, r"(card|it).{0,30}(arriv\w*|deliver\w*|mail\w*|post\w*|ship\w*|come)|(deliver\w*|arriv\w*|mailed|shipping)"),
    ("card_services", 3, r"expir\w*|contactless|\bvisa\b|mastercard|accept\w*|\bcountr(y|ies)\b|\bwhere can i use"),
    ("card_services", 3, r"declin\w*|rejected|won'?t work|not working|does ?n[o']?t work|did ?n[o']?t work|stopped working|can'?t use (my |the )?card|unable to use"),
    ("card_services", 3, r"(link|add|connect)\w* (my |a |the )?(\w+ )?card|card.{0,20}(to|in) (the )?app"),
    ("card_services", 1, r"\bcards?\b"),
]
TIE_BREAK = ["fees_exchange", "account_security", "cash_atm", "top_up_balance",
             "payments_transfers", "card_services", "other"]
_COMPILED = [(q, w, re.compile(p, re.IGNORECASE)) for q, w, p in KEYWORD_RULES]


def keyword_predict(text):
    scores = {q: 0.0 for q in QUEUES}
    for q, w, rx in _COMPILED:
        if rx.search(text):
            scores[q] += w
    total = sum(scores.values())
    if total == 0:
        probs = {q: (1.0 if q == "other" else 0.0) for q in QUEUES}
        return "other", probs
    probs = {q: scores[q] / total for q in QUEUES}
    best = max(probs.values())
    pred = next(q for q in TIE_BREAK if probs[q] == best)
    return pred, probs


# ---------------------------------------------------------------------------
# (b), (c) TF-IDF + logistic regression
# ---------------------------------------------------------------------------
C_GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def make_pipeline():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                 lowercase=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                 lowercase=True, min_df=1)),
    ])
    return Pipeline([("tfidf", feats),
                     ("lr", LogisticRegression(max_iter=5000, random_state=SEED))])


def train_tfidf_lr(train_rows):
    X = [r["text"] for r in train_rows]
    y = [r["queue"] for r in train_rows]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(make_pipeline(), {"lr__C": C_GRID}, cv=cv, scoring="neg_log_loss",
                      n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)
    cv_info = {
        "best_C": gs.best_params_["lr__C"],
        "cv_scoring": "neg_log_loss",
        "cv_mean_by_C": {str(c): round(float(s), 4)
                         for c, s in zip(C_GRID, gs.cv_results_["mean_test_score"])},
        "n_train": len(X),
    }
    return gs.best_estimator_, cv_info


# ---------------------------------------------------------------------------
def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def run_method(name, eval_rows):
    info = {}
    if name == "keyword":
        info = {"n_rules": len(KEYWORD_RULES)}
        predict = keyword_predict
        keyword_predict("warm up")
    else:
        train_file = {"tfidf_lr_20": "train_fewshot20.jsonl",
                      "tfidf_lr_full": "train_full.jsonl"}[name]
        t0 = time.perf_counter()
        model, info = train_tfidf_lr(read_jsonl(DATA / train_file))
        info["train_file"] = train_file
        info["train_seconds"] = round(time.perf_counter() - t0, 2)
        classes = list(model.classes_)
        assert set(classes) == set(QUEUES), classes

        def predict(text):
            p = model.predict_proba([text])[0]
            probs = {q: 0.0 for q in QUEUES}
            probs.update({c: float(v) for c, v in zip(classes, p)})
            best = max(probs.values())
            return next(q for q in QUEUES if probs[q] == best), probs

        predict("warm up")

    out = RESULTS / f"pred_{name}.jsonl"
    preds = []
    with out.open("w") as f:
        for r in eval_rows:
            t0 = time.perf_counter()
            pred, probs = predict(r["text"])
            ms = (time.perf_counter() - t0) * 1000
            row = {"id": r["id"], "pred": pred,
                   "probs": {q: round(probs[q], 6) for q in QUEUES},
                   "confidence": round(probs[pred], 6), "latency_ms": round(ms, 4)}
            preds.append(pred)
            f.write(json.dumps(row) + "\n")
    y = [r["queue"] for r in eval_rows]
    acc = accuracy_score(y, preds)
    mf1 = f1_score(y, preds, labels=QUEUES, average="macro", zero_division=0)
    info.update({"accuracy": round(acc, 4), "macro_f1": round(mf1, 4), "out": str(out.relative_to(ROOT))})
    print(f"{name:14s} accuracy={acc:.4f} macro_f1={mf1:.4f}  -> {out.relative_to(ROOT)}  {json.dumps({k: v for k, v in info.items() if k not in ('accuracy', 'macro_f1', 'out')})}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["keyword", "tfidf_lr_20", "tfidf_lr_full"])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    eval_rows = read_jsonl(DATA / "eval.jsonl")
    infos = {m: run_method(m, eval_rows) for m in args.methods}
    info_path = RESULTS / "baselines_info.json"
    old = json.loads(info_path.read_text()) if info_path.exists() else {}
    old.update(infos)
    info_path.write_text(json.dumps(old, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
