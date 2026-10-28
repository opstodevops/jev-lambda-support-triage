"""Build the labelled evaluation data for the confidence-aware triage experiment.

    python -I eval/prepare_data.py            # from the repo root

Inputs that are checked in (hand-authored, NOT regenerated here):
    data/queues.json                 queue id -> description (fed verbatim to the models)
    data/intent_map.json             Banking77 intent -> {queue, self_serve, ambiguous, rationale}
    data/clinc_oos_exclusions.json   CLINC150 out-of-scope texts that look like banking requests
    data/reworded_questions.json     alternative instruction / queue wordings (validated only)

Raw downloads (re-downloaded if missing, treated as untrusted data):
    data/raw/banking77/{train,test}.csv   PolyAI-LDN/task-specific-datasets (CC-BY-4.0)
    data/raw/clinc150/data_full.json      clinc/oos-eval (CC-BY-3.0)

Outputs: data/eval.jsonl, data/eval_small.jsonl, data/train_fewshot20.jsonl,
         data/train_full.jsonl, data/stats.json

All sampling uses random.Random(SEED) with SEED = 7. Each output gets its own fresh
Random(SEED) instance and inputs are sorted deterministically before sampling, so
every file is reproducible on its own.
"""
import collections
import csv
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

SEED = 7
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_B77 = DATA / "raw" / "banking77"
RAW_CLINC = DATA / "raw" / "clinc150"

B77_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
SOURCES = {
    RAW_B77 / "train.csv": f"{B77_BASE}/train.csv",
    RAW_B77 / "test.csv": f"{B77_BASE}/test.csv",
    RAW_CLINC / "data_full.json": CLINC_URL,
}
# SHA-256 of the files used for the published results (2026-10-14). A mismatch is reported, not fatal.
SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "data_full.json": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
}

EVAL_PER_INTENT = 10
EVAL_OOS = 100
FEWSHOT_PER_QUEUE = 20
SMALL_PER_QUEUE = 20


def fetch_raw():
    for path, url in SOURCES.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=120) as r:
                path.write_bytes(r.read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if SHA256.get(path.name) != digest:
            print(f"WARNING: {path.name} sha256 {digest} differs from the published build", file=sys.stderr)


def load_banking77(split):
    rows = list(csv.DictReader((RAW_B77 / f"{split}.csv").open(newline="", encoding="utf-8")))
    return [{"idx": i, "text": r["text"].strip(), "intent": r["category"]} for i, r in enumerate(rows)]


def load_clinc_oos(split, excluded):
    data = json.loads((RAW_CLINC / "data_full.json").read_text(encoding="utf-8"))
    rows = [{"idx": i, "text": t.strip()} for i, (t, lab) in enumerate(data[split]) if lab == "oos"]
    texts = {r["text"] for r in rows}
    missing = [t for t in excluded if t not in texts]
    if missing:
        raise SystemExit(f"exclusion texts not found in {split}: {missing}")
    return [r for r in rows if r["text"] not in set(excluded)]


def b77_record(r, split, intent_map):
    m = intent_map[r["intent"]]
    return {"id": f"b77-{split}-{r['idx']:05d}", "text": r["text"], "source": "banking77",
            "intent": r["intent"], "queue": m["queue"], "self_serve": m["self_serve"],
            "ambiguous": m["ambiguous"]}


def oos_record(r, split):
    # ambiguous=False: near-domain OOS texts were removed via clinc_oos_exclusions.json.
    return {"id": f"clinc-{split}-{r['idx']:04d}", "text": r["text"], "source": "clinc_oos",
            "intent": "oos", "queue": "other", "self_serve": None, "ambiguous": False}


def round_robin(rng, groups, n):
    """Take n items spread as evenly as possible over groups (dict key -> list).

    Each pass visits the groups in a shuffled order and takes one random unused item
    from each group that still has items, until n items are taken.
    """
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(groups.items())}
    out = []
    while len(out) < n and any(pools.values()):
        keys = [k for k in sorted(pools) if pools[k]]
        rng.shuffle(keys)
        for k in keys:
            if len(out) == n:
                break
            out.append(pools[k].pop())
    if len(out) < n:
        raise SystemExit(f"not enough items: wanted {n}, got {len(out)}")
    return out


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def counts(rows):
    c = collections.Counter(r["queue"] for r in rows)
    return {q: c.get(q, 0) for q in QUEUES} | {"total": len(rows)}


def main():
    global QUEUES
    QUEUES = list(json.loads((DATA / "queues.json").read_text()))
    intent_map = json.loads((DATA / "intent_map.json").read_text())
    excl = json.loads((DATA / "clinc_oos_exclusions.json").read_text())
    reworded = json.loads((DATA / "reworded_questions.json").read_text())

    fetch_raw()
    train, test = load_banking77("train"), load_banking77("test")
    oos_train = load_clinc_oos("oos_train", excl["oos_train"])
    oos_test = load_clinc_oos("oos_test", excl["oos_test"])

    # Validate the checked-in mapping against the data.
    intents = sorted({r["intent"] for r in train} | {r["intent"] for r in test})
    assert len(intents) == 77 and set(intents) == set(intent_map), "intent_map does not match Banking77 intents"
    assert all(m["queue"] in QUEUES and m["queue"] != "other" for m in intent_map.values())
    assert set(reworded["queues_alt"]) == set(QUEUES) and len(reworded["instructions"]) == 3

    # eval.jsonl: 10 per intent from Banking77 test + 100 CLINC oos_test, then shuffled.
    rng = random.Random(SEED)
    by_intent = collections.defaultdict(list)
    for r in test:
        by_intent[r["intent"]].append(r)
    ev = [b77_record(r, "test", intent_map)
          for intent in intents for r in rng.sample(by_intent[intent], EVAL_PER_INTENT)]
    ev += [oos_record(r, "oos_test") for r in rng.sample(oos_test, EVAL_OOS)]
    rng.shuffle(ev)
    write_jsonl(DATA / "eval.jsonl", ev)

    # train_fewshot20.jsonl: 20 per banking queue, spread across that queue's intents, + 20 oos_train.
    rng = random.Random(SEED)
    by_q_intent = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in train:
        by_q_intent[intent_map[r["intent"]]["queue"]][r["intent"]].append(r)
    few = []
    for q in QUEUES:
        if q == "other":
            few += [oos_record(r, "oos_train") for r in rng.sample(oos_train, FEWSHOT_PER_QUEUE)]
        else:
            few += [b77_record(r, "train", intent_map)
                    for r in round_robin(rng, by_q_intent[q], FEWSHOT_PER_QUEUE)]
    write_jsonl(DATA / "train_fewshot20.jsonl", few)

    # train_full.jsonl: all Banking77 train + all (non-excluded) oos_train.
    full = [b77_record(r, "train", intent_map) for r in train] + [oos_record(r, "oos_train") for r in oos_train]
    write_jsonl(DATA / "train_full.jsonl", full)

    # eval_small.jsonl: 20 per queue drawn from eval.jsonl, spread across intents; kept in eval.jsonl order.
    rng = random.Random(SEED)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sorted(ev, key=lambda r: r["id"]):
        groups[r["queue"]][r["intent"]].append(r)
    chosen = set()
    for q in QUEUES:
        chosen |= {r["id"] for r in round_robin(rng, groups[q], SMALL_PER_QUEUE)}
    small = [r for r in ev if r["id"] in chosen]
    write_jsonl(DATA / "eval_small.jsonl", small)

    amb = sorted(i for i, m in intent_map.items() if m["ambiguous"])
    stats = {
        "seed": SEED,
        "eval": counts(ev),
        "eval_ambiguous_rows": sum(bool(r["ambiguous"]) for r in ev),
        "eval_self_serve_true_rows": sum(r["self_serve"] is True for r in ev),
        "eval_small": counts(small),
        "train_fewshot20": counts(few),
        "train_full": counts(full),
        "intents_per_queue": counts([{"queue": m["queue"]} for m in intent_map.values()]),
        "ambiguous_intents": amb,
        "n_ambiguous_intents": len(amb),
        "self_serve_intents": sum(m["self_serve"] for m in intent_map.values()),
        "clinc_oos_excluded": {"oos_test": len(excl["oos_test"]), "oos_train": len(excl["oos_train"])},
        "duplicate_texts_in_eval": len(ev) - len({r["text"].lower() for r in ev}),
        "eval_texts_also_in_train_full": len({r["text"].lower() for r in ev} & {r["text"].lower() for r in full}),
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
