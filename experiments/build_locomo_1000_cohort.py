"""Cohort: 1000-question stratified LoCoMo selection for the overnight run.

Selection is deterministic (seed 20261023): for each conversation we take every
eligible question (evidence inside D1..D30, same rules as locomo_cohort.build),
then downsample the eligible pool to hit 1000 total while keeping the global
category distribution and per-conversation proportionality.

Also emits a LongMemEval cohort of ~1000 questions built from the S + oracle
files (see build_longmemeval_cohort.py); this script only handles LoCoMo.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from locomo_cohort import CATEGORY_NAMES, DATASET, available_ids, evidence_ids

TARGET = 1000
MAX_SESSION = 30
SEED = 20261023
OUT = ROOT / "result" / "locomo_1000_cohort_20261023.json"


def main() -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    eligible: list[dict] = []
    for sample_index, sample in enumerate(data):
        ids = available_ids(sample, MAX_SESSION)
        for qa_index, qa in enumerate(sample["qa"]):
            references = evidence_ids(qa)
            if not references or not all(ref in ids for ref in references):
                continue
            if qa["category"] == 5 and not qa.get("adversarial_answer"):
                continue
            if qa["category"] != 5 and qa.get("answer") is None:
                continue
            eligible.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample["sample_id"],
                    "qa_index": qa_index,
                    "category_id": qa["category"],
                    "category": CATEGORY_NAMES[qa["category"]],
                    "question": qa["question"],
                    "evidence": references,
                }
            )
    eligible.sort(key=lambda item: (item["sample_index"], item["qa_index"]))
    print(f"eligible pool: {len(eligible)}")
    print("pool categories:", dict(Counter(x["category"] for x in eligible)))

    rng = random.Random(SEED)
    if len(eligible) > TARGET:
        # Proportional quota per category, then per conversation inside a
        # category, so the 1000-question set mirrors the eligible pool.
        by_cat: dict[str, list[dict]] = {}
        for item in eligible:
            by_cat.setdefault(item["category"], []).append(item)
        selected = []
        remaining = TARGET
        cats = sorted(by_cat)
        quotas = {c: len(by_cat[c]) * TARGET // len(eligible) for c in cats}
        quotas[cats[-1]] += TARGET - sum(quotas.values())
        for cat in cats:
            pool = by_cat[cat]
            quota = quotas[cat]
            if len(pool) <= quota:
                chosen = list(pool)
            else:
                chosen = rng.sample(pool, quota)
            selected.extend(chosen)
            remaining -= len(chosen)
        # If any conversation quota broke totals, top up from leftovers.
        if remaining > 0:
            chosen_keys = {(x["sample_id"], x["qa_index"]) for x in selected}
            leftovers = [x for x in eligible if (x["sample_id"], x["qa_index"]) not in chosen_keys]
            selected.extend(rng.sample(leftovers, remaining))
        eligible = selected
    eligible.sort(key=lambda item: (item["sample_index"], item["qa_index"]))
    assert len({(x["sample_id"], x["qa_index"]) for x in eligible}) == len(eligible)

    signature = hashlib.sha256(
        json.dumps([(x["sample_id"], x["qa_index"]) for x in eligible], separators=(",", ":")).encode()
    ).hexdigest()
    cohort = {
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "selection_sha256": signature,
        "protocol": (
            f"stratified {len(eligible)}-question selection with seed {SEED}; official evidence "
            f"fully within sessions D1-D{MAX_SESSION}; category and per-conversation "
            "distribution proportional to the eligible pool"
        ),
        "max_session": MAX_SESSION,
        "question_count": len(eligible),
        "conversation_count": len({x["sample_id"] for x in eligible}),
        "categories": dict(Counter(x["category"] for x in eligible)),
        "per_conversation": dict(Counter(x["sample_id"] for x in eligible)),
        "questions": eligible,
    }
    OUT.write_text(json.dumps(cohort, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}: {cohort['question_count']} questions")
    print("categories:", cohort["categories"])
    print("per conversation:", cohort["per_conversation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
