"""Cohort: every eligible question in conv-48/49/50 at max_session=30.

These three conversations have never been benchmarked. Unlike the sampling
cohorts (locomo_cohort.py), this set takes the full eligible pool of the three
target conversations - no quota, no round-robin - because the goal is complete
coverage of unseen dialogue, not a sample of the whole dataset.

Eligibility rules are identical to locomo_cohort.build: every gold evidence
turn must resolve inside sessions D1..D30, distractors need their adversarial
answer, and other questions need an answer.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from locomo_cohort import CATEGORY_NAMES, DATASET, available_ids, evidence_ids

TARGETS = ("conv-48", "conv-49", "conv-50")
MAX_SESSION = 30
OUT = ROOT / "result" / "locomo_637_cohort_conv48-50_20261023.json"


def main() -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    selected: list[dict] = []
    for sample_index, sample in enumerate(data):
        if sample.get("sample_id") not in TARGETS:
            continue
        ids = available_ids(sample, MAX_SESSION)
        for qa_index, qa in enumerate(sample["qa"]):
            references = evidence_ids(qa)
            if not references or not all(ref in ids for ref in references):
                continue
            if qa["category"] == 5 and not qa.get("adversarial_answer"):
                continue
            if qa["category"] != 5 and qa.get("answer") is None:
                continue
            selected.append(
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
    selected.sort(key=lambda item: (item["sample_index"], item["qa_index"]))
    assert len({(x["sample_id"], x["qa_index"]) for x in selected}) == len(selected)
    signature = hashlib.sha256(
        json.dumps([(x["sample_id"], x["qa_index"]) for x in selected], separators=(",", ":")).encode()
    ).hexdigest()
    cohort = {
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "selection_sha256": signature,
        "protocol": (
            "conv-48/49/50 full coverage; official evidence fully within sessions D1-D30; "
            "no quota sampling - every eligible question included"
        ),
        "max_session": MAX_SESSION,
        "question_count": len(selected),
        "conversation_count": len({x["sample_id"] for x in selected}),
        "categories": dict(Counter(x["category"] for x in selected)),
        "per_conversation": dict(Counter(x["sample_id"] for x in selected)),
        "questions": selected,
    }
    OUT.write_text(json.dumps(cohort, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}: {cohort['question_count']} questions")
    print("categories:", cohort["categories"])
    print("per conversation:", cohort["per_conversation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
