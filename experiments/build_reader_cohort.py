"""42-question reader A/B cohort drawn from the 150-question conv-48/49/50 run.

Adversarial questions are excluded on purpose: the reader fallback replaces
abstentions, and on adversarial questions an abstention is the correct answer.

Quotas target the measured bottleneck (evidence-in-hand generation failures):
multi-hop and temporal are over-sampled relative to their share, single-hop is
the bulk, open-domain is small.  Deterministic: cohort order decides the picks,
round-robin across the three conversations.

Usage: python experiments/build_reader_cohort.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "result" / "locomo_150_cohort_conv48-50_20261023.json"
OUT = ROOT / "result" / "locomo_42_reader_cohort_20261023.json"

QUOTAS = {"single-hop": 15, "temporal": 10, "multi-hop": 12, "open-domain": 5}


def main() -> int:
    cohort = json.loads(SOURCE.read_text(encoding="utf-8"))
    questions = cohort["questions"]
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in questions:
        if item["category"] in QUOTAS:
            buckets[item["category"]][item["sample_id"]].append(item)

    selected: list[dict] = []
    for name, quota in QUOTAS.items():
        per_conv = buckets[name]
        order = sorted(per_conv)
        cursors = {conv: 0 for conv in order}
        taken = 0
        while taken < quota:
            progressed = False
            for conv in order:
                items = per_conv[conv]
                if cursors[conv] < len(items):
                    selected.append(items[cursors[conv]])
                    cursors[conv] += 1
                    taken += 1
                    progressed = True
                    if taken >= quota:
                        break
            if not progressed:
                break

    selected.sort(key=lambda item: (item["sample_index"], item["qa_index"]))
    keys = [(x["sample_id"], x["qa_index"]) for x in selected]
    assert len(set(keys)) == len(keys)
    # Inherit the 150-cohort metadata (dataset path, hashes, max_session) so the
    # runner and the protocol signature accept this subset unchanged.
    subset = dict(cohort)
    subset.update({
        "protocol": (
            "42-question non-adversarial reader A/B subset of the 150-question "
            "conv-48/49/50 cohort; bottleneck-stratified quotas; "
            "conversation round-robin within category"
        ),
        "questions": selected,
        "question_count": len(selected),
        "categories": dict(Counter(x["category"] for x in selected)),
        "per_conversation": dict(Counter(x["sample_id"] for x in selected)),
    })
    OUT.write_text(json.dumps(subset, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(selected)} questions")
    print("categories:", dict(Counter(x["category"] for x in selected)))
    print("per conversation:", dict(Counter(x["sample_id"] for x in selected)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
