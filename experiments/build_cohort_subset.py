"""Stratified subset of the 637-question conv-48/49/50 cohort.

Keeps the category mix of the full pool (single-hop 43.6%, temporal 16.8%,
adversarial 21.0%, multi-hop 14.1%, open-domain 4.4%) and draws round-robin
across the three conversations so no dialogue dominates a category.

Deterministic: no RNG, the 637 cohort order decides the picks.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

SOURCE = ROOT / "result" / "locomo_637_cohort_conv48-50_20261023.json"
OUT = ROOT / "result" / "locomo_150_cohort_conv48-50_20261023.json"
TARGET = 150


def main() -> int:
    cohort = json.loads(SOURCE.read_text(encoding="utf-8"))
    questions = cohort["questions"]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in questions:
        by_category[item["category"]].append(item)

    pool = len(questions)
    quotas = {name: int(TARGET * len(items) / pool) for name, items in by_category.items()}
    for name in sorted(by_category, key=lambda k: -len(by_category[k])):
        if sum(quotas.values()) >= TARGET:
            break
        quotas[name] += 1
    while sum(quotas.values()) > TARGET:
        name = max(quotas, key=lambda k: (quotas[k], -len(by_category[k])))
        quotas[name] -= 1

    selected: list[dict] = []
    for name, items in by_category.items():
        buckets: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            buckets[item["sample_id"]].append(item)
        order = sorted(buckets)
        cursor = 0
        taken = 0
        while taken < quotas[name]:
            for _ in range(len(order)):
                position = (cursor + _) % len(order)
                queue = buckets[order[position]]
                if queue:
                    selected.append(queue.pop(0))
                    taken += 1
                    cursor = (position + 1) % len(order)
                    break
            else:
                break
    selected.sort(key=lambda item: (item["sample_index"], item["qa_index"]))
    assert len({(x["sample_id"], x["qa_index"]) for x in selected}) == len(selected)
    signature = hashlib.sha256(
        json.dumps([(x["sample_id"], x["qa_index"]) for x in selected], separators=(",", ":")).encode()
    ).hexdigest()
    subset = dict(cohort)
    subset.update({
        "protocol": (
            f"stratified {TARGET}-question subset of the 637-question conv-48/49/50 cohort; "
            "category quotas proportional to the full pool; conversation round-robin within category"
        ),
        "question_count": len(selected),
        "categories": dict(Counter(x["category"] for x in selected)),
        "per_conversation": dict(Counter(x["sample_id"] for x in selected)),
        "selection_sha256": signature,
        "questions": selected,
    })
    OUT.write_text(json.dumps(subset, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}: {subset['question_count']} questions")
    print("categories:", subset["categories"])
    print("per conversation:", subset["per_conversation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
