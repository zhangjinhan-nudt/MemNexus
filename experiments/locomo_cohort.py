"""Deterministic LoCoMo cohort of any size, scoped to an evidence horizon.

Same rules as the frozen 100-question cohort, only parameterised: a question is
eligible when every gold evidence turn resolves inside sessions D1..D<max_session>
of its own conversation, distractors need their adversarial answer, and answers
must exist otherwise. Selection is round-robin across conversations inside each
official category, so a large cohort cannot be dominated by one long dialogue.

The cohort is a *question set*, not a filter on system output: no gold answer is
passed to either memory system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"

# Official LoCoMo categories. The evaluator keys on these ids, so the label must
# follow the id and never the other way round.
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def evidence_ids(qa: dict) -> list[str]:
    return [
        f"D{int(session)}:{int(turn)}"
        for entry in qa.get("evidence", [])
        for session, turn in re.findall(r"D(\d+):(\d+)", str(entry))
    ]


def available_ids(sample: dict, max_session: int) -> set[str]:
    return {
        str(turn["dia_id"])
        for key, turns in sample["conversation"].items()
        if key.startswith("session_")
        and isinstance(turns, list)
        and int(key.split("_")[1]) <= max_session
        for turn in turns
    }


def build(size: int, max_session: int, seed_note: str) -> dict:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    candidates: dict[int, list[dict]] = {category: [] for category in CATEGORY_NAMES}
    for sample_index, sample in enumerate(data):
        ids = available_ids(sample, max_session)
        for qa_index, qa in enumerate(sample["qa"]):
            references = evidence_ids(qa)
            if not references or not all(ref in ids for ref in references):
                continue
            if qa["category"] == 5 and not qa.get("adversarial_answer"):
                continue
            if qa["category"] != 5 and qa.get("answer") is None:
                continue
            candidates[qa["category"]].append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample.get("sample_id", str(sample_index)),
                    "qa_index": qa_index,
                    "category_id": qa["category"],
                    "category": CATEGORY_NAMES[qa["category"]],
                    "question": qa["question"],
                    "evidence": references,
                }
            )

    pool = sum(len(items) for items in candidates.values())
    if pool < size:
        raise ValueError(f"only {pool} eligible questions, requested {size}")

    # Quotas follow the eligible pool, so a category is never inflated beyond
    # what the horizon can support; the remainder goes to the largest pools.
    quotas = {category: int(size * len(items) / pool) for category, items in candidates.items()}
    for category in sorted(candidates, key=lambda key: -len(candidates[key])):
        if sum(quotas.values()) >= size:
            break
        quotas[category] += 1
    while sum(quotas.values()) > size:
        category = max(quotas, key=lambda key: (quotas[key], -len(candidates[key])))
        quotas[category] -= 1
    for category, target in quotas.items():
        if target == 0:
            continue
        if len(candidates[category]) < target:
            raise ValueError(
                f"category {category}: only {len(candidates[category])} eligible, need {target}"
            )

    selected: list[dict] = []
    buckets: dict[int, dict[int, list[dict]]] = {
        category: {} for category in CATEGORY_NAMES
    }
    for category, items in candidates.items():
        for item in items:
            buckets[category].setdefault(item["sample_index"], []).append(item)
    taken = Counter()
    # One item per conversation per turn, cycling with a cursor. Taking from the
    # first non-empty conversation instead would drain the earliest dialogues
    # and leave the last ones unrepresented, which is exactly the bias the
    # round-robin is there to remove.
    cursor = {category: 0 for category in CATEGORY_NAMES}
    while len(selected) < size:
        progressed = False
        for category in CATEGORY_NAMES:
            if taken[category] >= quotas[category]:
                continue
            order = sorted(buckets[category])
            for step in range(len(order)):
                position = (cursor[category] + step) % len(order)
                queue = buckets[category][order[position]]
                if not queue:
                    continue
                selected.append(queue.pop(0))
                taken[category] += 1
                cursor[category] = (position + 1) % len(order)
                progressed = True
                break
        if not progressed:
            break
    selected.sort(key=lambda item: (item["sample_index"], item["qa_index"]))
    if len({(x["sample_id"], x["qa_index"]) for x in selected}) != len(selected):
        raise AssertionError("cohort must not repeat a question")

    # sample_index stays: both runners group conversations by it.
    signature = hashlib.sha256(
        json.dumps([(x["sample_id"], x["qa_index"]) for x in selected], separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "selection_sha256": signature,
        "protocol": (
            f"{seed_note}; official evidence fully within sessions D1-D{max_session}; "
            "pool-proportional category quotas; conversation round-robin"
        ),
        "max_session": max_session,
        "question_count": len(selected),
        "conversation_count": len({x["sample_id"] for x in selected}),
        "categories": dict(Counter(x["category"] for x in selected)),
        "eligible_pool": {CATEGORY_NAMES[k]: len(v) for k, v in candidates.items()},
        "questions": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--max-session", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--note", default="large comparison cohort")
    args = parser.parse_args()
    cohort = build(args.size, args.max_session, args.note)
    Path(args.out).write_text(json.dumps(cohort, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}: {cohort['question_count']} questions over "
          f"{cohort['conversation_count']} conversations")
    print("categories:", cohort["categories"])
    print("eligible pool:", cohort["eligible_pool"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
