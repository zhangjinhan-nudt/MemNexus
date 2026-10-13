"""Freeze a second 100-question LoCoMo cohort disjoint from the first one."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from locomo_100_protocol import CATEGORY_NAMES, DATASET, MAX_SESSION, QUOTAS, ROOT

OUTPUT = ROOT / "result" / "locomo_100_cohort_v2.json"
PREVIOUS = ROOT / "result" / "locomo_100_cohort.json"


def evidence_ids(qa: dict) -> list[str]:
    return [
        f"D{int(session)}:{int(turn)}"
        for entry in qa.get("evidence", [])
        for session, turn in re.findall(r"D(\d+):(\d+)", str(entry))
    ]


def available_ids(sample: dict) -> set[str]:
    return {
        str(turn["dia_id"])
        for key, turns in sample["conversation"].items()
        if key.startswith("session_")
        and isinstance(turns, list)
        and int(key.split("_")[1]) <= MAX_SESSION
        for turn in turns
    }


def main() -> None:
    max_session = 5
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    excluded = {(item["sample_id"], item["qa_index"]) for item in previous["questions"]}
    candidates: dict[int, list[dict]] = {category: [] for category in CATEGORY_NAMES}
    for sample_index, sample in enumerate(data):
        valid_turns = {
            str(turn["dia_id"])
            for key, turns in sample["conversation"].items()
            if key.startswith("session_")
            and isinstance(turns, list)
            and int(key.split("_")[1]) <= max_session
            for turn in turns
        }
        sample_id = sample.get("sample_id", str(sample_index))
        for qa_index, qa in enumerate(sample["qa"]):
            key = (sample_id, qa_index)
            references = evidence_ids(qa)
            if (
                key in excluded
                or not references
                or not all(ref in valid_turns for ref in references)
            ):
                continue
            if qa["category"] == 5 and not qa.get("adversarial_answer"):
                continue
            if qa["category"] != 5 and qa.get("answer") is None:
                continue
            candidates[qa["category"]].append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "category_id": qa["category"],
                    "category": CATEGORY_NAMES[qa["category"]],
                    "question": qa["question"],
                    "evidence": references,
                }
            )

    selected: list[dict] = []
    for category, target in QUOTAS.items():
        buckets: dict[int, list[dict]] = {}
        for item in candidates[category]:
            buckets.setdefault(item["sample_index"], []).append(item)
        for items in buckets.values():
            items.sort(key=lambda item: item["qa_index"])
        while sum(item["category_id"] == category for item in selected) < target:
            progressed = False
            for sample_index in sorted(buckets):
                if (
                    buckets[sample_index]
                    and sum(item["category_id"] == category for item in selected)
                    < target
                ):
                    selected.append(buckets[sample_index].pop(0))
                    progressed = True
            if not progressed:
                raise ValueError(
                    f"Category {category} has fewer than {target} disjoint candidates"
                )

    selected.sort(key=lambda item: (item["sample_index"], item["qa_index"]))
    keys = {(item["sample_id"], item["qa_index"]) for item in selected}
    assert len(selected) == 100 and keys.isdisjoint(excluded)
    dataset_sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    selection_sha = hashlib.sha256(
        json.dumps(sorted(keys), separators=(",", ":")).encode()
    ).hexdigest()
    cohort = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_sha,
        "selection_sha256": selection_sha,
        "protocol": "official evidence fully within sessions D1-D2; same quotas and round-robin as cohort v1; disjoint question keys",
        "max_session": max_session,
        "question_count": 100,
        "categories": dict(Counter(item["category"] for item in selected)),
        "conversation_count": len({item["sample_id"] for item in selected}),
        "excluded_selection_sha256": previous["selection_sha256"],
        "excluded_question_count": len(excluded),
        "questions": selected,
    }
    OUTPUT.write_text(
        json.dumps(cohort, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selection_sha256": selection_sha,
                "excluded_selection_sha256": previous["selection_sha256"],
                "categories": cohort["categories"],
                "conversation_count": cohort["conversation_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
