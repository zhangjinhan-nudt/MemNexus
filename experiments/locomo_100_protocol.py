"""Frozen, evidence-scoped 100-question LoCoMo cohort and cost preflight.

This module is read-only. It creates a deterministic cohort from the official
LoCoMo file; no gold answer is passed to either memory system. All selected
evidence references must resolve to raw dialogue turns in sessions D1–D2.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}
QUOTAS = {1: 13, 2: 25, 3: 9, 4: 28, 5: 25}
MAX_SESSION = 2


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


def select_cohort() -> dict:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    candidates: dict[int, list[dict]] = {category: [] for category in CATEGORY_NAMES}
    for sample_index, sample in enumerate(data):
        ids = available_ids(sample)
        for qa_index, qa in enumerate(sample["qa"]):
            references = evidence_ids(qa)
            # Keep malformed/dangling gold mappings out of the cohort. Do not
            # silently repair them or let answer text choose the sample.
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
    for category, target in QUOTAS.items():
        if len(candidates[category]) < target:
            raise ValueError(
                f"Category {category}: only {len(candidates[category])} eligible, need {target}"
            )

    # Round-robin by conversation within each category, then restore official
    # conversation/QA order for execution. This uses indices, never gold text.
    selected = []
    for category, target in QUOTAS.items():
        buckets: dict[int, list[dict]] = {}
        for item in candidates[category]:
            buckets.setdefault(item["sample_index"], []).append(item)
        while (
            len([item for item in selected if item["category_id"] == category]) < target
        ):
            progressed = False
            for sample_index in sorted(buckets):
                if (
                    buckets[sample_index]
                    and len(
                        [item for item in selected if item["category_id"] == category]
                    )
                    < target
                ):
                    selected.append(buckets[sample_index].pop(0))
                    progressed = True
            if not progressed:
                raise AssertionError("Cohort selection exhausted unexpectedly")
    selected.sort(key=lambda item: (item["sample_index"], item["qa_index"]))
    if (
        len(selected) != 100
        or len({(x["sample_id"], x["qa_index"]) for x in selected}) != 100
    ):
        raise AssertionError("Cohort must contain exactly 100 unique questions")
    dataset_digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    signature = hashlib.sha256(
        json.dumps(
            [(x["sample_id"], x["qa_index"]) for x in selected], separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_digest,
        "selection_sha256": signature,
        "protocol": "official evidence fully within sessions D1-D2; category quotas; conversation round-robin",
        "max_session": MAX_SESSION,
        "question_count": len(selected),
        "categories": dict(Counter(x["category"] for x in selected)),
        "conversation_count": len({x["sample_id"] for x in selected}),
        "questions": selected,
    }


def estimate_cost(cohort: dict, max_session: int = MAX_SESSION) -> dict:
    """Scale real English pilot usage; return a planning range in RMB."""
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    sample_indices = sorted({item["sample_index"] for item in cohort["questions"]})
    turns = []
    for index in sample_indices:
        sample = data[index]
        turns.append(
            sum(
                len(value)
                for key, value in sample["conversation"].items()
                if key.startswith("session_")
                and isinstance(value, list)
                and int(key.split("_")[1]) <= max_session
            )
        )
    # SimpleMem.process_window advances five turns and keeps 15 overlapping
    # turns; finalize() then processes that remaining tail as one more window.
    windows = sum(1 if n < 20 else (n - 20) // 5 + 2 for n in turns)
    n_questions = cohort["question_count"]

    def rmb(input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * 2.4 + output_tokens * 9.6) / 1_000_000

    # Pilot: 5 SimpleMem windows and 7 MemNexus extraction batches across
    # 35 turns; 5 QA pairs. Ratios are estimates, live provider usage governs.
    simple_build = rmb(windows * 8173 / 5, windows * 13147 / 5)
    nexus_build = rmb(
        sum(math.ceil(n / 5) for n in turns) * 23378 / 7,
        sum(math.ceil(n / 5) for n in turns) * 5254 / 7,
    )
    simple_qa = rmb(n_questions * 35421 / 5, n_questions * 2502 / 5)
    nexus_qa = rmb(n_questions * 15657 / 5, n_questions * 1281 / 5)
    judge = rmb(n_questions * 2 * 1352 / 10, n_questions * 2 * 288 / 10)
    base = simple_build + nexus_build + simple_qa + nexus_qa + judge
    return {
        "conversations": len(sample_indices),
        "raw_turns": sum(turns),
        "simplemem_windows": windows,
        "simplemem_build_rmb": round(simple_build, 4),
        "memnexus_build_rmb": round(nexus_build, 4),
        "simplemem_answers_rmb": round(simple_qa, 4),
        "memnexus_answers_rmb": round(nexus_qa, 4),
        "judge_rmb": round(judge, 4),
        "point_estimate_rmb": round(base, 4),
        "with_25pct_margin_rmb": round(base * 1.25, 4),
        "max_session": max_session,
        "basis": "measured five-question English pilot usage scaled by selected raw turns, windows and 100 questions",
    }


if __name__ == "__main__":
    cohort = select_cohort()
    output = ROOT / "result" / "locomo_100_cohort.json"
    output.write_text(
        json.dumps(cohort, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selection_sha256": cohort["selection_sha256"],
                "categories": cohort["categories"],
                "estimate": estimate_cost(cohort),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
