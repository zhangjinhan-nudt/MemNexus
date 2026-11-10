"""Convert LongMemEval (S + oracle) into the LoCoMo-style sample format used by
experiments/run_locomo_pilot.py, and emit a 1000-question cohort file
(500 S-haystack + 500 oracle-haystack question evaluations).

Output:
  benchmark-data/official/longmemeval/longmemeval1000.json   (dataset)
  result/longmemeval_1000_cohort_20261023.json               (cohort)

Protocol notes:
  - S setting: the question's own ~48-session haystack (official LongMemEval-S).
  - oracle setting: the gold answer sessions only (upper bound sanity check).
  - Both systems build memory from identical turns per unit; units are
    independent and resumable (stage_result.json skip in the runner).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "benchmark-data" / "official" / "LongMemEval"
OUT_DIR = ROOT / "benchmark-data" / "official" / "longmemeval"
OUT_DATASET = OUT_DIR / "longmemeval1000.json"
OUT_COHORT = ROOT / "result" / "longmemeval_1000_cohort_20261023.json"


def convert(question: dict, setting: str) -> dict:
    """setting: 'S' (full haystack) or 'O' (oracle gold sessions)."""
    sessions = question["haystack_sessions"]
    conversation: dict = {}
    for index, session in enumerate(sessions, 1):
        conversation[f"session_{index}"] = [
            {
                "dia_id": f"s{index}-t{turn_index}",
                "speaker": "user" if turn["role"] == "user" else "assistant",
                "text": turn["content"],
            }
            for turn_index, turn in enumerate(session, 1)
        ]
        date_text = question["haystack_dates"][index - 1] if index <= len(question["haystack_dates"]) else ""
        conversation[f"session_{index}_date_time"] = date_text
    qa = {
        "question": question["question"],
        "answer": question["answer"],
        "question_date": question["question_date"],
        "category": 4,  # placeholder: the pilot requires an int category key;
        # run_longmemeval_bench overwrites row["category"] with question_type.
        "category_id": 4,
        "dataset": "LongMemEval-S",
        "evidence": question.get("answer_session_ids", []),
    }
    return {
        "sample_id": f"{question['question_id']}_{setting}",
        "dataset": "LongMemEval-S",
        "setting": setting,
        "question_type": question["question_type"],
        "question_date": question["question_date"],
        "conversation": conversation,
        "qa": [qa],
    }


def main() -> int:
    s_data = json.loads((SRC / "longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    o_data = json.loads((SRC / "longmemeval_oracle.json").read_text(encoding="utf-8"))
    assert len(s_data) == len(o_data) == 500

    samples: list[dict] = []
    for question in s_data:
        samples.append(convert(question, "S"))
    for question in o_data:
        samples.append(convert(question, "O"))

    questions = []
    for sample_index, sample in enumerate(samples):
        questions.append(
            {
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "qa_index": 0,
                "category_id": 0,
                "category": sample["question_type"],
                "question": sample["qa"][0]["question"],
                "evidence": sample["qa"][0]["evidence"],
                "setting": sample["setting"],
            }
        )

    signature = hashlib.sha256(
        json.dumps([(x["sample_id"], x["qa_index"]) for x in questions], separators=(",", ":")).encode()
    ).hexdigest()
    dataset_payload = samples
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATASET.write_text(json.dumps(dataset_payload, ensure_ascii=False), encoding="utf-8")
    cohort = {
        "dataset": str(OUT_DATASET.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(OUT_DATASET.read_bytes()).hexdigest(),
        "selection_sha256": signature,
        "protocol": (
            "all 500 LongMemEval-S questions under both haystack settings "
            "(S: official ~48-session haystack; O: oracle gold sessions only); "
            "1000 question evaluations total"
        ),
        "max_session": 64,
        "question_count": len(questions),
        "conversation_count": len(samples),
        "categories": dict(Counter(x["category"] for x in questions)),
        "settings": dict(Counter(x["setting"] for x in questions)),
        "questions": questions,
    }
    OUT_COHORT.write_text(json.dumps(cohort, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_DATASET} ({len(samples)} samples)")
    print(f"wrote {OUT_COHORT} ({len(questions)} questions)")
    print("categories:", cohort["categories"])
    print("settings:", cohort["settings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
