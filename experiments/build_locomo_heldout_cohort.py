"""Held-out cohort：eligible 池(D1-D30) − 母 cohort 1000 = 962 题。

用途：E6 cap=2 与 SimpleMem 的 held-out 验证。cap=2 的选参（错误分析、配额扫描、
子集 A/B）全部发生在母 cohort 的 1000 题上；held-out 提供未见过题目的外推检验，
是"tuned on test"质疑的直接回应。

选择零自由度：补集是唯一确定的（eligible 规则与母 cohort 构造完全相同，
不引入任何新抽样、新阈值）。结构/哈希算法与 build_locomo_1000_cohort.py 一致，
可被 run_parallel_bench / benchmark100 直接消费。
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from locomo_cohort import CATEGORY_NAMES, DATASET, available_ids, evidence_ids

MAX_SESSION = 30
MOTHER = ROOT / "result/locomo_1000_cohort_20261023.json"
OUT = ROOT / "result/locomo_heldout_962_cohort_20261110.json"


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
            eligible.append({
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "qa_index": qa_index,
                "category_id": qa["category"],
                "category": CATEGORY_NAMES[qa["category"]],
                "question": qa["question"],
                "evidence": references,
            })
    eligible.sort(key=lambda x: (x["sample_index"], x["qa_index"]))

    mother = json.loads(MOTHER.read_text(encoding="utf-8"))
    mother_keys = {(q["sample_id"], q["qa_index"]) for q in mother["questions"]}
    held = [q for q in eligible if (q["sample_id"], q["qa_index"]) not in mother_keys]

    # 自检：补集性质必须严格成立
    assert len({(q["sample_id"], q["qa_index"]) for q in held}) == len(held), "held-out 存在重复键"
    assert len(held) + len(mother_keys) == len(eligible), "补集与母集不互斥完备"
    assert not ({(q["sample_id"], q["qa_index"]) for q in held} & mother_keys), "与母 cohort 有交集"

    signature = hashlib.sha256(
        json.dumps([(q["sample_id"], q["qa_index"]) for q in held], separators=(",", ":")).encode()
    ).hexdigest()
    cohort = {
        "dataset": mother["dataset"],
        "dataset_sha256": mother["dataset_sha256"],
        "selection_sha256": signature,
        "protocol": (
            f"held-out complement: eligible pool (official evidence fully within "
            f"sessions D1-D{MAX_SESSION}, answer present, same rules as the 1000-question "
            f"development cohort) minus that cohort; zero selection freedom; {len(held)} questions"
        ),
        "max_session": MAX_SESSION,
        "question_count": len(held),
        "conversation_count": len({q["sample_id"] for q in held}),
        "categories": dict(Counter(q["category"] for q in held)),
        "per_conversation": dict(Counter(q["sample_id"] for q in held)),
        "questions": held,
    }
    OUT.write_text(json.dumps(cohort, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"eligible {len(eligible)} - mother {len(mother_keys)} = held-out {len(held)}")
    print("categories:", cohort["categories"])
    print("per conversation:", cohort["per_conversation"])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
