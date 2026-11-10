"""Per-question LLM call counts, by call site, for both systems.

Reads the real provider-request ledgers written by the previous paid run and
groups requests by the system prompt that produced them.  Attribution to a
system comes from the attempt folder, so no model is called here.
"""

from __future__ import annotations

import collections
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = str(ROOT / "result" / "bench260" / "attempts" / "*" / "usage.calls.jsonl")

# Signature substrings -> short label.  Matched against the system message.
SITES = [
    ("professional information extraction assistant", "建库·信息抽取"),
    ("professional Q&A assistant", "答题·最终作答"),
    ("query analysis assistant", "规划·查询分析"),
    ("query generation specialist", "规划·生成检索式"),
    ("information completeness evaluator", "反思·充分性判定"),
    ("missing information query generator", "反思·补缺检索式"),
    ("intelligent information requirement analyst", "规划·信息需求分析"),
]


def classify(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") != "system":
            continue
        text = re.sub(r"\s+", " ", str(message.get("content") or ""))
        for needle, label in SITES:
            if needle in text:
                return label
        return "其他:" + text[:40]
    return "(无 system 提示)"


def main() -> None:
    per_system = collections.defaultdict(collections.Counter)
    qa_per_system = collections.defaultdict(collections.Counter)
    questions = collections.Counter()

    for path in sorted(glob.glob(PATTERN)):
        folder = Path(path).parent.name
        system = "MemNexus" if folder.endswith("memnexus") else "SimpleMem"
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        requests = [r for r in rows if r.get("event") == "request"]
        for request in requests:
            messages = (request.get("body") or {}).get("messages") or []
            label = classify(messages)
            per_system[system][label] += 1
            if "建库" not in label:
                qa_per_system[system][label] += 1

    # Question counts per conversation from the rows file.
    payload = json.loads((ROOT / "result" / "bench260" / "rows.json").read_text(encoding="utf-8"))
    for row in payload:
        if row["system"] == "MemNexus":
            questions[row["sample_id"]] += 1
    total_questions = sum(questions.values())

    print("=" * 78)
    print("大模型调用分布（来自真实调用账本，本次零调用）")
    print("=" * 78)
    print(f"  配对题量：{total_questions} 题\n")
    for system in ("SimpleMem", "MemNexus"):
        counts = per_system[system]
        qa = qa_system = qa_per_system[system]
        build = sum(v for k, v in counts.items() if "建库" in k)
        qa_total = sum(qa.values())
        print(f"[{system}]")
        print(f"  建库阶段 {build:4d} 次   问答阶段 {qa_total:4d} 次"
              f"   → 问答 {qa_total / total_questions:.2f} 次/题")
        for label, count in counts.most_common():
            stage = "建库" if "建库" in label else "问答"
            print(f"     {stage}  {count:4d}  {label}")
        print()


if __name__ == "__main__":
    main()
