"""Normalised SimpleMem stage-call counts, excluding aborted first rounds.

Three SimpleMem conversations (conv-30, conv-41, conv-47) raised
``RuntimeError: SimpleMem produced an invalid answer`` part-way through and the
runner started again.  The request ledger is opened in append mode, so the dead
round's requests are still there.  Everything before the ``failure.json``
mtime belongs to that dead round and must be dropped, otherwise the per-question
call rate is inflated.
"""

from __future__ import annotations

import collections
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ROOT / "result" / "bench260" / "attempts"

SIMPLEMEM_SITES = [
    ("professional information extraction assistant", "建库·信息抽取"),
    ("professional Q&A assistant", "答题·最终作答"),
    ("query analysis assistant", "规划·查询分析"),
    ("query generation specialist", "规划·生成检索式"),
    ("information completeness evaluator", "反思·充分性判定"),
    ("missing information query generator", "反思·补缺检索式"),
    ("intelligent information requirement analyst", "规划·信息需求分析"),
]

QUESTIONS_PER_CONV = {"conv-26": 32, "conv-30": 29, "conv-41": 20, "conv-42": 29,
                      "conv-43": 28, "conv-44": 18, "conv-47": 22}


def label(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") != "system":
            continue
        text = re.sub(r"\s+", " ", str(message.get("content") or ""))
        for needle, name in SIMPLEMEM_SITES:
            if needle in text:
                return name
        return "其他"
    return "(无 system 提示)"


def main() -> None:
    total = collections.Counter()
    qa_only = collections.Counter()
    for path in sorted(glob.glob(str(ATTEMPTS / "conv-*_simplemem" / "usage.calls.jsonl"))):
        folder = Path(path).parent
        conv = folder.name.split("_")[0]
        failure = folder / "failure.json"
        cutoff = failure.stat().st_mtime if failure.exists() else 0.0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") != "request":
                continue
            if cutoff and record.get("time_unix", 0) < cutoff:
                continue  # dead round
            name = label((record.get("body") or {}).get("messages") or [])
            total[name] += 1
            if "建库" not in name:
                qa_only[name] += 1

    questions = sum(QUESTIONS_PER_CONV.values())
    build = sum(v for k, v in total.items() if "建库" in k)
    qa = sum(qa_only.values())
    print("=" * 74)
    print(f"[SimpleMem] 剔除 3 次失败首轮后   建库 {build} 次   问答 {qa} 次 "
          f"({qa / questions:.2f} 次/题)")
    print("=" * 74)
    for name, count in total.most_common():
        print(f"   {count:5d}  {name}   ({count / questions:.2f}/题)")
    print()
    groups = collections.Counter()
    for name, count in total.items():
        groups[name.split("·")[0]] += count
    print("   按阶段归并:")
    for name, count in groups.most_common():
        print(f"     {count:5d}  {name}   ({count / questions:.2f}/题)")


if __name__ == "__main__":
    main()
