"""Break the provider-request ledger down by pipeline stage, for both systems.

SimpleMem prompts carry a system message naming the stage, so they are labelled
from it.  MemNexus prompts carry no system message, so the stage is read off the
first user message, which has exactly one of three openings:

    "Plan retrieval for this factual question ..."        -> planning
    "Answer the original question using the numbered ..." -> answering
    "你负责 SimpleMem 式用户意图感知查询规划 ..."           -> planning (CN)

Nothing here calls a model; the ledger is read from disk.
"""

from __future__ import annotations

import collections
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = str(ROOT / "result" / "bench260" / "attempts" / "*" / "usage.calls.jsonl")
QUESTIONS = 178

SIMPLEMEM_SITES = [
    ("professional information extraction assistant", "建库·信息抽取"),
    ("professional Q&A assistant", "答题·最终作答"),
    ("query analysis assistant", "规划·查询分析"),
    ("query generation specialist", "规划·生成检索式"),
    ("information completeness evaluator", "反思·充分性判定"),
    ("missing information query generator", "反思·补缺检索式"),
    ("intelligent information requirement analyst", "规划·信息需求分析"),
]

MEMNEXUS_OPENINGS = [
    ("plan retrieval for this factual question", "规划·生成计划"),
    ("answer the original question using the numbered", "答题·生成答案表"),
    ("你负责 simplemem 式用户意图感知查询规划", "规划·生成计划"),
]


def simplemem_label(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") != "system":
            continue
        text = re.sub(r"\s+", " ", str(message.get("content") or ""))
        for needle, label in SIMPLEMEM_SITES:
            if needle in text:
                return label
        return "其他"
    return "(无 system 提示)"


def memnexus_label(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()
        lowered = text.casefold()
        for needle, label in MEMNEXUS_OPENINGS:
            if lowered.startswith(needle):
                return label
        return "其他"
    return "(空)"


def main() -> None:
    per_system = collections.defaultdict(collections.Counter)
    per_attempt = collections.defaultdict(collections.Counter)

    for path in sorted(glob.glob(PATTERN)):
        folder = Path(path).parent.name
        system = "MemNexus" if folder.endswith("memnexus") else "SimpleMem"
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for request in (r for r in rows if r.get("event") == "request"):
            messages = (request.get("body") or {}).get("messages") or []
            label = (
                memnexus_label(messages)
                if system == "MemNexus"
                else simplemem_label(messages)
            )
            per_system[system][label] += 1
            # `attempt` is an index *inside* one conversation folder, so it must
            # be scoped by folder or two conversations collide on the same key.
            per_attempt[(system, folder, request.get("attempt"))][label] += 1

    summary = {}
    for system in ("SimpleMem", "MemNexus"):
        counts = per_system[system]
        build = sum(v for k, v in counts.items() if "建库" in k)
        qa = sum(v for k, v in counts.items() if "建库" not in k)
        print("=" * 74)
        print(f"[{system}]  建库 {build} 次   问答 {qa} 次  "
              f"({qa / QUESTIONS:.2f} 次/题)")
        print("=" * 74)
        for label, count in counts.most_common():
            print(f"   {count:4d}  {label}")
        print()

        stage_calls = collections.Counter()
        for (sysname, _folder, attempt), counter in per_attempt.items():
            if sysname != system or attempt is None:
                continue
            stage_calls[sum(counter.values())] += 1
        total_attempts = sum(stage_calls.values())
        print(f"   每题调用次数分布（{total_attempts} 个 attempt）:")
        for calls in sorted(stage_calls):
            print(f"     {calls} 次: {stage_calls[calls]:4d} 题 "
                  f"({stage_calls[calls] / total_attempts:5.1%})")
        print()
        summary[system] = {"build": build, "qa": qa,
                           "per_qa_call": round(qa / QUESTIONS, 2),
                           "distribution": dict(sorted(stage_calls.items()))}

    Path(ROOT / "result" / "call_breakdown_20261022.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written: result/call_breakdown_20261022.json")


if __name__ == "__main__":
    main()
