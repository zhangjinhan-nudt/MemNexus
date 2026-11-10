"""Collect the evidence behind "why MemNexus is more accurate and cheaper".

Reads only what is already on disk:

  result/bench260/rows.json                  per-question metrics for both systems
  result/bench260/paper_metrics.json         paper-protocol aggregate
  result/bench260/attempts/*/usage.calls.jsonl   full provider request ledger

Nothing here calls a model.
"""

from __future__ import annotations

import collections
import glob
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "result" / "bench260" / "rows.json"
PAPER = ROOT / "result" / "bench260" / "paper_metrics.json"
LEDGER = str(ROOT / "result" / "bench260" / "attempts" / "*" / "usage.calls.jsonl")

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
        lowered = re.sub(r"\s+", " ", str(message.get("content") or "")).strip().casefold()
        for needle, label in MEMNEXUS_OPENINGS:
            if lowered.startswith(needle):
                return label
        return "其他"
    return "(空)"


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    by_system = collections.defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)

    print("=" * 78)
    print("  A.  provider 请求次数（逐题字段 model_calls.provider_attempts）")
    print("=" * 78)
    attempts = {}
    for system in ("MemNexus", "SimpleMem"):
        values = [
            (r.get("model_calls") or {}).get("provider_attempts")
            for r in by_system[system]
        ]
        values = [v for v in values if v is not None]
        attempts[system] = values
        counter = collections.Counter(values)
        print(f"[{system}]  n={len(values)}  mean={statistics.mean(values):.2f}  "
              f"median={statistics.median(values)}  total={sum(values)}")
        print("     分布:", dict(sorted(counter.items())))
    ratio = statistics.mean(attempts["SimpleMem"]) / statistics.mean(attempts["MemNexus"])
    print(f"\n  => SimpleMem 是 MemNexus 的 {ratio:.2f} 倍 provider 请求\n")

    print("=" * 78)
    print("  B.  MemNexus 内部阶段调用（逐题字段 runtime_model_calls）")
    print("=" * 78)
    stage_totals = collections.Counter()
    for row in by_system["MemNexus"]:
        stage = row.get("runtime_model_calls") or {}
        for key, value in stage.items():
            stage_totals[key] += int(value or 0)
    for key, value in stage_totals.most_common():
        print(f"   {value:5d}  {key}   ({value / len(by_system['MemNexus']):.2f}/题)")
    print()

    print("=" * 78)
    print("  C.  SimpleMem 各阶段调用（按 system prompt 归类账本）")
    print("=" * 78)
    sm_labels = collections.Counter()
    mn_labels = collections.Counter()
    for path in sorted(glob.glob(LEDGER)):
        folder = Path(path).parent.name
        system = "MemNexus" if folder.endswith("memnexus") else "SimpleMem"
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") != "request":
                continue
            messages = (record.get("body") or {}).get("messages") or []
            label = (memnexus_label(messages) if system == "MemNexus"
                     else simplemem_label(messages))
            (mn_labels if system == "MemNexus" else sm_labels)[label] += 1
    total_sm = sum(sm_labels.values())
    for label, count in sm_labels.most_common():
        print(f"   {count:5d}  {label}  ({count / 178:.2f}/题)")
    print(f"   {'-' * 5}")
    print(f"   {total_sm:5d}  合计  ({total_sm / 178:.2f}/题)")
    print()

    print("=" * 78)
    print("  D.  token / 成本 / 延迟（逐题字段）")
    print("=" * 78)
    for system in ("MemNexus", "SimpleMem"):
        sub = by_system[system]
        inp = [ (r.get("model_calls") or {}).get("input_tokens") or 0 for r in sub ]
        out = [ (r.get("model_calls") or {}).get("output_tokens") or 0 for r in sub ]
        cost = [ (r.get("model_calls") or {}).get("cost_rmb") or 0.0 for r in sub ]
        lat = [ (r.get("latency_ms") or 0) for r in sub ]
        print(f"[{system}]")
        print(f"   input  {statistics.mean(inp):8.0f} tok/题   合计 {sum(inp):>9,}")
        print(f"   output {statistics.mean(out):8.0f} tok/题   合计 {sum(out):>9,}")
        print(f"   总     {statistics.mean(inp)+statistics.mean(out):8.0f} tok/题   合计 {sum(inp)+sum(out):>9,}")
        print(f"   成本   {statistics.mean(cost):8.4f} 元/题   合计 {sum(cost):.4f}")
        print(f"   延迟   {statistics.mean(lat)/1000:8.2f} s/题")
        print()

    print("=" * 78)
    print("  E.  分题型 F1（论文口径聚合结果）")
    print("=" * 78)
    if PAPER.exists():
        paper = json.loads(PAPER.read_text(encoding="utf-8"))
        print(json.dumps(paper, ensure_ascii=False, indent=2)[:3000])
    else:
        print("(missing)", PAPER)


if __name__ == "__main__":
    main()
