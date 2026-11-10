"""Cheapest end-to-end proof that freeze -> replay actually works.

The protocol says no A/B may be read before "same frozen base twice gives the
same answers" is demonstrated. Doing that on 150 questions costs a full run;
doing it on one conversation and two questions costs about one yuan and catches
every wiring bug that matters (snapshot not hit, session not found, build
charged twice, reopen changing the base).

What it does, in order:

1. build one conversation from scratch and freeze it (paying the build once);
2. thaw that same base and answer the same two questions again (paying QA only);
3. compare predictions and report what each run was charged.

Exit code 0 means the pipeline is wired correctly and the two replays agreed.

Usage::

    python experiments/smoke_snapshot_pipeline.py --sample conv-26 --questions 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "MemNexus" / "memory-system" / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="conv-26")
    parser.add_argument("--questions", type=int, default=2)
    parser.add_argument("--cache", type=Path, default=ROOT / "result" / "smoke_snapshots",
                        help="where to keep the frozen base between the two runs")
    args = parser.parse_args()

    # Same env the paper variant sets, so the smoke exercises the real path.
    from agent_memory.settings import Settings  # noqa: F401  (loads the private .env)
    os.environ["MEMORY_SEMANTIC_EXECUTION"] = "false"
    os.environ["MEMORY_INPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "2.4"
    os.environ["MEMORY_OUTPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "9.6"
    os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "8192"

    from run_locomo_pilot import run_memnexus

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    sample = next(row for row in dataset if str(row.get("sample_id")) == args.sample)
    questions = sample["qa"][: args.questions]
    print(f"sample={args.sample}  questions={len(questions)}  cache={args.cache}")

    async def run(tag: str, require_snapshot: bool) -> tuple[dict, list[dict]]:
        with tempfile.TemporaryDirectory() as folder:
            return await run_memnexus(
                sample,
                questions,
                10.0,
                usage_ledger=Path(folder) / "usage.json",
                snapshot_dir=args.cache,
                require_snapshot=require_snapshot,
            )

    first_stage, first_rows = asyncio.run(run("build", require_snapshot=False))
    print()
    print(f"第一次  回放={first_stage['memory_base_replayed']}  "
          f"记忆={first_stage['memory_count']}  "
          f"建库花费=¥{first_stage['build_usage'].get('cost_rmb', 0):.4f}  "
          f"sha={(first_stage['memory_snapshot_sha256'] or '')[:12]}")
    if first_stage["memory_base_replayed"]:
        print("  注意：第一次就命中了旧快照，这次没有重新建库。删掉缓存目录可重来。")

    second_stage, second_rows = asyncio.run(run("replay", require_snapshot=True))
    print(f"第二次  回放={second_stage['memory_base_replayed']}  "
          f"记忆={second_stage['memory_count']}  "
          f"建库花费=¥{second_stage['build_usage'].get('cost_rmb', 0):.4f}  "
          f"sha={(second_stage['memory_snapshot_sha256'] or '')[:12]}")

    ok = True
    if not second_stage["memory_base_replayed"]:
        print("失败：第二次没有回放快照")
        ok = False
    if second_stage["memory_count"] != first_stage["memory_count"]:
        print(f"失败：两次记忆条数不同 {first_stage['memory_count']} vs {second_stage['memory_count']}")
        ok = False
    if second_stage["build_usage"].get("cost_rmb", 0) != 0:
        print("失败：回放仍然产生了建库花费")
        ok = False
    if first_stage["memory_snapshot_sha256"] != second_stage["memory_snapshot_sha256"]:
        print("失败：两次用的不是同一份快照")
        ok = False

    print()
    for index, (a, b) in enumerate(zip(first_rows, second_rows), 1):
        same = a["prediction"] == b["prediction"]
        ok = ok and same
        print(f"Q{index} [{a['category']}] {a['question'][:60]}")
        print(f"   建库后: {str(a['prediction'])[:90]}")
        print(f"   回放后: {str(b['prediction'])[:90]}")
        print(f"   一致: {same}   证据数: {a['retrieval_count']} vs {b['retrieval_count']}")

    print()
    print("结论：" + ("冻结基座可回放且问答一致，可以进入正式实验。" if ok else "管线有问题，先修再谈实验。"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
