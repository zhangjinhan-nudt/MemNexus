"""用 SimpleMem 原论文 Appendix A.4 的判分口径，对已有运行重新判分。

旧判决（`benchmark100.py` 曾用的 strict 口径）保留在原 `judge_*` 目录里，本脚本
写到独立目录，两份结果可直接对比。

判分口径的唯一来源是 `benchmark100.JUDGE_INSTRUCTION`（论文 Listing 4 的逐字移植），
因此本脚本不重复定义任何判分规则，只负责遍历、重试与落盘。

用法：
    python experiments/rejudge_with_paper_prompt.py result/benchmark100/20261020_161439_654687
    python experiments/rejudge_with_paper_prompt.py <run目录> --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "SimpleMem"))

import benchmark100 as B  # noqa: E402

OUT = ROOT / "result" / "judge_paper_20261020"


def unit_of(row: dict) -> str:
    return f"{row['sample_id']}_{row['qa_index']}_{row['system'].lower()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=1.5, help="每条间隔秒，避免触发每分钟限额")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    run_dir = args.run
    rows: list[dict] = []
    for path in sorted((run_dir / "attempts").iterdir()):
        if not path.is_dir() or path.name.startswith("judge_"):
            continue
        stage = path / "stage_result.json"
        if not stage.exists():
            continue
        rows.extend(json.loads(stage.read_text(encoding="utf-8"))["rows"])

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"run {run_dir.name}  待判 {len(rows)} 条  ->  {OUT.relative_to(ROOT)}")
    print(f"判分口径: SimpleMem Appendix A.4（generous）")

    if args.limit:
        rows = rows[: args.limit]
    if args.dry_run:
        print(f"[dry-run] 首条: {rows[0]['question'][:70]}")
        return 0

    cap = 50.0
    # judge_one 用 folder.parents[1] 作为"本轮到目前为止的花费"根目录。必须给它
    # 一个独立的空根，否则会扫描整个 result/ 下所有历史运行并误判预算耗尽。
    ledger_root = OUT / "_ledger"
    ledger_root.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for row in rows:
        name = unit_of(row)
        target = OUT / f"{name}.json"
        if target.exists():
            skipped += 1
            continue
        value = None
        for retry in range(6):
            try:
                probe = ledger_root / "attempts" / name
                probe.mkdir(parents=True, exist_ok=True)
                value = B.judge_one(row, probe, cap)
                break
            except Exception as error:  # noqa: BLE001 - 限流与网络抖动退避，其余直接抛
                text = str(error).lower()
                transient = any(token in text for token in
                                ("429", "rate limit", "rate_limit", "connection error",
                                 "proxyerror", "502", "503", "504", "timed out", "timeout"))
                if not transient:
                    raise
                wait = min(30 * (retry + 1), 180)
                print(f"  可重试错误（{type(error).__name__}），等待 {wait}s（第 {retry + 1} 次）", flush=True)
                time.sleep(wait)
        if value is None:
            raise SystemExit("连续限流，已判部分保留，稍后重跑即可续判")
        value.update(
            sample_id=row["sample_id"],
            qa_index=row["qa_index"],
            system=row["system"],
            question=row["question"],
            gold_answer=row["gold_answer"],
            prediction=row["prediction"],
        )
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        time.sleep(args.delay)
        if done % 20 == 0:
            print(f"  已判 {done}（跳过 {skipped}）", flush=True)

    print(f"完成：新判 {done} 条，跳过 {skipped} 条，共 {len(rows)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
