"""对已完成的 unit 单独判分——运行中断后的补救。

正常情况下判分是 benchmark100.py 的最后一段，只在整个 run 跑完后才进入。
当 run 因异常中断（如 2026-10-20 的 query_type 崩溃）时，已完成的 17 个 unit
结果仍然完整，可以直接判分，不必重跑。

判分逻辑完全复用 benchmark100.judge_one，只补上"遍历哪些 rows"这一层。

用法：
    python experiments/judge_partial_run.py result/benchmark100/20261020_161439_654687
    python experiments/judge_partial_run.py <run目录> --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "SimpleMem"))

import benchmark100 as B  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="只判前 N 条，用于试跑")
    parser.add_argument("--delay", type=float, default=1.5, help="每条之间的间隔秒，避免触发每分钟限额")
    args = parser.parse_args()

    run_dir = args.run
    report = B.read(run_dir / "run.json")
    stages = report.get("stages") or {}
    rows = [row for unit in stages for row in B.read(run_dir / stages[unit])["rows"]]
    total = len(rows)
    print(f"run {run_dir.name}  已完成 unit {len(stages)}  待判 rows {total}")
    print(f"已判 {len(report.get('judgments') or {})} 条，预算 ¥{report['budget_rmb']}")

    random.Random(report["judge_seed"]).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    if args.dry_run:
        print(f"[dry-run] 将判 {len(rows)} 条，首条：{rows[0]['question'][:70]}")
        return 0

    cap = report["budget_rmb"]
    done = 0
    for row in rows:
        unit = f"{row['sample_id']}_{row['qa_index']}_{row['system'].lower()}"
        if unit in report["judgments"]:
            continue
        completed = list((run_dir / "attempts").glob(f"judge_{unit}__*/judgment.json"))
        if completed:
            report["judgments"][unit] = sorted(completed)[-1].relative_to(run_dir).as_posix()
            B.save(run_dir / "run.json", report)
            continue
        attempt = B.new_attempt(run_dir, "judge_" + unit)
        report["active_unit"] = "judge_" + unit
        B.save(run_dir / "run.json", report)
        value = None
        for retry in range(6):
            try:
                value = B.judge_one(row, attempt, cap)
                break
            except Exception as error:  # noqa: BLE001 - 网关限流要退避，其他错误直接抛
                text = str(error).lower()
                if "429" not in text and "rate limit" not in text and "rate_limit" not in text:
                    raise
                wait = min(30 * (retry + 1), 180)
                print(f"  429 限流，等待 {wait}s 后重试（第 {retry + 1} 次）", flush=True)
                time.sleep(wait)
        if value is None:
            raise SystemExit("连续限流，已判的部分保留在 run.json，稍后重跑本脚本即可续判")
        value.update(
            sample_id=row["sample_id"], qa_index=row["qa_index"], system=row["system"],
            prediction_sha256=hashlib.sha256(row["prediction"].encode()).hexdigest(),
        )
        B.save(attempt / "judgment.json", value)
        report["judgments"][unit] = (attempt / "judgment.json").relative_to(run_dir).as_posix()
        B.save(run_dir / "run.json", report)
        done += 1
        time.sleep(args.delay)
        if done % 10 == 0:
            print(f"  已判 {len(report['judgments'])}/{total}；累计 ¥{B.all_spend(run_dir):.4f}", flush=True)

    report["active_unit"] = ""
    B.save(run_dir / "run.json", report)
    print(f"完成：判分 {len(report['judgments'])} 条，累计 ¥{B.all_spend(run_dir):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
