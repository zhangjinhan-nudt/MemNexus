"""How many paired questions fit in a wall-clock budget? Measured, not guessed.

Throughput is taken from the previous real run (result/bench260): per-question
latency and cost for each system, plus the one-off cost of building a memory
base for a conversation that has no frozen snapshot yet.
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "result" / "bench260" / "rows.json"
SNAPSHOTS = ROOT / "result" / "small_ab_20261020" / "snapshots"
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"

HOURS = 7.0
BUDGET_S = HOURS * 3600


def per_system(rows):
    agg = collections.defaultdict(list)
    for row in rows:
        agg[row["system"]].append(
            (
                (row.get("latency_ms") or 0) / 1000.0,
                (row.get("model_calls", {}) or {}).get("cost_rmb", 0) or 0,
            )
        )
    return agg


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    agg = per_system(rows)

    print("=" * 74)
    print(f"实测吞吐（来自 result/bench260，{len(rows)} 条记录）")
    print("=" * 74)
    unit = {}
    for system in ("SimpleMem", "MemNexus"):
        lat = [x[0] for x in agg[system]]
        cost = [x[1] for x in agg[system]]
        mean_lat = statistics.mean(lat)
        med_lat = statistics.median(lat)
        mean_cost = statistics.mean(cost)
        unit[system] = (mean_lat, mean_cost)
        print(
            f"  {system:10s} n={len(lat):3d}  均值 {mean_lat:6.2f}s  中位 {med_lat:6.2f}s  "
            f"均值成本 ¥{mean_cost:.5f}/题"
        )

    pair_s = unit["SimpleMem"][0] + unit["MemNexus"][0]
    pair_c = unit["SimpleMem"][1] + unit["MemNexus"][1]

    print(f"\n  一「对」题（两系统各答一次）= {pair_s:.2f}s / ¥{pair_c:.5f}")
    print(f"  {HOURS:.0f} 小时 = {BUDGET_S:.0f}s（串行，runner 固定 workers=1）")

    print()
    print("=" * 74)
    print("场景 A：复用现有 D1–D5 冻结基座（无建库费）")
    print("=" * 74)
    have = sorted(p.stem for p in SNAPSHOTS.glob("*.json"))
    print(f"  已有快照：{len(have)} 段 -> {have}")
    n_a = int(BUDGET_S / pair_s)
    print(f"  理论上限 {n_a} 对 = {n_a*2} 次问答，约 ¥{n_a*pair_c:.2f}")
    print("  ⚠️ 但 D1–D5 视界下全数据集合格题池只有 337 题 → 题量本身是瓶颈，跑不满 7 小时")
    print("     预计实际耗时 337 × {:.2f}s = {:.1f} h，花费约 ¥{:.2f}".format(
        pair_s, 337 * pair_s / 3600, 337 * pair_c))

    print()
    print("=" * 74)
    print("场景 B：全 sessions 重新冻结（对齐论文口径，1986 题）")
    print("=" * 74)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    turns = []
    for sample in data:
        conv = sample["conversation"]
        n = sum(
            len(v)
            for k, v in conv.items()
            if k.startswith("session_") and isinstance(v, list)
        )
        turns.append(n)
    print(f"  10 段对话原始轮数：{turns}  合计 {sum(turns)} turns")
    print("  已知锚点：conv-26 D1–D5 = 92 turns，建库 ¥0.3095、19 个抽取批次")
    per_turn = 0.3095 / 92
    build_cost = sum(turns) * per_turn
    print(f"  按 ¥{per_turn:.5f}/turn 外推，全 sessions 建库 ≈ ¥{build_cost:.2f}（一次性）")
    print()
    print(f"  QA 部分 7 h 可做 {n_a} 对 = {n_a*2} 次问答，约 ¥{n_a*pair_c:.2f}")
    print(f"  ⇒ 全量 1986 对需要 {1986*pair_s/3600:.1f} h，7 小时只能覆盖 "
          f"{min(n_a,1986)/1986:.0%}")

    print()
    print("=" * 74)
    print("场景 C（推荐）：7 h 内单侧重复，换可判读性")
    print("=" * 74)
    print("  与其 7 h 跑 780 对不同题，不如跑 N 题 × R 遍，把 run 级误差测出来。")
    for n in (100, 150, 200, 250):
        for r in (2, 3):
            total = n * r * pair_s
            if total <= BUDGET_S:
                print(f"    n={n:4d} × R={r}  ->  {total/3600:5.2f} h  "
                      f"¥{n*r*pair_c:7.2f}  ✔")
    print()
    print("  参照：分辨 +2.34 分机制效应需 n≈1291 对（单次运行）。")
    print("  若加 R=3 且逐题取均值，等效样本量提升 √3≈1.73 倍 → 需 n≈745 对。")


if __name__ == "__main__":
    main()
