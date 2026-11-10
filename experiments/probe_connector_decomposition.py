"""How much of the F1 gap is the list connector, and how much is the mechanism?

LoCoMo's official multi-hop scorer splits the prediction on commas:

    preds = [p.strip() for p in prediction.split(",") if p.strip()]

so a prediction written as "A, B, C" enters the scorer as three candidates and
each gold item is matched against the best of them.  Written as "A; B; C" it
enters as one candidate.  MemNexus renders ``set`` operators with ", " while
SimpleMem tends to use "; ", so part of the observed gap is a formatting
convention rather than better memory.

This rewrites MemNexus predictions with "; " and re-scores, without touching
anything retrieved.  The residual is the mechanism.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"


def swap_connector(text: str) -> str:
    return str(text or "").replace(", ", "; ")


def average_f1(rows: list[dict]) -> tuple[float, dict[str, float]]:
    by_type = defaultdict(list)
    for row in rows:
        if row.get("category_id") == pp.ADVERSARIAL_CATEGORY:
            continue
        by_type[pp.REASONING_TYPES[row["category_id"]]].append(pp.f1_of(row))
    scores = {name: statistics.mean(values) for name, values in by_type.items()}
    return statistics.mean(scores.values()), scores


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    paired = pp.common_subset(rows, ("SimpleMem", "MemNexus"))
    questions = len(paired) // 2

    sm = [r for r in paired if r["system"] == "SimpleMem"]
    mn = [r for r in paired if r["system"] == "MemNexus"]
    mn_swapped = [dict(r, prediction=swap_connector(r.get("prediction"))) for r in mn]

    base_mn, by_mn = average_f1(mn)
    base_sm, by_sm = average_f1(sm)
    swap_mn, by_swap = average_f1(mn_swapped)

    print("=" * 78)
    print(f"  F1 分解（{questions} 对配对题，SimpleMem 论文口径）")
    print("=" * 78)
    print(f"{'':14s}{'SimpleMem':>12s}{'MemNexus':>12s}{'Δ':>10s}")
    print("-" * 78)
    for name in ("MultiHop", "Temporal", "OpenDomain", "SingleHop"):
        print(f"{name:14s}{by_sm[name] * 100:11.2f}%{by_mn[name] * 100:11.2f}%"
              f"{(by_mn[name] - by_sm[name]) * 100:+9.2f}")
    print("-" * 78)
    print(f"{'Average':14s}{base_sm * 100:11.2f}%{base_mn * 100:11.2f}%"
          f"{(base_mn - base_sm) * 100:+9.2f}")
    print()

    total = (base_mn - base_sm) * 100
    connector = (base_mn - swap_mn) * 100
    mechanism = (swap_mn - base_sm) * 100
    print("  把 MemNexus 的列表连接符 ', ' 换回 '; '（同模型、同预测、只改写法）:")
    print(f"     MemNexus 原样            {base_mn * 100:6.2f}")
    print(f"     MemNexus 换连接符后      {swap_mn * 100:6.2f}   (-{connector:.2f})")
    print(f"     SimpleMem                {base_sm * 100:6.2f}")
    print()
    print(f"  => 总差 {total:+.2f} = 连接符 {connector:+.2f} + 机制 {mechanism:+.2f}")
    print(f"     连接符占比 {connector / total * 100:.0f}%")
    print()

    changed = sum(
        1 for a, b in zip(mn, mn_swapped, strict=True)
        if str(a.get("prediction") or "") != str(b.get("prediction") or "")
    )
    print(f"  受影响的预测条数: {changed}/{questions}")


if __name__ == "__main__":
    main()
