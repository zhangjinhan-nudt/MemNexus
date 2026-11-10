"""Symmetric connector normalisation: re-score BOTH systems under one convention.

The one-sided counterfactual (rewrite only MemNexus) is biased: it also mangles
commas that are ordinary prose punctuation, not list separators.  Here both
systems are rewritten into the *same* convention, so the comparison stays
apples-to-apples:

  as_written  each system as recorded
  to_comma    "; " -> ", " in both
  to_semi     ", " -> "; " in both

If MemNexus still leads under ``to_semi``, the lead is a mechanism property.
If it disappears, the lead was the rendering convention.
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
ORDER = ("MultiHop", "Temporal", "OpenDomain", "SingleHop")


def rewrite(rows: list[dict], mode: str) -> list[dict]:
    if mode == "as_written":
        return rows
    source, target = ("; ", ", ") if mode == "to_comma" else (", ", "; ")
    return [dict(r, prediction=str(r.get("prediction") or "").replace(source, target))
            for r in rows]


def per_type(rows: list[dict]) -> dict[str, float]:
    bucket = defaultdict(list)
    for row in rows:
        if row.get("category_id") == pp.ADVERSARIAL_CATEGORY:
            continue
        bucket[pp.REASONING_TYPES[row["category_id"]]].append(pp.f1_of(row))
    return {name: statistics.mean(values) for name, values in bucket.items()}


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    paired = pp.common_subset(rows, ("SimpleMem", "MemNexus"))
    questions = len(paired) // 2
    systems = {
        "SimpleMem": [r for r in paired if r["system"] == "SimpleMem"],
        "MemNexus": [r for r in paired if r["system"] == "MemNexus"],
    }

    print("=" * 82)
    print(f"  连接符归一化重打分（{questions} 对，SimpleMem 论文官方 F1）")
    print("=" * 82)
    print(f"{'类型':<12s}{'SM 原样':>10s}{'MN 原样':>10s}{'Δ原样':>9s}"
          f"{'|':^3s}{'SM→分号':>10s}{'MN→分号':>10s}{'Δ分号':>9s}")
    print("-" * 82)

    results = {}
    for mode in ("as_written", "to_semi"):
        results[mode] = {name: per_type(rewrite(rows_, mode))
                         for name, rows_ in systems.items()}

    for name in ORDER:
        sm0, mn0 = results["as_written"]["SimpleMem"][name], results["as_written"]["MemNexus"][name]
        sm1, mn1 = results["to_semi"]["SimpleMem"][name], results["to_semi"]["MemNexus"][name]
        print(f"{name:<12s}{sm0 * 100:9.2f}%{mn0 * 100:9.2f}%{(mn0 - sm0) * 100:+8.2f}"
              f"{'|':^3s}{sm1 * 100:9.2f}%{mn1 * 100:9.2f}%{(mn1 - sm1) * 100:+8.2f}")

    avg = {mode: {name: statistics.mean(v.values())
                  for name, v in results[mode].items()} for mode in results}
    print("-" * 82)
    for name in ("SimpleMem", "MemNexus"):
        print(f"{name:<12s}{avg['as_written'][name] * 100:9.2f}%{'':9s}{'':8s}"
              f"{'|':^3s}{avg['to_semi'][name] * 100:9.2f}%")
    print("-" * 82)
    total = (avg["as_written"]["MemNexus"] - avg["as_written"]["SimpleMem"]) * 100
    semi = (avg["to_semi"]["MemNexus"] - avg["to_semi"]["SimpleMem"]) * 100
    print(f"{'Δ':<12s}{total:+9.2f}{'':17s}{'|':^3s}{semi:+9.2f}")
    print()
    print(f"  两侧统一为分号后，MemNexus 的领先从 {total:+.2f} 变为 {semi:+.2f}")
    print(f"  => 机制净贡献约 {semi:+.2f}；写法（连接符）贡献约 {total - semi:+.2f}")


if __name__ == "__main__":
    main()
