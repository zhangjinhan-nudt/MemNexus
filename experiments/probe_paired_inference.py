"""Paired inference on the 178 matched questions, per reasoning type and overall.

Each question is answered by both systems, so the difference is a within-question
paired comparison.  Reports the paired mean, a bootstrap 95% CI on the paper's
``Average`` statistic (equal weight over the four reasoning types), and a
Wilcoxon signed-rank p-value computed without SciPy.

Both systems' predictions go through the same rewrite, so the comparison never
conflates "different writing convention" with "different memory quality".
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"
BOOTSTRAP = 20000
SEED = 20261022
ORDER = ("MultiHop", "Temporal", "OpenDomain", "SingleHop")


def paired_differences(rows: list[dict], rewrite=None) -> dict[str, list[float]]:
    """reasoning type -> list of (MemNexus F1 - SimpleMem F1) over matched questions."""
    buckets: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        if row.get("category_id") == pp.ADVERSARIAL_CATEGORY:
            continue
        key = (row["sample_id"], row["qa_index"])
        buckets[key]["type"] = pp.REASONING_TYPES[row["category_id"]]
        buckets[key][row["system"]] = row

    out = defaultdict(list)
    for entry in buckets.values():
        mn, sm = entry.get("MemNexus"), entry.get("SimpleMem")
        if mn is None or sm is None:
            continue
        if rewrite is not None:
            mn = dict(mn, prediction=rewrite(str(mn.get("prediction") or "")))
            sm = dict(sm, prediction=rewrite(str(sm.get("prediction") or "")))
        out[entry["type"]].append(pp.f1_of(mn) - pp.f1_of(sm))
    return out


def average_point(grouped: dict[str, list[float]]) -> float:
    """The metric itself: unweighted mean over the four reasoning-type means."""
    return statistics.fmean(statistics.fmean(v) for v in grouped.values() if v)


def bootstrap_ci(grouped: dict[str, list[float]], draws: int = BOOTSTRAP) -> tuple[float, float]:
    """Resample questions within each type, then take the equal-weight mean.

    Resampling per type (rather than pooling all 140 differences) keeps the CI
    aligned with the statistic the paper actually reports.
    """
    rng = random.Random(SEED)
    names = [name for name, values in grouped.items() if values]
    if not names:
        return (0.0, 0.0)
    means = []
    for _ in range(draws):
        draw = []
        for name in names:
            values = grouped[name]
            count = len(values)
            draw.append(statistics.fmean(values[rng.randrange(count)] for _ in range(count)))
        means.append(statistics.fmean(draw))
    means.sort()
    return (means[int(0.025 * draws)], means[int(0.975 * draws) - 1])


def wilcoxon_p(values: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank test, normal approximation with ties."""
    nonzero = [v for v in values if abs(v) > 1e-12]
    count = len(nonzero)
    if count < 5:
        return None
    order = sorted(range(count), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * count
    position = 0
    while position < count:
        end = position
        while (end + 1 < count
               and abs(abs(nonzero[order[end + 1]]) - abs(nonzero[order[position]])) < 1e-12):
            end += 1
        average = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    w_plus = sum(ranks[i] for i in range(count) if nonzero[i] > 0)
    mean = count * (count + 1) / 4
    sd = (count * (count + 1) * (2 * count + 1) / 24) ** 0.5
    z = (w_plus - mean) / sd if sd else 0.0
    return math.erfc(abs(z) / math.sqrt(2))


def report(title: str, grouped: dict[str, list[float]]) -> None:
    print("=" * 88)
    print(f"  {title}")
    print("=" * 88)
    print(f"{'类型':<12s}{'n':>4s}{'配对Δ均值':>12s}{'95% CI':>24s}{'p(Wilcoxon)':>14s}")
    print("-" * 88)
    total = 0
    for name in ORDER:
        values = grouped.get(name) or []
        total += len(values)
        if not values:
            continue
        mean = statistics.fmean(values)
        low, high = bootstrap_ci({name: values})
        p = wilcoxon_p(values)
        p_text = "n/a" if p is None else f"{p:.4f}"
        print(f"{name:<12s}{len(values):>4d}{mean * 100:+11.2f}"
              f"   [{low * 100:+7.2f}, {high * 100:+7.2f}]{p_text:>14s}")
    print("-" * 88)
    point = average_point(grouped)
    low, high = bootstrap_ci(grouped)
    print(f"{'Average':<12s}{total:>4d}{point * 100:+11.2f}"
          f"   [{low * 100:+7.2f}, {high * 100:+7.2f}]")
    print()


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    paired = pp.common_subset(rows, ("SimpleMem", "MemNexus"))
    report("配对差（原始写法）", paired_differences(paired))
    report("配对差（两侧统一为分号列表）",
           paired_differences(paired, rewrite=lambda text: text.replace(", ", "; ")))


if __name__ == "__main__":
    main()
