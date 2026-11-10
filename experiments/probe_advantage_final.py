"""One authoritative pass over the paired LoCoMo run, on the clean 175 pairs.

`result/bench260/rows.json` holds 356 rows (178 per system).  Three MemNexus rows
never produced an answer (`answer_path == "failed"`, empty prediction), so the
honest paired set is **175 questions**, not 178.  Every number below is computed
on that set so it is comparable with the previously reported headline.

Prints: paper-protocol F1/BLEU per type, the connector decomposition, paired
bootstrap CIs, provider-request rates, token/cost/latency, and the per-question
gain composition.
"""

from __future__ import annotations

import json
import math
import random
import re
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
SEED = 20261023
ORDER = ("MultiHop", "Temporal", "OpenDomain", "SingleHop")

DECLINE = re.compile(
    r"not mentioned|not specified|insufficient|not provided|no information|"
    r"cannot be determined|not in the (?:provided |given )?(?:context|conversation)|"
    r"does not (?:mention|specify|provide)|unknown",
    re.IGNORECASE,
)
IN = 2.4e-6
OUT = 9.6e-6


def clean_pairs(rows: list[dict]) -> list[tuple[dict, dict]]:
    """(SimpleMem row, MemNexus row) for questions both systems actually answered."""
    buckets: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        key = (row["sample_id"], row["qa_index"])
        buckets[key][row["system"]] = row
    pairs = []
    for entry in buckets.values():
        sm, mn = entry.get("SimpleMem"), entry.get("MemNexus")
        if sm is None or mn is None:
            continue
        if not str(mn.get("prediction") or "").strip():
            continue  # provider-side failure, not a system decision
        pairs.append((sm, mn))
    return pairs


def rewrite(row: dict, mode: str | None) -> dict:
    if mode is None:
        return row
    source, target = ("; ", ", ") if mode == "to_comma" else (", ", "; ")
    return dict(row, prediction=str(row.get("prediction") or "").replace(source, target))


def type_means(pairs, mode: str | None) -> dict[str, float]:
    bucket: dict[str, list[float]] = defaultdict(list)
    for sm, mn in pairs:
        cid = sm["category_id"]
        if cid == pp.ADVERSARIAL_CATEGORY:
            continue
        name = pp.REASONING_TYPES[cid]
        bucket[name].append(pp.f1_of(rewrite(mn, mode)) - pp.f1_of(rewrite(sm, mode)))
    return {name: statistics.fmean(values) for name, values in bucket.items()}


def bootstrap_ci(grouped: dict[str, list[float]], draws: int = BOOTSTRAP) -> tuple[float, float]:
    rng = random.Random(SEED)
    names = [n for n, v in grouped.items() if v]
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


def diff_groups(pairs, mode: str | None) -> dict[str, list[float]]:
    bucket: dict[str, list[float]] = defaultdict(list)
    for sm, mn in pairs:
        cid = sm["category_id"]
        if cid == pp.ADVERSARIAL_CATEGORY:
            continue
        name = pp.REASONING_TYPES[cid]
        bucket[name].append(pp.f1_of(rewrite(mn, mode)) - pp.f1_of(rewrite(sm, mode)))
    return bucket


def scores(pairs, mode: str | None) -> dict[str, dict[str, float]]:
    out = defaultdict(lambda: defaultdict(list))
    for sm, mn in pairs:
        cid = sm["category_id"]
        for system, row in (("SimpleMem", sm), ("MemNexus", mn)):
            out[system][pp.REASONING_TYPES.get(cid, "Adversarial")].append(
                (pp.f1_of(rewrite(row, mode)), pp.bleu1_of(rewrite(row, mode))))
    return {system: {name: (statistics.fmean([v[0] for v in values]),
                                             statistics.fmean([v[1] for v in values]))
                     for name, values in groups.items()}
            for system, groups in out.items()}


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    all_pairs = clean_pairs(rows)
    print(f"rows.json: {len(rows)} 行；剔除 3 条 MN 空预测后 = {len(all_pairs)} 对配对题\n")

    base = scores(all_pairs, None)
    print("=" * 92)
    print("  A. 论文口径主表（每格：F1 / BLEU-1，百分数）")
    print("=" * 92)
    print(f"{'类型':<12s}{'SimpleMem':>20s}{'MemNexus':>20s}{'Δ F1':>12s}{'Δ BLEU':>12s}")
    print("-" * 92)
    for name in ORDER:
        sm_f1, sm_b = base["SimpleMem"][name]
        mn_f1, mn_b = base["MemNexus"][name]
        print(f"{name:<12s}{sm_f1 * 100:14.2f} /{sm_b * 100:5.2f}"
              f"{mn_f1 * 100:14.2f} /{mn_b * 100:5.2f}"
              f"{(mn_f1 - sm_f1) * 100:+12.2f}{(mn_b - sm_b) * 100:+12.2f}")
    avg = {s: (statistics.fmean([base[s][n][0] for n in ORDER]),
                statistics.fmean([base[s][n][1] for n in ORDER])) for s in ("SimpleMem", "MemNexus")}
    print("-" * 92)
    print(f"{'Average':<12s}{avg['SimpleMem'][0] * 100:14.2f} /{avg['SimpleMem'][1] * 100:5.2f}"
          f"{avg['MemNexus'][0] * 100:14.2f} /{avg['MemNexus'][1] * 100:5.2f}"
          f"{(avg['MemNexus'][0] - avg['SimpleMem'][0]) * 100:+12.2f}"
          f"{(avg['MemNexus'][1] - avg['SimpleMem'][1]) * 100:+12.2f}")
    print()

    print("=" * 92)
    print("  B. 连接符分解与配对 bootstrap（20000 次）")
    print("=" * 92)
    for label, mode in (("原样", None), ("两侧统一分号", "to_semi")):
        means = type_means(all_pairs, mode)
        groups = diff_groups(all_pairs, mode)
        low, high = bootstrap_ci(groups)
        print(f"  {label:<14s} Average Δ = {statistics.fmean(means.values()) * 100:+.2f}"
              f"   95% CI [{low * 100:+.2f}, {high * 100:+.2f}]"
              f"   {'跨 0 不显著' if low < 0 < high else '不跨 0'}")
    print()

    print("=" * 92)
    print("  C. 效率：provider 请求 / token / 成本 / 延迟")
    print("=" * 92)
    print(f"{'系统':<12s}{'请求/题':>10s}{'input/题':>11s}{'output/题':>11s}{'总 tok/题':>12s}{'延迟 s/题':>12s}")
    print("-" * 92)
    for system in ("SimpleMem", "MemNexus"):
        subset = [r for r in rows if r["system"] == system
                  and (r["sample_id"], r["qa_index"]) in
                  {(p[0]["sample_id"], p[0]["qa_index"]) for p in all_pairs}]
        calls = [(r.get("model_calls") or {}) for r in subset]
        attempts = sum(c.get("provider_attempts") or 0 for c in calls)
        inp = sum(c.get("input_tokens") or 0 for c in calls)
        out = sum(c.get("output_tokens") or 0 for c in calls)
        latency = statistics.fmean([r.get("latency_ms") or 0 for r in subset]) / 1000
        print(f"{system:<12s}{attempts / len(subset):>10.2f}{inp / len(subset):>11.0f}"
              f"{out / len(subset):>11.0f}{(inp + out) / len(subset):>12.0f}{latency:>12.2f}")
    print()

    print("=" * 92)
    print("  D. 逐题增益构成（175 对剔除对抗题后的 137 道）")
    print("=" * 92)
    tally = defaultdict(lambda: defaultdict(int))
    for sm, mn in all_pairs:
        if sm["category_id"] == pp.ADVERSARIAL_CATEGORY:
            continue
        mn_f1, sm_f1 = pp.f1_of(mn), pp.f1_of(sm)
        sm_declined = bool(DECLINE.search(str(sm.get("prediction") or "")))
        mn_declined = bool(DECLINE.search(str(mn.get("prediction") or "")))
        item = tally[pp.REASONING_TYPES[sm["category_id"]]]
        item["n"] += 1
        if sm_declined and not mn_declined and mn_f1 > 0.05:
            item["recall_win"] += 1
        elif mn_f1 - sm_f1 > 0.05:
            item["evidence_miss" if sm_f1 < 0.05 and not sm_declined else "wording_win"] += 1
        elif mn_f1 - sm_f1 < -0.05:
            item["loss"] += 1
        else:
            item["tie"] += 1
    print(f"{'类型':<12s}{'n':>5s}{'召回胜':>9s}{'证据选错':>10s}{'输出胜':>9s}{'MN更差':>9s}{'平':>7s}")
    print("-" * 92)
    grand = defaultdict(int)
    for name in ORDER:
        item = tally[name]
        print(f"{name:<12s}{item['n']:>5d}{item['recall_win']:>9d}{item['evidence_miss']:>10d}"
              f"{item['wording_win']:>9d}{item['loss']:>9d}{item['tie']:>7d}")
        for key in ("n", "recall_win", "evidence_miss", "wording_win", "loss", "tie"):
            grand[key] += item[key]
    print("-" * 92)
    print(f"{'合计':<12s}{grand['n']:>5d}{grand['recall_win']:>9d}{grand['evidence_miss']:>10d}"
          f"{grand['wording_win']:>9d}{grand['loss']:>9d}{grand['tie']:>7d}")
    print()


if __name__ == "__main__":
    main()
