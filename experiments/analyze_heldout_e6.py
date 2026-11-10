"""E6 cap=2 held-out 962 题的配对分析（与母 cohort 分析完全同口径）。

输入：
  - result/e6_heldout_cap2_20261110/rows.json  （MN cap=2，held-out）
  - result/e6_heldout_sm_20261110/rows.json    （SimpleMem，held-out）
  - result/bench1000_locomo_20261023/rows.json （母 cohort 基线，用于外推对照）

口径：
  - 官方 F1（paper_protocol，Porter 词干 / multi-hop 逗号切最佳 / open-domain 取分号前）
  - 四类等权 Average（不含 adversarial）+ AdvSR（adversarial_success）
  - 簇 bootstrap（10 对话簇）95% CI + 符号检验
  - 格式归一化分解（双侧 `, `→`; `，与母 cohort 分析同协议）
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import paper_protocol as pp  # noqa: E402

MN_HELD = ROOT / "result/e6_heldout_cap2_20261110/rows.json"
SM_HELD = ROOT / "result/e6_heldout_sm_20261110/rows.json"
MOTHER = ROOT / "result/bench1000_locomo_20261023/rows.json"

CATS = ("temporal", "single-hop", "multi-hop", "open-domain")
SEED = 20261110
BOOTSTRAP = 2000


def load_index(path: Path, system: str) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(r["sample_id"], r["qa_index"]): r
            for r in rows if r.get("system") == system}


def average(values_by_cat: dict[str, list[float]]) -> float:
    return statistics.fmean([statistics.fmean(v) for v in values_by_cat.values()])


def cluster_bootstrap_avg(delta_by_conv: dict[str, dict[str, list[float]]],
                          rng: random.Random) -> tuple[float, float, float]:
    """四类等权 Average Δ 的簇 bootstrap。返回 (点估计, lo, hi)。"""
    convs = sorted(delta_by_conv)

    def avg_of(pick):
        by = defaultdict(list)
        for c in pick:
            for cat, vals in delta_by_conv[c].items():
                by[cat].extend(vals)
        return average(by)

    point = avg_of(convs)
    bs = sorted(avg_of([rng.choice(convs) for _ in convs]) for _ in range(BOOTSTRAP))
    return point, bs[int(0.025 * BOOTSTRAP)], bs[int(0.975 * BOOTSTRAP) - 1]


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    wins = sum(1 for x in deltas if x > 1e-12)
    losses = sum(1 for x in deltas if x < -1e-12)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)
    return wins, losses, p


def norm(text) -> str:
    return str(text or "").replace(", ", "; ")


def f1_with_prediction(row: dict, prediction: str) -> float:
    copy = dict(row)
    copy["prediction"] = prediction
    return pp.f1_of(copy)


def main() -> int:
    for path in (MN_HELD, SM_HELD):
        if not path.exists():
            print(f"[FAIL] 缺少输入：{path}", file=sys.stderr)
            return 3

    mn = load_index(MN_HELD, "MemNexus")
    sm = load_index(SM_HELD, "SimpleMem")
    keys = [k for k in mn if k in sm]
    nonadv = [k for k in keys if mn[k]["category"] != "adversarial"]
    adv = [k for k in keys if mn[k]["category"] == "adversarial"]
    empty_mn = [k for k in keys if not str(mn[k].get("prediction") or "").strip()]
    empty_sm = [k for k in keys if not str(sm[k].get("prediction") or "").strip()]
    print(f"配对 {len(keys)}（非对抗 {len(nonadv)} / 对抗 {len(adv)}）")
    if empty_mn or empty_sm:
        print(f"[WARN] 空预测：MN {len(empty_mn)} / SM {len(empty_sm)}", file=sys.stderr)

    # ---------- 官方口径 ----------
    f1_mn = {k: pp.f1_of(mn[k]) for k in nonadv}
    f1_sm = {k: pp.f1_of(sm[k]) for k in nonadv}
    by_cat_mn = defaultdict(list)
    by_cat_sm = defaultdict(list)
    for k in nonadv:
        by_cat_mn[mn[k]["category"]].append(f1_mn[k])
        by_cat_sm[mn[k]["category"]].append(f1_sm[k])
    a_mn, a_sm = average(by_cat_mn), average(by_cat_sm)
    print()
    print("=== 官方口径（四类等权 Average）===")
    print(f"  MN cap=2: {a_mn:.4f} | SM: {a_sm:.4f} | Δ {100*(a_mn-a_sm):+.2f}pp")
    print(f"  pooled: MN {statistics.fmean(f1_mn.values()):.4f} | "
          f"SM {statistics.fmean(f1_sm.values()):.4f}")

    delta_by_conv = defaultdict(lambda: defaultdict(list))
    for k in nonadv:
        delta_by_conv[k[0]][mn[k]["category"]].append(f1_mn[k] - f1_sm[k])
    rng = random.Random(SEED)
    point, lo, hi = cluster_bootstrap_avg(delta_by_conv, rng)
    verdict = "不跨零" if lo > 0 else "跨零"
    print(f"  AverageΔ 簇bootstrap95%CI [{100*lo:+.2f}, {100*hi:+.2f}] {verdict}")
    wins, losses, p = sign_test([f1_mn[k] - f1_sm[k] for k in nonadv])
    print(f"  符号检验: 胜 {wins} / 负 {losses} / 平 {len(nonadv)-wins-losses}  p≈{p:.3f}")

    print()
    print("  分题型（MN | SM | Δ）:")
    for cat in CATS:
        vals_mn = by_cat_mn.get(cat, [])
        vals_sm = by_cat_sm.get(cat, [])
        if not vals_mn:
            continue
        print(f"    {cat:<12} n={len(vals_mn):<4} {statistics.fmean(vals_mn):.4f} | "
              f"{statistics.fmean(vals_sm):.4f} | "
              f"{100*(statistics.fmean(vals_mn)-statistics.fmean(vals_sm)):+.2f}pp")

    # ---------- 格式归一化 ----------
    dn = {k: f1_with_prediction(mn[k], norm(mn[k].get("prediction")))
             - f1_with_prediction(sm[k], norm(sm[k].get("prediction")))
          for k in nonadv}
    by_cat_nor = defaultdict(list)
    for k in nonadv:
        by_cat_nor[mn[k]["category"]].append(dn[k])
    a_nor = average(by_cat_nor)
    nor_by_conv = defaultdict(lambda: defaultdict(list))
    for k in nonadv:
        nor_by_conv[k[0]][mn[k]["category"]].append(dn[k])
    point_n, lo_n, hi_n = cluster_bootstrap_avg(nor_by_conv, rng)
    print()
    print("=== 格式归一化（双侧 `, `→`; `）===")
    print(f"  AverageΔ {100*a_nor:+.2f}pp  CI [{100*lo_n:+.2f}, {100*hi_n:+.2f}] "
          f"{'不跨零' if lo_n > 0 else '跨零'}")

    # ---------- AdvSR ----------
    sr_mn = [1.0 if pp.adversarial_success(mn[k]) else 0.0 for k in adv]
    sr_sm = [1.0 if pp.adversarial_success(sm[k]) else 0.0 for k in adv]
    print()
    print(f"=== AdvSR（n={len(adv)}）===")
    print(f"  MN cap=2 {statistics.fmean(sr_mn):.4f} | SM {statistics.fmean(sr_sm):.4f} | "
          f"Δ {100*(statistics.fmean(sr_mn)-statistics.fmean(sr_sm)):+.2f}pp")

    # ---------- 效率 ----------
    mn_rows = [mn[k] for k in keys]
    sm_rows = [sm[k] for k in keys]

    def mean_field(rows, *path):
        vals = []
        for r in rows:
            v = r
            for part in path:
                v = (v or {}).get(part) if isinstance(v, dict) else None
            vals.append(v or 0)
        return statistics.fmean(vals)

    print()
    print("=== 效率（每题）===")
    print(f"  input tokens : MN {mean_field(mn_rows,'model_calls','input_tokens'):.0f} | "
          f"SM {mean_field(sm_rows,'model_calls','input_tokens'):.0f}")
    print(f"  延迟 ms      : MN {mean_field(mn_rows,'latency_ms'):.0f} | "
          f"SM {mean_field(sm_rows,'latency_ms'):.0f}")
    print(f"  模型请求     : MN {mean_field(mn_rows,'model_calls','provider_attempts'):.2f} | "
          f"SM {mean_field(sm_rows,'model_calls','provider_attempts'):.2f}")

    # ---------- 外推保持率（对照母 cohort）----------
    mother = json.loads(MOTHER.read_text(encoding="utf-8"))
    m_mn = {(r["sample_id"], r["qa_index"]): r for r in mother if r["system"] == "MemNexus"}
    # 母 cohort 上 cap2 的结果
    cap2_rows = json.loads((ROOT / "result/e6_full_cap2_20261110/rows.json").read_text(encoding="utf-8"))
    c2 = {(r["sample_id"], r["qa_index"]): r for r in cap2_rows if r["system"] == "MemNexus"}
    mkeys = [k for k in c2 if k in m_mn and c2[k]["category"] != "adversarial"]
    by_m = defaultdict(list)
    by_c = defaultdict(list)
    for k in mkeys:
        by_m[c2[k]["category"]].append(pp.f1_of(c2[k]) - pp.f1_of(load_index(MOTHER, "SimpleMem")[k]))
    # 母 cohort 的 cap2-vs-SM Average Δ
    mother_avg_delta = average(by_m)
    held_delta = a_mn - a_sm
    print()
    print("=== 外推保持率 ===")
    print(f"  母 cohort (1000题) AverageΔ: {100*mother_avg_delta:+.2f}pp")
    print(f"  held-out  (962题) AverageΔ: {100*held_delta:+.2f}pp")
    if abs(mother_avg_delta) > 1e-9:
        print(f"  保持率: {100*held_delta/mother_avg_delta:.0f}%")

    # ---------- 汇总 JSON ----------
    out = ROOT / "result/e6_heldout_summary_20261110.json"
    out.write_text(json.dumps({
        "paired": len(keys), "nonadversarial": len(nonadv), "adversarial": len(adv),
        "average_mn": round(a_mn, 4), "average_sm": round(a_sm, 4),
        "average_delta_pp": round(100 * (a_mn - a_sm), 2),
        "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
        "normalized_average_delta_pp": round(100 * a_nor, 2),
        "normalized_ci95": [round(100 * lo_n, 2), round(100 * hi_n, 2)],
        "adv_sr_mn": round(statistics.fmean(sr_mn), 4) if adv else None,
        "adv_sr_sm": round(statistics.fmean(sr_sm), 4) if adv else None,
        "sign_test": {"wins": wins, "losses": losses, "p": round(p, 4)},
        "mother_cohort_delta_pp": round(100 * mother_avg_delta, 2),
        "retention_pct": round(100 * held_delta / mother_avg_delta, 0) if abs(mother_avg_delta) > 1e-9 else None,
        "by_category": {
            cat: {
                "n": len(by_cat_mn.get(cat, [])),
                "mn": round(statistics.fmean(by_cat_mn[cat]), 4) if by_cat_mn.get(cat) else None,
                "sm": round(statistics.fmean(by_cat_sm[cat]), 4) if by_cat_sm.get(cat) else None,
            } for cat in CATS if by_cat_mn.get(cat)
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 汇总写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
