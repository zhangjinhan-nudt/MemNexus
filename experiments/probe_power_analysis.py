"""Is the current sample large enough to support ANY conclusion?

Two questions asked together, because they decide whether a full run is worth
paying for:

  Q1. EFFECT SIZE -- given the per-item paired differences we already have,
      what is the standard error of the headline gap, and what sample size
      would be needed to call a 2.34-point mechanism effect "real"?

  Q2. COVERAGE -- how much of the full 1986-question LoCoMo set do the 175
      paired items represent, per reasoning type, and what is the standard
      error on the weakest type (open-domain, n=9)?

Uses only the recorded run; no paid calls.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
ADVERSARIAL = 5
CATS = ("multi-hop", "temporal", "open-domain", "single-hop")


def main() -> int:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r.get("sample_id"), r.get("qa_index")), {})[r.get("system")] = r
    pairs = [(v["MemNexus"], v["SimpleMem"]) for v in by_key.values()
             if v.get("MemNexus") and v.get("SimpleMem")
             and str(v["MemNexus"].get("prediction") or "").strip()
             and str(v["SimpleMem"].get("prediction") or "").strip()]

    print("=== Q1. effect size and standard error of the headline gap ===")
    per_cat: dict[str, list[float]] = {c: [] for c in CATS}
    for mn, sm in pairs:
        if mn.get("category_id") == ADVERSARIAL:
            continue
        per_cat[mn.get("category")].append(100 * (pp.f1_of(mn) - pp.f1_of(sm)))
    for c in CATS:
        d = per_cat[c]
        if len(d) < 2:
            continue
        m = statistics.mean(d)
        sd = statistics.stdev(d)
        se = sd / math.sqrt(len(d))
        print(f"  {c:<13} n={len(d):>3}  mean delta={m:+7.2f}  "
              f"sd={sd:6.2f}  se={se:5.2f}  95%CI=[{m-1.96*se:+7.2f},{m+1.96*se:+7.2f}]")

    # Average is the unweighted mean of the four category means
    cat_means = [statistics.mean(per_cat[c]) for c in CATS if per_cat[c]]
    avg = statistics.mean(cat_means)
    # SE of the unweighted mean of four independent category means
    se_cat = []
    for c in CATS:
        d = per_cat[c]
        if len(d) >= 2:
            se_cat.append(statistics.stdev(d) / math.sqrt(len(d)))
    se_avg = math.sqrt(sum(s * s for s in se_cat)) / len(se_cat)
    print(f"\n  Average gap = {avg:+.2f}, SE(Average) = {se_avg:.2f}, "
          f"95%CI = [{avg-1.96*se_avg:+.2f},{avg+1.96*se_avg:+.2f}]")
    n_needed = None
    if se_avg > 0:
        # n scales as 1/sqrt(n); to make the CI half-width <= effect
        effect = 2.34
        scale = (1.96 * se_avg / effect) ** 2
        n_needed = math.ceil(len(pairs) * scale)
    print(f"  To resolve a {2.34:.2f}-point mechanism effect at 95% confidence "
          f"you would need roughly n = {n_needed} paired items "
          f"(you have {len(pairs)}).")
    print()

    print("=== Q2. coverage of the full 1986-question set ===")
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    full: dict[str, int] = {c: 0 for c in CATS}
    full_adv = 0
    for conv in data:
        for q in conv.get("qa", []):
            cid = q.get("category")
            if cid == ADVERSARIAL:
                full_adv += 1
            else:
                full[CATS[cid - 1]] = full.get(CATS[cid - 1], 0) + 1
    done: dict[str, int] = {c: len(per_cat[c]) for c in CATS}
    print(f"{'type':<14}{'done':>6}{'full':>7}{'cover':>8}{'max SE*':>10}")
    for c in CATS:
        cov = done[c] / full[c] if full[c] else 0.0
        # worst-case SE for a proportion-like mean with p=0.5
        se_max = 0.5 / math.sqrt(done[c]) * 100 if done[c] else float("nan")
        print(f"{c:<14}{done[c]:>6}{full[c]:>7}{100*cov:>7.1f}%{se_max:>10.1f}")
    print(f"{'adversarial':<14}{sum(1 for mn,_ in pairs if mn.get('category_id')==ADVERSARIAL):>6}"
          f"{full_adv:>7}")
    print()
    print("  * max SE = worst-case standard error of that type's mean F1 (p=0.5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
