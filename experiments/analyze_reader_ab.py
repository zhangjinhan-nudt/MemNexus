"""Paired A/B of the LLM reader variants against the 150-run baseline rows.

The reader variants only change the MemNexus answering stage on a 42-question
non-adversarial subset, so the baseline rows are filtered to the same question
keys and compared pairwise.  Telemetry: how often the reader actually fired
(hint on every question, fallback only on abstentions) and how often its quote
verified.

Usage: python experiments/analyze_reader_ab.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "result" / "locomo_42_reader_cohort_20261023.json"
BASE = ROOT / "result" / "bench150_conv48-50_20261023" / "rows_baseline.json"
VARIANTS = {"fallback": ROOT / "result" / "bench42_reader" / "fallback" / "rows.json",
            "hint": ROOT / "result" / "bench42_reader" / "hint" / "rows.json"}
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def key_of(row: dict) -> tuple:
    return (row["sample_id"], row["qa_index"])


def main() -> int:
    cohort = load(COHORT)
    keys = {(q["sample_id"], q["qa_index"]) for q in cohort["questions"]}
    cat_of = {(q["sample_id"], q["qa_index"]): q["category_id"] for q in cohort["questions"]}

    base = {key_of(r): r for r in load(BASE)
            if r["system"] == "MemNexus" and key_of(r) in keys}
    print(f"baseline rows available for {len(base)}/{len(keys)} cohort questions")

    summary = {}
    for name, path in VARIANTS.items():
        if not path.exists():
            print(f"{name}: rows.json missing, skipped")
            continue
        rows = {key_of(r): r for r in load(path) if r["system"] == "MemNexus" and key_of(r) in keys}
        shared = sorted(set(base) & set(rows))
        diffs, by_cat = [], {c: [] for c in CATEGORY.values()}
        fired = verified = 0
        better = worse = same = 0
        for k in shared:
            fb, fv = float(base[k]["f1"]), float(rows[k]["f1"])
            diffs.append(fv - fb)
            by_cat[CATEGORY[cat_of[k]]].append(fv - fb)
            if fv > fb + 1e-9:
                better += 1
            elif fv < fb - 1e-9:
                worse += 1
            else:
                same += 1
            warns = rows[k].get("warnings") or []
            if any(w.startswith("reader_fallback") for w in warns):
                fired += 1
            if any(w == "reader_hint" for w in warns):
                fired += 1
            trace = rows[k].get("reasoning") or {}
            if trace.get("reader_fallback") or trace.get("reader_hint"):
                verified += 1
        n = len(diffs)
        mean = statistics.mean(diffs) if n else 0.0
        sd = statistics.stdev(diffs) if n > 1 else 0.0
        se = sd / (n ** 0.5) if n else 0.0
        calls = statistics.mean(
            (rows[k].get("model_calls") or {}).get("provider_attempts", 0) for k in shared)
        summary[name] = {
            "n": n, "mean_delta": round(mean, 3), "se": round(se, 3),
            "ci": (round(mean - 1.96 * se, 3), round(mean + 1.96 * se, 3)),
            "better": better, "same": same, "worse": worse,
            "fired": fired, "verified": verified, "calls_per_q": round(calls, 2),
            "by_category": {c: round(statistics.mean(v), 3) for c, v in by_cat.items() if v},
        }
        print(f"\n{name}: n={n} ΔF1={mean:+.3f} CI [{mean-1.96*se:+.3f},{mean+1.96*se:+.3f}]")
        print(f"  better={better} same={same} worse={worse} | reader fired={fired} "
              f"verified={verified} | calls/q={calls:.2f} (baseline 2.39)")
        print("  by category:", summary[name]["by_category"])

    base_f1 = statistics.mean(float(base[k]["f1"]) for k in base)
    print(f"\nbaseline F1 on these {len(base)} questions: {base_f1:.3f}")
    (ROOT / "result" / "reader_ab_20261023.json").write_text(
        json.dumps({"baseline_f1": round(base_f1, 4), "variants": summary},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote result/reader_ab_20261023.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
