"""Paired comparison: temporal_rescue variant vs baseline (same 150 questions)."""

from __future__ import annotations

import json
import math
from collections import defaultdict

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "result" / "bench150_conv48-50_20261023" / "rows_baseline.json"
VARIANT = ROOT / "result" / "bench150_temporal_rescue" / "rows.json"
OUT = ROOT / "result" / "temporal_rescue_ab_20261023.json"
NAMES = {"1": "multi-hop", "2": "temporal", "3": "open-domain", "4": "single-hop", "5": "adversarial"}


def keyed(rows, system):
    return {(r["sample_id"], r["qa_index"]): r for r in rows if r["system"] == system}


def paired(keys, a, b):
    d = [float(b[k]["f1"]) - float(a[k]["f1"]) for k in keys]
    n = len(d)
    mean = sum(d) / n
    se = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5 / math.sqrt(n)
    return {
        "n": n, "delta": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se,
        "better": sum(1 for x in d if x > 1e-9), "worse": sum(1 for x in d if x < -1e-9),
    }


def main() -> int:
    base = keyed(json.loads(BASELINE.read_text(encoding="utf-8")), "MemNexus")
    var = keyed(json.loads(VARIANT.read_text(encoding="utf-8")), "MemNexus")
    keys = sorted(set(base) & set(var))
    report = {"overall": paired(keys, base, var)}
    by_cat = defaultdict(list)
    for k in keys:
        by_cat[str(base[k]["category_id"])].append(k)
    report["by_category"] = {
        NAMES.get(c, c): paired(ks, base, var) for c, ks in sorted(by_cat.items())
    }
    fired = [k for k in keys if any(w == "temporal_rescue" for w in (var[k].get("warnings") or []))]
    report["fired"] = {
        "count": len(fired),
        "questions": [
            {
                "key": list(k),
                "delta": float(var[k]["f1"]) - float(base[k]["f1"]),
                "gold": str(base[k]["gold_answer"])[:80],
                "answer": (var[k].get("reasoning") or {}).get("temporal_rescue", {}).get("answer", ""),
                "base_pred": (base[k]["prediction"] or "")[:80],
            }
            for k in fired
        ],
    }
    if fired:
        report["fired"]["paired"] = paired(fired, base, var)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    overall = report["overall"]
    print(f"overall ΔF1 = {overall['delta']:+.4f} CI [{overall['lo']:+.4f},{overall['hi']:+.4f}] "
          f"n={overall['n']} {overall['better']}W/{overall['worse']}L")
    for name, stat in report["by_category"].items():
        print(f"  {name:12s} ΔF1={stat['delta']:+.4f} CI [{stat['lo']:+.4f},{stat['hi']:+.4f}] "
              f"n={stat['n']} {stat['better']}W/{stat['worse']}L")
    print(f"fired: {report['fired']['count']}")
    for q in report["fired"]["questions"]:
        print(f"  {q['key']} Δ={q['delta']:+.2f} gold={q['gold'][:40]!r} got={q['answer'][:40]!r}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
