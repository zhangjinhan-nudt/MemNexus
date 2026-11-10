"""Upper bound on the Average gain available from fixing enumeration length alone.

For each reasoning category we split the paired items by whether the prediction
enumerates more items than the gold answer (splitting on the connectors the
official scorer honours).  We then ask: if every "longer" item had instead scored
at the mean of the "equal" items in its own category, how much would the paper's
``Average`` move?

This is a CEILING, not a forecast: it assumes the extra items are pure loss and
that a fix exists which removes them without touching anything else.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"
ADVERSARIAL = 5
CATS = ("multi-hop", "temporal", "open-domain", "single-hop")


def split_items(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;]", str(text or "")) if p.strip()]


def main() -> int:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r.get("sample_id"), r.get("qa_index")), {})[r.get("system")] = r
    pairs = [(v.get("MemNexus"), v.get("SimpleMem")) for v in by_key.values()
             if v.get("MemNexus") and v.get("SimpleMem")
             and str(v["MemNexus"].get("prediction") or "").strip()
             and str(v["SimpleMem"].get("prediction") or "").strip()]

    base = defaultdict(list)
    gain = defaultdict(list)
    for mn, sm in pairs:
        if mn.get("category_id") == ADVERSARIAL:
            continue
        f1 = 100 * pp.f1_of(mn)
        cat = mn.get("category")
        p = len(split_items(mn.get("prediction")))
        g = len(split_items(mn.get("gold_answer")))
        bucket = "longer" if p > g else ("shorter" if p < g else "equal")
        base[(cat, bucket)].append(f1)

    print(f"{'category':<14}{'bucket':<9}{'n':>5}{'mean F1':>10}")
    for cat in CATS:
        for b in ("equal", "longer", "shorter"):
            v = base[(cat, b)]
            if v:
                print(f"{cat:<14}{b:<9}{len(v):>5}{statistics.mean(v):>10.2f}")
    print()

    # The category mean is taken over ALL items in the category, so the
    # denominator is the full category count (equal + longer + shorter), not
    # just the two buckets being compared.
    per_cat_gain = []
    print(f"{'category':<14}{'n_cat':>7}{'n_longer':>9}{'gap':>9}{'cat gain':>10}")
    for cat in CATS:
        eq, lg = base[(cat, "equal")], base[(cat, "longer")]
        n_cat = sum(len(base[(cat, b)]) for b in ("equal", "longer", "shorter"))
        if not lg or not eq:
            per_cat_gain.append(0.0)
            print(f"{cat:<14}{n_cat:>7}{len(lg):>9}{'-':>9}{0.0:>10.2f}")
            continue
        gap = statistics.mean(eq) - statistics.mean(lg)
        g = len(lg) * gap / n_cat
        per_cat_gain.append(g)
        print(f"{cat:<14}{n_cat:>7}{len(lg):>9}{gap:>9.2f}{g:>10.2f}")
    print()
    print(f"CEILING on Average from fixing enumeration length: "
          f"{statistics.mean(per_cat_gain):+.2f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
