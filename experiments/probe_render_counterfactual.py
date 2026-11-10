"""Counterfactual: how much of the gap is reachable by stricter ANSWER WRITING.

The audit shows the MemNexus-loses group writes roughly three times longer
answers than the tie group.  This script asks: if we changed only how the value
is *rendered* -- never what was retrieved -- how much would Average move?

Variants (purely mechanical, no model calls):

  as_written  : the recorded prediction
  first_item  : keep only the first comma-separated value  (naive truncation)
  novel_only  : keep the first value plus later values whose content tokens
                are mostly new                            (selective drop)
  oracle_k    : keep the best prefix, chosen with the gold answer (UPPER BOUND,
                not deployable -- included to bound the lever)
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"

STOP = {
    "the", "is", "a", "an", "what", "was", "of", "my", "me", "please", "and",
    "to", "in", "on", "at", "for", "did", "does", "do", "are", "he", "she",
    "they", "it", "that", "this", "with", "how", "many", "when", "which",
    "who", "had", "has", "have", "been", "be", "her", "his",
}

# Clause boundaries that in practice begin a justification rather than a value.
CLAUSE = re.compile(r"\s+(?:which|because|since|so that|but|while)\b", re.IGNORECASE)


def tokens(text) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").casefold()) if t not in STOP and len(t) > 1}


def items_of(text) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def variants(row: dict) -> dict[str, str]:
    prediction = str(row.get("prediction") or "").strip()
    options = {"as_written": prediction}
    items = items_of(prediction)
    if not items:
        return options
    options["first_item"] = items[0]
    kept = [items[0]]
    seen = tokens(items[0])
    for item in items[1:]:
        item_tokens = tokens(item)
        if not item_tokens:
            continue
        novelty = len(item_tokens - seen) / len(item_tokens)
        if novelty >= 0.6:
            kept.append(item)
            seen |= item_tokens
    options["novel_only"] = ", ".join(kept)
    for k in range(1, len(items) + 1):
        options[f"prefix_{k}"] = ", ".join(items[:k])
    cut = CLAUSE.search(prediction)
    if cut and cut.start() > 12:
        options["clause_cut"] = prediction[: cut.start()].strip().rstrip(",")
    else:
        options["clause_cut"] = prediction
    return options


def main() -> int:
    import paper_protocol as pp

    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    keyed: dict = {}
    for row in rows:
        keyed.setdefault((row.get("sample_id"), row.get("qa_index")), {})[row.get("system")] = row
    pairs = []
    for value in keyed.values():
        mn, sm = value.get("MemNexus"), value.get("SimpleMem")
        if not mn or not sm:
            continue
        if not str(mn.get("prediction") or "").strip():
            continue
        if not str(sm.get("prediction") or "").strip():
            continue
        pairs.append((mn, sm))

    def average_of(scorer) -> dict[str, float]:
        out: dict[str, float] = {}
        for system in ("MemNexus", "SimpleMem"):
            by_type: dict[int, list[float]] = defaultdict(list)
            for mn, sm in pairs:
                row = mn if system == "MemNexus" else sm
                if row.get("category_id") == 5:
                    continue
                by_type[row["category_id"]].append(scorer(row))
            out[system] = statistics.mean(statistics.mean(v) for v in by_type.values())
        return out

    base = average_of(pp.f1_of)

    def build(name: str):
        def scorer(row: dict) -> float:
            options = variants(row)
            if name == "oracle_k":
                keys = [key for key in options if key.startswith("prefix_")]
                if not keys:
                    return pp.f1_of(row)
                # pick the prefix with the highest official F1, NOT the largest string
                best_key = max(keys, key=lambda k: pp.f1_of({**row, "prediction": options[k]}))
                text = options[best_key]
            elif name in options:
                text = options[name]
            else:
                text = options["as_written"]
            return pp.f1_of({**row, "prediction": text})

        return scorer

    print("Baseline (official F1, four reasoning types equally weighted, %d pairs)" % len(pairs))
    print(f"  MemNexus  {100 * base['MemNexus']:.2f}")
    print(f"  SimpleMem {100 * base['SimpleMem']:.2f}")
    print(f"  gap       {100 * (base['MemNexus'] - base['SimpleMem']):+.2f}")
    print()
    print("Counterfactual: change ONLY the rendering of the value")
    for name in ("first_item", "novel_only", "clause_cut", "oracle_k"):
        value = average_of(build(name))["MemNexus"]
        print(f"  {name:12s} {100 * value:6.2f}   ({100 * (value - base['MemNexus']):+.2f} vs as written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
