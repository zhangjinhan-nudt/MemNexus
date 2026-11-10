"""Split the MemNexus losses by the stage that created them.

Definitions
-----------
gold_cov  = |gold tokens ∩ evidence tokens| / |gold tokens|
            -> low  : the fact never reached the model        (L1/L4)
hit       = |gold tokens ∩ prediction tokens| / |gold tokens|
            -> high : the model knew the answer               (L7 render)
noise     = |prediction tokens - gold tokens| / |prediction tokens|
            -> high : the answer is buried in extra text     (L7 render)

A loss with high hit and high noise is a RENDERING loss, not a knowledge loss.
Fixing it does not require a better memory system, only a stricter writer.
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
STOP = {
    "the", "is", "a", "an", "what", "was", "of", "my", "me", "please", "and",
    "to", "in", "on", "at", "for", "did", "does", "do", "are", "he", "she",
    "they", "it", "that", "this", "with", "how", "many", "when", "which",
    "who", "had", "has", "have", "been", "be", "her", "his",
}


def tokens(text) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").casefold()) if t not in STOP and len(t) > 1}


def evidence_text(row: dict) -> str:
    chunks = []
    for src in row.get("retrieved_sources") or []:
        chunks.append(str(src.get("content") or ""))
        for ev in ((src.get("metadata") or {}).get("evidence") or []):
            chunks.append(str(ev.get("text") or ""))
    return " ".join(chunks)


def main() -> int:
    import paper_protocol as pp

    rows = json.loads((ROOT / "result" / "bench260" / "rows.json").read_text(encoding="utf-8"))
    by: dict = {}
    for r in rows:
        by.setdefault((r.get("sample_id"), r.get("qa_index")), {})[r.get("system")] = r
    pairs = []
    for v in by.values():
        mn, sm = v.get("MemNexus"), v.get("SimpleMem")
        if not mn or not sm:
            continue
        if not str(mn.get("prediction") or "").strip() or not str(sm.get("prediction") or "").strip():
            continue
        mn["f1"] = pp.f1_of(mn)
        sm["f1"] = pp.f1_of(sm)
        pairs.append((mn, sm))

    losses = [(mn, sm) for mn, sm in pairs if mn["f1"] - sm["f1"] < -0.005]
    buckets = defaultdict(list)
    for mn, sm in losses:
        g = tokens(mn.get("gold_answer"))
        p = tokens(mn.get("prediction"))
        e = tokens(evidence_text(mn))
        cov = len(g & e) / len(g) if g else 1.0
        hit = len(g & p) / len(g) if g else 1.0
        noise = len(p - g) / len(p) if p else 0.0
        if hit >= 0.6 and noise >= 0.6:
            buckets["R 渲染层：答案正确但被多余文本稀释"].append((mn, sm, cov, hit, noise))
        elif hit >= 0.6:
            buckets["P 表述层：答案在，词面不同"].append((mn, sm, cov, hit, noise))
        elif cov < 0.34:
            buckets["K 表示/召回层：金标事实未进证据"].append((mn, sm, cov, hit, noise))
        else:
            buckets["V 绑定层：证据在手取错值"].append((mn, sm, cov, hit, noise))

    print(f"MemNexus losses: {len(losses)}")
    print()
    for key in sorted(buckets, key=lambda k: -len(buckets[k])):
        items = buckets[key]
        print(f"{key}  ->  {len(items)} items")
        for mn, sm, cov, hit, noise in items:
            print(f"    {mn['sample_id']}/{mn['qa_index']:>3} {mn['category']:<12} "
                  f"cov={cov:.2f} hit={hit:.2f} noise={noise:.2f}  "
                  f"G={str(mn.get('gold_answer'))[:44]!r}")
            print(f"        MN={str(mn.get('prediction'))[:96]!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
