"""Offline: how many of the 175 paired questions would enter the execution path?

`runtime._answer_impl` picks the output path like this:

    synthesis = answer_synthesis_enabled and hasattr(model,"synthesize")
                and plan.response_intent == "direct_answer" and grounding_enabled
    complex_route = classify_question(query, entities).kind in
                    {adversarial,count,set,relation,causal,inference}
    semantic = synthesis and semantic_execution_enabled
               and hasattr(model,"bind_answer")
               and (not semantic_execution_complex_only or complex_route)

`semantic_execution_enabled` defaults to false, so `semantic` is false and every
row lands on `source_bound_synthesis`.  This probe measures what would change if
the flag were turned on, without calling any model.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus" / "memory-system" / "src"))

from agent_memory.reasoning import classify_question  # noqa: E402

ROWS = ROOT / "result" / "bench260" / "rows.json"
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
MAX_SESSION = 5
COMPLEX = {"adversarial", "count", "set", "relation", "causal", "inference"}


def speakers(sample: dict) -> list[str]:
    conversation = sample["conversation"]
    names = []
    for key, turns in conversation.items():
        if not (key.startswith("session_") and isinstance(turns, list)):
            continue
        if int(key.split("_")[1]) > MAX_SESSION:
            continue
        names.extend(turn.get("speaker", "") for turn in turns)
    return [name for name in names if name]


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    entities = {sample.get("sample_id", str(i)): speakers(sample) for i, sample in enumerate(data)}

    nexus = [r for r in rows if r["system"] == "MemNexus" and r.get("reasoning")]
    kinds = collections.Counter()
    by_category = collections.defaultdict(collections.Counter)
    complex_by_category = collections.Counter()
    for row in nexus:
        query = row.get("evaluation_query") or row["question"]
        route = classify_question(query, entities.get(row["sample_id"], []))
        kinds[route.kind] += 1
        by_category[row["category"]][route.kind] += 1
        if route.kind in COMPLEX:
            complex_by_category[row["category"]] += 1

    total = sum(kinds.values())
    print(f"MemNexus rows with reasoning: {total}")
    print("\nclassify_question kind distribution:")
    for kind, count in kinds.most_common():
        marker = " <- complex" if kind in COMPLEX else ""
        print(f"  {kind:16s} {count:4d}  {count/total:6.1%}{marker}")

    complex_total = sum(complex_by_category.values())
    print(f"\nwould enter semantic execution if enabled: {complex_total}/{total} "
          f"({complex_total/total:.1%})")
    print("\nby official category:")
    for category in sorted(by_category):
        total_c = sum(by_category[category].values())
        print(f"  {category:12s} complex {complex_by_category[category]:3d}/{total_c:3d}"
              f"   {dict(by_category[category].most_common())}")


if __name__ == "__main__":
    main()
