"""Reproduce the two unresolved-relative-time failures off-line.

Pulls the recorded evidence (retrieved_sources -> source_events) for conv-42/8
and conv-47/4 and runs the real ``resolve_relative_value`` / ``derive_times``
against it, so we can see WHY the calendar form was not produced.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MN = ROOT / "MemNexus" / "memory-system" / "src"
if str(MN) not in sys.path:
    sys.path.insert(0, str(MN))

from agent_memory.synthesis import (  # noqa: E402
    derive_times,
    execution_contract,
    render_phrase,
    render_table,
    resolve_relative_value,
)

ROWS = ROOT / "result" / "bench260" / "rows.json"
TARGETS = [("conv-42", 8), ("conv-47", 4)]


def main() -> int:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    for sid, qi in TARGETS:
        row = next((r for r in rows
                    if r.get("system") == "MemNexus"
                    and r.get("sample_id") == sid
                    and r.get("qa_index") == qi), None)
        if row is None:
            print(f"!! {sid}/{qi} not found")
            continue
        print("=" * 78)
        print(f"{sid}/{qi}  [{row.get('category')}]")
        print(f"  Q    : {row.get('question')}")
        print(f"  GOLD : {row.get('gold_answer')}")
        print(f"  MN   : {row.get('prediction')}")
        plan = row.get("plan") or {}
        print(f"  operator in plan.answer_contract: "
              f"{[o.get('operator') for o in ((plan.get('answer_contract') or {}).get('outputs') or [])]}")
        print(f"  query_type={plan.get('query_type')}  temporal_mode={plan.get('temporal_mode')}")

        srcs = row.get("retrieved_sources") or []
        print(f"  retrieved_sources: {len(srcs)}")
        for i, s in enumerate(srcs, 1):
            events = (s.get("metadata") or {}).get("source_events") or []
            print(f"    [{i}] speaker={s.get('speaker')!r} events={len(events)}")
            for e in events[:6]:
                print(f"         occurred_at={e.get('occurred_at')!r} "
                      f"context_only={e.get('context_only')!r}")
                print(f"         text={str(e.get('text') or '')[:150]!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
