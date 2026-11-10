"""Offline regression probe for the two defect fixes (zero cost).

Both fixes were unit-tested, but a unit test only says the new rule fires on a
handwritten string.  This probe replays them against the 178 real MemNexus rows
from `result/bench260` and reports exactly what would change, per question type:

  Fix 1 (temporal): `derive_times` now also resolves "N units before/after
  <written date>".  Anchored expressions were previously dropped, so the
  evidence carried no resolvable time and temporal ranking had nothing to work
  with.

  Fix 2 (synthesis): `normalize_answer_text` now truncates a polar question's
  answer to its leading Yes/No token.  The contract asks for a single polar
  token, but the generator appends a justification, which is a pure precision
  loss.

No model is called.  Numbers below are a *ceiling*: they assume every row still
produces the same pre-fix text, which is only true if retrieval is unchanged.
"""

from __future__ import annotations

import collections
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus" / "memory-system" / "src"))

from agent_memory.synthesis import normalize_answer_text  # noqa: E402
from agent_memory.temporal import derive_times  # noqa: E402

ROWS = ROWS = ROOT / "result" / "bench260" / "rows.json"
POLAR_LEAD = ("yes", "no")


def _is_polar(question: str) -> bool:
    import re

    return bool(
        re.match(
            r"^(?:does|do|did|is|are|was|were|has|have|had|can|could|will|would|should|am)\b",
            question.strip(),
            re.IGNORECASE,
        )
    )


def token_f1(pred: str, gold: str) -> float:
    import re
    from collections import Counter

    tok = lambda t: re.findall(r"[a-z0-9]+|[一-鿿]", str(t).casefold())  # noqa: E731
    p, g = tok(pred), tok(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    nexus = [r for r in rows if r["system"] == "MemNexus"]

    print("=" * 72)
    print("Fix 2 — polar-answer truncation (normalize_answer_text)")
    print("=" * 72)
    polar = [r for r in nexus if _is_polar(r.get("evaluation_query") or r["question"])]
    changed = []
    for row in polar:
        query = row.get("evaluation_query") or row["question"]
        before = str(row["prediction"]).strip()
        after = normalize_answer_text(before, query)
        if after != before:
            f1_before = token_f1(before, row["gold_answer"])
            f1_after = token_f1(after, row["gold_answer"])
            changed.append((row, before, after, f1_before, f1_after))
    print(f"polar-question rows: {len(polar)}   would be rewritten: {len(changed)}")
    delta = 0.0
    for row, before, after, f1_before, f1_after in changed:
        delta += f1_after - f1_before
        print(f"\n  [{row['category']}] {row['question'][:88]}")
        print(f"    gold  : {str(row['gold_answer'])[:76]}")
        print(f"    before: {before[:100]}")
        print(f"    after : {after[:100]}")
        print(f"    F1 {f1_before:.3f} -> {f1_after:.3f}  ({f1_after - f1_before:+.3f})")
    if polar:
        print(f"\n  sum of F1 change over polar rows: {delta:+.3f}"
              f"   (mean {delta / len(polar):+.4f} over {len(polar)} rows)")

    print()
    print("=" * 72)
    print("Fix 1 — relative date resolution (derive_times)")
    print("=" * 72)
    total_sources = 0
    total_derivations = 0
    new_rule_hits = []
    by_category = collections.Counter()
    for row in nexus:
        for source in row.get("retrieved_sources") or []:
            content = source.get("content") if isinstance(source, dict) else str(source)
            if not content:
                continue
            total_sources += 1
            anchor = datetime(2023, 6, 1, tzinfo=UTC)
            derived = derive_times(content, anchor)
            total_derivations += len(derived)
            fresh = [d for d in derived if d.rule == "offset_from_written_date"]
            if fresh:
                by_category[row["category"]] += 1
                for item in fresh:
                    new_rule_hits.append((row, content, item))

    print(f"retrieved source records replayed: {total_sources}")
    print(f"temporal derivations now produced: {total_derivations}")
    print(f"records gaining a NEW derivation from the fix: {len(new_rule_hits)}"
          f"  ({len(new_rule_hits) / max(1, total_sources):.2%} of sources)")
    print("\n  by question category:", dict(by_category))
    print("\n  examples of what the fix now resolves:")
    for row, content, item in new_rule_hits[:8]:
        print(f"    [{row['category']}] {item.expression!r} -> "
              f"{item.start}..{item.end_exclusive} (precision={item.precision})")
        print(f"        from: ...{content[max(0, item.span_start - 40):item.span_end + 30]}...")


if __name__ == "__main__":
    main()
