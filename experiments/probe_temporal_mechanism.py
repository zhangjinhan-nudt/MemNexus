"""Is the temporal gain concentrated where rule-based time parsing applies?

The overall decomposition says the only statistically significant mechanism gain
is on Temporal (+11.41, p=0.009).  The candidate mechanism is ``derive_times``,
which resolves relative/anchored expressions ("five years before July 2023",
"two weeks before 2022-03-18", "last Friday") into concrete date intervals --
work SimpleMem delegates to free-form LLM generation.

If that is the mechanism, the gain should concentrate on questions whose gold
answer is a date/year, and on questions that carry a temporal cue.  It should
not appear on temporal questions whose gold is a bare entity.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"

MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december")
DATE_LIKE = re.compile(
    rf"\b(?:\d{{1,2}}\s+(?:{MONTHS})|(?:{MONTHS})\s+\d{{1,2}}|\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|\b(?:19|20)\d{{2}}\b)\b",
    re.IGNORECASE,
)
RELATIVE = re.compile(
    r"\b(?:last|next|past|previous|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|"
    fr"friday|saturday|sunday|{MONTHS})\b"
    r"|\b\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|later|before|after|prior to|earlier)\b"
    r"|\b(?:yesterday|tomorrow|today)\b",
    re.IGNORECASE,
)


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    paired = pp.common_subset(rows, ("SimpleMem", "MemNexus"))
    buckets: dict[tuple, dict] = {}
    for row in paired:
        key = (row["sample_id"], row["qa_index"])
        buckets.setdefault(key, {})[row["system"]] = row

    groups = {"gold 是日期/年份": [], "gold 非日期": []}
    cue = {"问题含时间线索": [], "问题无时间线索": []}
    for entry in buckets.values():
        mn, sm = entry.get("MemNexus"), entry.get("SimpleMem")
        if mn is None or sm is None or mn.get("category_id") != 2:
            continue
        diff = pp.f1_of(mn) - pp.f1_of(sm)
        gold = str(mn.get("gold_answer") or "")
        question = str(mn.get("question") or "")
        groups["gold 是日期/年份" if DATE_LIKE.search(gold) else "gold 非日期"].append(diff)
        cue["问题含时间线索" if RELATIVE.search(question) else "问题无时间线索"].append(diff)

    for title, data in (("按金标答案形态", groups), ("按问题措辞", cue)):
        print("=" * 76)
        print(f"  {title}")
        print("=" * 76)
        for name, values in data.items():
            if not values:
                continue
            wins = sum(1 for v in values if v > 1e-9)
            losses = sum(1 for v in values if v < -1e-9)
            print(f"  {name:<16s} n={len(values):3d}   ΔF1={statistics.fmean(values) * 100:+7.2f}"
                  f"   胜{wins:3d} 负{losses:3d}")
        print()


if __name__ == "__main__":
    main()
