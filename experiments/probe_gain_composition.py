"""What kind of win is it?  Split each paired difference into three cases.

For every matched question, compare the two predictions against the gold F1:

  recall_win   SimpleMem declined ("not mentioned" / "insufficient" / ...) while
               MemNexus produced something that scores -- MemNexus found evidence
               SimpleMem did not retrieve.
  quality_win  both answered, MemNexus scores higher -- closer/cleaner wording,
               or better selection among the retrieved candidates.
  loss / tie   everything else.

The split matters because the two cases justify different claims: recall wins
support a claim about the memory/retrieval layer, quality wins support a claim
about the answer-contract / validation layer.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
ROWS = ROOT / "result" / "bench260" / "rows.json"
ORDER = ("MultiHop", "Temporal", "OpenDomain", "SingleHop")

DECLINE = re.compile(
    r"not mentioned|not specified|insufficient|not provided|no information|"
    r"cannot be determined|not in the (?:provided |given )?(?:context|conversation)|"
    r"does not (?:mention|specify|provide)|unknown",
    re.IGNORECASE,
)


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    paired = pp.common_subset(rows, ("SimpleMem", "MemNexus"))
    buckets: dict[tuple, dict] = defaultdict(dict)
    for row in paired:
        key = (row["sample_id"], row["qa_index"])
        buckets[key][row["system"]] = row

    stats = defaultdict(lambda: defaultdict(float))
    for entry in buckets.values():
        mn, sm = entry.get("MemNexus"), entry.get("SimpleMem")
        if mn is None or sm is None:
            continue
        cid = mn["category_id"]
        if cid == pp.ADVERSARIAL_CATEGORY:
            continue
        name = pp.REASONING_TYPES[cid]
        mn_f1, sm_f1 = pp.f1_of(mn), pp.f1_of(sm)
        stats[name]["n"] += 1
        stats[name]["delta"] += mn_f1 - sm_f1
        sm_declined = bool(DECLINE.search(str(sm.get("prediction") or "")))
        mn_declined = bool(DECLINE.search(str(mn.get("prediction") or "")))
        if sm_declined and not mn_declined and mn_f1 > 0.05:
            stats[name]["recall_win"] += 1
        elif mn_f1 - sm_f1 > 0.05:
            # SimpleMem also produced content; did it overlap the gold at all?
            if sm_f1 < 0.05 and not sm_declined:
                stats[name]["evidence_miss"] += 1
            else:
                stats[name]["wording_win"] += 1
        elif mn_f1 - sm_f1 < -0.05:
            stats[name]["loss"] += 1
        else:
            stats[name]["tie"] += 1

    print("=" * 88)
    print("  增益构成：MemNexus 是怎么赢的（178 对配对题）")
    print("=" * 88)
    print(f"{'类型':<12s}{'n':>4s}{'ΔF1':>9s}{'SM拒答':>9s}{'SM答偏':>9s}"
          f"{'MN更近':>9s}{'MN更差':>9s}{'打平':>7s}")
    print("-" * 88)
    for name in ORDER:
        item = stats[name]
        count = int(item["n"])
        if not count:
            continue
        print(f"{name:<12s}{count:>4d}{item['delta'] / count * 100:+8.2f}"
              f"{int(item['recall_win']):>9d}{int(item['evidence_miss']):>9d}"
              f"{int(item['wording_win']):>9d}{int(item['loss']):>9d}"
              f"{int(item['tie']):>7d}")
    print("-" * 88)
    print("  SM拒答 = SimpleMem 抽答/拒答，MemNexus 拿出了内容（召回层差异）")
    print("  SM答偏 = SimpleMem 也答了，但内容几乎不沾金标（证据选错）")
    print("  MN更近 = 两边都答到点上，MemNexus 更接近/更简练（输出层差异）")
    print()


def _unused() -> None:  # pragma: no cover - keeps linters honest about the helper
    return None


if __name__ == "__main__":
    main()
