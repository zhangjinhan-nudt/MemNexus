"""Is per-question latency flat in memory-base size? It decides whether 7 h of
full-sessions data costs what a linear extrapolation says.

The previous run built every base from D1-D5 only. A full-sessions run is
several times larger, so if retrieval latency grows with memory count the
throughput estimate has to be corrected, not just scaled by question count.
"""

from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = str(ROOT / "result" / "bench260" / "attempts" / "*" / "stage_result.json")

SYSTEMS = ("SimpleMem", "MemNexus")


def main() -> None:
    points = defaultdict(list)
    for path in sorted(glob.glob(PATTERN)):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stage = payload["stage"]
        rows = payload["rows"]
        if not rows:
            continue
        system = "MemNexus" if stage.get("system") == "MemNexus" else "SimpleMem"
        lat = [((r.get("latency_ms") or 0) / 1000.0) for r in rows]
        points[system].append(
            (
                stage.get("memory_count", 0),
                stage.get("raw_turns", 0),
                statistics.mean(lat),
                len(rows),
            )
        )

    print("=" * 78)
    print("延迟 vs 记忆库规模（按段对话聚合）")
    print("=" * 78)
    for system in SYSTEMS:
        items = sorted(points[system])
        print(f"\n[{system}]")
        print(f"  {'conv':<10s} {'mem':>5s} {'turns':>6s} {'n':>4s} {'lat_mean':>9s} {'s/题/千条记忆':>14s}")
        for mem, turns, lat, n in items:
            norm = lat / (mem / 1000.0) if mem else 0.0
            print(f"  {'':<10s} {mem:5d} {turns:6d} {n:4d} {lat:9.2f} {norm:14.2f}")
        lats = [x[2] for x in items]
        mems = [x[0] for x in items]
        print(f"  记忆数范围 {min(mems)}–{max(mems)}，延迟范围 {min(lats):.2f}–{max(lats):.2f}s")
        if max(mems) > min(mems):
            lo = [x[2] for x in items if x[0] <= statistics.median(mems)]
            hi = [x[2] for x in items if x[0] > statistics.median(mems)]
            print(f"  小库均值 {statistics.mean(lo):.2f}s vs 大库均值 {statistics.mean(hi):.2f}s"
                  f"  （比值 {statistics.mean(hi)/statistics.mean(lo):.2f}×）")


if __name__ == "__main__":
    main()
