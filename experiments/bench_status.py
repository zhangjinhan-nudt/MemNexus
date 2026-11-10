"""One-glance progress for the conv-48/49/50 benchmark.

Usage:  python experiments\\bench_status.py [out_dir]
        (out_dir defaults to the current 150-question run)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "result" / "bench150_conv48-50_20261023"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    attempts = out / "attempts"
    print(f"run dir: {out}")
    if not attempts.exists():
        print("not started yet")
        return 0
    for unit in sorted(attempts.iterdir()):
        done = (unit / "stage_result.json").exists()
        failed = (unit / "failure.json").exists()
        rows = 0
        rows_file = unit / "usage.rows.jsonl"
        if rows_file.exists():
            rows = sum(1 for line in rows_file.read_text(encoding="utf-8").splitlines() if line.strip())
        state = "DONE" if done else ("retrying after failure" if failed else "running")
        cost = ""
        if done:
            payload = json.loads((unit / "stage_result.json").read_text(encoding="utf-8"))
            usage = payload.get("stage", {}).get("total_usage", {})
            cost = f"  cost ¥{usage.get('cost_rmb', 0):.2f}"
        print(f"  {unit.name:28s} {state:22s} {rows:4d} questions answered{cost}")
    total = sum(
        sum(1 for line in (u / "usage.rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
        for u in attempts.iterdir() if (u / "usage.rows.jsonl").exists()
    )
    print(f"\n[{time.strftime('%H:%M:%S')}] answered so far: {total} / 150 (x2 systems = 300 rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
