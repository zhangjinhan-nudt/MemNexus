"""Overnight 637-question full-cohort run (both systems).

Launched by the MemNexus637 scheduled task so it survives session teardown.
Env:
  MEMORY_TEMPORAL_RESCUE  1 = run with the temporal rescue flag on
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\zhangjinhan\AppData\Local\Programs\Python\Python313\python.exe")
COHORT = "result/locomo_637_cohort_conv48-50_20261023.json"
OUT = "result/bench637_night"
LOG = ROOT / "result" / "bench637_night.log"

COHORT_PATH = ROOT / COHORT
if not COHORT_PATH.exists():
    print(f"missing cohort {COHORT_PATH}", flush=True)
    sys.exit(2)


def main() -> int:
    environment = dict(os.environ)
    environment.update({
        "NO_PROXY": "127.0.0.1,localhost",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "MEMORY_TEMPORAL_RESCUE": os.environ.get("NIGHT_TEMPORAL_RESCUE", "0"),
    })
    command = [
        str(PYTHON), "experiments/run_parallel_bench.py",
        "--cohort", COHORT, "--out", OUT,
        "--snapshots", "result/memnexus_snapshots/full30",
        "--concurrency", "2", "--stagger", "30",
        "--budget-per-conversation", "60",
    ]
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"start cohort={COHORT} out={OUT} rescue={environment['MEMORY_TEMPORAL_RESCUE']}\n")
        log.flush()
        return subprocess.Popen(command, cwd=str(ROOT), env=environment,
                                stdout=log, stderr=subprocess.STDOUT).wait()


if __name__ == "__main__":
    sys.exit(main())
