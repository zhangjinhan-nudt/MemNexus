"""Repair run: fills in the units a rate-limited run could not finish.

Runs the same cohort and the same output directory, so units whose
``stage_result.json`` exists are skipped; only the failed ones re-run. Uses
concurrency 1 and a 60s stagger to stay well inside the provider's rate limit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["BENCH_CONCURRENCY"] = "2"
os.environ["BENCH_STAGGER"] = "30"

import launch_bench637  # noqa: E402  (env must be set before import-time reads)

if __name__ == "__main__":
    raise SystemExit(launch_bench637.main())
