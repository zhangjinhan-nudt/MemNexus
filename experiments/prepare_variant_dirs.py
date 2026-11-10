"""Prepare per-variant output directories for the LLM-variant A/B.

Root cause this fixes: run_parallel_bench.py skips any unit whose
``stage_result.json`` exists in ``<out>/attempts/``. The previous design ran
all three variants inside the *same* out dir as the baseline, so every unit
was already complete and all variant runs finished instantly as no-ops
(rows_bridge/gate/execution were byte-identical copies of the baseline).

Fix: each variant gets its own out dir seeded with ONLY the SimpleMem attempt
units from the baseline (SM rows are flag-independent). The runner's skip
check then correctly skips SM and re-executes the three MemNexus units per
variant under the variant's env flags.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "result" / "bench150_conv48-50_20261023"
VARIANTS_ROOT = ROOT / "result" / "bench150_variants"
VARIANTS = ["bridge", "gate", "execution"]


def main() -> int:
    attempts = BASELINE / "attempts"
    if not attempts.exists():
        print("baseline attempts dir missing:", attempts)
        return 1
    sm_units = sorted(p for p in attempts.iterdir() if p.name.endswith("_simplemem"))
    print(f"seed SimpleMem units: {len(sm_units)}")
    for name in VARIANTS:
        target = VARIANTS_ROOT / name
        target_attempts = target / "attempts"
        if target_attempts.exists():
            print(f"{name}: exists, keeping (resume-safe)")
            continue
        target_attempts.mkdir(parents=True)
        for unit in sm_units:
            shutil.copytree(unit, target_attempts / unit.name)
        print(f"{name}: seeded {len(sm_units)} SimpleMem units -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
