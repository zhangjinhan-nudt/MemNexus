"""Sequential LLM-variant A/B on one benchmark cohort.

Each variant only changes the MemNexus side, so the SimpleMem attempt units of
the baseline run are reused as-is: every variant gets its OWN out directory
(``result/bench150_variants/<variant>/``) seeded with the baseline SimpleMem
units, and the runner's skip check (``stage_result.json`` exists) then skips
exactly those units while re-executing the three MemNexus units.

Baseline rows are snapshotted before the first variant runs, so variant
results can never overwrite the reference.

IMPORTANT: do NOT run variants inside the baseline out dir. Sharing it makes
every unit already-complete, so all variant runs exit instantly as no-ops and
the snapshots become byte-identical copies of the baseline.

Usage: python experiments\\run_variants.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\zhangjinhan\AppData\Local\Programs\Python\Python313\python.exe")
BASELINE_OUT = ROOT / "result" / "bench150_conv48-50_20261023"
VARIANTS_ROOT = ROOT / "result" / "bench150_variants"
COHORT = "result/locomo_150_cohort_conv48-50_20261023.json"
SNAPSHOTS = "result/memnexus_snapshots/full30"
LOG = ROOT / "result" / "variants.log"

VARIANTS: list[tuple[str, dict[str, str]]] = [
    ("bridge", {"MEMORY_LLM_BRIDGE": "1"}),
    ("gate", {"MEMORY_GATE_FULL_SCAN": "1"}),
    ("execution", {"MEMORY_SEMANTIC_EXECUTION": "1"}),
]


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def snapshot(out: Path, name: str) -> None:
    rows = out / "rows.json"
    if rows.exists():
        shutil.copyfile(rows, out / f"rows_{name}.json")
    log(f"snapshotted {name}")


def finished_units(out: Path) -> int:
    attempts = out / "attempts"
    if not attempts.exists():
        return 0
    return sum(1 for unit in attempts.iterdir() if (unit / "stage_result.json").exists())


def run_variant(name: str, flags: dict[str, str]) -> int:
    environment = dict(os.environ)
    environment.update(flags)
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["PYTHONUNBUFFERED"] = "1"
    out_rel = (VARIANTS_ROOT / name).relative_to(ROOT)
    command = [
        str(PYTHON), "experiments/run_parallel_bench.py",
        "--cohort", COHORT, "--out", str(out_rel),
        "--snapshots", SNAPSHOTS,
        "--concurrency", "2", "--stagger", "30",
        "--budget-per-conversation", "60",
    ]
    log(f"variant {name}: " + " ".join(command) + " env=" + json.dumps(flags))
    bench_log = (ROOT / "result" / f"variant_{name}.log").open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                               stdout=bench_log, stderr=subprocess.STDOUT)
    return process.wait()


def snapshot(name: str) -> None:
    rows = VARIANTS_ROOT / name / "rows.json"
    if rows.exists():
        shutil.copyfile(rows, VARIANTS_ROOT / name / f"rows_{name}.json")
    log(f"snapshotted {name}")


def main() -> int:
    log(f"variant driver start (per-variant dirs under {VARIANTS_ROOT.name})")
    expected = 6
    baseline_units = BASELINE_OUT / "attempts"
    waited = 0.0
    while (baseline_units.exists() and
           sum(1 for u in baseline_units.iterdir() if (u / "stage_result.json").exists()) < expected
           and waited < 8 * 3600):
        time.sleep(60)
        waited += 60
    if not baseline_units.exists():
        log("baseline out dir missing; stopping")
        return 2
    rows_baseline = BASELINE_OUT / "rows_baseline.json"
    if not rows_baseline.exists():
        shutil.copyfile(BASELINE_OUT / "rows.json", rows_baseline)
        log("snapshotted baseline")
    for name, flags in VARIANTS:
        code = run_variant(name, flags)
        log(f"variant {name} exited {code}")
        snapshot(name)
    log("all variants finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
