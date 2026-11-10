"""Overnight 150-item A/B driver.

Each (cell, repeat) runs in its own subprocess so environment switches cannot
leak between cells and one crash cannot kill the night. Memory bases are
replayed from a frozen snapshot directory, so no build cost is paid.

Usage (driver):
    python experiments/overnight_run.py --deadline 07:00 --budget 28

Usage (single cell, used by the driver):
    python experiments/overnight_run.py --cell default --rep 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
COHORT_RUN = ROOT / "result" / "benchmark150_paper" / "20261020_000448_049755"
DEFAULT_ROOT = ROOT / "result" / "overnight_20261021"
DEFAULT_SNAPSHOTS = ROOT / "result" / "memnexus_snapshots" / "main"

# Cell definitions. Every switch is written out explicitly so the run is
# self-documenting; an empty dict means "code defaults, no override".
CELLS = {
    "default": {},
    "no_grounding": {"MEMORY_GROUNDING_ENABLED": "false"},
    "no_normalize": {"MEMORY_ANSWER_NORMALIZE": "false"},
}

# Interleaved so that a truncated night still leaves every cell with one run.
PLAN = [
    ("default", 1),
    ("no_grounding", 1),
    ("no_normalize", 1),
    ("default", 2),
    ("no_grounding", 2),
    ("no_normalize", 2),
    ("default", 3),
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def cohort():
    """The 150 questions of the paper run, in official order."""
    rows = [r for r in read(COHORT_RUN / "results.json")["rows"] if r["system"] == "MemNexus"]
    seen, out = set(), []
    for row in rows:
        key = (row["sample_id"], row["qa_index"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "sample_id": row["sample_id"],
                "qa_index": row["qa_index"],
                "category": row["category"],
                "category_id": row["category_id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
            }
        )
    return out


async def run_cell(cell: str, rep: int, root: Path, snapshots: Path):
    for name, value in CELLS[cell].items():
        os.environ[name] = value
    os.environ["MEMORY_INPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "2.4"
    os.environ["MEMORY_OUTPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "9.6"
    os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "8192"

    from run_locomo_pilot import run_memnexus  # imported after env is set

    out = root / f"{cell}_r{rep}"
    out.mkdir(parents=True, exist_ok=True)
    items = cohort()
    write(out / "questions.json", items)

    dataset = {s["sample_id"]: s for s in read(DATASET)}
    by_sample: dict[str, list[dict]] = {}
    for item in items:
        by_sample.setdefault(item["sample_id"], []).append(item)

    status_path = out / "status.json"
    status = read(status_path) if status_path.exists() else {
        "cell": cell, "rep": rep, "env": CELLS[cell],
        "started": datetime.now().isoformat(timespec="seconds"),
        "samples": {}, "cost_rmb": 0.0,
    }

    for sample_id, sample_items in by_sample.items():
        stage_path = out / f"{sample_id}.json"
        if stage_path.exists():
            continue
        sample = dataset[sample_id]
        qa = [sample["qa"][it["qa_index"]] for it in sample_items]
        t0 = time.perf_counter()
        try:
            stage, rows = await run_memnexus(
                sample, qa, 2.0,
                usage_ledger=out / f"{sample_id}_usage.json",
                max_session=None,
                snapshot_dir=snapshots,
                resilient=True,
                answer_timeout=300.0,
            )
        except Exception as error:  # noqa: BLE001 - one bad sample must not end the night
            status["samples"][sample_id] = {"error": f"{type(error).__name__}: {error}"[:400]}
            write(status_path, status)
            continue
        if len(rows) != len(sample_items):
            raise RuntimeError(f"{sample_id}: {len(rows)} rows for {len(sample_items)} questions")
        for item, row in zip(sample_items, rows, strict=True):
            row["qa_index"] = item["qa_index"]
            row["sample_id"] = item["sample_id"]
        write(stage_path, {"stage": stage, "rows": rows, "questions": sample_items})
        cost = float(stage.get("total_usage", {}).get("cost_rmb", 0.0))
        status["samples"][sample_id] = {
            "questions": len(rows),
            "cost_rmb": round(cost, 4),
            "replayed": stage.get("memory_base_replayed"),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        status["cost_rmb"] = round(status["cost_rmb"] + cost, 4)
        write(status_path, status)
        print(f"[{cell} r{rep}] {sample_id}: {len(rows)} q, ¥{cost:.4f}, "
              f"{status['samples'][sample_id]['seconds']}s", flush=True)

    status["finished"] = datetime.now().isoformat(timespec="seconds")
    write(status_path, status)
    print(f"[{cell} r{rep}] done ¥{status['cost_rmb']:.4f}", flush=True)


def finished(root: Path, cell: str, rep: int) -> bool:
    status = root / f"{cell}_r{rep}" / "status.json"
    return status.exists() and "finished" in read(status)


def spawn(cell: str, rep: int, root: Path, snapshots: Path) -> subprocess.Popen:
    log = root / f"{cell}_r{rep}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--cell", cell, "--rep", str(rep),
        "--root", str(root), "--snapshots", str(snapshots),
    ]
    print(f"[driver] start {cell} r{rep} -> {log.name}", flush=True)
    return subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)


def total_cost(root: Path) -> float:
    cost = 0.0
    for status_path in root.glob("*/status.json"):
        cost += float(read(status_path).get("cost_rmb", 0.0))
    return cost


def report(root: Path, log_path: Path | None = None):
    script = Path(__file__).resolve().parent / "overnight_report.py"
    print("[driver] generating report", flush=True)
    subprocess.run([sys.executable, str(script), "--root", str(root)], cwd=str(ROOT), check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    parser.add_argument("--deadline", default="07:00", help="HH:MM local stop time")
    parser.add_argument("--budget", type=float, default=28.0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    snapshots = Path(args.snapshots)

    if args.cell:
        asyncio.run(run_cell(args.cell, args.rep, root, snapshots))
        return

    root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(root)
        return

    hour, minute = (int(x) for x in args.deadline.split(":"))
    now = datetime.now()
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    print(f"[driver] deadline {deadline:%Y-%m-%d %H:%M}, budget ¥{args.budget:.0f}", flush=True)

    for cell, rep in PLAN:
        if finished(root, cell, rep):
            print(f"[driver] skip {cell} r{rep}", flush=True)
            continue
        if datetime.now() > deadline - timedelta(minutes=25):
            print("[driver] deadline guard: stop starting new runs", flush=True)
            break
        if total_cost(root) > args.budget:
            print(f"[driver] budget guard: ¥{total_cost(root):.2f} > ¥{args.budget:.0f}", flush=True)
            break
        proc = spawn(cell, rep, root, snapshots)
        try:
            proc.wait(timeout=4200)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[driver] {cell} r{rep} exceeded 70 min, killed", flush=True)
            continue
        if proc.returncode != 0:
            print(f"[driver] {cell} r{rep} exited {proc.returncode}", flush=True)

    summary = {"cells": {p.parent.name: read(p) for p in root.glob("*/status.json")},
               "cost_rmb": round(total_cost(root), 4),
               "finished": datetime.now().isoformat(timespec="seconds")}
    write(root / "driver_summary.json", summary)
    print(f"[driver] all runs done, total ¥{summary['cost_rmb']:.4f}", flush=True)
    report(root)


if __name__ == "__main__":
    main()
