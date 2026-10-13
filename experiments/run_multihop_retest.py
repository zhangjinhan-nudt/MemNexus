"""Live rerun of the frozen category-1 development questions after retrieval changes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from locomo_100_protocol import DATASET, ROOT
from run_locomo_100 import ledger_total, save
from run_locomo_pilot import judge_rows, run_memnexus

COHORT = ROOT / "result/locomo_100_cohort_v2.json"
OLD = ROOT / "result/locomo_100_disjoint_v2_complete.json"


async def run(output: Path) -> int:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    assert cohort["selection_sha256"] == old["selection_sha256"]
    questions = [item for item in cohort["questions"] if item["category_id"] == 1]
    assert len(questions) == 13
    by_index: dict[int, list[dict]] = defaultdict(list)
    for item in questions:
        by_index[item["sample_index"]].append(item)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    ledger_dir = output.with_name(output.stem + "_ledgers")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    files = [
        *(ROOT / "MemNexus/memory-system/src/agent_memory").glob("*.py"),
        ROOT / "experiments/run_locomo_pilot.py",
        Path(__file__),
    ]
    report = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "scope": "13 previously tested category-1 development questions; not held-out evidence",
        "dataset_sha256": cohort["dataset_sha256"],
        "selection_sha256": cohort["selection_sha256"],
        "old_result": str(OLD),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "protocol": {
            "model": "deepseek-v4-flash",
            "judge": "deepseek-v4-flash",
            "embedding": "Qwen/Qwen3-Embedding-0.6B",
            "session_horizon": 5,
            "answer_cache": False,
            "planner_mode": "model",
            "question_count": len(questions),
            "question_keys": [
                (item["sample_id"], item["qa_index"]) for item in questions
            ],
        },
        "stages": [],
        "rows": [],
        "judge_usage": [],
        "run_cost_rmb": 0.0,
    }
    save(output, report)
    started = time.perf_counter()
    try:
        for sample_index, items in sorted(by_index.items()):
            sample = data[sample_index]
            selected = [sample["qa"][item["qa_index"]] for item in items]
            sample_id = sample["sample_id"]
            stage_ledger = ledger_dir / f"{sample_id}_memnexus.json"
            stage, rows = await run_memnexus(
                sample, selected, 100.0, usage_ledger=stage_ledger, max_session=5
            )
            for row, item in zip(rows, items, strict=True):
                row["qa_index"] = item["qa_index"]
                row["gold_evidence"] = item["evidence"]
            report["stages"].append(stage)
            report["rows"].extend(rows)
            report["run_cost_rmb"] = ledger_total(ledger_dir)
            save(output, report)
            judge_ledger = ledger_dir / f"{sample_id}_judge.json"
            judge_usage = judge_rows(rows, 100.0, usage_ledger=judge_ledger)
            report["judge_usage"].append({"sample_id": sample_id, **judge_usage})
            report["run_cost_rmb"] = ledger_total(ledger_dir)
            save(output, report)
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "questions": len(items),
                        "scored": sum(
                            row["judge_result"] in {"CORRECT", "WRONG"}
                            for row in report["rows"]
                        ),
                        "cost_rmb": report["run_cost_rmb"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        assert len(report["rows"]) == 13
        assert all(
            row["judge_result"] in {"CORRECT", "WRONG"} for row in report["rows"]
        )
        report["status"] = "completed"
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        save(output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "correct": sum(
                        row["judge_result"] == "CORRECT" for row in report["rows"]
                    ),
                    "total": len(report["rows"]),
                    "cost_rmb": report["run_cost_rmb"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:  # noqa: BLE001 - live budget exceptions inherit BaseException
        report["status"] = "aborted"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["upper_run_cost_rmb"] = ledger_total(ledger_dir)
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        save(output, report)
        print(
            json.dumps(
                {"status": "aborted", "failure": report["failure"]}, ensure_ascii=False
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "result/locomo_multihop_retest_20261013.json",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.output)))
