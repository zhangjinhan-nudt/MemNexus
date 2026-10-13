"""Complete an interrupted LoCoMo judge stage from saved real predictions.

The source run and its ledgers remain untouched. A request with no response is
charged at its recorded reservation. Only missing judge rows are submitted.
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

from run_locomo_pilot import judge_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "result/locomo_100_disjoint_v2_20261013.json"
OUTPUT = ROOT / "result/locomo_100_disjoint_v2_recovered.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = read(SOURCE)
    if original["status"] != "running" or original["judge_usage"]:
        raise ValueError("Expected one interrupted uncommitted judge stage")
    sample_id = original["stages"][0]["sample_id"]
    assert len(original["stages"]) == 2
    assert {stage["system"] for stage in original["stages"]} == {"SimpleMem", "MemNexus"}
    assert len(original["rows"]) == 2 * sum(
        item["sample_id"] == sample_id for item in original["cohort"]["questions"]
    )

    source_dir = SOURCE.with_name(SOURCE.stem + "_ledgers")
    target_dir = OUTPUT.with_name(OUTPUT.stem + "_ledgers")
    target_dir.mkdir(parents=True, exist_ok=False)
    for path in source_dir.iterdir():
        shutil.copyfile(path, target_dir / path.name)

    prior_path = target_dir / f"{sample_id}_judge.json"
    prior = read(prior_path)
    trace_path = target_dir / f"{sample_id}_judge.calls.jsonl"
    old_trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    old_judged_path = target_dir / f"{sample_id}_judge.rows.jsonl"
    judged_rows = [json.loads(line) for line in old_judged_path.read_text(encoding="utf-8").splitlines()]
    keyed_judgments = {(row["system"], row["qa_index"]): row for row in judged_rows}
    assert len(keyed_judgments) == len(judged_rows)
    assert len([event for event in old_trace if event["event"] == "request"]) == prior["provider_attempts"]
    assert len([event for event in old_trace if event["event"] == "response"]) + 1 == prior["provider_attempts"]
    assert prior["reserved_rmb"] > 0

    report = deepcopy(original)
    for row in report["rows"]:
        saved = keyed_judgments.get((row["system"], row["qa_index"]))
        if saved:
            assert row["prediction"] == saved["prediction"]
            for field in ("judge_result", "judge_reason", "judge_raw_response"):
                row[field] = saved[field]
    pending = [row for row in report["rows"] if row["judge_result"] is None]
    pending.sort(key=lambda row: (row["qa_index"], row["system"] != "SimpleMem"))
    assert len(pending) == 2
    assert {row["system"] for row in pending} == {"SimpleMem", "MemNexus"}

    # Use a separate trace while the two real judge requests are in flight.
    extra_path = ROOT / "result/locomo_100_disjoint_v2_recovery_judge.json"
    if extra_path.exists():
        raise FileExistsError(extra_path)
    extra = judge_rows(pending, component_budget=10.0, usage_ledger=extra_path)
    if any(row["judge_result"] not in {"CORRECT", "WRONG"} for row in pending):
        raise ValueError("Recovery judge did not score both remaining rows")

    old_attempts = prior["provider_attempts"]
    old_reservation = prior.pop("reserved_rmb")
    prior["cost_rmb"] += old_reservation
    prior["unknown_usage_upper_bound_rmb"] = old_reservation
    prior["provider_errors"] = prior.get("provider_errors", 0) + 1
    merged = dict(prior)
    for key, value in extra.items():
        if isinstance(value, (int, float)):
            merged[key] = merged.get(key, 0) + value
    merged["reserved_rmb"] = 0.0
    save(prior_path, merged)

    error_event = {
        "event": "error",
        "attempt": old_attempts,
        "error_type": "InterruptedUnknownUsage",
        "charged_upper_bound_rmb": old_reservation,
    }
    extra_trace = [json.loads(line) for line in extra_path.with_suffix(".calls.jsonl").read_text(encoding="utf-8").splitlines()]
    for event in extra_trace:
        event["attempt"] += old_attempts
    merged_trace = [*old_trace, error_event, *extra_trace]
    trace_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in merged_trace) + "\n",
        encoding="utf-8",
    )
    old_judged_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in [*judged_rows, *pending]) + "\n",
        encoding="utf-8",
    )
    report["judge_usage"] = [{"sample_id": sample_id, **merged}]
    report["completed_conversations"] = 1
    report["run_cost_rmb"] = sum(
        read(path).get("cost_rmb", 0) + read(path).get("reserved_rmb", 0)
        for path in target_dir.glob("*.json")
        if not path.name.endswith((".stage.json", ".memory.json"))
    )
    report["upper_run_cost_rmb"] = report["run_cost_rmb"]
    report["status"] = "aborted"
    report["failure"] = {"type": "InterruptedProcess", "message": "judge stage recovered from saved predictions"}
    report["recovery"] = {
        "source": str(SOURCE),
        "original_judgments": len(judged_rows),
        "new_judgments": len(pending),
        "interrupted_attempt": old_attempts,
        "unknown_usage_upper_bound_rmb": old_reservation,
        "supplemental_ledger": str(extra_path),
    }
    save(OUTPUT, report)
    print(json.dumps({
        "output": str(OUTPUT),
        "judged": len(judged_rows) + len(pending),
        "cost_rmb": report["run_cost_rmb"],
        "interrupted_attempt_upper_bound_rmb": old_reservation,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
