"""One-shot, auditable migration from text keys to (sample_id, qa_index).

Legacy checkpoints did not store qa_index.  Keep uniquely identifiable rows,
quarantine only ambiguous records, and invalidate only non-complete stage
results so the affected questions are regenerated rather than copied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_locomo_pilot import checkpoint_row_key, valid_checkpoint_row


def repair_records(records: list[dict], items: list[dict]) -> tuple[list[dict], int]:
    candidates: dict[tuple[str, str], set[int]] = {}
    for item in items:
        key = (str(item["sample_id"]), str(item["question"]))
        candidates.setdefault(key, set()).add(item["qa_index"])
    repaired = []
    dropped = 0
    for row in records:
        if not valid_checkpoint_row(row):
            repaired.append(row)
            continue
        if row.get("qa_index") is not None:
            repaired.append(row)
            continue
        key = (str(row["sample_id"]), str(row["question"]))
        matches = candidates.get(key, set())
        if len(matches) != 1:
            dropped += 1
            continue
        row = dict(row)
        row["qa_index"] = next(iter(matches))
        repaired.append(row)
    return repaired, dropped


def repair_run(run: Path, cohort_path: Path) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = {}
    for item in cohort["questions"]:
        groups.setdefault(item["sample_id"], []).append(item)
    report = {"repaired_rows": 0, "quarantined_rows": 0, "invalidated_units": []}
    for unit in sorted((run / "attempts").iterdir()):
        if not unit.is_dir():
            continue
        rows_path = unit / "usage.rows.jsonl"
        sample_id = unit.name.rsplit("_", 1)[0]
        items = groups.get(sample_id, [])
        if rows_path.exists():
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            repaired, dropped = repair_records(rows, items)
            if dropped:
                report["quarantined_rows"] += dropped
            migrated = sum(valid_checkpoint_row(row) and row.get("qa_index") is None for row in rows) - dropped
            if repaired != rows:
                report["repaired_rows"] += migrated
                backup = run / "repair_backups" / "pre_qa_index" / unit.name
                backup.mkdir(parents=True, exist_ok=True)
                (backup / "usage.rows.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
                )
                rows_path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repaired), encoding="utf-8"
                )
        stage_path = unit / "stage_result.json"
        if stage_path.exists():
            payload = json.loads(stage_path.read_text(encoding="utf-8"))
            keys = [(str(row.get("sample_id")), str(row.get("qa_index"))) for row in payload.get("rows", [])]
            expected = {(str(item["sample_id"]), str(item["qa_index"])) for item in items}
            if len(keys) != len(set(keys)) or set(keys) != expected:
                target = run / "repair_backups" / "pre_qa_index" / unit.name / "stage_result.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                stage_path.replace(target)
                report["invalidated_units"].append(unit.name)
    report["status"] = "ok"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(repair_run(args.run, args.cohort), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
