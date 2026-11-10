"""Smoke gate: 3 questions on the frozen conv-26 base, before any larger run.

Checks protocol R2 in this order:
  1. forced_abstain == 0   -- no answer text may come from a crash fallback.
  2. provider repair count -- must not have exploded after the two fixes.
  3. only then look at F1.

The question set is fixed here and chosen only to cover the three code paths
that were changed (temporal derivation, polar-answer normalisation, list/set
rendering).  It is not selected from any result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_protocol import bleu1_of, f1_of
from run_locomo_pilot import (
    CATEGORY_NAMES,
    ORACLE_MAX_SESSION,
    evaluation_query,
    raw_turns,
    run_memnexus,
    run_simplemem,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
# The frozen base the 175-question paired run actually consumed (conv-26 sha
# f3f384cf..., 92 turns = sessions D1-D5).  Replaying it keeps the smoke run
# comparable to that data and costs nothing to build.
SNAPSHOTS = ROOT / "result" / "small_ab_20261020" / "snapshots"
MAX_SESSION = 5

# (category name, index within that category in dataset order)
SMOKE_SET = (
    ("temporal", 0),      # exercises derive_times / relative-date rendering
    ("open-domain", 1),   # polar "Would ...?" -> normalise_answer_text
    ("multi-hop", 5),     # list-valued output -> operator / join choice
)


def select_questions(sample: dict) -> list[dict]:
    by_name = {name: cid for cid, name in CATEGORY_NAMES.items()}
    selected = []
    for name, index in SMOKE_SET:
        rows = [qa for qa in sample["qa"] if qa["category"] == by_name[name]]
        if index >= len(rows):
            raise RuntimeError(f"conv-26 has only {len(rows)} {name} questions")
        selected.append(rows[index])
    return selected


def summarise(rows: list[dict]) -> dict:
    forced = [row for row in rows if row.get("forced_abstain")]
    errors = [row for row in rows if row.get("error")]
    repairs = sum(int(row.get("model_calls", {}).get("errors", 0)) for row in rows)
    return {
        "questions": len(rows),
        "forced_abstain": len(forced),
        "runtime_errors": len(errors),
        "provider_repairs": repairs,
        "empty_predictions": sum(1 for row in rows if not str(row.get("prediction") or "").strip()),
        "mean_f1": round(sum(f1_of(row) for row in rows) / max(1, len(rows)), 4),
        "mean_bleu1": round(sum(bleu1_of(row) for row in rows) / max(1, len(rows)), 4),
        "input_tokens": sum(int(row.get("model_calls", {}).get("input_tokens", 0)) for row in rows),
        "output_tokens": sum(int(row.get("model_calls", {}).get("output_tokens", 0)) for row in rows),
        "cost_rmb": round(sum(float(row.get("model_calls", {}).get("cost_rmb", 0)) for row in rows), 6),
        "rows": [
            {
                "category": row["category"],
                "question": row["question"],
                "gold": row["gold_answer"],
                "prediction": row["prediction"],
                "f1": round(f1_of(row), 4),
                "bleu1": round(bleu1_of(row), 4),
                "forced_abstain": bool(row.get("forced_abstain")),
                "answer_path": (row.get("reasoning") or {}).get("answer_path"),
                "provider_repairs": int(row.get("model_calls", {}).get("errors", 0)),
                "latency_ms": row.get("latency_ms"),
                "error": row.get("error"),
            }
            for row in rows
        ],
    }


async def main(args) -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    sample = data[0]
    questions = select_questions(sample)
    ledger = args.output.with_suffix("")
    report = {
        "status": "running",
        "scope": "smoke_gate_after_defect_fixes_not_a_result",
        "sample_id": sample.get("sample_id"),
        "max_session": MAX_SESSION,
        "raw_turns": len(raw_turns(sample, MAX_SESSION)),
        "snapshot_dir": str(SNAPSHOTS.relative_to(ROOT)),
        "questions": [
            {"category": CATEGORY_NAMES[qa["category"]], "question": qa["question"]}
            for qa in questions
        ],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    simple_stage, simple_rows = run_simplemem(
        sample, questions, args.budget, usage_ledger=ledger, max_session=MAX_SESSION
    )
    report["simplemem_stage"] = simple_stage
    report["SimpleMem"] = summarise(simple_rows)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    nexus_stage, nexus_rows = await run_memnexus(
        sample,
        questions,
        args.budget,
        usage_ledger=ledger,
        max_session=MAX_SESSION,
        snapshot_dir=SNAPSHOTS,
        require_snapshot=args.require_snapshots,
        resilient=args.resilient,
    )
    report["memnexus_stage"] = nexus_stage
    report["MemNexus"] = summarise(nexus_rows)

    gates = []
    for system in ("SimpleMem", "MemNexus"):
        block = report[system]
        gates.append(
            {
                "system": system,
                "forced_abstain": block["forced_abstain"],
                "runtime_errors": block["runtime_errors"],
                "provider_repairs": block["provider_repairs"],
                "pass": block["forced_abstain"] == 0
                and block["runtime_errors"] == 0
                and block["provider_repairs"] <= args.max_repairs,
            }
        )
    report["gates"] = gates
    report["gate_pass"] = all(gate["pass"] for gate in gates)
    report["status"] = "completed"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"sample = {report['sample_id']}  raw_turns = {report['raw_turns']}")
    for system in ("SimpleMem", "MemNexus"):
        block = report[system]
        print(
            f"\n[{system}] forced_abstain={block['forced_abstain']} "
            f"runtime_errors={block['runtime_errors']} "
            f"repairs={block['provider_repairs']}  "
            f"F1={block['mean_f1']:.4f} BLEU1={block['mean_bleu1']:.4f} "
            f"¥{block['cost_rmb']:.4f}"
        )
        for row in block["rows"]:
            print(f"  - {row['category']:12s} path={row['answer_path']}")
            print(f"      Q   {row['question'][:100]}")
            print(f"      gold {str(row['gold'])[:80]}")
            print(f"      pred {str(row['prediction'])[:100]}")
            print(f"      f1={row['f1']:.4f} bleu1={row['bleu1']:.4f} repairs={row['provider_repairs']}")
    print(f"\nGATE: {'PASS' if report['gate_pass'] else 'FAIL'}")
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "result" / "smoke_fix_validation.json",
    )
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--require-snapshots", action="store_true")
    parser.add_argument("--resilient", action="store_true")
    parser.add_argument("--max-repairs", type=int, default=3)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
