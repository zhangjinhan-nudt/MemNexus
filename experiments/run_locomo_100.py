"""Paired, budget-capped 100-question comparison on official LoCoMo data.

Run only after reviewing result/locomo_100_cohort.json. The 100 questions are
fixed by locomo_100_protocol.py before any model request. Ten conversations
are independently rebuilt by each system from raw sessions D1-D2. A live
ledger is written after every provider response, and the run stops if any
window, answer, judge call, or cost accounting fails.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from locomo_100_protocol import DATASET, ROOT, estimate_cost, select_cohort
from run_locomo_pilot import judge_rows, run_memnexus, run_simplemem


def save(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pending.replace(path)


def money_in_ledger(path: Path) -> float:
    if not path.exists():
        return 0.0
    ledger = json.loads(path.read_text(encoding="utf-8"))
    return float(ledger.get("cost_rmb", 0.0)) + float(ledger.get("reserved_rmb", 0.0))


def ledger_total(folder: Path) -> float:
    return sum(
        money_in_ledger(path)
        for path in folder.glob("*.json")
        if not path.name.endswith((".stage.json", ".memory.json"))
    )


def selected_questions(sample: dict, items: list[dict]) -> list[dict]:
    return [sample["qa"][item["qa_index"]] for item in items]


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def exact_mcnemar_p(better: int, worse: int) -> float:
    discordant = better + worse
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(better, worse) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap_ci(
    pairs: list[tuple[int, int]], seed: int = 20261012, n: int = 5000
) -> list[float]:
    import random

    rng = random.Random(seed)
    size = len(pairs)
    differences = []
    for _ in range(n):
        drawn = [pairs[rng.randrange(size)] for _ in range(size)]
        differences.append(sum(mem - simple for simple, mem in drawn) / size)
    return [percentile(differences, 0.025), percentile(differences, 0.975)]


def summary(rows: list[dict], stages: list[dict]) -> dict:
    if len(rows) != 200:
        raise ValueError(f"Expected 200 answered and judged rows, found {len(rows)}")
    by_system = {}
    by_key = defaultdict(dict)
    for row in rows:
        if row.get("judge_result") not in {"CORRECT", "WRONG"}:
            raise ValueError("Every row must have a valid judge result")
        by_key[(row["sample_id"], row["qa_index"])][row["system"]] = row
    if len(by_key) != 100 or any(
        set(pair) != {"SimpleMem", "MemNexus"} for pair in by_key.values()
    ):
        raise ValueError("Paired result keys are incomplete")

    for system in ("SimpleMem", "MemNexus"):
        subset = [row for row in rows if row["system"] == system]
        correct = sum(row["judge_result"] == "CORRECT" for row in subset)
        q_cost = sum(float(row["model_calls"]["cost_rmb"]) for row in subset)
        build_cost = sum(
            float(stage["build_usage"]["cost_rmb"])
            for stage in stages
            if stage["system"] == system
        )
        by_category = {}
        for category in sorted({row["category"] for row in subset}):
            items = [row for row in subset if row["category"] == category]
            by_category[category] = {
                "n": len(items),
                "correct": sum(item["judge_result"] == "CORRECT" for item in items),
                "accuracy": sum(item["judge_result"] == "CORRECT" for item in items)
                / len(items),
            }
        by_system[system] = {
            "n": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset),
            "input_tokens": sum(
                int(row["model_calls"]["input_tokens"]) for row in subset
            ),
            "output_tokens": sum(
                int(row["model_calls"]["output_tokens"]) for row in subset
            ),
            "provider_attempts": sum(
                int(row["model_calls"]["provider_attempts"]) for row in subset
            ),
            "answer_cost_rmb": q_cost,
            "build_cost_rmb": build_cost,
            "main_model_cost_rmb": q_cost + build_cost,
            "cost_per_correct_answer_rmb": (q_cost + build_cost) / correct
            if correct
            else None,
            "latency_p50_ms": statistics.median(row["latency_ms"] for row in subset),
            "latency_p95_ms": percentile([row["latency_ms"] for row in subset], 0.95),
            "by_category": by_category,
        }

    pairs = [
        (
            int(pair["SimpleMem"]["judge_result"] == "CORRECT"),
            int(pair["MemNexus"]["judge_result"] == "CORRECT"),
        )
        for pair in by_key.values()
    ]
    mem_only = sum(simple == 0 and mem == 1 for simple, mem in pairs)
    simple_only = sum(simple == 1 and mem == 0 for simple, mem in pairs)
    return {
        "systems": by_system,
        "paired_accuracy_difference_mem_minus_simple": sum(
            mem - simple for simple, mem in pairs
        )
        / len(pairs),
        "paired_bootstrap_95pct_ci": paired_bootstrap_ci(pairs),
        "discordant_mem_only_correct": mem_only,
        "discordant_simple_only_correct": simple_only,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(mem_only, simple_only),
    }


async def run(args) -> int:
    os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "8192"
    cohort = (
        json.loads(args.cohort.resolve().read_text(encoding="utf-8"))
        if args.cohort
        else select_cohort()
    )
    if cohort.get("question_count") != 100 or len(cohort.get("questions", [])) != 100:
        raise ValueError("The selected cohort must contain exactly 100 questions")
    official = json.loads(DATASET.read_text(encoding="utf-8"))
    items_by_index = defaultdict(list)
    for item in cohort["questions"]:
        items_by_index[item["sample_index"]].append(item)
    out = args.output.resolve()
    ledger_dir = out.parent / (out.stem + "_ledgers")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    code_files = [
        *(ROOT / "MemNexus/memory-system/src/agent_memory").glob("*.py"),
        *(ROOT / "SimpleMem/simplemem/core").rglob("*.py"),
        ROOT / "SimpleMem/main.py",
        ROOT / "experiments/run_locomo_pilot.py",
        Path(__file__),
        ROOT / "experiments/locomo_100_protocol.py",
    ]
    report = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": cohort["dataset_sha256"],
        "selection_sha256": cohort["selection_sha256"],
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code_files
        },
        "interpretation": "development-cohort rerun; not held-out generalization or a controlled single-component ablation",
        "cohort": cohort,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model": "deepseek-v4-flash",
            "judge": "deepseek-v4-flash",
            "embedding": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dimension": 1024,
            "output_language": "en",
            "answer_cache": False,
            "parallelism": 1,
            "max_session": args.max_session,
            "grounding": {"enabled": True, "semantic_raw": False, "raw_limit": 3},
            "max_tokens": 4096,
            "temperature": 0,
            "thinking": False,
            "length_recovery": "one retry at max_tokens=8192 only on provider finish_reason=length; both systems and judge",
            "usd_converted_conservative_input_output_rmb_per_million": [2.4, 9.6],
            "verified_official_peak_input_output_rmb_per_million": [2, 8],
            "price_source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
            "prices_checked_date": "2026-10-12",
        },
        "budget": {
            "cumulative_cap_rmb": args.cumulative_cap,
            "prior_reserved_rmb": args.prior_reserved,
            "run_cap_rmb": args.cumulative_cap - args.prior_reserved,
            "cost_basis": "provider prompt/completion usage at conservative peak prices",
        },
        "preflight_cost_estimate": estimate_cost(cohort, max_session=args.max_session),
        "stages": [],
        "rows": [],
        "judge_usage": [],
        "run_cost_rmb": 0.0,
    }
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {out}")
    if report["budget"]["run_cap_rmb"] <= 0:
        raise ValueError("No approved budget remains")
    if args.resume_from:
        previous_path = args.resume_from.resolve()
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous["selection_sha256"] != report["selection_sha256"]:
            raise ValueError("Cannot resume a different cohort")
        allowed_changes = {
            "experiments/run_locomo_100.py",
            "SimpleMem/simplemem/core/utils/llm_client.py",
            "MemNexus/memory-system/src/agent_memory/providers.py",
        }
        changed = [
            p
            for p, digest in previous["source_sha256"].items()
            if report["source_sha256"].get(p) != digest
        ]
        if {p.replace("\\", "/") for p in changed} - allowed_changes:
            raise ValueError(f"Resume changes evaluated algorithms: {changed}")
        completed = {j["sample_id"] for j in previous["judge_usage"]}
        for sample_id in completed:
            expected = 2 * sum(q["sample_id"] == sample_id for q in cohort["questions"])
            selected = [r for r in previous["rows"] if r["sample_id"] == sample_id]
            if len(selected) != expected or any(
                r["judge_result"] not in {"CORRECT", "WRONG"} for r in selected
            ):
                raise ValueError(
                    "Resume requires complete paired conversation and judgments"
                )
        report["rows"] = [r for r in previous["rows"] if r["sample_id"] in completed]
        report["stages"] = [
            s for s in previous["stages"] if s["sample_id"] in completed
        ]
        report["judge_usage"] = previous["judge_usage"]
        previous_ledgers = previous_path.with_name(previous_path.stem + "_ledgers")
        for source in previous_ledgers.iterdir():
            if any(source.name.startswith(sample_id + "_") for sample_id in completed):
                shutil.copyfile(source, ledger_dir / source.name)
        report["run_cost_rmb"] = ledger_total(ledger_dir)
        report["resumed_from"] = {
            "path": str(previous_path),
            "completed_samples": sorted(completed),
            "previous_source_sha256": previous["source_sha256"],
            "changed_files": changed,
            "policy_change": "only truncation recovery and checkpoint continuation; completed untruncated calls retained",
            "excluded_incomplete_cost_rmb": previous["upper_run_cost_rmb"]
            - previous["run_cost_rmb"],
        }
    save(out, report)
    started = time.perf_counter()
    active_ledger: Path | None = None
    resume_base_cost = 0.0
    if args.resume_from:
        resume_base_cost = report["run_cost_rmb"]
    try:
        for sample_index in sorted(items_by_index):
            sample = official[sample_index]
            items = items_by_index[sample_index]
            questions = selected_questions(sample, items)
            sample_id = sample.get("sample_id", str(sample_index))
            if any(j["sample_id"] == sample_id for j in report["judge_usage"]):
                continue
            remaining = report["budget"]["run_cap_rmb"] - max(
                0.0, report["run_cost_rmb"] - resume_base_cost
            )
            if remaining < 0.05:
                raise RuntimeError("Insufficient reserved budget for next conversation")

            active_ledger = ledger_dir / f"{sample_id}_simplemem.json"
            stage, rows = run_simplemem(
                sample, questions, remaining, usage_ledger=active_ledger,
                max_session=args.max_session,
            )
            for row, item in zip(rows, items, strict=True):
                row["qa_index"] = item["qa_index"]
                row["gold_evidence"] = item["evidence"]
            report["stages"].append(stage)
            report["rows"].extend(rows)
            report["run_cost_rmb"] = ledger_total(ledger_dir)
            save(out, report)

            remaining = report["budget"]["run_cap_rmb"] - max(
                0.0, report["run_cost_rmb"] - resume_base_cost
            )
            active_ledger = ledger_dir / f"{sample_id}_memnexus.json"
            stage, rows = await run_memnexus(
                sample, questions, remaining, usage_ledger=active_ledger,
                max_session=args.max_session,
            )
            for row, item in zip(rows, items, strict=True):
                row["qa_index"] = item["qa_index"]
                row["gold_evidence"] = item["evidence"]
            report["stages"].append(stage)
            report["rows"].extend(rows)
            report["run_cost_rmb"] = ledger_total(ledger_dir)
            save(out, report)

            # Judge in a fixed alternating order within each conversation; the
            # judge sees no system label. Neither prediction is post-edited.
            paired = [row for row in report["rows"] if row["sample_id"] == sample_id]
            paired.sort(
                key=lambda row: (
                    row["qa_index"],
                    row["system"]
                    != ("SimpleMem" if sample_index % 2 == 0 else "MemNexus"),
                )
            )
            active_ledger = ledger_dir / f"{sample_id}_judge.json"
            judge_usage = judge_rows(
                paired,
                report["budget"]["run_cap_rmb"]
                - max(0.0, report["run_cost_rmb"] - resume_base_cost),
                usage_ledger=active_ledger,
            )
            report["judge_usage"].append({"sample_id": sample_id, **judge_usage})
            report["run_cost_rmb"] = ledger_total(ledger_dir)
            report["completed_conversations"] = sample_index + 1
            save(out, report)
            print(
                json.dumps(
                    {
                        "conversation": sample_id,
                        "paired_questions": len(items),
                        "cumulative_run_cost_rmb": round(report["run_cost_rmb"], 4),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        report["summary"] = summary(report["rows"], report["stages"])
        report["status"] = "completed"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        save(out, report)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_cost_rmb": report["run_cost_rmb"],
                    "summary": report["summary"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:  # noqa: BLE001 - BudgetExceededError intentionally inherits BaseException
        # Provider-ledger cost can be greater than the latest stage checkpoint
        # when a failure interrupts an in-progress conversation.
        partial = money_in_ledger(active_ledger) if active_ledger else 0.0
        report["status"] = "aborted"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["incomplete_stage_ledger_rmb"] = partial
        report["upper_run_cost_rmb"] = ledger_total(ledger_dir)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(out, report)
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "failure": report["failure"],
                    "upper_run_cost_rmb": report["upper_run_cost_rmb"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "result" / "locomo_100_results.json"
    )
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--max-session", type=int, default=2)
    parser.add_argument("--cumulative-cap", type=float, default=18.0)
    parser.add_argument("--prior-reserved", type=float, default=8.0)
    parser.add_argument("--resume-from", type=Path)
    options = parser.parse_args()
    raise SystemExit(asyncio.run(run(options)))
