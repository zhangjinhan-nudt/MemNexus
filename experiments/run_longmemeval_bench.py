"""Run the LongMemEval 1000-question cohort with both systems, in parallel.

Modeled on run_parallel_bench.py but for per-question units: each question is
its own conversation (own haystack, own scope), so units are independent and
resumable. Every unit writes its own stage_result.json; a rerun skips finished
units. MemNexus freezes/replays memory snapshots like the LoCoMo runner.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "SimpleMem"))

SYSTEMS = ("MemNexus", "SimpleMem")


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def configure_environment(budget: float) -> None:
    os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = str(budget)
    os.environ["MEMORY_INPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "2.4"
    os.environ["MEMORY_OUTPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "9.6"
    os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "8192"
    os.environ.setdefault("MEMORY_MODEL_TIMEOUT", "180")


async def run_unit(system: str, sample: dict, question: dict, attempt: Path,
                   budget: float, max_session: int, snapshots: Path | None) -> dict:
    from run_locomo_pilot import run_memnexus, run_simplemem

    ledger = attempt / "usage.json"
    started = time.perf_counter()
    questions = [sample["qa"][0]]
    if system == "MemNexus":
        stage, rows = await run_memnexus(
            sample, questions, budget, usage_ledger=ledger, max_session=max_session,
            snapshot_dir=snapshots, require_snapshot=False, resilient=True, answer_timeout=300.0,
            resume=True,
        )
    else:
        stage, rows = await asyncio.to_thread(
            run_simplemem, sample, questions, budget, usage_ledger=ledger, max_session=max_session,
            resume=True,
        )
    if len(rows) != 1:
        raise ValueError(f"{system}/{sample.get('sample_id')}: {len(rows)} rows")
    if not str(rows[0].get("prediction") or "").strip():
        raise RuntimeError(f"{system}/{sample.get('sample_id')}: empty prediction is not a completed unit")
    row = rows[0]
    item = question
    row.update(qa_index=item["qa_index"], category_id=0,
               category=item["category"], gold_evidence=item["evidence"])
    stage["elapsed_s"] = round(time.perf_counter() - started, 1)
    write(attempt / "stage_result.json", {"stage": stage, "rows": rows})
    return {"stage": stage, "rows": rows}


async def execute(args) -> dict:
    cohort = read(ROOT / args.cohort)
    dataset = read(ROOT / cohort["dataset"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    snapshots = ROOT / args.snapshots if args.snapshots else None
    if snapshots:
        snapshots.mkdir(parents=True, exist_ok=True)

    order = {q["sample_index"]: q for q in cohort["questions"]}
    indices = sorted(order)
    # Oracle units are minutes; S units are tens of minutes. Run all O units
    # first so the cheap half of the cohort is banked before the slow half.
    indices.sort(key=lambda i: (order[i]["setting"] != "O", i))

    gate = asyncio.Semaphore(args.concurrency)
    progress_lock = asyncio.Lock()
    done: Counter = Counter()

    async def run_unit_with_retries(system, sample, question, attempt) -> str:
        status = "not started"
        for round_index in range(args.unit_retries):
            async with gate:
                try:
                    result = await run_unit(system, sample, question, attempt,
                                            args.budget_per_question, cohort["max_session"], snapshots)
                    write(attempt / "ok.json", {"rows": len(result["rows"])})
                    return (f"1 question "
                            f"RMB {result['stage'].get('total_usage', {}).get('cost_rmb', 0):.3f} "
                            f"{result['stage'].get('elapsed_s', 0)}s")
                except BaseException as error:  # provider budget/truncation errors inherit BaseException
                    if isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)) or type(error).__name__ == "LedgerPersistenceError":
                        raise
                    status = f"FAILED {type(error).__name__}: {error}"
                    write(attempt / "failure.json", {"error": str(error),
                                                     "type": type(error).__name__,
                                                     "round": round_index + 1})
                    if type(error).__name__ == "BudgetExceededError":
                        break
            if round_index + 1 < args.unit_retries:
                await asyncio.sleep(20 * (round_index + 1))
        return status

    async def one(index: int) -> None:
        question = order[index]
        sample = dataset[index]
        await asyncio.sleep(args.stagger * (index % max(1, args.concurrency)))
        for system in SYSTEMS:
            unit = f"{sample.get('sample_id')}_{system.lower()}"
            attempt = out / "attempts" / unit
            if (attempt / "stage_result.json").exists():
                continue
            attempt.mkdir(parents=True, exist_ok=True)
            status = await run_unit_with_retries(system, sample, question, attempt)
            async with progress_lock:
                done[system] += 1
                try:
                    print(f"[{time.strftime('%H:%M:%S')}] {unit}: {status}", flush=True)
                except (UnicodeEncodeError, OSError):
                    safe = f"[{time.strftime('%H:%M:%S')}] {unit}: {status}".encode(
                        "ascii", "replace").decode("ascii")
                    print(safe, flush=True)

    await asyncio.gather(*(one(index) for index in indices))

    rows = []
    for index in indices:
        sample_id = dataset[index].get("sample_id")
        for system in SYSTEMS:
            path = out / "attempts" / f"{sample_id}_{system.lower()}" / "stage_result.json"
            if path.exists():
                rows += read(path)["rows"]
    write(out / "rows.json", rows)
    summary = {
        "cohort": args.cohort,
        "selection_sha256": cohort["selection_sha256"],
        "max_session": cohort["max_session"],
        "questions": len(indices),
        "systems": dict(done),
        "rows": len(rows),
        "categories": dict(Counter(row["category"] for row in rows)),
        "settings": dict(Counter(row.get("setting", "") for row in rows)),
    }
    write(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def collect(args) -> dict:
    out = ROOT / args.out
    rows = []
    units = 0
    for path in sorted((out / "attempts").glob("*/stage_result.json")):
        payload = read(path)
        rows += payload["rows"]
        units += 1
    write(out / "rows.json", rows)
    summary = {"units": units, "rows": len(rows),
               "categories": dict(Counter(row.get("category") for row in rows))}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def judge(args) -> dict:
    """Blind grading with the shared judge instruction, in a bounded pool."""
    import benchmark100 as bench
    import config
    from concurrent.futures import ThreadPoolExecutor
    from simplemem.core.utils.llm_client import LLMClient

    out = ROOT / args.out
    rows = read(out / "rows.json")
    folder = out / "judgments"
    folder.mkdir(parents=True, exist_ok=True)

    def one(row: dict) -> dict:
        key = hashlib.sha256(
            f"{row['sample_id']}|{row['qa_index']}|{row['system']}|{row['prediction']}".encode()
        ).hexdigest()[:20]
        path = folder / f"{key}.json"
        if path.exists():
            return read(path)
        client = LLMClient(api_key=config.JUDGE_API_KEY,
                           model=config.JUDGE_MODEL, base_url=config.JUDGE_BASE_URL,
                           enable_thinking=False, use_streaming=False)
        payload = json.dumps({k: row[k] for k in ("question", "gold_answer", "prediction")},
                             ensure_ascii=False)
        messages = [{"role": "user", "content": bench.JUDGE_INSTRUCTION + "\n" + payload}]
        try:
            raw = client.chat_completion(messages, response_format={"type": "json_object"}, max_retries=2)
            verdict = bench.parse_judgment(raw)
            value = {"judge_result": verdict["label"], "judge_reason": verdict["reason"],
                     "sample_id": row["sample_id"], "qa_index": row["qa_index"],
                     "system": row["system"]}
        except Exception as error:
            value = {"judge_result": "error", "judge_reason": f"{type(error).__name__}: {error}",
                     "sample_id": row["sample_id"], "qa_index": row["qa_index"],
                     "system": row["system"]}
        finally:
            client.client.close()
        write(path, value)
        return value

    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        verdicts = list(pool.map(one, rows))
    for row, verdict in zip(rows, verdicts, strict=True):
        row["judge_result"] = verdict["judge_result"]
        row["judge_reason"] = verdict["judge_reason"]
    write(out / "rows_judged.json", rows)
    tally = Counter((v["system"], v["judge_result"]) for v in verdicts)
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(tally.items())}, indent=1))
    return {"judged": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshots", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--budget-per-question", type=float, default=8.0)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-workers", type=int, default=6)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--unit-retries", type=int, default=3)
    parser.add_argument("--stagger", type=float, default=2.0)
    parser.add_argument("--only-setting", default="")
    args = parser.parse_args()

    configure_environment(max(200.0, args.budget_per_question * 12))
    if args.collect:
        result = collect(args)
    elif args.judge:
        result = judge(args)
    else:
        result = asyncio.run(execute(args))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
