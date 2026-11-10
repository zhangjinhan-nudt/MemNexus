"""Run a LoCoMo cohort with conversations in parallel.

The frozen benchmark runner walks conversations one after another, which makes
a 500-question comparison an overnight job. Conversations are independent
(separate database, separate scope, separate questions), so the only shared
resource is the provider rate limit. This runner keeps the same per-question
bookkeeping and writes the same ``stage_result.json`` shape, but overlaps
conversations behind a semaphore and judges afterwards in a bounded pool.

Nothing about the measurement changes: same prompts, same config, same
official cohort, same judge instruction.
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


def resolve_systems(only_system: str) -> tuple[str, ...]:
    """决定本次执行哪些系统。

    空串 → 全部（默认，行为与改动前完全一致）；单个系统名 → 只执行它。
    另一个系统的已有 stage_result.json 会在收集阶段被复用（见 execute 的收集循环），
    因此"只改 MemNexus"的实验可以固定 SimpleMem 基线、每轮省一半成本与时间，
    同时消除网关漂移带来的混杂。
    """
    if not only_system:
        return SYSTEMS
    if only_system not in SYSTEMS:
        raise ValueError(f"--only-system 必须是 {SYSTEMS} 之一或留空，收到 {only_system!r}")
    return (only_system,)


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


def cohort_questions(sample: dict, items: list[dict]) -> list[dict]:
    """Carry the cohort identity into raw QA objects for safe checkpointing."""
    return [
        {**sample["qa"][item["qa_index"]], "qa_index": item["qa_index"]}
        for item in items
    ]


async def run_unit(system: str, sample: dict, questions: list[dict], items: list[dict],
                   attempt: Path, budget: float, max_session: int,
                   snapshots: Path | None) -> dict:
    from run_locomo_pilot import run_memnexus, run_simplemem

    ledger = attempt / "usage.json"
    started = time.perf_counter()
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
    if len(rows) != len(items):
        raise ValueError(f"{system}/{sample.get('sample_id')}: {len(rows)} rows for {len(items)} questions")
    if any(not str(row.get("prediction") or "").strip() for row in rows):
        raise RuntimeError(f"{system}/{sample.get('sample_id')}: empty predictions are not a completed unit")
    for row, item in zip(rows, items, strict=True):
        row.update(qa_index=item["qa_index"], category_id=item["category_id"],
                   category=item["category"], gold_evidence=item["evidence"])
    stage["elapsed_s"] = round(time.perf_counter() - started, 1)
    write(attempt / "stage_result.json", {"stage": stage, "rows": rows})
    return {"stage": stage, "rows": rows}


async def execute(args) -> dict:
    from run_locomo_pilot import CATEGORY_NAMES

    CATEGORY_NAMES.update({1: "multi-hop", 2: "temporal", 3: "open-domain",
                           4: "single-hop", 5: "adversarial"})
    cohort = read(ROOT / args.cohort)
    dataset = read(ROOT / cohort["dataset"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    snapshots = ROOT / args.snapshots if args.snapshots else None
    if snapshots:
        snapshots.mkdir(parents=True, exist_ok=True)

    groups: dict[int, list[dict]] = {}
    for question in cohort["questions"]:
        groups.setdefault(question.get("sample_index", 0), []).append(question)
    limit = args.limit_conversations
    indices = sorted(groups)[:limit] if limit else sorted(groups)
    if args.only_sample:
        wanted = set(args.only_sample.split(","))
        indices = [index for index in indices
                   if dataset[index].get("sample_id") in wanted]

    gate = asyncio.Semaphore(args.concurrency)
    progress_lock = asyncio.Lock()
    done = Counter()
    # 只执行选中的系统；未选中的系统若有已完成单元，会在下方收集阶段被复用。
    active_systems = resolve_systems(args.only_system)

    async def one(index: int) -> None:
        items = sorted(groups[index], key=lambda item: item["qa_index"])
        sample = dataset[index]
        questions = [sample["qa"][item["qa_index"]] for item in items]
        # Stagger conversation starts: the provider rejects a simultaneous burst
        # with HTTP 429, which would fail every unit at once.
    async def run_unit_with_retries(system, sample, questions, items, attempt) -> str:
        """Run one conversation x system unit, retrying transient failures.

        The backoff wait happens *outside* the gate: a unit that is waiting for
        the provider to cool down must not hold a concurrency slot, otherwise
        one throttled unit stalls every other conversation.
        """
        status = "not started"
        for round_index in range(args.unit_retries):
            async with gate:
                try:
                    result = await run_unit(system, sample, questions, items, attempt,
                                            args.budget_per_conversation, cohort["max_session"], snapshots)
                    write(attempt / "ok.json", {"rows": len(result["rows"])})
                    return (f"{len(result['rows'])} questions "
                            f"RMB {result['stage'].get('total_usage', {}).get('cost_rmb', 0):.3f}")
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
        items = sorted(groups[index], key=lambda item: item["qa_index"])
        sample = dataset[index]
        questions = cohort_questions(sample, items)
        # Stagger conversation starts: the provider rejects a simultaneous burst
        # with HTTP 429, which would fail every unit at once.
        await asyncio.sleep(args.stagger * (index % max(1, args.concurrency)))
        for system in active_systems:
            unit = f"{sample.get('sample_id')}_{system.lower()}"
            attempt = out / "attempts" / unit
            if (attempt / "stage_result.json").exists():
                continue
            attempt.mkdir(parents=True, exist_ok=True)
            status = await run_unit_with_retries(system, sample, questions, items, attempt)
            async with progress_lock:
                done[system] += 1
                # A status line must never be able to kill the run: the console
                # may be GBK while the text can contain any character.
                try:
                    print(f"[{time.strftime('%H:%M:%S')}] {unit}: {status}", flush=True)
                except (UnicodeEncodeError, OSError):
                    safe = f"[{time.strftime('%H:%M:%S')}] {unit}: {status}".encode(
                        "ascii", "replace").decode("ascii")
                    print(safe, flush=True)

    await asyncio.gather(*(one(index) for index in indices))

    rows, reused_systems = collect_run_rows(
        out=out, dataset=dataset, indices=indices, groups=groups,
        active_systems=active_systems,
    )
    write(out / "rows.json", rows)
    summary = {
        "cohort": args.cohort,
        "selection_sha256": cohort["selection_sha256"],
        "max_session": cohort["max_session"],
        "questions": len(cohort["questions"]) if not limit else sum(len(groups[i]) for i in indices),
        "systems": dict(done),
        "only_system": args.only_system or None,
        "active_systems": list(active_systems),
        "reused_units": dict(reused_systems),
        "rows": len(rows),
        "categories": dict(Counter(row["category"] for row in rows)),
    }
    write(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def collect_run_rows(out: Path, dataset: list[dict], indices: list[int],
                     groups: dict[int, list[dict]],
                     active_systems: tuple[str, ...]) -> tuple[list[dict], Counter]:
    """合并所有已完成单元的行，返回 (rows, 被复用的系统计数)。

    始终遍历 SYSTEMS 全集：本次未执行的系统若已有 stage_result.json，即作为基线被复用。
    对被复用的单元强制校验 (sample_id, qa_index) 键集与当前 cohort 完全一致，
    防止把上一次不同 cohort 的结果静默混入 rows.json——这种污染会让配对统计悄悄失真。
    """
    rows: list[dict] = []
    reused_systems: Counter = Counter()
    for index in indices:
        sample_id = dataset[index].get("sample_id")
        expected_keys = {(sample_id, item["qa_index"]) for item in groups[index]}
        for system in SYSTEMS:
            path = out / "attempts" / f"{sample_id}_{system.lower()}" / "stage_result.json"
            if not path.exists():
                continue
            unit_rows = read(path)["rows"]
            if system not in active_systems:
                actual_keys = {(row.get("sample_id"), row.get("qa_index")) for row in unit_rows}
                if actual_keys != expected_keys:
                    raise ValueError(
                        f"复用 {system}/{sample_id} 的既有结果与当前 cohort 不匹配："
                        f"缺 {sorted(expected_keys - actual_keys)[:5]}、"
                        f"多 {sorted(actual_keys - expected_keys)[:5]}。"
                        f"复用基线必须来自同一 cohort，请检查 --out 目录。"
                    )
                reused_systems[system] += 1
            rows += unit_rows
    return rows, reused_systems


def collect(args) -> dict:
    """Rebuild rows.json from every finished attempt, so a resumed run merges."""
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
    """Blind grading with the paper's single user message, in a bounded pool."""
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
        if path.exists() and read(path).get("judge_result") in {"CORRECT", "WRONG"}:
            return read(path)
        client = LLMClient(api_key=config.JUDGE_API_KEY or config.OPENAI_API_KEY,
                           model=config.JUDGE_MODEL, base_url=config.JUDGE_BASE_URL,
                           enable_thinking=False, use_streaming=False)
        client.ledger_path = str(folder / f"{key}.usage.json")
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
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit-conversations", type=int, default=0)
    parser.add_argument("--budget-per-conversation", type=float, default=60.0)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-workers", type=int, default=6)
    parser.add_argument("--only-sample", default="")
    parser.add_argument("--only-system", default="",
                        help="只执行指定系统（MemNexus 或 SimpleMem）；留空=两者都跑。"
                             "另一系统的既有结果会在收集阶段被复用（须同 out 目录、同 cohort）。")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--unit-retries", type=int, default=3)
    parser.add_argument("--stagger", type=float, default=6.0)
    args = parser.parse_args()

    configure_environment(max(200.0, args.budget_per_conversation * 12))
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
