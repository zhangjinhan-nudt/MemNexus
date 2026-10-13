"""Run a real, budget-capped LoCoMo pilot through SimpleMem and MemNexus.

This is a development pilot, not a full-dataset paper result. It uses the first
official LoCoMo conversation and the first question of each official category,
chosen before any model is called. Each system builds its own memory from the
same raw turns. All calls use the real configured provider and usage metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from contextlib import AsyncExitStack
from copy import deepcopy
from datetime import UTC
from pathlib import Path

import httpx
import numpy as np
from dateutil import parser as date_parser

ROOT = Path(__file__).resolve().parents[1]
SIMPLEMEM = ROOT / "SimpleMem"
MEMNEXUS = ROOT / "MemNexus" / "memory-system"
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}
INPUT_PRICE = 2.4
OUTPUT_PRICE = 9.6
PROBE_USAGE = {"input_tokens": 27, "output_tokens": 3}
FAILED_EMPTY_RUN_UPPER_BOUND_RMB = 3.5
FAILED_PARSER_RUN_USAGE = {
    "provider_attempts": 6,
    "input_tokens": 8958,
    "output_tokens": 12288,
    "cost_rmb": 0.139464,
}
ORACLE_MAX_SESSION = 2


class RemoteQwenEmbedding:
    """Synchronous adapter using the exact same local service as MemNexus."""

    model_name = "Qwen/Qwen3-Embedding-0.6B"
    dimension = 1024
    model_type = "qwen3_remote"
    supports_query_prompt = True

    def __init__(self, endpoint: str = "http://127.0.0.1:8099/v1/embeddings") -> None:
        self.endpoint = endpoint
        self.calls = 0
        self.inputs = 0

    def _encode(self, texts, input_type: str):
        if isinstance(texts, str):
            texts = [texts]
        response = httpx.post(
            self.endpoint,
            json={
                "model": self.model_name,
                "input": list(texts),
                "encoding_format": "float",
                "input_type": input_type,
            },
            timeout=180,
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        self.calls += 1
        self.inputs += len(texts)
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise RuntimeError("Unexpected Qwen embedding shape")
        return vectors

    def encode_documents(self, texts):
        return self._encode(texts, "document")

    def encode_query(self, texts):
        return self._encode(texts, "query")

    def encode_single(self, text, is_query=False):
        return self._encode([text], "query" if is_query else "document")[0]


def usage_delta(after: dict, before: dict) -> dict:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in set(after) | set(before)
        if isinstance(after.get(key, 0), (int, float))
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(text).casefold())


def token_f1(prediction: str, reference: str) -> float:
    pred, gold = tokenize(prediction), tokenize(reference)
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, reference: str) -> float:
    pred, gold = tokenize(prediction), tokenize(reference)
    if not pred or not gold:
        return 0.0
    clipped = sum((Counter(pred) & Counter(gold)).values())
    precision = clipped / len(pred)
    brevity = 1.0 if len(pred) >= len(gold) else math.exp(1 - len(gold) / len(pred))
    return brevity * precision


def evaluation_query(qa: dict) -> tuple[str, str]:
    if qa["category"] != 5:
        return qa["question"], qa["answer"]
    query = (
        qa["question"]
        + "\nChoose exactly one option based only on the conversation:\n"
        + "A. Not mentioned in the conversation\n"
        + f"B. {qa['adversarial_answer']}"
    )
    return query, "Not mentioned in the conversation"


def load_fixed_sample(per_category: int = 1) -> tuple[dict, list[dict]]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    sample = data[0]
    selected = []
    for category in sorted(CATEGORY_NAMES):
        category_rows = [qa for qa in sample["qa"] if qa["category"] == category]
        selected.extend(deepcopy(qa) for qa in category_rows[:per_category])
    return sample, selected


def raw_turns(sample: dict, max_session: int | None = None) -> list[dict]:
    conversation = sample["conversation"]
    turns = []
    for key in sorted(
        (
            key
            for key, value in conversation.items()
            if key.startswith("session_") and isinstance(value, list)
        ),
        key=lambda value: int(value.split("_")[1]),
    ):
        if max_session is not None and int(key.split("_")[1]) > max_session:
            continue
        date_text = conversation.get(f"{key}_date_time", "")
        for turn in conversation[key]:
            text = turn.get("text", "")
            if turn.get("blip_caption"):
                text = f"[Image: {turn['blip_caption']}] " + text
            if text.strip():
                turns.append(
                    {
                        "dia_id": turn["dia_id"],
                        "speaker": turn["speaker"],
                        "text": text.strip(),
                        "date_time": date_text,
                    }
                )
    if not turns:
        raise RuntimeError("Official sample contains no usable turns")
    return turns


def save(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def checkpoint(ledger, suffix, value, *, append=False):
    if ledger is None:
        return
    path = ledger.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def run_simplemem(
    sample: dict,
    questions: list[dict],
    component_budget: float,
    *,
    usage_ledger: Path | None = None,
    max_session: int | None = None,
) -> tuple[dict, list[dict]]:
    os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = str(component_budget)
    os.environ["EXPERIMENT_USAGE_LEDGER_PATH"] = str(
        usage_ledger or ROOT / "result" / "simplemem_usage_live.json"
    )
    sys.path.insert(0, str(SIMPLEMEM))
    from main import SimpleMemSystem
    from simplemem.core.models.memory_entry import Dialogue

    turns = raw_turns(
        sample, ORACLE_MAX_SESSION if max_session is None else max_session
    )
    embedder = RemoteQwenEmbedding()
    with tempfile.TemporaryDirectory(prefix="simplemem-locomo-pilot-") as folder:
        system = SimpleMemSystem(
            db_path=str(Path(folder) / "lancedb"),
            clear_db=True,
            enable_parallel_processing=False,
            max_parallel_workers=1,
            enable_parallel_retrieval=False,
            max_retrieval_workers=1,
            embedding_model=embedder,
        )
        build_before = dict(system.llm_client.usage)
        started = time.perf_counter()
        dialogues = [
            Dialogue(
                dialogue_id=index,
                speaker=turn["speaker"],
                content=turn["text"],
                timestamp=turn["date_time"],
            )
            for index, turn in enumerate(turns, 1)
        ]
        system.add_dialogues(dialogues)
        system.finalize()
        if system.memory_builder.failed_windows:
            raise RuntimeError(
                f"SimpleMem extraction failed in {system.memory_builder.failed_windows} windows"
            )
        build_ms = (time.perf_counter() - started) * 1000
        memories = system.get_all_memories()
        if not memories:
            raise RuntimeError("SimpleMem memory build produced zero memories")
        build_usage = usage_delta(system.llm_client.usage, build_before)
        checkpoint(
            usage_ledger,
            ".memory.json",
            {
                "sample_id": sample["sample_id"],
                "raw_turns": turns,
                "memories": [entry.model_dump(mode="json") for entry in memories],
                "build_usage": build_usage,
            },
        )
        rows = []
        for qa in questions:
            query, gold = evaluation_query(qa)
            before = dict(system.llm_client.usage)
            started = time.perf_counter()
            contexts = system.hybrid_retriever.retrieve(
                query, enable_reflection=False if qa["category"] == 5 else None
            )
            retrieval_ms = (time.perf_counter() - started) * 1000
            generation_started = time.perf_counter()
            prediction = system.answer_generator.generate_answer(query, contexts)
            generation_ms = (time.perf_counter() - generation_started) * 1000
            rows.append(
                {
                    "system": "SimpleMem",
                    "sample_id": sample.get("sample_id", "0"),
                    "category": CATEGORY_NAMES[qa["category"]],
                    "category_id": qa["category"],
                    "question": qa["question"],
                    "evaluation_query": query,
                    "gold_answer": gold,
                    "prediction": prediction,
                    "retrieved_sources": [
                        entry.lossless_restatement for entry in contexts
                    ],
                    "retrieval_count": len(contexts),
                    "model_calls": usage_delta(system.llm_client.usage, before),
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": generation_ms,
                    "latency_ms": retrieval_ms + generation_ms,
                    "f1": token_f1(prediction, gold),
                    "bleu1": bleu1(prediction, gold),
                    "judge_result": None,
                }
            )
            if not prediction or prediction == "Failed to generate answer":
                raise RuntimeError(
                    "SimpleMem produced an invalid answer; aborting paired experiment"
                )
            checkpoint(usage_ledger, ".rows.jsonl", rows[-1], append=True)
        stage = {
            "system": "SimpleMem",
            "sample_id": sample["sample_id"],
            "raw_turns": len(turns),
            "memory_count": len(memories),
            "build_ms": build_ms,
            "build_usage": build_usage,
            "total_usage": dict(system.llm_client.usage),
            "embedding_calls": embedder.calls,
            "embedding_inputs": embedder.inputs,
            "failed_windows": system.memory_builder.failed_windows,
        }
        checkpoint(usage_ledger, ".stage.json", stage)
    return stage, rows


def parse_occurred_at(value: str):
    parsed = date_parser.parse(value, fuzzy=True) if value else None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


async def run_memnexus(
    sample: dict,
    questions: list[dict],
    component_budget: float,
    *,
    usage_ledger: Path | None = None,
    max_session: int | None = None,
):
    os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = str(component_budget)
    os.environ["EXPERIMENT_USAGE_LEDGER_PATH"] = str(
        usage_ledger or ROOT / "result" / "memnexus_usage_live.json"
    )
    os.environ["MEMORY_EXPERIMENT_DROP_QUESTION_CANDIDATES"] = "1"
    sys.path.insert(0, str(MEMNEXUS / "src"))
    from agent_memory.models import Event, Scope, TurnInput
    from agent_memory.runtime import MemoryRuntime
    from agent_memory.settings import Settings

    turns = raw_turns(
        sample, ORACLE_MAX_SESSION if max_session is None else max_session
    )
    async with AsyncExitStack() as resources:
        folder = resources.enter_context(
            tempfile.TemporaryDirectory(prefix="memnexus-locomo-pilot-")
        )
        runtime = MemoryRuntime.from_settings(
            Settings(
                db_path=Path(folder) / "memory.sqlite3",
                pending_turns=5,
                context_turns=15,
                model_timeout=180,
                model_max_tokens=4096,
                planner_mode="model",
                acceleration_enabled=False,
                output_language="en",
                evidence_grounding_enabled=True,
                grounding_semantic_enabled=False,
                grounding_max_raw_sources=3,
            )
        )
        resources.push_async_callback(runtime.close)
        if runtime.model is None or runtime.retriever.embedder is None:
            raise RuntimeError("MemNexus model or Qwen embedding is not configured")
        scope = Scope(
            tenant_id="locomo-pilot", owner_id=str(sample.get("sample_id", "sample-0"))
        )
        session = runtime.store.create_session(scope, "LoCoMo official pilot")
        build_before = dict(runtime.model.usage)
        started = time.perf_counter()
        extraction_batches = 0
        for index, turn in enumerate(turns, 1):
            result = await runtime.append_turn(
                scope,
                session.session_id,
                TurnInput(
                    request_id=f"dialogue-{index}",
                    events=[
                        Event(
                            role="user",
                            speaker=turn["speaker"],
                            content=f"{turn['speaker']}: {turn['text']}",
                            occurred_at=parse_occurred_at(turn["date_time"]),
                        )
                    ],
                ),
            )
            if result["extraction_due"]:
                extracted = await runtime.extract(scope, session.session_id)
                if extracted["status"] != "idle":
                    extraction_batches += 1
        while True:
            extracted = await runtime.extract(scope, session.session_id)
            if extracted["status"] == "idle":
                break
            extraction_batches += 1
        build_ms = (time.perf_counter() - started) * 1000
        memories = runtime.store.memories(scope)
        if not memories:
            raise RuntimeError("MemNexus memory build produced zero memories")
        build_usage = usage_delta(runtime.model.usage, build_before)
        checkpoint(
            usage_ledger,
            ".memory.json",
            {
                "sample_id": sample["sample_id"],
                "turns": [
                    turn.model_dump(mode="json")
                    for turn in runtime.store.turns(scope, session.session_id)
                ],
                "memories": [memory.model_dump(mode="json") for memory in memories],
                "build_usage": build_usage,
            },
        )
        rows = []
        for qa in questions:
            query, gold = evaluation_query(qa)
            before = dict(runtime.model.usage)
            answer = await runtime.answer(
                scope,
                session.session_id,
                query,
                top_k=20,
                timeout=180,
                accelerate=False,
                prepare=False,
            )
            rows.append(
                {
                    "system": "MemNexus",
                    "sample_id": sample.get("sample_id", "0"),
                    "category": CATEGORY_NAMES[qa["category"]],
                    "category_id": qa["category"],
                    "question": qa["question"],
                    "evaluation_query": query,
                    "gold_answer": gold,
                    "prediction": answer.generated_text,
                    "retrieved_sources": [
                        source.model_dump(mode="json") for source in answer.sources
                    ],
                    "retrieval_count": answer.retrieval_count,
                    "model_calls": usage_delta(runtime.model.usage, before),
                    "runtime_model_calls": answer.model_calls.model_dump(mode="json"),
                    "grounding": answer.grounding.model_dump(mode="json"),
                    "reasoning": answer.reasoning,
                    "retrieval_ms": answer.acceleration.retrieval_ms,
                    "generation_ms": answer.acceleration.generation_ms,
                    "latency_ms": answer.elapsed_ms,
                    "f1": token_f1(answer.generated_text, gold),
                    "bleu1": bleu1(answer.generated_text, gold),
                    "judge_result": None,
                    "citations": answer.citations,
                    "warnings": answer.warnings,
                    "plan": answer.plan.model_dump(mode="json")
                    if answer.plan
                    else None,
                }
            )
            checkpoint(usage_ledger, ".rows.jsonl", rows[-1], append=True)
        stage = {
            "system": "MemNexus",
            "sample_id": sample["sample_id"],
            "raw_turns": len(turns),
            "memory_count": len(memories),
            "extraction_batches": extraction_batches,
            "build_ms": build_ms,
            "build_usage": build_usage,
            "total_usage": dict(runtime.model.usage),
            "rejected_question_candidates": runtime.model.rejected_question_candidates,
            "validation_rejections": dict(runtime.model.validation_rejections),
            "collection_queries_enabled": runtime.settings.collection_queries_enabled,
            "collection_index_limit": min(runtime.settings.max_candidates, 120),
        }
        checkpoint(usage_ledger, ".stage.json", stage)
    return stage, rows


def judge_rows(
    rows: list[dict],
    component_budget: float,
    *,
    usage_ledger: Path | None = None,
) -> dict:
    os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = str(component_budget)
    os.environ["EXPERIMENT_USAGE_LEDGER_PATH"] = str(
        usage_ledger or ROOT / "result" / "judge_usage_live.json"
    )
    if str(SIMPLEMEM) not in sys.path:
        sys.path.insert(0, str(SIMPLEMEM))
    import config
    from simplemem.core.utils.llm_client import LLMClient

    judge = LLMClient(
        api_key=config.JUDGE_API_KEY or config.OPENAI_API_KEY,
        model=config.JUDGE_MODEL,
        base_url=config.JUDGE_BASE_URL,
        enable_thinking=False,
        use_streaming=False,
    )
    for row in rows:
        prompt = (
            "Judge whether the prediction correctly answers the question according to the gold answer. "
            "Ignore style and accept semantically equivalent concise answers. Reject contradictions, "
            "unsupported additions that change the answer, or missing core facts. Return one JSON object "
            "with label CORRECT or WRONG and a short reason.\n"
            + json.dumps(
                {
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "prediction": row["prediction"],
                },
                ensure_ascii=False,
            )
        )
        response = judge.chat_completion(
            [
                {"role": "system", "content": "You are a strict blinded QA evaluator."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_retries=2,
        )
        parsed = judge.extract_json(response)
        label = str(parsed.get("label", "")).upper()
        if label not in {"CORRECT", "WRONG"}:
            raise RuntimeError("Judge returned an invalid label")
        row["judge_result"] = label
        row["judge_reason"] = str(parsed.get("reason", ""))
        row["judge_raw_response"] = response
        checkpoint(usage_ledger, ".rows.jsonl", row, append=True)
    return dict(judge.usage)


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for system in ("SimpleMem", "MemNexus"):
        selected = [row for row in rows if row["system"] == system]
        result[system] = {
            "questions": len(selected),
            "accuracy": sum(row["judge_result"] == "CORRECT" for row in selected)
            / len(selected),
            "mean_f1": sum(row["f1"] for row in selected) / len(selected),
            "mean_bleu1": sum(row["bleu1"] for row in selected) / len(selected),
            "mean_latency_ms": sum(row["latency_ms"] for row in selected)
            / len(selected),
            "input_tokens": sum(
                row["model_calls"].get("input_tokens", 0) for row in selected
            ),
            "output_tokens": sum(
                row["model_calls"].get("output_tokens", 0) for row in selected
            ),
            "answer_cost_rmb": sum(
                row["model_calls"].get("cost_rmb", 0) for row in selected
            ),
            "by_category": {
                row["category"]: {
                    "judge_result": row["judge_result"],
                    "f1": row["f1"],
                    "bleu1": row["bleu1"],
                }
                for row in selected
            },
        }
    return result


async def main(args) -> int:
    sample, questions = load_fixed_sample(args.per_category)
    report = {
        "status": "running",
        "scope": "oracle_evidence_horizon_development_pilot_not_paper_main_result",
        "dataset": str(DATASET.relative_to(ROOT)),
        "sample_id": sample.get("sample_id", "0"),
        "selection": (
            f"first official conversation; first {args.per_category} QA(s) in dataset order for each category; "
            "official raw sessions D1-D2 because all selected gold evidence lies in that horizon"
        ),
        "oracle_max_session": ORACLE_MAX_SESSION,
        "questions": [
            {"category": CATEGORY_NAMES[qa["category"]], "question": qa["question"]}
            for qa in questions
        ],
        "budget": {
            "combined_limit_rmb": 8.0,
            "failed_empty_simplemem_run_upper_bound_rmb": FAILED_EMPTY_RUN_UPPER_BOUND_RMB,
            "failed_parser_simplemem_run_usage": FAILED_PARSER_RUN_USAGE,
            "simplemem_cap_rmb": 1.2,
            "memnexus_cap_rmb": 1.2,
            "judge_cap_rmb": 0.3,
            "pricing": {
                "input_rmb_per_million": INPUT_PRICE,
                "output_rmb_per_million": OUTPUT_PRICE,
                "basis": "DeepSeek Flash peak cache-miss USD prices converted at conservative 8 RMB/USD",
            },
        },
        "stages": [],
        "rows": [],
    }
    save(report, args.output)
    simple_stage, simple_rows = run_simplemem(sample, questions, 1.2)
    report["stages"].append(simple_stage)
    report["rows"].extend(simple_rows)
    save(report, args.output)
    nexus_stage, nexus_rows = await run_memnexus(sample, questions, 1.2)
    report["stages"].append(nexus_stage)
    report["rows"].extend(nexus_rows)
    save(report, args.output)
    report["judge_usage"] = judge_rows(report["rows"], 0.3)
    report["aggregate"] = aggregate(report["rows"])
    probe_cost = (
        PROBE_USAGE["input_tokens"] * INPUT_PRICE
        + PROBE_USAGE["output_tokens"] * OUTPUT_PRICE
    ) / 1_000_000
    report["probe_usage"] = {**PROBE_USAGE, "cost_rmb": probe_cost}
    component_cost = sum(stage["total_usage"]["cost_rmb"] for stage in report["stages"])
    successful_cost = component_cost + report["judge_usage"]["cost_rmb"] + probe_cost
    conservative_total = (
        successful_cost
        + FAILED_EMPTY_RUN_UPPER_BOUND_RMB
        + FAILED_PARSER_RUN_USAGE["cost_rmb"]
    )
    report["successful_run_cost_rmb"] = successful_cost
    report["conservative_total_cost_upper_bound_rmb"] = conservative_total
    if conservative_total > 8:
        raise RuntimeError("Combined experiment budget exceeded")
    report["status"] = "completed"
    save(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "aggregate": report["aggregate"],
                "successful_run_cost_rmb": successful_cost,
                "conservative_total_cost_upper_bound_rmb": conservative_total,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "result" / "locomo_pilot.json"
    )
    parser.add_argument("--per-category", type=int, default=1)
    options = parser.parse_args()
    if options.per_category < 1:
        parser.error("--per-category must be >= 1")
    raise SystemExit(asyncio.run(main(options)))
