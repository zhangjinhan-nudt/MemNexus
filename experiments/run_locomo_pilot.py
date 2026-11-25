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
from contextlib import AsyncExitStack, nullcontext
from copy import deepcopy
from datetime import UTC
from pathlib import Path

import httpx
import numpy as np
from dateutil import parser as date_parser

from paper_protocol import JUDGE_INSTRUCTION, paper_table

ROOT = Path(__file__).resolve().parents[1]
SIMPLEMEM = ROOT / "SimpleMem"
MEMNEXUS = ROOT / "MemNexus" / "memory-system"
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
# Official LoCoMo categories; the evaluator keys on these ids.
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
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


# Answers that a provider error produced rather than the system deciding.
# Protocol R2: read forced_abstain before accuracy. A legitimate abstention
# ("Not mentioned in the conversation.") is a decision and must NOT be flagged,
# because on adversarial questions it is the correct answer.
FORCED_ABSTAIN_MARKERS = {"failed to generate answer"}


def is_forced_abstain(prediction: str) -> bool:
    text = (prediction or "").strip()
    return not text or text.casefold() in FORCED_ABSTAIN_MARKERS


# Transient provider/network failures: worth retrying at both the question
# level (3 attempts with backoff) and, if they persist, the unit level.
TRANSIENT_ERROR_NAMES = {
    "ReadError", "ConnectError", "ReadTimeout", "ConnectTimeout",
    "APIConnectionError", "APITimeoutError",
}


def is_transient_error(exc: BaseException) -> bool:
    """网络/限流类瞬时错误：重试有意义。与确定性校验错误（ValueError 族）
    相对——后者在 temp=0 下重试必然复现，必须降级而非重试。"""
    status = getattr(exc, "status_code", None)
    return (
        status == 429
        or isinstance(status, int) and status >= 500
        or isinstance(exc, (TimeoutError, ConnectionError))
        or type(exc).__name__ in TRANSIENT_ERROR_NAMES
    )


def is_deterministic_output_error(exc: BaseException | None) -> bool:
    """确定性输出校验错误：降级为该题拒答行，而非杀死整单元。

    判据：ValueError 族（ModelOutputError 是其子类，synthesis 的日期/引用
    校验也抛 ValueError）且非网络瞬时错误。temp=0 下同一证据集重试必然
    复现同一拒绝——实测 conv-44 连续 6 轮单元重试全败于同一道题。
    预算类（BudgetExceededError 继承 BaseException）与未知异常不在此列，
    仍走 raise，由人工或单元级重试处理。
    """
    return isinstance(exc, ValueError) and not is_transient_error(exc)


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
    if qa.get("dataset") == "LongMemEval-S":
        return qa["question"] + "\nQuestion date: " + qa["question_date"], str(qa["answer"])
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


def _max_turn_chars() -> int:
    """Admission limit for one turn; must match MEMORY_MAX_TURN_CHARS.

    The runtime compares the *serialised* turn against this value, so the
    splitter keeps 2500 characters of headroom for the JSON wrapper and the
    surrounding event metadata.
    """
    return max(1000, int(os.getenv("MEMORY_MAX_TURN_CHARS", "6000")) - 2500)


def _split_oversized(text: str, limit: int | None = None) -> list[str]:
    """Split one over-long utterance into <=limit chunks at natural boundaries.

    LongMemEval-S contains utterances up to ~5.8k characters. The runtime
    rejects a single turn whose estimate exceeds the ingest window, which would
    fail the whole question. Splitting is applied to *both* systems identically
    (both read turns from this function), so it is input normalisation, not a
    memory advantage. Boundaries: paragraph, then sentence, then hard cut.
    """
    if len(text) <= (limit := limit or _max_turn_chars()):
        return [text]
    units: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            continue
        if len(para) <= limit:
            units.append(para)
            continue
        buffer = ""
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            piece = sentence
            while len(piece) > limit:
                units.append(piece[:limit])
                piece = piece[limit:]
            if not piece:
                continue
            if len(buffer) + len(piece) + 1 <= limit:
                buffer = f"{buffer} {piece}".strip()
            else:
                if buffer:
                    units.append(buffer)
                buffer = piece
        if buffer:
            units.append(buffer)
    chunks: list[str] = []
    buffer = ""
    for unit in units:
        if len(buffer) + len(unit) + 1 <= limit:
            buffer = f"{buffer}\n{unit}".strip() if buffer else unit
        else:
            if buffer:
                chunks.append(buffer)
            buffer = unit
    if buffer:
        chunks.append(buffer)
    return chunks or [text[:limit]]


def raw_turns(sample: dict, max_session: int | None = None) -> list[dict]:
    conversation = sample["conversation"]
    turns = []
    split_long = sample.get("dataset") == "LongMemEval-S"
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
                pieces = _split_oversized(text) if split_long else [text]
                for piece_index, piece in enumerate(pieces):
                    turns.append(
                        {
                            "dia_id": (
                                turn["dia_id"]
                                if len(pieces) == 1
                                else f"{turn['dia_id']}#c{piece_index + 1}"
                            ),
                            "speaker": turn["speaker"],
                            "text": piece if split_long else piece.strip(),
                            "date_time": date_text,
                            "source_session": key,
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


def checkpoint_row_key(row: dict) -> tuple[str, str, str, str, str]:
    """Stable question identity; text alone is not unique in LoCoMo."""
    return (
        str(row.get("sample_id", "")),
        str(row.get("qa_index", "")),
        str(row.get("question", "")),
        str(row.get("evaluation_query", "")),
        str(row.get("gold_answer", "")),
    )


def valid_checkpoint_row(row: dict) -> bool:
    prediction = str(row.get("prediction") or "").strip()
    return bool(prediction and prediction != "Failed to generate answer" and not row.get("error"))


def load_resumable_rows(usage_ledger: Path | None, sample_id: str) -> dict[tuple[str, str, str, str, str], dict]:
    """Read valid rows only; failed/empty attempts remain in the audit log."""
    if usage_ledger is None:
        return {}
    path = usage_ledger.with_suffix(".rows.jsonl")
    if not path.exists():
        return {}
    completed = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed checkpoint {path}:{line_number}") from error
            key = checkpoint_row_key(row)
            if key[0] == str(sample_id) and valid_checkpoint_row(row):
                completed[key] = row
    return completed


def validate_qa_history_scope(record: dict, source_sessions: tuple[str, ...], source_read_scope: str) -> None:
    """Reject changed QA permissions without rewriting historical metadata."""
    containers = [record]
    if isinstance(record.get("reasoning"), dict):
        containers.append(record["reasoning"])
    scopes = [item["source_read_scope"] for item in containers if "source_read_scope" in item] or ["quotes"]
    grants = [item["authorized_source_sessions"] for item in containers
              if "authorized_source_sessions" in item] or [[]]
    if any(value != source_read_scope for value in scopes) or any(
        not isinstance(value, (list, tuple)) or not all(isinstance(sid, str) for sid in value)
        or tuple(sorted(value)) != source_sessions for value in grants
    ):
        raise RuntimeError(
            "Checkpoint used a different QA history scope. Reuse the build snapshot in a new QA results directory."
        )


def validate_completed_qa_history(result: dict, source_read_scope: str) -> None:
    """Gate current-run reuse; historical read-only collectors do not call this."""
    stage = result.get("stage", {})
    if stage.get("source_read_scope", "quotes") != source_read_scope:
        raise RuntimeError("Completed unit used a different QA history scope; use a new QA results directory.")
    if source_read_scope == "events":
        history = stage.get("history_session_ids")
        qa_session = stage.get("qa_session_id")
        if (not isinstance(history, list) or not history
                or not all(isinstance(sid, str) and sid for sid in history)
                or len(set(history)) != len(history) or qa_session not in history
                or len(history) != stage.get("internal_session_count")):
            raise RuntimeError("Completed unit lacks a verifiable full QA history grant; use a new results directory.")
        expected = tuple(sorted(set(history) - {qa_session}))
    else:
        expected = ()
    validate_qa_history_scope(stage, expected, source_read_scope)
    for row in result["rows"]:
        validate_qa_history_scope(row, expected, source_read_scope)


def run_simplemem(
    sample: dict,
    questions: list[dict],
    component_budget: float,
    *,
    usage_ledger: Path | None = None,
    max_session: int | None = None,
    resume: bool = False,
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
    if sample.get("expected_raw_turns") is not None and len(turns) != sample["expected_raw_turns"]:
        raise ValueError("Raw conversation coverage mismatch before any provider request")
    embedder = RemoteQwenEmbedding()
    # Preserve the native index beside an explicit audit ledger. No rebuild or
    # algorithm change: this only prevents successful/failed builds disappearing.
    persistent_folder = usage_ledger.with_suffix(".index") if usage_ledger else None
    if persistent_folder:
        persistent_folder.mkdir(parents=True, exist_ok=False)
    with (nullcontext(str(persistent_folder)) if persistent_folder
          else tempfile.TemporaryDirectory(prefix="simplemem-locomo-pilot-")) as folder:
        system = SimpleMemSystem(
            db_path=str(Path(folder) / "lancedb"),
            clear_db=True,
            enable_parallel_processing=False,
            max_parallel_workers=1,
            enable_parallel_retrieval=False,
            max_retrieval_workers=1,
            embedding_model=embedder,
        )
        # Conversation units run in threads: environment variables are process
        # global and cannot carry per-unit accounting identity. Bind the client
        # before its first provider request (initialization only builds objects).
        system.llm_client.ledger_path = str(
            usage_ledger or ROOT / "result" / "simplemem_usage_live.json"
        )
        system.llm_client.cost_limit_rmb = component_budget
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
        completed = load_resumable_rows(usage_ledger, str(sample.get("sample_id", "0"))) if resume else {}
        rows = []
        for qa in questions:
            query, gold = evaluation_query(qa)
            key = checkpoint_row_key({
                "sample_id": sample.get("sample_id", "0"), "qa_index": qa["qa_index"],
                "question": qa["question"], "evaluation_query": query, "gold_answer": gold,
            })
            if key in completed:
                rows.append(completed[key])
                continue
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
                    "qa_index": qa["qa_index"],
                    "category": CATEGORY_NAMES[qa["category"]],
                    "category_id": qa["category"],
                    "question": qa["question"],
                    "evaluation_query": query,
                    "gold_answer": gold,
                    "prediction": prediction,
                    "forced_abstain": is_forced_abstain(prediction),
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


async def build_memory_base(
    runtime, scope, session, turns: list[dict], longmemeval: bool
):
    """Ingest unchanged turns; LME source boundaries own extraction context.

    Return the batch count and final history session used for QA.
    """
    from agent_memory.models import Event, TurnInput

    batches = 0
    by_topic = {s.topic: s for s in runtime.store.sessions(scope)}
    if len(by_topic) != len(runtime.store.sessions(scope)):
        raise ValueError("Ambiguous session topics in this benchmark component")
    async def drain(active):
        nonlocal batches
        while True:
            extracted = await runtime.extract(scope, active.session_id)
            if extracted["status"] == "idle":
                break
            batches += 1
    for index, turn in enumerate(turns, 1):
        if longmemeval:
            topic = "LongMemEval-S history: " + turn["source_session"]
            if session.topic != topic:
                await drain(session)
                if topic not in by_topic:
                    by_topic[topic] = runtime.store.create_session(scope, topic)
                session = by_topic[topic]
        content = turn["text"] if longmemeval else f"{turn['speaker']}: {turn['text']}"
        chunks = [content]
        for part, chunk in enumerate(chunks):
            result = await runtime.append_turn(
                scope,
                session.session_id,
                TurnInput(
                    request_id=f"dialogue-{index}-{part}",
                    events=[
                        Event(
                            role=("assistant" if longmemeval and turn["speaker"] == "assistant" else "user"),
                            speaker=turn["speaker"],
                            content=chunk,
                            occurred_at=parse_occurred_at(turn["date_time"]),
                        )
                    ],
                ),
            )
            if result["extraction_due"]:
                extracted = await runtime.extract(scope, session.session_id)
                if extracted["status"] != "idle":
                    batches += 1
    await drain(session)
    return batches, session


def restore_budget_usage(client, usage_ledger: Path | None, *, resume: bool) -> None:
    """Retain paid/unknown attempts across process retries of the same unit."""
    if not resume or usage_ledger is None or not usage_ledger.exists():
        return
    previous = json.loads(usage_ledger.read_text(encoding="utf-8"))
    required = {"cost_rmb", "input_tokens", "output_tokens", "provider_attempts"}
    if not isinstance(previous, dict) or not required <= previous.keys():
        raise ValueError("Cannot resume an incomplete usage ledger")
    for key, value in previous.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid prior usage field: {key}")
    if previous.get("reserved_rmb", 0):
        raise RuntimeError("Unsettled prior provider reservation; reconcile before resuming")
    # unknown_usage_upper_bound_rmb has already been included in cost_rmb by
    # the provider. Do not add it a second time or reset request attempt IDs.
    client.usage.update(previous)


def benchmark_memory_settings(database: Path):
    """The effective settings used for a native benchmark runtime."""
    from agent_memory.settings import Settings
    return Settings(
        db_path=database,
        pending_turns=5,
        context_turns=int(os.getenv("MEMORY_CONTEXT_TURNS", "15")),
        model_timeout=180,
        model_max_tokens=int(os.getenv("EXPERIMENT_MEMNEXUS_MAX_TOKENS", "4096")),
        planner_mode=os.getenv("MEMORY_PLANNER_MODE", "model"),
        acceleration_enabled=False,
        output_language="en",
        evidence_grounding_enabled=True,
        grounding_semantic_enabled=False,
    )


async def run_memnexus(
    sample: dict,
    questions: list[dict],
    component_budget: float,
    *,
    usage_ledger: Path | None = None,
    max_session: int | None = None,
    snapshot_dir: Path | None = None,
    require_snapshot: bool = False,
    resilient: bool = False,
    answer_timeout: float = 180.0,
    resume: bool = False,
):
    os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = str(component_budget)
    os.environ["EXPERIMENT_USAGE_LEDGER_PATH"] = str(
        usage_ledger or ROOT / "result" / "memnexus_usage_live.json"
    )
    os.environ["MEMORY_EXPERIMENT_DROP_QUESTION_CANDIDATES"] = "1"
    sys.path.insert(0, str(MEMNEXUS / "src"))
    from agent_memory import snapshot as memory_snapshot
    from agent_memory.models import Scope
    from agent_memory.providers import BudgetExceededError as MemNexusBudgetExceededError
    from agent_memory.runtime import MemoryRuntime

    turns = raw_turns(
        sample, ORACLE_MAX_SESSION if max_session is None else max_session
    )
    if sample.get("expected_raw_turns") is not None and len(turns) != sample["expected_raw_turns"]:
        raise ValueError("Raw conversation coverage mismatch before any provider request")
    longmemeval = sample.get("dataset") == "LongMemEval-S"
    sample_id = str(sample.get("sample_id", "sample-0"))
    async with AsyncExitStack() as resources:
        # A missing snapshot previously meant that every provider failure threw
        # away the partially built SQLite base together with its temporary
        # directory.  Keep only that build in the unit directory so a retry can
        # resume through the existing request_id idempotency contract.  Frozen
        # snapshots still use an isolated temporary replay database.
        persistent_build = bool(
            resume and usage_ledger is not None and snapshot_dir is not None
            and not memory_snapshot.available(snapshot_dir, sample_id)
        )
        if persistent_build:
            database = usage_ledger.with_suffix(".work.sqlite3")
        else:
            folder = resources.enter_context(
                tempfile.TemporaryDirectory(prefix="memnexus-locomo-pilot-")
            )
            database = Path(folder) / "memory.sqlite3"
        # Replay happens before the runtime opens the file, so the runtime does
        # no ingestion at all and QA sees exactly the frozen memory set.
        replayed = None
        snapshot_thawed = False
        if snapshot_dir is not None:
            if memory_snapshot.available(snapshot_dir, sample_id):
                replayed = memory_snapshot.thaw(snapshot_dir, sample_id, database)
                snapshot_thawed = True
            elif require_snapshot:
                raise RuntimeError(
                    f"No memory snapshot for {sample_id!r} in {snapshot_dir} "
                    "and --require-snapshots forbids rebuilding it"
                )
        runtime = MemoryRuntime.from_settings(benchmark_memory_settings(database))
        resources.push_async_callback(runtime.close)
        if runtime.model is None or runtime.retriever.embedder is None:
            raise RuntimeError("MemNexus model or Qwen embedding is not configured")
        runtime.model.ledger_path = str(
            usage_ledger or ROOT / "result" / "memnexus_usage_live.json"
        )
        runtime.model.cost_limit_rmb = component_budget
        restore_budget_usage(runtime.model, usage_ledger, resume=resume)
        scope = Scope(tenant_id="locomo-pilot", owner_id=sample_id)
        build_before = dict(runtime.model.usage)
        started = time.perf_counter()
        sessions = runtime.store.sessions(scope)
        if replayed is not None:
            if not sessions:
                raise RuntimeError(f"Snapshot for {sample_id!r} contains no session")
            selected_id = replayed.get("qa_session_id")
            if selected_id:
                session = runtime.store.get_session(scope, selected_id)
            elif len(sessions) == 1:
                # Old flat snapshots remain explicitly old flat snapshots.
                session = sessions[0]
            else:
                raise RuntimeError("A multi-session snapshot needs an explicit QA session")
            extraction_batches = int(replayed.get("extraction_batches", 0))
            build_ms = 0.0
        else:
            if longmemeval:
                topic = "LongMemEval-S history: " + turns[0]["source_session"]
                allowed_topics = {"LongMemEval-S history: " + t["source_session"] for t in turns}
                if any(s.topic not in allowed_topics for s in sessions):
                    raise RuntimeError("Old flat build cannot be resumed as a session-preserving build; use a new component")
                matching = [s for s in sessions if s.topic == topic]
                if len(matching) > 1:
                    raise RuntimeError("Duplicate history session identity")
                session = matching[0] if matching else runtime.store.create_session(scope, topic)
            elif persistent_build and sessions:
                if len(sessions) != 1:
                    raise RuntimeError(
                        f"Resumable build for {sample_id!r} has {len(sessions)} sessions"
                    )
                session = sessions[0]
            else:
                session = runtime.store.create_session(scope, "LoCoMo official pilot")
            # In this ingestion-only database, append and extraction each add
            # one revision. Subtract committed turns to count prior extraction
            # windows exactly, including short boundary flushes. No QA or
            # manual state mutation occurs before a build is frozen.
            committed_before = sum(s.revision - len(runtime.store.turns(scope, s.session_id))
                                   for s in sessions)
            if committed_before < 0:
                raise RuntimeError("Inconsistent ingestion revision accounting")
            extraction_batches, session = await build_memory_base(
                runtime, scope, session, turns, longmemeval
            )
            extraction_batches += committed_before
            build_ms = (time.perf_counter() - started) * 1000
        memories = runtime.store.memories(scope)
        all_sessions = runtime.store.sessions(scope)
        if replayed is not None:
            history_ids = replayed.get("history_session_ids") or [session.session_id]
        elif longmemeval:
            by_topic = {s.topic: s.session_id for s in all_sessions}
            history_ids = [by_topic["LongMemEval-S history: " + key]
                           for key in dict.fromkeys(t["source_session"] for t in turns)]
        else:
            history_ids = [session.session_id]
        stored_turns = [turn for sid in history_ids for turn in runtime.store.turns(scope, sid)]
        if len(stored_turns) != len(turns):
            raise RuntimeError("Snapshot/history session coverage does not match the full source history")
        # The benchmark supplies the complete haystack for this question. Its
        # source boundaries are available to both systems independently of labels.
        source_sessions = tuple(sorted(set(history_ids) - {session.session_id})) if longmemeval else ()
        source_read_scope = "events" if longmemeval else "quotes"
        history_access = {"authorized_source_sessions": list(source_sessions),
                          "source_read_scope": source_read_scope}
        degraded_raw_only = not memories
        if degraded_raw_only and not runtime.store.turns(scope, session.session_id):
            raise RuntimeError("MemNexus memory build produced neither memories nor raw turns")
        # A replay must report no build spend; anything else would silently
        # double-count money that was paid on the run that froze the base.
        build_usage = (
            usage_delta(runtime.model.usage, runtime.model.usage)
            if replayed is not None
            else usage_delta(runtime.model.usage, build_before)
        )
        if snapshot_dir is not None and replayed is None:
            # Raw turns are usable evidence even if no fact was compressed.
            # Record this degraded build explicitly; it is not a successful
            # extraction and must remain visible in benchmark accounting.
            replayed = memory_snapshot.freeze(
                runtime.store,
                snapshot_dir,
                sample_id,
                metadata={
                    "system": "MemNexus",
                    "model": os.getenv("MEMORY_MODEL_NAME", ""),
                    "raw_turns": len(turns),
                    "memory_count": len(memories),
                    "degraded_raw_only": degraded_raw_only,
                    "extraction_batches": extraction_batches,
                    "build_usage": build_usage,
                    "session_layout": "dataset_sessions" if longmemeval else "single_session",
                    "qa_session_id": session.session_id,
                    "history_session_ids": history_ids,
                },
            )
        checkpoint(
            usage_ledger,
            ".memory.json",
            {
                "sample_id": sample["sample_id"],
                "turns": [
                    turn.model_dump(mode="json")
                    for turn in stored_turns
                ],
                "memories": [memory.model_dump(mode="json") for memory in memories],
                "build_usage": build_usage,
            },
        )
        completed = load_resumable_rows(usage_ledger, sample_id) if resume else {}
        rows = []
        for qa in questions:
            query, gold = evaluation_query(qa)
            key = checkpoint_row_key({
                "sample_id": sample_id, "qa_index": qa["qa_index"],
                "question": qa["question"], "evaluation_query": query, "gold_answer": gold,
            })
            if key in completed:
                validate_qa_history_scope(completed[key], source_sessions, source_read_scope)
                rows.append(completed[key])
                continue
            before = dict(runtime.model.usage)
            answer, error = None, None
            for attempt in range(3):
                try:
                    answer = await runtime.answer(
                        scope,
                        session.session_id,
                        query,
                        top_k=20,
                        timeout=answer_timeout,
                        accelerate=False,
                        prepare=False,
                        source_session_ids=source_sessions,
                        **({"source_read_scope": "events"} if longmemeval else {}),
                    )
                    break
                except (Exception, MemNexusBudgetExceededError) as caught:
                    error = caught
                    if not resilient:
                        raise
                    if not is_transient_error(caught) or attempt == 2:
                        break
                    await asyncio.sleep((3, 12, 30)[attempt])
            if answer is None and resume:
                # 2026-11-10 修复：确定性输出校验错误（synthesis/providers 的
                # ValueError 族，如日期端点校验、证据引用校验）在 temp=0 下
                # 重试必然复现；此前无条件 raise 会把该对话已答对的题全部丢弃
                # （实测 conv-44 连丢 79 题、6 轮单元重试全败于同一道题）。
                # 现在此类错误降级为该题的拒答行（非空 prediction +
                # error 审计字段），单元继续作答其余题；
                # transient 网络错误与预算/未知错误仍走 raise，由单元级重试兜底。
                if not is_deterministic_output_error(error):
                    raise RuntimeError(
                        f"MemNexus answer failed for {sample_id}: {type(error).__name__}: {error}"
                    ) from error
            if answer is None:
                rows.append({
                    "system": "MemNexus",
                    "sample_id": sample.get("sample_id", "0"),
                    "qa_index": qa["qa_index"],
                    "category": CATEGORY_NAMES[qa["category"]],
                    "category_id": qa["category"],
                    "question": qa["question"],
                    "evaluation_query": query,
                    "gold_answer": gold,
                    **history_access,
                    # 降级拒答措辞必须非空：跑批器把空预测判为单元失败
                    # （"empty predictions are not a completed unit"），
                    # 那正是本次修复要消除的传染路径。非对抗题上该行 F1≈0，
                    # 只丢这一题的分；对抗题上会计为 AdvSR 成功，分析时用
                    # answer_error 字段区分真实拒答与降级拒答。
                    "prediction": "Not mentioned in the conversation",
                    # The answer request raised and no text was produced at all:
                    # this is the forced-abstain case the gate exists to catch.
                    "forced_abstain": True,
                    "retrieved_sources": [],
                    "retrieval_count": 0,
                    "model_calls": usage_delta(runtime.model.usage, before),
                    "runtime_model_calls": {},
                    "grounding": {},
                    "reasoning": {"answer_path": "failed", **history_access},
                    "retrieval_ms": 0,
                    "generation_ms": 0,
                    "latency_ms": 0,
                    "f1": 0.0,
                    "bleu1": 0.0,
                    "judge_result": None,
                    "citations": [],
                    "warnings": [f"{type(error).__name__}: {error}"],
                    "plan": None,
                    "error": f"{type(error).__name__}: {error}",
                })
                checkpoint(usage_ledger, ".rows.jsonl", rows[-1], append=True)
                continue
            if resume and not str(answer.generated_text or "").strip():
                raise RuntimeError(f"MemNexus returned an empty answer for {sample_id}")
            rows.append(
                {
                    "system": "MemNexus",
                    "sample_id": sample.get("sample_id", "0"),
                    "qa_index": qa["qa_index"],
                    "category": CATEGORY_NAMES[qa["category"]],
                    "category_id": qa["category"],
                    "question": qa["question"],
                    "evaluation_query": query,
                    "gold_answer": gold,
                    **history_access,
                    "prediction": answer.generated_text,
                    "forced_abstain": is_forced_abstain(answer.generated_text)
                    or bool(answer.reasoning.get("forced_abstain")),
                    "retrieved_sources": [
                        source.model_dump(mode="json") for source in answer.sources
                    ],
                    "retrieval_count": answer.retrieval_count,
                    "model_calls": usage_delta(runtime.model.usage, before),
                    "runtime_model_calls": answer.model_calls.model_dump(mode="json"),
                    "grounding": answer.grounding.model_dump(mode="json"),
                    "reasoning": {**answer.reasoning, **history_access},
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
            "internal_session_count": len(all_sessions),
            "session_layout": ((replayed or {}).get("session_layout", "single_session")
                               if snapshot_thawed else "dataset_sessions" if longmemeval else "single_session"),
            "qa_session_id": session.session_id,
            "history_session_ids": history_ids,
            "build_ms": build_ms,
            **history_access,
            "build_usage": build_usage,
            "total_usage": dict(runtime.model.usage),
            "rejected_question_candidates": runtime.model.rejected_question_candidates,
            "validation_rejections": dict(runtime.model.validation_rejections),
            "answer_pipeline": "source_bound_synthesis",
            "degraded_raw_only": degraded_raw_only,
            "collection_index_limit": min(runtime.settings.max_candidates, 120),
            # Provenance for the freeze: a run is only comparable to another run
            # if both consumed the same sha256, and only honest about its cost
            # if memory_base_replayed says whether the build was actually paid.
            "memory_base_replayed": snapshot_thawed,
            "memory_snapshot_sha256": (replayed or {}).get("sha256"),
            "memory_snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else None,
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
        # Appendix A.4 verbatim.  The paper uses this judge for LongMemEval-S
        # only, and it deliberately asks for a generous grader; tightening the
        # wording changes the measured accuracy, so it is shared from one place.
        prompt = JUDGE_INSTRUCTION.format(
            question=row["question"],
            gold_answer=row["gold_answer"],
            generated_answer=row["prediction"],
        )
        response = judge.chat_completion(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_retries=2,
        )
        try:
            parsed = judge.extract_json(response)
        except (ValueError, TypeError, json.JSONDecodeError):
            # Some providers emit a long reasoning preamble despite JSON mode.
            # Recover only a complete final label object; never infer a label
            # from prose or silently mark an unparseable judgment.
            matches = re.findall(r"\{\s*\"label\"\s*:\s*\"(?:CORRECT|WRONG)\"\s*,\s*\"reason\"\s*:\s*\"(?:\\.|[^\"])*\"\s*\}", response, re.DOTALL)
            if not matches:
                raise
            parsed = json.loads(matches[-1])
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
    # Paper protocol (§3.1): the headline numbers are per-reasoning-type F1/BLEU-1
    # with an unweighted Average, plus a separate adversarial success rate.  The
    # legacy mean_f1/accuracy fields stay for continuity but are diagnostics only.
    paper = paper_table(rows)
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
            "paper_average_f1": (paper["systems"].get(system) or {}).get("average_f1"),
            "paper_average_bleu1": (paper["systems"].get(system) or {}).get("average_bleu1"),
            "paper_adversarial_sr": (paper["systems"].get(system) or {}).get("adversarial_sr"),
            "paper_f1_by_type": (paper["systems"].get(system) or {}).get("f1_by_type"),
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
