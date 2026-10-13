"""Budget-safe entry point for MemNexus/SimpleMem comparison experiments.

The current entry point implements the non-paid P0 layer: official dataset
validation, immutable manifest generation, configuration fairness checks and a
combined RMB budget preflight. Paid execution is deliberately refused until
both an API key and non-zero, user-verified provider prices are configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCOMO = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
LONGMEM_ORACLE = ROOT / "benchmark-data" / "official" / "LongMemEval" / "longmemeval_oracle.json"
LONGMEM_S = ROOT / "benchmark-data" / "official" / "LongMemEval" / "longmemeval_s_cleaned.json"
SIMPLEMEM_CONFIG = ROOT / "SimpleMem" / "config.py"
MEMNEXUS_ENV = ROOT / "MemNexus" / "memory-system" / ".env"

LOCOMO_CATEGORIES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


class PreflightError(RuntimeError):
    """The experiment is not safe or not fair enough to start."""


@dataclass(frozen=True)
class LockedProtocol:
    main_model: str = "deepseek-v4-flash"
    judge_model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int = 1024
    embedding_encoding: str = "query_prompt_for_query_plain_for_document"
    locomo_window_turns: int = 20
    stride_turns: int = 5
    retrieval_min_k: int = 3
    retrieval_max_k: int = 20
    accelerate: bool = False
    planner_mode: str = "model"
    output_language: str = "en"
    max_retrieval_rounds: int = 1
    combined_budget_rmb: float = 8.0


@dataclass
class BudgetLedger:
    limit_rmb: float
    input_price_per_million: float
    output_price_per_million: float
    spent_rmb: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.limit_rmb <= 0:
            raise ValueError("Budget must be positive")
        if self.input_price_per_million <= 0 or self.output_price_per_million <= 0:
            raise PreflightError(
                "Paid execution requires current non-zero input/output prices; "
                "zero-price defaults never authorize a provider call"
            )

    def projected_cost(self, input_tokens: int, output_tokens: int) -> float:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        return (
            input_tokens * self.input_price_per_million
            + output_tokens * self.output_price_per_million
        ) / 1_000_000

    def reserve(self, input_tokens: int, max_output_tokens: int) -> float:
        """Fail before a call unless its worst-case completion fits the ceiling."""
        projected = self.projected_cost(input_tokens, max_output_tokens)
        if self.spent_rmb + projected > self.limit_rmb:
            raise PreflightError(
                f"Call would exceed combined budget: {self.spent_rmb + projected:.6f} "
                f"> {self.limit_rmb:.6f} RMB"
            )
        return projected

    def commit(self, input_tokens: int, output_tokens: int) -> float:
        actual = self.projected_cost(input_tokens, output_tokens)
        if self.spent_rmb + actual > self.limit_rmb:
            raise PreflightError("Provider usage exceeded the reserved combined budget")
        self.spent_rmb += actual
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        return actual


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise PreflightError(f"Dataset must be a non-empty JSON list: {path}")
    return value


def audit_locomo(path: Path = LOCOMO) -> dict[str, Any]:
    data = _load_json(path)
    categories: Counter[str] = Counter()
    qa_count = evidence_count = bad_evidence = unparseable_evidence = 0
    sample_ids: set[str] = set()
    for sample_index, sample in enumerate(data):
        sample_id = str(sample.get("sample_id", sample_index))
        if sample_id in sample_ids:
            raise PreflightError(f"Duplicate LoCoMo sample_id: {sample_id}")
        sample_ids.add(sample_id)
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            raise PreflightError(f"LoCoMo {sample_id} has no conversation object")
        dialogue_ids = {
            (int(match.group(1)), int(match.group(2)))
            for key, session in conversation.items()
            if key.startswith("session_") and isinstance(session, list)
            for turn in session
            if isinstance(turn, dict)
            if (match := re.fullmatch(r"D(\d+):(\d+)", str(turn.get("dia_id", ""))))
        }
        for qa_index, qa in enumerate(sample.get("qa", [])):
            if not qa.get("question"):
                raise PreflightError(f"LoCoMo {sample_id}/{qa_index} has no question")
            category = qa.get("category")
            if category not in LOCOMO_CATEGORIES:
                raise PreflightError(f"Unknown LoCoMo category: {category}")
            gold = qa.get("adversarial_answer") if category == 5 else qa.get("answer")
            if gold is None:
                raise PreflightError(f"LoCoMo {sample_id}/{qa_index} has no gold answer")
            categories[LOCOMO_CATEGORIES[category]] += 1
            qa_count += 1
            for evidence_entry in qa.get("evidence", []):
                references = re.findall(r"D(\d+):(\d+)", str(evidence_entry))
                if not references:
                    unparseable_evidence += 1
                for session_number, turn_number in references:
                    evidence_count += 1
                    if (int(session_number), int(turn_number)) not in dialogue_ids:
                        bad_evidence += 1
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "conversations": len(data),
        "questions": qa_count,
        "categories": dict(sorted(categories.items())),
        "evidence_references": evidence_count,
        "unresolved_evidence_references": bad_evidence,
        "unparseable_evidence_entries": unparseable_evidence,
    }


def audit_longmemeval(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    types: Counter[str] = Counter()
    seen: set[str] = set()
    evidence_sessions = unresolved = 0
    for index, sample in enumerate(data):
        question_id = str(sample.get("question_id", ""))
        if not question_id or question_id in seen:
            raise PreflightError(f"Invalid/duplicate LongMemEval question_id at row {index}")
        seen.add(question_id)
        if not sample.get("question") or sample.get("answer") is None:
            raise PreflightError(f"LongMemEval {question_id} lacks question/gold answer")
        question_type = str(sample.get("question_type", "unknown"))
        types[question_type] += 1
        raw_haystack_ids = sample.get("haystack_session_ids", [])
        haystack_ids = {str(item) for item in raw_haystack_ids}
        sessions = sample.get("haystack_sessions", [])
        if not isinstance(sessions, list) or len(sessions) != len(raw_haystack_ids):
            raise PreflightError(f"LongMemEval {question_id} session/id count mismatch")
        for answer_id in sample.get("answer_session_ids", []):
            evidence_sessions += 1
            if str(answer_id) not in haystack_ids:
                unresolved += 1
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "questions": len(data),
        "question_types": dict(sorted(types.items())),
        "evidence_session_references": evidence_sessions,
        "unresolved_evidence_session_references": unresolved,
    }


def _dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def fairness_preflight(protocol: LockedProtocol) -> dict[str, Any]:
    simple = runpy.run_path(str(SIMPLEMEM_CONFIG))
    nexus = _dotenv(MEMNEXUS_ENV)
    checks = {
        "main_model": simple.get("LLM_MODEL") == nexus.get("MEMORY_MODEL_NAME") == protocol.main_model,
        "judge_model": simple.get("JUDGE_MODEL") == nexus.get("MEMORY_JUDGE_MODEL_NAME") == protocol.judge_model,
        "temperature": float(simple.get("TEMPERATURE", -1)) == protocol.temperature,
        "embedding_model": simple.get("EMBEDDING_MODEL") == nexus.get("MEMORY_EMBEDDING_NAME") == protocol.embedding_model,
        "embedding_dimension": int(simple.get("EMBEDDING_DIMENSION", -1)) == protocol.embedding_dimension,
        "embedding_encoding": bool(simple.get("EMBEDDING_QUERY_PROMPT")) and nexus.get("MEMORY_EMBEDDING_ENCODING") == "query_prompt",
        "window": int(simple.get("WINDOW_SIZE", -1)) == protocol.locomo_window_turns,
        "stride": (
            int(simple.get("WINDOW_SIZE", -1)) - int(simple.get("OVERLAP_SIZE", -1))
            == protocol.stride_turns
        ),
        "retrieval_max": int(simple.get("SEMANTIC_TOP_K", -1)) == protocol.retrieval_max_k,
        "combined_budget": float(simple.get("EXPERIMENT_BUDGET_RMB", -1)) == float(nexus.get("MEMORY_BUDGET_RMB", -1)) == protocol.combined_budget_rmb,
        "output_language": nexus.get("MEMORY_OUTPUT_LANGUAGE", "en").lower() == protocol.output_language,
        "keys_configured": all(
            bool(str(value or "").strip())
            for value in (
                simple.get("OPENAI_API_KEY"),
                simple.get("JUDGE_API_KEY"),
                nexus.get("MEMORY_MODEL_API_KEY"),
                nexus.get("MEMORY_JUDGE_API_KEY"),
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PreflightError("Fairness preflight failed: " + ", ".join(failed))
    return checks


def build_manifest() -> dict[str, Any]:
    protocol = LockedProtocol()
    checks = fairness_preflight(protocol)
    return {
        "schema_version": 1,
        "protocol": asdict(protocol),
        "fairness_checks": checks,
        "datasets": {
            "locomo": audit_locomo(),
            "longmemeval_oracle": audit_longmemeval(LONGMEM_ORACLE),
            "longmemeval_s": audit_longmemeval(LONGMEM_S),
        },
        "result_record_required_fields": [
            "sample_id",
            "question",
            "gold_answer",
            "prediction",
            "retrieved_sources",
            "model_calls",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "cost_rmb",
            "judge_result",
        ],
    }


def paid_preflight() -> dict[str, Any]:
    """Validate authority and cost inputs without making a network request."""
    protocol = LockedProtocol()
    fairness_preflight(protocol)
    simple = runpy.run_path(str(SIMPLEMEM_CONFIG))
    nexus = _dotenv(MEMNEXUS_ENV)
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        key = str(simple.get("OPENAI_API_KEY", "")).strip() or nexus.get("MEMORY_MODEL_API_KEY", "")
    if not key:
        raise PreflightError("API key is blank; paid execution is safely disabled")
    input_price = float(os.getenv("INPUT_PRICE_RMB_PER_MILLION_TOKENS", simple.get("INPUT_PRICE_RMB_PER_MILLION_TOKENS", 0)))
    output_price = float(os.getenv("OUTPUT_PRICE_RMB_PER_MILLION_TOKENS", simple.get("OUTPUT_PRICE_RMB_PER_MILLION_TOKENS", 0)))
    ledger = BudgetLedger(protocol.combined_budget_rmb, input_price, output_price)
    return {"budget_rmb": ledger.limit_rmb, "prices_configured": True, "api_key_configured": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "paid-preflight"), default="audit")
    parser.add_argument("--output", type=Path, default=ROOT / "result" / "benchmark_manifest.json")
    args = parser.parse_args()
    try:
        report = build_manifest() if args.mode == "audit" else paid_preflight()
    except PreflightError as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, ensure_ascii=False))
        return 2
    if args.mode == "audit":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "mode": args.mode, "report": report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
