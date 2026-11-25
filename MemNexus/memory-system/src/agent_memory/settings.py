"""Validated configuration; process environment overrides the local project .env."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Local developer settings. Existing process environment always wins.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_int(name: str, default: int) -> int:
    """允许 0 的整数配置：用于"关闭该通道"是合法取值的开关型上限。"""
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path(os.getenv("MEMORY_DB_PATH", "data/memory.sqlite3")))
    local_tenant: str = field(default_factory=lambda: os.getenv("MEMORY_LOCAL_TENANT", "local"))
    local_owner: str = field(default_factory=lambda: os.getenv("MEMORY_LOCAL_OWNER", "demo"))
    pending_turns: int = 5
    context_turns: int = 15
    window_token_limit: int = field(default_factory=lambda: _positive_int("MEMORY_WINDOW_TOKEN_LIMIT", 6000))
    # Admission limit for a single turn, decoupled from the short-term window:
    # how much text we accept per turn is an input-validation question, while
    # window_token_limit governs how much context the window keeps. LongMemEval-S
    # contains utterances far above the old window-derived ceiling (~4.2k), and
    # rejecting them would fail the whole question rather than degrade one turn.
    max_turn_chars: int = field(default_factory=lambda: _positive_int("MEMORY_MAX_TURN_CHARS", 6000))
    # Ablation switch for the short->long promotion pipeline (see LongTermMemory).
    promotion_enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_PROMOTION_ENABLED", "true").lower() in {"1", "true", "yes"}
    )
    # Long-history benchmarks can produce a session summary larger than a short
    # chat window allows. Default False keeps the strict contract (raise); set
    # MEMORY_STATE_CLAMP=true to truncate instead of failing the whole question.
    state_clamp_enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_STATE_CLAMP", "false").lower() in {"1", "true", "yes"}
    )
    prompt_token_limit: int = field(default_factory=lambda: _positive_int("MEMORY_PROMPT_TOKEN_LIMIT", 10000))
    complex_prompt_token_limit: int = field(
        default_factory=lambda: _positive_int("MEMORY_COMPLEX_PROMPT_TOKEN_LIMIT", 22000)
    )
    retrieval_token_limit: int = 3000
    max_top_k: int = 20
    max_candidates: int = 120
    # Indexed-sweep quota: caps how many ranked hits the indexed sweep may add
    # per question. Measured on the 1000-question run: indexed_sweep items take
    # 64.4% of retrieved characters with a 1.0% hit rate, while the offline
    # replay shows cap=2 keeps R@10 (0.726->0.734) and cuts retrieval chars 49%.
    # Default 32 = the behaviour of the frozen baseline run; A/B on the E4
    # targeted cohort decides whether a smaller cap earns its keep end-to-end.
    indexed_sweep_limit: int = field(
        default_factory=lambda: _non_negative_int("MEMORY_INDEXED_SWEEP_LIMIT", 32)
    )
    answer_normalize_enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ANSWER_NORMALIZE", "true").lower() in {"1", "true", "yes"}
    )
    allow_assistant_evidence: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "true").lower() in {"1", "true", "yes"}
    )
    evidence_grounding_enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_GROUNDING_ENABLED", "true").lower() in {"1", "true", "yes"}
    )
    grounding_semantic_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("MEMORY_GROUNDING_SEMANTIC", "false").lower() in {"1", "true", "yes"}
        )
    )
    activation_ttl_seconds: int = 300
    maintenance_interval: float = field(
        default_factory=lambda: float(os.getenv("MEMORY_MAINTENANCE_INTERVAL", "300"))
    )
    model_timeout: float = field(default_factory=lambda: _positive_float("MEMORY_MODEL_TIMEOUT", 30.0))
    model_max_tokens: int = field(default_factory=lambda: _positive_int("MEMORY_MODEL_MAX_TOKENS", 4096))
    api_key: str = field(default_factory=lambda: os.getenv("MEMORY_API_KEY", ""), repr=False)
    deployment_mode: str = field(default_factory=lambda: os.getenv("MEMORY_DEPLOYMENT_MODE", "local"))
    planner_mode: str = field(default_factory=lambda: os.getenv("MEMORY_PLANNER_MODE", "model"))
    output_language: str = field(
        default_factory=lambda: os.getenv("MEMORY_OUTPUT_LANGUAGE", "en").strip().lower()
    )
    acceleration_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("MEMORY_ACCELERATION_ENABLED", "true").lower() in {"1", "true", "yes"}
        )
    )
    inference_cache_ttl_seconds: float = field(
        default_factory=lambda: _positive_float("MEMORY_INFERENCE_CACHE_TTL", 60)
    )
    inference_cache_max_entries: int = field(
        default_factory=lambda: _positive_int("MEMORY_INFERENCE_CACHE_ENTRIES", 128)
    )

    def __post_init__(self):
        if not math.isfinite(self.maintenance_interval) or self.maintenance_interval < 0:
            raise ValueError("maintenance_interval must be finite and non-negative")
        for name in (
            "pending_turns",
            "window_token_limit",
            "max_turn_chars",
            "prompt_token_limit",
            "complex_prompt_token_limit",
            "retrieval_token_limit",
            "max_top_k",
            "max_candidates",
            "activation_ttl_seconds",
            "model_max_tokens",
            "inference_cache_max_entries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.context_turns, bool)
            or not isinstance(self.context_turns, int)
            or self.context_turns < 0
        ):
            raise ValueError("context_turns must be a non-negative integer")
        if not math.isfinite(self.model_timeout) or self.model_timeout <= 0:
            raise ValueError("model_timeout must be finite and positive")
        if not math.isfinite(self.inference_cache_ttl_seconds) or self.inference_cache_ttl_seconds <= 0:
            raise ValueError("inference_cache_ttl_seconds must be finite and positive")
        if self.deployment_mode not in {"local", "production"}:
            raise ValueError("MEMORY_DEPLOYMENT_MODE must be local or production")
        if self.planner_mode not in {"model", "adaptive", "rule"}:
            raise ValueError("MEMORY_PLANNER_MODE must be model, adaptive or rule")
        if self.output_language not in {"en", "zh"}:
            raise ValueError("MEMORY_OUTPUT_LANGUAGE must be en or zh")
        if self.deployment_mode == "production" and len(self.api_key) < 32:
            raise ValueError("Production HTTP deployment requires MEMORY_API_KEY of at least 32 characters")
        for name in ("local_tenant", "local_owner"):
            value = getattr(self, name)
            if not value.strip() or len(value) > 100:
                raise ValueError(f"{name} must contain 1–100 characters")
