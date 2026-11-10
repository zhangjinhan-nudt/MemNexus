"""Offline, gated final analysis for the frozen LoCoMo-1000 paired run.

The script never runs a benchmark or a judge.  Official scoring comes from
``paper_protocol``; mechanical error features come from ``audit_errors_20261022``;
all provider-ledger USD accounting comes from ``token_breakdown``.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_errors_20261022 as error_audit  # noqa: E402
import paper_protocol as pp  # noqa: E402
import token_breakdown as tb  # noqa: E402

SYSTEMS = ("MemNexus", "SimpleMem")
CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
EXPECTED_CATEGORIES = {
    "temporal": 163,
    "multi-hop": 141,
    "open-domain": 46,
    "single-hop": 425,
    "adversarial": 225,
}
PAPER_NAMES = {1: "MultiHop", 2: "Temporal", 3: "OpenDomain", 4: "SingleHop"}
PLAN_KEYS = (
    "category", "route", "query_type", "recall_policy", "depth", "subject",
    "predicate", "temporal_mode", "response_intent",
)
GROUNDING_KEYS = (
    "supplemental_rounds", "supplemental_queries", "raw_sources_selected",
    "initial_missing_info", "final_missing_info", "temporal_derivations",
    "attribution_conflicts", "coverage_measurement",
)
TOLERANCE = 1e-12
DEFAULT_SEED = 20261025


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def item_key(row: dict) -> tuple[Any, Any]:
    return row.get("sample_id"), row.get("qa_index")


def full_key(row: dict) -> tuple[Any, Any, Any]:
    return row.get("sample_id"), row.get("qa_index"), row.get("system")


def key_text(key: Iterable[Any]) -> str:
    return "/".join(str(value) for value in key)


def sort_key(row: dict) -> tuple[str, int, str]:
    try:
        qa_index = int(row.get("qa_index"))
    except (TypeError, ValueError):
        qa_index = -1
    return str(row.get("sample_id")), qa_index, str(row.get("system"))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def valid_answer(row: dict) -> bool:
    prediction = str(row.get("prediction") or "")
    return bool(
        isinstance(row, dict)
        and row.get("system") in SYSTEMS
        and prediction.strip()
        and not row.get("error")
        and "failed to generate answer" not in prediction.casefold()
    )


def load_stage_records(run_dir: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    unreadable: list[dict] = []
    attempts = run_dir / "attempts"
    paths = sorted(attempts.glob("*/stage_result.json")) if attempts.exists() else []
    for path in paths:
        try:
            payload = read_json(path)
            if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
                raise ValueError("stage_result.json must be an object with a rows list")
            if not isinstance(payload.get("stage"), dict):
                raise ValueError("stage_result.json has no stage object")
            records.append({"path": path, "unit": path.parent.name, "payload": payload})
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            unreadable.append({
                "path": str(path), "error": f"{type(error).__name__}: {error}",
            })
    return records, unreadable


def count_raw_turns(sample: dict, max_session: int) -> int:
    """Mirror the runner's pure raw_turns coverage rule without importing runners."""
    conversation = sample.get("conversation") or {}
    count = 0
    for key, turns in conversation.items():
        if not key.startswith("session_") or not isinstance(turns, list):
            continue
        try:
            session = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue
        if session > max_session:
            continue
        for turn in turns:
            text = str(turn.get("text") or "")
            if turn.get("blip_caption"):
                text = f"[Image: {turn['blip_caption']}] " + text
            count += int(bool(text.strip()))
    return count


def cohort_raw_turns(cohort: dict, cohort_path: Path) -> dict[str, int]:
    direct = cohort.get("raw_turns")
    if isinstance(direct, dict):
        return {str(sample): int(value) for sample, value in direct.items()}
    per_sample = cohort.get("per_sample")
    if isinstance(per_sample, dict) and all(
        isinstance(value, dict) and "raw_turns" in value for value in per_sample.values()
    ):
        return {str(sample): int(value["raw_turns"]) for sample, value in per_sample.items()}

    dataset_value = cohort.get("dataset")
    if not dataset_value:
        raise ValueError("cohort has neither raw_turns nor dataset")
    dataset_path = Path(dataset_value)
    candidates = [dataset_path] if dataset_path.is_absolute() else [
        ROOT / dataset_path, cohort_path.parent / dataset_path,
    ]
    dataset_path = next((path for path in candidates if path.exists()), candidates[0])
    dataset = read_json(dataset_path)
    by_index = {index: sample for index, sample in enumerate(dataset)}
    by_id = {str(sample.get("sample_id")): sample for sample in dataset}
    result: dict[str, int] = {}
    for question in cohort.get("questions") or []:
        sample_id = str(question.get("sample_id"))
        sample = by_index.get(question.get("sample_index"))
        if sample is None or str(sample.get("sample_id")) != sample_id:
            sample = by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"dataset has no sample {sample_id}")
        result[sample_id] = count_raw_turns(sample, int(cohort["max_session"]))
    return result


def validate_gate(
    cohort: dict,
    stage_records: list[dict],
    unreadable: list[dict],
    *,
    expected_raw_turns: dict[str, int] | None = None,
    raw_turns_error: str | None = None,
    run_summary: dict | None = None,
    summary_error: str | None = None,
    snapshot_dir: Path | None = None,
    require_full: bool = True,
) -> dict:
    """Validate completeness without silently deduplicating or repairing rows."""
    audit: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": now(),
        "gate_passed": True,
        "checks": [],
        "failures": [],
        "warnings": [],
        "expected_categories": EXPECTED_CATEGORIES,
        "snapshot_checks": [],
    }
    failures: list[dict] = audit["failures"]

    def check(name: str, passed: bool, code: str, message: str, details: Any = None) -> None:
        audit["checks"].append({"name": name, "passed": bool(passed), "details": details})
        if not passed:
            failures.append({"code": code, "message": message, "details": details})

    questions = cohort.get("questions") if isinstance(cohort.get("questions"), list) else []
    expected_keys = {(q.get("sample_id"), q.get("qa_index")) for q in questions}
    cohort_key_counts = Counter(item_key(q) for q in questions)
    cohort_duplicates = [key_text(key) for key, count in cohort_key_counts.items() if count > 1]
    check("cohort_unique", not cohort_duplicates, "duplicate_cohort_key",
          "cohort 内 (sample_id, qa_index) 不唯一", cohort_duplicates)
    if require_full:
        check("cohort_1000", len(questions) == 1000 and len(expected_keys) == 1000,
              "cohort_not_1000", "cohort 必须恰好包含 1000 个唯一题键", len(questions))
        check("cohort_categories", cohort.get("categories") == EXPECTED_CATEGORIES,
              "cohort_category_quota", "cohort 类别配额不符合冻结的 1000 题协议",
              cohort.get("categories"))

    rows = [row for record in stage_records for row in record["payload"]["rows"]]
    malformed_rows = [index for index, row in enumerate(rows) if not isinstance(row, dict)]
    rows = [row for row in rows if isinstance(row, dict)]
    check("row_objects", not malformed_rows, "malformed_row", "rows 中存在非对象记录", malformed_rows)
    key_counts = Counter(full_key(row) for row in rows)
    duplicates = [key_text(key) for key, count in key_counts.items() if count > 1]
    check("unique_row_keys", not duplicates, "duplicate_row_key",
          "attempts 中 (sample_id, qa_index, system) 必须唯一", duplicates)

    by_system = {system: [row for row in rows if row.get("system") == system] for system in SYSTEMS}
    for system, system_rows in by_system.items():
        observed = {item_key(row) for row in system_rows}
        missing = sorted(expected_keys - observed, key=lambda key: (str(key[0]), str(key[1])))
        extra = sorted(observed - expected_keys, key=lambda key: (str(key[0]), str(key[1])))
        check(f"key_alignment_{system}", not missing and not extra, "key_misalignment",
              f"{system} 与 cohort 题键未完全对齐",
              {"missing": [key_text(key) for key in missing],
               "extra": [key_text(key) for key in extra]})
        invalid = [key_text(full_key(row)) for row in system_rows if not valid_answer(row)]
        check(f"valid_answers_{system}", not invalid, "invalid_answer",
              f"{system} 存在空答案、error 或 Failed to generate answer", invalid)
        actual_categories = dict(Counter(row.get("category") for row in system_rows))
        expected_categories = dict(Counter(q.get("category") for q in questions))
        check(f"categories_{system}", actual_categories == expected_categories,
              "category_count_mismatch", f"{system} 分类数量与 cohort 不一致", actual_categories)
        if require_full:
            check(f"count_1000_{system}", len(system_rows) == 1000,
                  "system_not_1000", f"{system} 必须有 1000 行", len(system_rows))

    expected_samples = {str(q.get("sample_id")) for q in questions}
    if require_full:
        check("ten_conversations", len(expected_samples) == 10 and cohort.get("conversation_count") == 10,
              "conversation_count", "cohort 必须覆盖 10 个对话", sorted(expected_samples))

    expected_units = {(str(sample), system) for sample in expected_samples for system in SYSTEMS}
    observed_units = [(str(record["payload"]["stage"].get("sample_id")),
                       record["payload"]["stage"].get("system")) for record in stage_records]
    observed_unit_set = set(observed_units)
    unit_counts = Counter(observed_units)
    duplicate_units = [key_text(unit) for unit, count in unit_counts.items() if count > 1]
    check("unique_units", not duplicate_units, "duplicate_unit", "同一对话/系统有多个完成单元", duplicate_units)
    check("expected_units", observed_unit_set == expected_units,
          "unit_misalignment", "完成单元未与 cohort 的 20 个对话×系统单元对齐",
          {"missing": [key_text(unit) for unit in sorted(expected_units - observed_unit_set)],
           "extra": [key_text(unit) for unit in sorted(observed_unit_set - expected_units)]})
    if require_full:
        check("twenty_units", len(stage_records) == 20, "unit_count",
              "必须有 20 个可读完成单元", len(stage_records))
    check("stage_json_readable", not unreadable, "unreadable_stage",
          "存在不可读或结构错误的 stage_result.json", unreadable)

    metadata_mismatches = []
    cohort_by_key = {item_key(question): question for question in questions}
    for row in rows:
        expected = cohort_by_key.get(item_key(row))
        if expected and (row.get("category_id") != expected.get("category_id")
                         or row.get("question") != expected.get("question")):
            metadata_mismatches.append(key_text(full_key(row)))
    check("row_cohort_metadata", not metadata_mismatches, "row_metadata_mismatch",
          "行的问题或类别与 cohort 不一致", metadata_mismatches)

    max_session = cohort.get("max_session")
    explicit_max_sessions = []
    for record in stage_records:
        payload = record["payload"]
        value = payload["stage"].get("max_session", payload.get("max_session"))
        if value is not None:
            explicit_max_sessions.append(value)
    summary_max_session = (run_summary or {}).get("max_session")
    max_session_ok = not summary_error and (
        summary_max_session == max_session
        or (summary_max_session is None and bool(explicit_max_sessions)
            and all(value == max_session for value in explicit_max_sessions))
    )
    check("max_session", max_session_ok, "max_session_mismatch",
          "run/stage 的 max_session 必须与 cohort 一致",
          {"cohort": max_session, "summary": summary_max_session,
           "explicit_stage_values": sorted(set(explicit_max_sessions), key=str),
           "summary_error": summary_error})

    raw_mismatches = []
    if expected_raw_turns is None:
        check("raw_turns_reference", False, "raw_turns_reference",
              "无法从 cohort/官方数据取得期望 raw_turns", raw_turns_error)
    else:
        for record in stage_records:
            stage = record["payload"]["stage"]
            sample = str(stage.get("sample_id"))
            expected = expected_raw_turns.get(sample)
            if expected is None or stage.get("raw_turns") != expected:
                raw_mismatches.append({"unit": record["unit"], "actual": stage.get("raw_turns"),
                                       "expected": expected})
        check("raw_turns", not raw_mismatches, "raw_turns_mismatch",
              "阶段 raw_turns 与 cohort 数据范围不一致", raw_mismatches)

    for record in stage_records:
        stage = record["payload"]["stage"]
        sample = str(stage.get("sample_id"))
        for row in record["payload"]["rows"]:
            if isinstance(row, dict) and (str(row.get("sample_id")) != sample
                                         or row.get("system") != stage.get("system")):
                failures.append({"code": "stage_row_mismatch",
                                 "message": f"{record['unit']} 的 stage 与 row 身份不一致", "details": None})
                break

    snapshot_dir = snapshot_dir or ROOT / "result" / "memnexus_snapshots" / "full30"
    for record in stage_records:
        stage = record["payload"]["stage"]
        if stage.get("system") != "MemNexus":
            continue
        sample = str(stage.get("sample_id"))
        result = {"unit": record["unit"], "sample_id": sample, "passed": False}
        manifest_path = snapshot_dir / f"{sample}.json"
        try:
            manifest = read_json(manifest_path)
            actual_sha = stage.get("memory_snapshot_sha256")
            manifest_sha = manifest.get("sha256")
            expected_turns = (expected_raw_turns or {}).get(sample)
            checks = {
                "stage_sha_present": bool(actual_sha),
                "sha_matches_manifest": bool(actual_sha) and actual_sha == manifest_sha,
                "manifest_sample": str(manifest.get("sample_id")) == sample,
                "manifest_raw_turns": manifest.get("raw_turns") == stage.get("raw_turns"),
                "cohort_raw_turns": expected_turns is not None and manifest.get("raw_turns") == expected_turns,
                "manifest_max_session": manifest.get("max_session", max_session) == max_session,
            }
            result.update({"manifest": str(manifest_path), "stage_sha256": actual_sha,
                           "manifest_sha256": manifest_sha, "checks": checks,
                           "passed": all(checks.values())})
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        audit["snapshot_checks"].append(result)
    check("memnexus_snapshots", all(item["passed"] for item in audit["snapshot_checks"]),
          "snapshot_mismatch", "MemNexus 快照 sha/raw_turns/max_session 与 full30 manifest 不一致",
          [item for item in audit["snapshot_checks"] if not item["passed"]])

    audit["counts"] = {
        "cohort_questions": len(questions),
        "conversations": len(expected_samples),
        "readable_units": len(stage_records),
        "unreadable_units": len(unreadable),
        "rows": len(rows),
        "rows_by_system": {system: len(values) for system, values in by_system.items()},
        "unique_question_keys_by_system": {
            system: len({item_key(row) for row in values}) for system, values in by_system.items()
        },
    }
    audit["gate_passed"] = not failures
    audit["status"] = "PASS" if audit["gate_passed"] else "FAIL"
    return audit


def inspect_run(run_dir: Path, cohort_path: Path, snapshot_dir: Path | None = None) -> tuple[dict, dict]:
    cohort = read_json(cohort_path)
    records, unreadable = load_stage_records(run_dir)
    try:
        expected_raw_turns = cohort_raw_turns(cohort, cohort_path)
        raw_turns_error = None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        expected_raw_turns = None
        raw_turns_error = f"{type(error).__name__}: {error}"
    summary_path = run_dir / "summary.json"
    summary = None
    summary_error = None
    if summary_path.exists():
        try:
            summary = read_json(summary_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            summary_error = f"{type(error).__name__}: {error}"
    else:
        summary_error = "summary.json missing"
    audit = validate_gate(
        cohort, records, unreadable, expected_raw_turns=expected_raw_turns,
        raw_turns_error=raw_turns_error, run_summary=summary,
        summary_error=summary_error, snapshot_dir=snapshot_dir,
    )
    return audit, {
        "cohort": cohort, "stage_records": records, "unreadable": unreadable,
        "rows": [row for record in records for row in record["payload"]["rows"]
                 if isinstance(row, dict)],
        "expected_raw_turns": expected_raw_turns, "summary": summary,
        "snapshot_dir": snapshot_dir or ROOT / "result" / "memnexus_snapshots" / "full30",
    }


def paired_rows(rows: list[dict], expected_keys: set[tuple[Any, Any]] | None = None) -> list[tuple[dict, dict]]:
    """Return only unambiguous, valid pairs; never choose one duplicate by score."""
    candidates = [row for row in rows if valid_answer(row)
                  and (expected_keys is None or item_key(row) in expected_keys)]
    # Reuse the protocol's common-subset definition before enforcing uniqueness.
    common = pp.common_subset(candidates, systems=SYSTEMS)
    counts = Counter(full_key(row) for row in rows
                     if isinstance(row, dict) and row.get("system") in SYSTEMS
                     and (expected_keys is None or item_key(row) in expected_keys))
    grouped: dict[tuple[Any, Any], dict[str, dict]] = defaultdict(dict)
    for row in common:
        if counts[full_key(row)] == 1:
            grouped[item_key(row)][row["system"]] = row
    pairs = [(values["MemNexus"], values["SimpleMem"]) for values in grouped.values()
             if all(system in values for system in SYSTEMS)]
    pairs.sort(key=lambda pair: sort_key(pair[0]))
    return pairs


def align_judges(rows: list[dict], judged_rows: list[dict] | None) -> tuple[dict, dict]:
    current = {full_key(row): row for row in rows}
    candidates: dict[tuple[Any, Any, Any], list[dict]] = defaultdict(list)
    stats = Counter()
    for judged in judged_rows or []:
        if not isinstance(judged, dict):
            stats["malformed"] += 1
            continue
        key = full_key(judged)
        row = current.get(key)
        if row is None or judged.get("prediction") != row.get("prediction"):
            stats["stale_or_unmatched"] += 1
            continue
        candidates[key].append(judged)
    aligned = {}
    for key, values in candidates.items():
        if len(values) == 1:
            aligned[key] = values[0]
            stats["matched"] += 1
        else:
            stats["ambiguous"] += len(values)
    stats["missing"] = len(current) - len(aligned)
    return aligned, dict(stats)


def judge_metrics(items: list[dict]) -> dict:
    result = {"auxiliary_only": True, "systems": {}}
    for system in SYSTEMS:
        prefix = "mn" if system == "MemNexus" else "sm"
        labels = [item.get(f"{prefix}_judge_result") for item in items]
        valid = [label for label in labels if label in ("CORRECT", "WRONG")]
        errors = [label for label in labels if isinstance(label, str)
                  and label not in ("CORRECT", "WRONG")]
        result["systems"][system] = {
            "correct": sum(label == "CORRECT" for label in valid),
            "wrong": sum(label == "WRONG" for label in valid),
            "denominator": len(valid),
            "accuracy": (sum(label == "CORRECT" for label in valid) / len(valid)) if valid else None,
            "error_or_invalid_excluded": len(errors),
            "unjudged_excluded": sum(label is None for label in labels),
        }
    return result


def build_paired_items(pairs: list[tuple[dict, dict]], judged_rows: list[dict] | None = None) -> tuple[list[dict], dict]:
    flat = [row for pair in pairs for row in pair]
    judges, judge_alignment = align_judges(flat, judged_rows)
    items = []
    for mn, sm in pairs:
        mn_f1, sm_f1 = pp.f1_of(mn), pp.f1_of(sm)
        mn_bleu, sm_bleu = pp.bleu1_of(mn), pp.bleu1_of(sm)
        delta = mn_f1 - sm_f1
        outcome = "win" if delta > TOLERANCE else "loss" if delta < -TOLERANCE else "tie"
        coverage = error_audit.gold_coverage(mn)
        hit = error_audit.pred_hit_rate(mn)
        noise = error_audit.pred_noise(mn)
        reasoning = mn.get("reasoning") or {}
        grounding = mn.get("grounding") or {}
        mn_calls, sm_calls = mn.get("model_calls") or {}, sm.get("model_calls") or {}
        mn_judge = judges.get(full_key(mn), {})
        sm_judge = judges.get(full_key(sm), {})
        item = {
            "key": key_text(item_key(mn)), "sample_id": mn.get("sample_id"),
            "qa_index": mn.get("qa_index"), "category_id": mn.get("category_id"),
            "category": mn.get("category"), "question": mn.get("question"),
            "gold_answer": mn.get("gold_answer"),
            "mn_prediction": mn.get("prediction"), "sm_prediction": sm.get("prediction"),
            "mn_f1": mn_f1, "sm_f1": sm_f1, "f1_delta": delta,
            "mn_bleu1": mn_bleu, "sm_bleu1": sm_bleu, "bleu1_delta": mn_bleu - sm_bleu,
            "outcome": outcome,
            "mn_input_tokens": mn_calls.get("input_tokens"),
            "mn_output_tokens": mn_calls.get("output_tokens"),
            "mn_latency_ms": mn.get("latency_ms"),
            "sm_input_tokens": sm_calls.get("input_tokens"),
            "sm_output_tokens": sm_calls.get("output_tokens"),
            "sm_latency_ms": sm.get("latency_ms"),
            "mn_retrieval_count": mn.get("retrieval_count"),
            "sm_retrieval_count": sm.get("retrieval_count"),
            "mn_gold_coverage": coverage, "mn_pred_hit_rate": hit, "mn_pred_noise": noise,
            "mn_evidence_chars": len(error_audit.evidence_text(mn)),
            "mn_answer_path": reasoning.get("answer_path", mn.get("answer_path")),
            "mn_plan": {key: (mn.get("plan") or {}).get(key) for key in PLAN_KEYS
                        if (mn.get("plan") or {}).get(key) is not None},
            "mn_grounding_supplemental_rounds": grounding.get("supplemental_rounds"),
            "mn_grounding": {key: grounding.get(key) for key in GROUNDING_KEYS if key in grounding},
            "mn_warnings": mn.get("warnings") or [], "sm_warnings": sm.get("warnings") or [],
            "mn_classify": error_audit.classify(mn, coverage),
            "classify_status": "heuristic_requires_manual_review_not_causal_truth",
            "mn_judge_result": mn_judge.get("judge_result"),
            "mn_judge_reason": mn_judge.get("judge_reason"),
            "sm_judge_result": sm_judge.get("judge_result"),
            "sm_judge_reason": sm_judge.get("judge_reason"),
        }
        items.append(item)
    return items, judge_alignment


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def interval(values: list[float]) -> list[float] | None:
    return [percentile(values, 0.025), percentile(values, 0.975)] if values else None


def observed_deltas(items: list[dict]) -> dict[str, float | None]:
    category_values = {}
    for cid, name in CATEGORIES.items():
        category_values[name] = mean(item["f1_delta"] for item in items if item["category_id"] == cid)
    return {
        "Average": mean(value for cid, value in ((cid, category_values[CATEGORIES[cid]]) for cid in range(1, 5))
                        if value is not None),
        **{f"category:{name}": category_values[name] for name in CATEGORIES.values()},
        "AdvSR": category_values["adversarial"],
    }


def cluster_bootstrap(items: list[dict], replicates: int = 5000, seed: int = DEFAULT_SEED) -> dict:
    """Conversation-cluster paired bootstrap; only four-category draws enter CIs."""
    rng = random.Random(seed)
    conversations = sorted({str(item["sample_id"]) for item in items})
    stats: dict[str, dict[int, list[float]]] = {}
    for conversation in conversations:
        stats[conversation] = {}
        for cid in CATEGORIES:
            values = [item["f1_delta"] for item in items
                      if str(item["sample_id"]) == conversation and item["category_id"] == cid]
            stats[conversation][cid] = [sum(values), len(values)]
    draws: dict[str, list[float]] = {
        "Average": [], **{f"category:{CATEGORIES[cid]}": [] for cid in range(1, 5)}, "AdvSR": [],
    }
    valid_four_category = 0
    for _ in range(replicates):
        selected = [rng.choice(conversations) for _ in conversations] if conversations else []
        values = {}
        for cid in CATEGORIES:
            count = sum(stats[conversation][cid][1] for conversation in selected)
            if count:
                values[cid] = sum(stats[conversation][cid][0] for conversation in selected) / count
        if not all(cid in values for cid in range(1, 5)):
            continue
        valid_four_category += 1
        draws["Average"].append(statistics.mean(values[cid] for cid in range(1, 5)))
        for cid in range(1, 5):
            draws[f"category:{CATEGORIES[cid]}"].append(values[cid])
        if 5 in values:
            draws["AdvSR"].append(values[5])
    return {
        "method": "paired conversation-cluster bootstrap with replacement; 95% percentile CI",
        "primary": True, "seed": seed, "requested_replicates": replicates,
        "n_clusters": len(conversations), "valid_four_category_replicates": valid_four_category,
        "observed_delta": observed_deltas(items),
        "ci95": {name: interval(values) for name, values in draws.items()},
        "valid_replicates_by_metric": {name: len(values) for name, values in draws.items()},
        "power_warning": "只有少量对话簇（完整队列为 10），CI 与检验功效有限，不能按 1000 个独立题解释。",
        "item_bootstrap_is_not_primary": True,
    }


def grouped_summaries(items: list[dict], field: str) -> dict:
    values = sorted({str(item[field]) for item in items})
    result = {}
    for value in values:
        group = [item for item in items if str(item[field]) == value]
        outcomes = Counter(item["outcome"] for item in group)
        result[value] = {
            "n": len(group), "mn_f1": mean(item["mn_f1"] for item in group),
            "sm_f1": mean(item["sm_f1"] for item in group),
            "f1_delta": mean(item["f1_delta"] for item in group),
            "mn_bleu1": mean(item["mn_bleu1"] for item in group),
            "sm_bleu1": mean(item["sm_bleu1"] for item in group),
            "win": outcomes["win"], "tie": outcomes["tie"], "loss": outcomes["loss"],
        }
    return result


def cause_counts(items: list[dict]) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        if item["outcome"] == "loss":
            counts[item["mn_classify"]][item["category"]] += 1
    return {cause: dict(counts_by_category) for cause, counts_by_category in sorted(counts.items())}


def connector_diagnostic(pairs: list[tuple[dict, dict]]) -> dict:
    original = [row for pair in pairs for row in pair]
    alternate = []
    for row in original:
        clone = dict(row)
        clone["prediction"] = str(clone.get("prediction") or "").replace(", ", "; ")
        alternate.append(clone)
    before = pp.paper_table(original, systems=SYSTEMS)
    after = pp.paper_table(alternate, systems=SYSTEMS)
    result = {
        "diagnostic_only": True,
        "symmetric_transform": "both systems: replace every ', ' with '; ' in prediction",
        "not_main_result_not_mechanism_contribution": True,
        "systems": {},
    }
    for system in SYSTEMS:
        before_rows = [row for row in original if row["system"] == system]
        changed = sum(
            abs(pp.f1_of(row) - pp.f1_of({**row, "prediction": str(row.get("prediction") or "").replace(", ", "; ")}))
            > TOLERANCE for row in before_rows
        )
        result["systems"][system] = {
            "changed_item_scores": changed,
            "before": before["systems"].get(system),
            "semicolon": after["systems"].get(system),
        }
    return result


def compute_metrics(items: list[dict], pairs: list[tuple[dict, dict]], bootstrap: int,
                    judged_alignment: dict, partial: bool) -> dict:
    flat = [row for pair in pairs for row in pair]
    table = pp.paper_table(flat, systems=SYSTEMS)
    pooled = {}
    for system in SYSTEMS:
        rows = [row for row in flat if row["system"] == system and row.get("category_id") != 5]
        pooled[system] = {
            "pooled_nonadversarial_f1": mean(pp.f1_of(row) for row in rows),
            "n": len(rows), "is_average": False,
        }
    outcomes = Counter(item["outcome"] for item in items)
    return {
        "schema_version": 1, "generated_at": now(),
        "status": "PARTIAL" if partial else "FINAL", "partial": partial,
        "paired_questions": len(items), "paper_table": table,
        "pooled_f1_diagnostic": pooled,
        "pooled_warning": "pooled F1 按非对抗逐题池化，不是四类等权 Average，也不是主 Average。",
        "outcomes": {"win": outcomes["win"], "tie": outcomes["tie"], "loss": outcomes["loss"],
                     "tolerance": TOLERANCE},
        "by_category": grouped_summaries(items, "category"),
        "by_conversation": grouped_summaries(items, "sample_id"),
        "cluster_bootstrap": cluster_bootstrap(items, bootstrap),
        "judge": judge_metrics(items), "judge_alignment": judged_alignment,
        "connector_diagnostic": connector_diagnostic(pairs),
        "loss_cause_by_category": cause_counts(items),
        "classify_warning": "classify 是词面/证据启发式，仅供人工复核，不是因果真值。",
    }


def sum_usage(records: list[dict]) -> dict:
    fields = ("provider_attempts", "provider_errors", "input_tokens", "output_tokens", "cost_rmb")
    return {field: sum(float((record or {}).get(field) or 0) for record in records) for field in fields}


def judge_ledger_audit(run_dir: Path) -> dict:
    attempts = run_dir / "attempts"
    paths = sorted(attempts.glob("judge_*/usage.calls.jsonl")) if attempts.exists() else []
    paths += sorted((run_dir / "judgments").glob("**/*.usage.calls.jsonl")) if (run_dir / "judgments").exists() else []
    if not paths:
        return {"status": "unavailable", "cost_usd": None,
                "reason": "没有独立 Judge usage.calls.jsonl；不从系统账本拆分或猜测 Judge 成本。"}
    scans = [tb.scan(path) for path in paths]
    totals = {field: sum(float(scan["totals"][field]) for scan in scans) for field in tb.STAT_FIELDS}
    totals["cost_usd"] = round(totals["cost_usd"], 12)
    for field in tb.STAT_FIELDS:
        if field != "cost_usd":
            totals[field] = int(totals[field])
    return {"status": "available", "ledger_count": len(paths), "totals": totals,
            "note": "仅独立 Judge ledger；未并入任何系统 gross/accepted。"}


def build_token_audit(run_dir: Path, cohort: dict, stage_records: list[dict], rows: list[dict],
                      snapshot_dir: Path, partial: bool) -> dict:
    samples = ",".join(sorted({str(question.get("sample_id")) for question in cohort.get("questions") or []}))
    report = tb.build_report(run_dir, samples=samples)
    report["status"] = "PARTIAL" if partial else "FINAL"
    report["partial_run"] = bool(report["partial_run"] or partial)
    report["accepted_qa_by_category"] = {}
    for system in SYSTEMS:
        report["accepted_qa_by_category"][system] = {}
        for cid, category in CATEGORIES.items():
            subset = [row for row in rows if row.get("system") == system
                      and row.get("category_id") == cid and row.get("category") == category]
            if subset:
                report["accepted_qa_by_category"][system][category] = tb._accepted_view(tb._accepted_rows(subset))

    accepted_build: dict[str, Any] = {}
    original_build: dict[str, Any] = {"MemNexus": {"units": []}, "SimpleMem": {"status": "not_applicable"}}
    retry_view: dict[str, Any] = {}
    for system in SYSTEMS:
        records = [record for record in stage_records if record["payload"]["stage"].get("system") == system]
        units = []
        for record in records:
            stage = record["payload"]["stage"]
            usage = stage.get("build_usage") or {}
            units.append({"unit": record["unit"], "sample_id": stage.get("sample_id"),
                          "memory_base_replayed": stage.get("memory_base_replayed"),
                          "build_usage_native_rmb": usage})
            if system == "MemNexus":
                manifest_path = snapshot_dir / f"{stage.get('sample_id')}.json"
                try:
                    manifest = read_json(manifest_path)
                    original_build["MemNexus"]["units"].append({
                        "sample_id": stage.get("sample_id"), "manifest": str(manifest_path),
                        "sha256": manifest.get("sha256"), "build_usage_native_rmb": manifest.get("build_usage") or {},
                    })
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    original_build["MemNexus"]["units"].append({
                        "sample_id": stage.get("sample_id"), "status": "unavailable",
                        "error": f"{type(error).__name__}: {error}",
                    })
        accepted_build[system] = {
            "currency_of_cost_field": "RMB (native stage build_usage; not converted)",
            "total": sum_usage([unit["build_usage_native_rmb"] for unit in units]), "units": units,
        }
        gross = report["per_system"].get(system, {})
        totals = gross.get("totals", {})
        retry_view[system] = {
            "gross_epochs": gross.get("epochs", 0), "gross_errors": totals.get("errors", 0),
            "gross_unsettled": totals.get("unsettled", 0),
            "gross_responses_without_usage": totals.get("responses_without_usage", 0),
            "incomplete_units": gross.get("incomplete_units"),
            "note": "重复建库、废弃重试和失败/未结请求保留在 gross；不与 accepted 相加，也不伪装成可精确归属的 accepted 差额。",
        }
    original_build["MemNexus"]["currency_of_cost_field"] = "RMB (native snapshot manifest; not converted)"
    original_build["MemNexus"]["total"] = sum_usage([
        unit.get("build_usage_native_rmb") for unit in original_build["MemNexus"]["units"]
        if unit.get("build_usage_native_rmb") is not None
    ])
    report["accepted_build"] = accepted_build
    report["snapshot_original_build"] = original_build
    report["rebuilds_and_discarded_retries"] = retry_view
    report["judge_cost"] = judge_ledger_audit(run_dir)
    report["gross_accepted_additive"] = False
    return report


def token_markdown(report: dict) -> str:
    base = tb.markdown_report(report)
    partial = report["status"] == "PARTIAL"
    lines = ["# PARTIAL — 部分 Token 审计（非最终结果）" if partial else "# LoCoMo1000 Token 审计", ""]
    if partial:
        lines += ["> **PARTIAL：GATE 未全部通过，以下仅是部分数据审计。**", ""]
    lines += [base, "", "## Accepted QA 按题型（独立口径，不与 gross 相加）", ""]
    for system, categories in report["accepted_qa_by_category"].items():
        lines += [f"### {system}", "", "| 类别 | 题数 | 输入 | 输出 | USD |", "|---|---:|---:|---:|---:|"]
        for category, values in categories.items():
            lines.append(f"| {category} | {values['questions']} | {values['prompt_tokens']} | "
                         f"{values['completion_tokens']} | {values['cost_usd']:.8f} |")
        lines.append("")
    lines += ["## 最终接受 build 与快照原始 build", "",
              "这些 `build_usage.cost_rmb` 是产物原生 RMB 字段，不做未定义汇率换算；也不与 USD gross/accepted 相加。", ""]
    for system, values in report["accepted_build"].items():
        lines.append(f"- {system} 最终接受 build：`{json.dumps(values['total'], ensure_ascii=False)}`")
    original = report["snapshot_original_build"]["MemNexus"]
    lines.append(f"- MemNexus 快照原始 build（能取得 manifest 的单元）：`{json.dumps(original['total'], ensure_ascii=False)}`")
    lines += ["", "## 重复建库 / 废弃重试（单列）", ""]
    for system, values in report["rebuilds_and_discarded_retries"].items():
        lines.append(f"- {system}: `{json.dumps(values, ensure_ascii=False)}`")
    lines += ["", "## Judge 成本", "", f"`{json.dumps(report['judge_cost'], ensure_ascii=False)}`", ""]
    return "\n".join(lines)


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}"


def signed_pp(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:+.2f}"


def ci_text(value: list[float] | None) -> str:
    return "N/A" if value is None else f"[{100 * value[0]:+.2f}, {100 * value[1]:+.2f}]"


def render_losses(items: list[dict], partial: bool) -> str:
    losses = sorted((item for item in items if item["outcome"] == "loss"),
                    key=lambda item: (item["f1_delta"], item["key"]))
    lines = ["# PARTIAL — MemNexus 全部输题（非最终结果）" if partial else "# MemNexus 全部输题审计", ""]
    if partial:
        lines += ["> **PARTIAL：仅覆盖当前无歧义配对中的输题。**", ""]
    lines += [f"共 **{len(losses)}** 题，按 F1 差（MN−SM）升序列出；不截断、不只取前 N 题。",
              "机械 classify 是启发式待人工复核，不是因果真值。", ""]
    cause_categories = cause_counts(items)
    lines += ["## 错因 × 题型（仅 MN 输题）", "",
              "| 错因 | " + " | ".join(CATEGORIES.values()) + " | 合计 |",
              "|---|" + "---:|" * 6]
    for cause, counts in cause_categories.items():
        lines.append(f"| {cause} | " + " | ".join(str(counts.get(category, 0)) for category in CATEGORIES.values())
                     + f" | {sum(counts.values())} |")
    lines.append("")
    for item in losses:
        lines += [
            f"## {item['key']} · {item['category']} · ΔF1={item['f1_delta']:+.6f}",
            "", f"- 问题：{item['question']}", f"- Gold：{item['gold_answer']}",
            f"- **MN**：{item['mn_prediction']}", f"- **SM**：{item['sm_prediction']}",
            f"- F1 / BLEU1：MN {item['mn_f1']:.6f}/{item['mn_bleu1']:.6f}；SM {item['sm_f1']:.6f}/{item['sm_bleu1']:.6f}",
            f"- 启发式错因（待人工复核）：{item['mn_classify']}",
            f"- 检索/覆盖：retrieved={item['mn_retrieval_count']}，gold_coverage={item['mn_gold_coverage']:.4f}，"
            f"pred_hit={item['mn_pred_hit_rate']:.4f}，pred_noise={item['mn_pred_noise']:.4f}",
            f"- answer_path：`{item['mn_answer_path']}`；plan：`{json.dumps(item['mn_plan'], ensure_ascii=False)}`",
            f"- grounding：`{json.dumps(item['mn_grounding'], ensure_ascii=False)}`",
            f"- warnings：MN `{json.dumps(item['mn_warnings'], ensure_ascii=False)}`；SM `{json.dumps(item['sm_warnings'], ensure_ascii=False)}`",
            f"- Judge（辅助）：MN={item['mn_judge_result']} / {item['mn_judge_reason']}；SM={item['sm_judge_result']} / {item['sm_judge_reason']}",
            "",
        ]
    return "\n".join(lines)


def optimization_section(metrics: dict) -> list[str]:
    counts = Counter()
    for cause, values in metrics["loss_cause_by_category"].items():
        counts[cause] = sum(values.values())
    groups = [
        ("检索覆盖", ("F-", "C-"), "只改检索召回/证据覆盖；固定答案器做 retrieval-only，再以 Gold-evidence 和上线下端到端消融。"),
        ("义务粒度", ("H-", "B-"), "只改问题义务拆解与完备性判定；分别做义务-only、Gold-evidence 和端到端消融。"),
        ("时间/集合执行", ("E-", "G-"), "只改时间或集合绑定/执行算子；固定相同检索，做 execution-only、Gold-evidence 和端到端消融。"),
        ("答案冗余", ("D-",), "只改最终渲染/去冗余；固定证据与绑定结果，做 renderer-only、Gold-evidence 和端到端消融。"),
    ]
    lines = ["## 可验证的优化优先级框架", "",
             "以下数量只是当前 MN 输题的机械启发式映射，不是机制结论，也不沿用任何旧题集结论。", "",
             "### P0：基础设施（不计算法贡献）", "",
             "1. 建库持久恢复与快照完整性校验：故障注入后从同一 qa_index 断点续跑。",
             "2. 以 `(sample_id, qa_index, system)` 为唯一键，重复即拒绝提交；重复文本不得去重。",
             "3. 限流退避与请求/响应账本闭合：注入 429/中断，验证无空答案混入、无重复接受行。",
             "P0 的预注册验收是恢复率、重复键拒绝率、账本闭合率；不得记作 Average 或机制收益。", "",
             "### P1：按本轮实际错因排序的单变量实验", ""]
    ranked = sorted(groups, key=lambda group: -sum(count for cause, count in counts.items()
                                                  if any(cause.startswith(prefix) for prefix in group[1])))
    for index, (name, prefixes, protocol) in enumerate(ranked, 1):
        mapped = {cause: count for cause, count in counts.items()
                  if any(cause.startswith(prefix) for prefix in prefixes)}
        lines += [
            f"#### P1-{index} {name}：当前映射 {sum(mapped.values())} 题", "",
            f"- 动态计数：`{json.dumps(mapped, ensure_ascii=False, sort_keys=True)}`；先人工复核，再决定是否实验。",
            f"- 单变量 A/B 与消融：{protocol}",
            "- 预注册：主指标为对应题型官方 F1 的 MN−SM cluster CI，次指标为四类等权 Average；不得用 pooled 或 Judge 替代。",
            "- 门槛：同 cohort、同预测协议下 AdvSR 不下降，accepted QA tokens/题不增加；另报 gross，禁止用 gross+accepted 制造收益。",
            "- 若检索-only 改善而 Gold-evidence/end-to-end 不改善，只能定位到候选环节，不能宣称端到端机制贡献。", "",
        ]
    return lines


def render_report(audit: dict, metrics: dict, token: dict) -> str:
    partial = metrics["partial"]
    lines = ["# PARTIAL — LoCoMo1000 部分分析（非最终结果）" if partial else "# LoCoMo1000 最终分析报告", ""]
    if partial:
        lines += ["> # **PARTIAL / 非最终结果**", "> **GATE 未全部通过；本报告只用于管线测试或部分数据诊断，禁止作为 LoCoMo1000 最终结论。**", ""]
    lines += ["## 数据口径与 GATE", "",
              f"- 状态：**{metrics['status']}**；无歧义且双侧有效的完全配对：**{metrics['paired_questions']}** 对。",
              f"- cohort：{audit['counts']['cohort_questions']} 题 / {audit['counts']['conversations']} 对话；可读单元 {audit['counts']['readable_units']}。",
              "- 主结果只用 `(sample_id, qa_index)` 完全配对；重复 `(sample_id, qa_index, system)` 不做择优去重。",
              "- Average 是 MultiHop/Temporal/OpenDomain/SingleHop 四类等权平均，**不含 adversarial**；AdvSR 独立报告。",
              "- pooled F1 是非对抗逐题池化诊断，**不是 Average**。Judge 仅为辅助诊断，error/未判/旧预测均不进分母。", ""]
    if audit["failures"]:
        lines += ["### 未通过的 GATE", ""]
        lines += [f"- `{failure['code']}`：{failure['message']}" for failure in audit["failures"]] + [""]
    table = metrics["paper_table"]
    names = [PAPER_NAMES[cid] for cid in range(1, 5)]
    lines += ["## 主表：官方分类 F1 / BLEU1", "",
              "| 系统 | " + " | ".join(f"{name} F1 | {name} BLEU1" for name in names)
              + " | **Average F1** | **Average BLEU1** | AdvSR |",
              "|---|" + "---:|" * 11]
    for system in SYSTEMS:
        values = table["systems"].get(system)
        if not values:
            continue
        cells = []
        for name in names:
            cells += [pct(values["f1_by_type"].get(name)), pct(values["bleu1_by_type"].get(name))]
        cells += [pct(values["average_f1"]), pct(values["average_bleu1"]), pct(values["adversarial_sr"])]
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    lines += ["", "数值以百分数显示；所有逐题分数由 `paper_protocol.f1_of/bleu1_of` 重算。", "",
              "### pooled 诊断（不是主 Average）", "",
              "| 系统 | 非对抗题数 | pooled F1 |", "|---|---:|---:|"]
    for system, values in metrics["pooled_f1_diagnostic"].items():
        lines.append(f"| {system} | {values['n']} | {pct(values['pooled_nonadversarial_f1'])} |")

    bootstrap = metrics["cluster_bootstrap"]
    lines += ["", "## 配对差与 conversation-cluster 95% CI", "",
              f"固定 seed={bootstrap['seed']}，请求 {bootstrap['requested_replicates']} 次，四类齐全重采样 {bootstrap['valid_four_category_replicates']} 次；单位均为百分点（MN−SM）。",
              f"**{bootstrap['power_warning']}**", "",
              "| 指标 | 观测差 | cluster 95% CI | 有效重采样 |", "|---|---:|---:|---:|"]
    for name in ("Average", "category:multi-hop", "category:temporal", "category:open-domain",
                 "category:single-hop", "AdvSR"):
        lines.append(f"| {name} | {signed_pp(bootstrap['observed_delta'][name])} | "
                     f"{ci_text(bootstrap['ci95'][name])} | {bootstrap['valid_replicates_by_metric'][name]} |")
    outcomes = metrics["outcomes"]
    lines += ["", f"逐题数值容差 1e-12：MN 胜 **{outcomes['win']}** / 平 **{outcomes['tie']}** / 负 **{outcomes['loss']}**。",
              "主 CI 按 conversation 整簇有放回抽样；未将逐题 bootstrap 当作主 CI。", "",
              "### 按题型与对话", "", "| 分组 | n | MN F1 | SM F1 | ΔF1(pp) | 胜/平/负 |", "|---|---:|---:|---:|---:|---:|"]
    for title, groups in (("题型", metrics["by_category"]), ("对话", metrics["by_conversation"])):
        for name, values in groups.items():
            lines.append(f"| {title}:{name} | {values['n']} | {pct(values['mn_f1'])} | {pct(values['sm_f1'])} | "
                         f"{signed_pp(values['f1_delta'])} | {values['win']}/{values['tie']}/{values['loss']} |")

    lines += ["", "## Token 与成本（gross / accepted 永不相加）", "",
              "- gross：全部已追加 ledger 请求，含重试、重建 epoch、失败和无 usage 请求；USD 单价与缓存规则直接来自 `token_breakdown.PRICING`。",
              "- accepted：独立读取 `stage_result.json` 的行级 `model_calls`；不是 gross 的子集加总证明，也不与 gross 相加。",
              "- 无 usage、失败、未结请求不猜 token 或费用。build 原生 RMB 与 ledger USD 不擅自换算。", "",
              "| 系统 | gross tokens | gross USD | accepted QA题数 | accepted tokens | accepted USD |",
              "|---|---:|---:|---:|---:|---:|"]
    for system in SYSTEMS:
        values = token["per_system"].get(system)
        if not values:
            continue
        gross, accepted = values["totals"], values["accepted"]
        lines.append(f"| {system} | {gross['total_tokens']} | {gross['cost_usd']:.8f} | {accepted['questions']} | "
                     f"{accepted['total_tokens']} | {accepted['cost_usd']:.8f} |")
    for system, values in token["accepted_build"].items():
        lines.append(f"- {system} 最终接受 build（原生字段）：`{json.dumps(values['total'], ensure_ascii=False)}`")
    lines.append(f"- MemNexus 快照原始 build：`{json.dumps(token['snapshot_original_build']['MemNexus']['total'], ensure_ascii=False)}`")
    lines.append(f"- Judge 成本：`{json.dumps(token['judge_cost'], ensure_ascii=False)}`")
    lines.append("重复建库/废弃重试的 epochs、errors、unsettled、无 usage 响应在 `token_audit.md` 单列。")

    lines += ["", "## 辅助 Judge", "", "| 系统 | CORRECT | WRONG | 有效分母 | accuracy | error/无效排除 | 未判排除 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for system, values in metrics["judge"]["systems"].items():
        lines.append(f"| {system} | {values['correct']} | {values['wrong']} | {values['denominator']} | "
                     f"{pct(values['accuracy'])} | {values['error_or_invalid_excluded']} | {values['unjudged_excluded']} |")

    lines += ["", "## 机械错因 × 题型（只统计 MN 输题）", "",
              "**classify 是启发式待人工复核，不是因果真值。** 所有输题及双方完整答案见 `mn_losses.md` 与 `paired_items.*`。", "",
              "| 错因 | " + " | ".join(CATEGORIES.values()) + " | 合计 |", "|---|" + "---:|" * 6]
    for cause, counts in metrics["loss_cause_by_category"].items():
        lines.append(f"| {cause} | " + " | ".join(str(counts.get(category, 0)) for category in CATEGORIES.values())
                     + f" | {sum(counts.values())} |")

    lines += ["", "## 对称连接符敏感性诊断（非主结果）", "",
              "两系统都执行同一 `', ' → '; '` 变换后重算；只诊断官方多跳切分敏感性，**不得当作主结果或机制贡献**。", "",
              "| 系统 | 原 Average F1 | 分号 Average F1 | Δ(pp) | 分数变化题数 |", "|---|---:|---:|---:|---:|"]
    for system, values in metrics["connector_diagnostic"]["systems"].items():
        before, after = values["before"], values["semicolon"]
        old = before.get("average_f1") if before else None
        new = after.get("average_f1") if after else None
        delta = new - old if old is not None and new is not None else None
        lines.append(f"| {system} | {pct(old)} | {pct(new)} | {signed_pp(delta)} | {values['changed_item_scores']} |")

    lines += [""] + optimization_section(metrics)
    lines += ["## 局限性", "",
              "- 完整队列只有 10 个 conversation 簇；1000 题不是 1000 个独立簇，功效与外部可推广性有限。",
              "- 部分模式下缺题、重复键、无效答案和快照问题仍存在，任何数值都只是 PARTIAL 诊断。",
              "- 官方 F1/BLEU1 是词面指标；Judge 缺失不妨碍官方分析，但有效 Judge 也只作辅助。",
              "- gold_coverage/pred_hit/pred_noise/classify 是词面启发式，需人工复核，不能证明检索或生成因果机制。",
              "- 连接符结果是对称敏感性诊断，不是系统机制差异。",
              "- 成本只覆盖可观测 usage；未知 usage 与无独立 ledger 的 Judge 成本保持不可用，不做估算。", ""]
    return "\n".join(lines)


def figure_svg(figure: Any) -> str:
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)
    text = buffer.getvalue()
    return text[text.index("<svg"):]


def render_dashboard(audit: dict, metrics: dict, token: dict, items: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "svg.fonttype": "path", "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "#64748b", "text.color": "#172033",
        "axes.labelcolor": "#172033", "xtick.color": "#475569", "ytick.color": "#475569",
    })
    colors = {"MemNexus": "#2563a6", "SimpleMem": "#d17a22"}
    figures = []

    def finish(fig: Any, title: str) -> None:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        figures.append(figure_svg(fig))

    table = metrics["paper_table"]
    names = [PAPER_NAMES[cid] for cid in range(1, 5)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = list(range(len(names)))
    for index, system in enumerate(SYSTEMS):
        values = table["systems"].get(system, {}).get("f1_by_type", {})
        heights = [100 * (values.get(name) or 0) for name in names]
        ax.bar([value + (index - .5) * .36 for value in x], heights, .36, label=system, color=colors[system])
    ax.set_xticks(x, names); ax.set_ylabel("官方 F1（%）"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    finish(fig, "分类官方 F1")

    boot = metrics["cluster_bootstrap"]
    observed = boot["observed_delta"]["Average"]
    ci = boot["ci95"]["Average"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if observed is not None:
        point = 100 * observed
        errors = [[point - 100 * ci[0]], [100 * ci[1] - point]] if ci else [[0], [0]]
        ax.errorbar([0], [point], yerr=errors, fmt="o", markersize=9, capsize=8, color="#2563a6")
        ax.axhline(0, color="#64748b", linewidth=1)
        ax.set_xlim(-.6, .6); ax.set_xticks([0], ["Average（四类等权，不含 Adv）"])
    else:
        ax.text(.5, .5, "无有效 cluster CI", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("MN−SM（百分点）"); ax.grid(axis="y", alpha=.2)
    finish(fig, "Average 差及 conversation-cluster 95% CI")

    groups = metrics["by_conversation"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    labels = list(groups)
    values = [100 * (groups[name]["f1_delta"] or 0) for name in labels]
    ax.bar(labels, values, color=["#2563a6" if value >= 0 else "#c44e52" for value in values])
    ax.axhline(0, color="#64748b", linewidth=1); ax.set_ylabel("ΔF1（百分点）"); ax.grid(axis="y", alpha=.2)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    finish(fig, "分对话配对 F1 差")

    outcomes = metrics["outcomes"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["MN胜", "平", "MN负"]
    ax.bar(labels, [outcomes["win"], outcomes["tie"], outcomes["loss"]], color=["#2563a6", "#94a3b8", "#c44e52"])
    ax.set_ylabel("题数"); ax.grid(axis="y", alpha=.2)
    finish(fig, "逐题胜 / 平 / 负（容差 1e-12）")

    stages = sorted({stage for system in SYSTEMS for stage, values in token["per_system"].get(system, {}).get("stages", {}).items()
                     if values["total_tokens"]})
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = list(range(len(stages)))
    for index, system in enumerate(SYSTEMS):
        values = token["per_system"].get(system, {}).get("stages", {})
        heights = [values.get(stage, {}).get("total_tokens", 0) for stage in stages]
        ax.bar([value + (index - .5) * .36 for value in x], heights, .36, label=system, color=colors[system])
    ax.set_xticks(x, stages); ax.set_ylabel("gross tokens"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    finish(fig, "Token 分阶段（gross；不与 accepted 相加）")

    totals = Counter({cause: sum(values.values()) for cause, values in metrics["loss_cause_by_category"].items()})
    ordered = sorted(totals, key=lambda cause: totals[cause])
    fig, ax = plt.subplots(figsize=(10, max(3.8, .55 * len(ordered) + 1.5)))
    if ordered:
        ax.barh(ordered, [totals[cause] for cause in ordered], color="#7c5aa6")
    else:
        ax.text(.5, .5, "当前配对中无 MN 输题", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("MN 输题数"); ax.grid(axis="x", alpha=.2)
    finish(fig, "机械错因分布（启发式，待人工复核）")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for system, prefix in (("MemNexus", "mn"), ("SimpleMem", "sm")):
        lengths = [len(str(item[f"{prefix}_prediction"] or "").split()) for item in items]
        scores = [100 * item[f"{prefix}_f1"] for item in items]
        ax.scatter(lengths, scores, s=14, alpha=.42, label=system, color=colors[system], edgecolors="none")
    ax.set_xlabel("预测长度（空白切分词数）"); ax.set_ylabel("官方 F1（%）"); ax.legend(frameon=False); ax.grid(alpha=.18)
    finish(fig, "预测长度与答案 F1")

    partial = metrics["partial"]
    title = "PARTIAL — LoCoMo1000 部分仪表盘" if partial else "LoCoMo1000 最终仪表盘"
    banner = ("<div class='partial'><strong>PARTIAL / 非最终结果：</strong> GATE 未全部通过；仅用于测试或部分诊断。</div>"
              if partial else "")
    cards = "\n".join(f"<section>{svg}</section>" for svg in figures)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px}}h1{{margin:0 0 18px}}.partial{{background:#fff1f2;border:3px solid #be123c;color:#9f1239;padding:18px;border-radius:10px;font-size:20px;margin-bottom:20px}}
.meta{{background:#fff;border:1px solid #dbe3ee;border-radius:10px;padding:14px;margin-bottom:18px}}section{{background:#fff;border:1px solid #dbe3ee;border-radius:12px;padding:12px;margin:18px 0;box-shadow:0 2px 8px #1720330a}}svg{{width:100%;height:auto;display:block}}footer{{color:#64748b;padding:12px 0}}
</style></head><body><main><h1>{html.escape(title)}</h1>{banner}
<div class="meta">状态：<strong>{metrics['status']}</strong>；完全配对 {metrics['paired_questions']} 题；{boot['n_clusters']} 个对话簇。主 CI 为 conversation-cluster bootstrap；错因为启发式，连接符诊断不属于此仪表盘主结论。</div>
{cards}<footer>自包含离线 HTML；SVG 内嵌，无外部 JavaScript、字体或网络依赖。详细口径见 REPORT.md。</footer>
</main></body></html>"""


def write_paired_files(items: list[dict], out_dir: Path) -> None:
    with (out_dir / "paired_items.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    fields = [
        "key", "sample_id", "qa_index", "category_id", "category", "question", "gold_answer",
        "mn_prediction", "sm_prediction", "mn_f1", "sm_f1", "f1_delta", "mn_bleu1", "sm_bleu1",
        "bleu1_delta", "outcome", "mn_input_tokens", "mn_output_tokens", "mn_latency_ms",
        "sm_input_tokens", "sm_output_tokens", "sm_latency_ms", "mn_retrieval_count", "sm_retrieval_count",
        "mn_gold_coverage", "mn_pred_hit_rate", "mn_pred_noise", "mn_answer_path", "mn_plan",
        "mn_grounding_supplemental_rounds", "mn_grounding", "mn_warnings", "sm_warnings", "mn_classify",
        "classify_status", "mn_judge_result", "mn_judge_reason", "sm_judge_result", "sm_judge_reason",
    ]
    with (out_dir / "paired_items.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            row = {field: item.get(field) for field in fields}
            for field in ("mn_plan", "mn_grounding", "mn_warnings", "sm_warnings"):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)


def load_judged_rows(run_dir: Path, audit: dict) -> list[dict] | None:
    path = run_dir / "rows_judged.json"
    if not path.exists():
        audit["warnings"].append("rows_judged.json 不存在；仅省略辅助 Judge，不影响官方分析。")
        return None
    try:
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError("rows_judged.json must be a list")
        return rows
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit["warnings"].append(f"rows_judged.json 不可读，忽略辅助 Judge：{type(error).__name__}: {error}")
        return None


def execute(run_dir: Path, cohort_path: Path, out_dir: Path, allow_partial: bool = False,
            bootstrap: int = 5000, snapshot_dir: Path | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit, context = inspect_run(run_dir, cohort_path, snapshot_dir)
    audit["allow_partial"] = allow_partial
    if not audit["gate_passed"] and not allow_partial:
        write_json(out_dir / "audit.json", audit)
        print(f"GATE FAIL：仅写入 {out_dir / 'audit.json'}；未生成最终结果。", file=sys.stderr)
        for failure in audit["failures"]:
            print(f"- {failure['code']}: {failure['message']}", file=sys.stderr)
        return 2

    partial = not audit["gate_passed"]
    audit["analysis_status"] = "PARTIAL" if partial else "FINAL"
    cohort = context["cohort"]
    expected_keys = {item_key(question) for question in cohort.get("questions") or []}
    pairs = paired_rows(context["rows"], expected_keys)
    judged_rows = load_judged_rows(run_dir, audit)
    items, judge_alignment = build_paired_items(pairs, judged_rows)
    metrics = compute_metrics(items, pairs, bootstrap, judge_alignment, partial)
    token = build_token_audit(run_dir, cohort, context["stage_records"], context["rows"],
                              context["snapshot_dir"], partial)

    ledger_models = {}
    for system in SYSTEMS:
        values = token["per_system"].get(system, {})
        models = sorted(values.get("models") or {})
        ledger_models[system] = models
        if len(models) > 1:
            audit["warnings"].append(
                f"{system} ledger 出现多个响应模型 {models}；含失败/重试请求，只作辅助警示，"
                "旧 cctoken 失败调用不等于接受答案混模型。"
            )
    audit["ledger_response_models_warning_only"] = ledger_models
    audit["analysis_pairs"] = len(items)
    audit["status"] = "PASS" if audit["gate_passed"] else "PARTIAL"
    metrics["audit_gate_passed"] = audit["gate_passed"]

    write_json(out_dir / "audit.json", audit)
    write_json(out_dir / "metrics.json", metrics)
    write_paired_files(items, out_dir)
    (out_dir / "mn_losses.md").write_text(render_losses(items, partial), encoding="utf-8")
    write_json(out_dir / "token_audit.json", token)
    (out_dir / "token_audit.md").write_text(token_markdown(token), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(audit, metrics, token), encoding="utf-8")
    (out_dir / "DASHBOARD.html").write_text(render_dashboard(audit, metrics, token, items), encoding="utf-8")
    print(f"{'PARTIAL' if partial else 'FINAL'}: pairs={len(items)}, outputs={out_dir}")
    return 0


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("result/bench1000_locomo_20261023"))
    parser.add_argument("--cohort", type=Path, default=Path("result/locomo_1000_cohort_20261023.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("result/LOCOMO1000_FINAL_20261025"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    return execute(resolve(args.run), resolve(args.cohort), resolve(args.out_dir),
                   args.allow_partial, args.bootstrap)


if __name__ == "__main__":
    raise SystemExit(main())
