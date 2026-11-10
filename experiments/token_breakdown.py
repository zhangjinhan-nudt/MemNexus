"""Audit LoCoMo provider ledgers without estimating unreported consumption.

``gross`` covers every appended request, including retries and rebuilt epochs.
``accepted`` is calculated independently from ``stage_result.json`` row-level
``model_calls``; the two are never added together. Units without a ledger or a
stage result remain visible, and a partially written final JSONL line is counted
as malformed instead of aborting a live-run audit.

A request whose attempt is non-increasing relative to the previous request starts
an epoch. Responses and errors only pair with the same attempt in that epoch.
No prompt/response content deduplication is performed.

USD pricing uses reported usage only. Cache fields are aliases which may repeat
in one response, so billable cached tokens are the maximum of the non-negative
``prompt_cache_hit_tokens``, ``cache_read_input_tokens`` and ``cached_tokens``
values, capped by ``prompt_tokens``. They are never summed. Requests with an
error, no response, or a response without usage have no token/cost estimate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGES = (
    "extraction",
    "plan",
    "answer",
    "bound_answer",
    "collection_extraction",
    "inference",
    "reflect",
    "other",
)
CACHE_FIELDS = (
    "prompt_cache_hit_tokens",
    "cache_read_input_tokens",
    "cached_tokens",
)
PRICING = {
    "currency": "USD",
    "per_million_tokens": {
        "uncached_input": 0.028,
        "cached_input": 0.0007,
        "output": 0.112,
    },
    "cached_token_policy": (
        "min(prompt_tokens, max(nonnegative prompt_cache_hit_tokens, "
        "cache_read_input_tokens, cached_tokens)); aliases are not summed"
    ),
    "formula": (
        "(uncached_prompt_tokens * 0.028 + cached_tokens * 0.0007 + "
        "completion_tokens * 0.112) / 1e6"
    ),
    "unknown_usage_policy": (
        "Errors, unsettled requests, and responses without usage are counted "
        "separately and receive zero estimated tokens/cost, not an actual-usage claim."
    ),
}
LABELS = (
    ("你负责短期记忆", "extraction"),
    ("You are a professional information extraction assistant", "extraction"),
    ("Plan retrieval", "plan"),
    ("你负责 SimpleMem 式用户意图感知查询规划", "plan"),
    ("You are an intelligent information requirement analyst", "plan"),
    ("You are a query generation specialist", "plan"),
    ("You are a query analysis assistant", "plan"),
    ("Answer the original question using the numbered evidence", "answer"),
    ("You are a professional Q&A assistant", "answer"),
    ("Fill each mandatory output in the query contract", "bound_answer"),
    ("Examine every numbered source for the typed query", "collection_extraction"),
    ("The direct evidence pass produced no answer", "inference"),
    ("You are an information completeness evaluator", "reflect"),
    ("You are a missing information query generator", "reflect"),
)
STAT_FIELDS = (
    "requests",
    "responses",
    "errors",
    "unsettled",
    "responses_with_usage",
    "responses_without_usage",
    "unmatched_responses",
    "unmatched_errors",
    "prompt_chars",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "uncached_prompt_tokens",
    "cost_usd",
)
ACCEPTED_FIELDS = (
    "questions",
    "rows_with_model_calls",
    "provider_attempts",
    "provider_errors",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "uncached_prompt_tokens",
    "cost_usd",
    "latency_ms",
    "latency_samples",
)


def label_of(prompt: str) -> str:
    head = prompt[:512].casefold()
    for prefix, label in LABELS:
        if prefix.casefold() in head:
            return label
    return "other"


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    return " ".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict)
    )


def _prefix(prompt: str) -> str:
    return " ".join(prompt.split())[:240] or "<empty prompt>"


def _empty_models() -> dict[str, dict[str, Any]]:
    return defaultdict(
        lambda: {"totals": Counter(), "stages": defaultdict(Counter)}
    )


def _empty_scan() -> dict[str, Any]:
    return {
        "totals": Counter(),
        "stages": defaultdict(Counter),
        "models": _empty_models(),
        "errors": Counter(),
        "other_prefixes": Counter(),
        "malformed_lines": 0,
        "epochs": 0,
    }


def _bump(
    audit: dict[str, Any], stage: str, model: str, field: str, value: float = 1
) -> None:
    audit["totals"][field] += value
    audit["stages"][stage][field] += value
    audit["models"][model]["totals"][field] += value
    audit["models"][model]["stages"][stage][field] += value


def usage_metrics(usage: dict[str, Any]) -> dict[str, float | int]:
    """Return reported tokens and deterministic USD cost for one usage object."""
    prompt = _int(usage.get("prompt_tokens"))
    completion = _int(usage.get("completion_tokens"))
    total = _int(usage.get("total_tokens"))
    cached = min(
        max(prompt, 0),
        max(0, max(_int(usage.get(field)) for field in CACHE_FIELDS)),
    )
    uncached = max(prompt - cached, 0)
    cost = (uncached * 0.028 + cached * 0.0007 + completion * 0.112) / 1e6
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": cached,
        "uncached_prompt_tokens": uncached,
        "cost_usd": cost,
    }


def scan(path: Path) -> dict[str, Any]:
    """Pair one ledger by request epochs and aggregate all observed outcomes."""
    audit = _empty_scan()
    epochs: dict[int, dict[Any, dict[str, Any]]] = defaultdict(dict)
    epoch = 0
    previous_attempt: Any = None
    saw_request = False
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        audit["read_error"] = f"{type(error).__name__}: {error}"
        return audit

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            audit["malformed_lines"] += 1
            continue
        if not isinstance(entry, dict):
            audit["malformed_lines"] += 1
            continue
        event = entry.get("event")
        attempt = entry.get("attempt")
        body = entry.get("body") if isinstance(entry.get("body"), dict) else {}

        if event == "request":
            if saw_request and (
                attempt is None
                or previous_attempt is None
                or attempt <= previous_attempt
            ):
                epoch += 1
            previous_attempt = attempt
            saw_request = True
            prompt = _prompt(body.get("messages"))
            stage = label_of(prompt)
            model = str(body.get("model") or "unknown")
            epochs[epoch][attempt] = {
                "stage": stage,
                "model": model,
                "settled": False,
            }
            _bump(audit, stage, model, "requests")
            _bump(audit, stage, model, "prompt_chars", len(prompt))
            if stage == "other":
                audit["other_prefixes"][(_prefix(prompt), model)] += 1
            continue

        if event not in {"response", "error"}:
            continue
        request = epochs[epoch].get(attempt)
        matched = request is not None
        if matched:
            stage = request["stage"]
            model = request["model"]
            request["settled"] = True
        else:
            stage = "other"
            model = str(body.get("model") or "unknown")

        if event == "response":
            _bump(audit, stage, model, "responses")
            if not matched:
                _bump(audit, stage, model, "unmatched_responses")
            usage = body.get("usage")
            if isinstance(usage, dict) and usage:
                _bump(audit, stage, model, "responses_with_usage")
                for field, value in usage_metrics(usage).items():
                    _bump(audit, stage, model, field, value)
            else:
                _bump(audit, stage, model, "responses_without_usage")
        else:
            _bump(audit, stage, model, "errors")
            if not matched:
                _bump(audit, stage, model, "unmatched_errors")
            error_type = str(entry.get("error_type") or "unknown")
            status_code = entry.get("status_code")
            audit["errors"][(model, stage, error_type, status_code)] += 1

    for requests in epochs.values():
        for request in requests.values():
            if not request["settled"]:
                _bump(audit, request["stage"], request["model"], "unsettled")
    audit["epochs"] = epoch + 1 if saw_request else 0
    return audit


def _merge_scan(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["totals"].update(source["totals"])
    for stage, values in source["stages"].items():
        target["stages"][stage].update(values)
    for model, values in source["models"].items():
        target["models"][model]["totals"].update(values["totals"])
        for stage, stage_values in values["stages"].items():
            target["models"][model]["stages"][stage].update(stage_values)
    target["errors"].update(source["errors"])
    target["other_prefixes"].update(source["other_prefixes"])
    target["malformed_lines"] += source["malformed_lines"]
    target["epochs"] += source["epochs"]


def _stats_view(values: Counter) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for field in STAT_FIELDS:
        value = values[field]
        result[field] = round(float(value), 12) if field == "cost_usd" else int(value)
    return result


def _models_view(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        model: {
            "totals": _stats_view(values["totals"]),
            "stages": {
                stage: _stats_view(values["stages"][stage])
                for stage in STAGES
                if values["stages"][stage]
            },
        }
        for model, values in sorted(models.items())
    }


def _scan_view(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "totals": _stats_view(audit["totals"]),
        "stages": {stage: _stats_view(audit["stages"][stage]) for stage in STAGES},
        "models": _models_view(audit["models"]),
        "epochs": audit["epochs"],
        "malformed_lines": audit["malformed_lines"],
    }


def _accepted_rows(rows: Any) -> Counter:
    accepted = Counter()
    if not isinstance(rows, list):
        return accepted
    for row in rows:
        if not isinstance(row, dict):
            continue
        accepted["questions"] += 1
        calls = row.get("model_calls")
        if not isinstance(calls, dict):
            continue
        accepted["rows_with_model_calls"] += 1
        accepted["provider_attempts"] += _int(calls.get("provider_attempts"))
        accepted["provider_errors"] += _int(calls.get("provider_errors"))
        prompt = _int(calls.get("input_tokens"))
        completion = _int(calls.get("output_tokens"))
        cached = min(
            max(prompt, 0),
            max(0, max(_int(calls.get(field)) for field in CACHE_FIELDS)),
        )
        uncached = max(prompt - cached, 0)
        accepted["prompt_tokens"] += prompt
        accepted["completion_tokens"] += completion
        accepted["total_tokens"] += (
            _int(calls["total_tokens"])
            if "total_tokens" in calls
            else prompt + completion
        )
        accepted["cached_tokens"] += cached
        accepted["uncached_prompt_tokens"] += uncached
        accepted["cost_usd"] += (
            uncached * 0.028 + cached * 0.0007 + completion * 0.112
        ) / 1e6
        if row.get("latency_ms") is not None:
            accepted["latency_ms"] += _float(row.get("latency_ms"))
            accepted["latency_samples"] += 1
    return accepted


def _accepted_view(values: Counter, questions_override: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ACCEPTED_FIELDS:
        value = values[field]
        if field in {"cost_usd", "latency_ms"}:
            result[field] = round(float(value), 12)
        else:
            result[field] = int(value)
    divisor = questions_override if questions_override > 0 else values["questions"]
    result["question_divisor"] = int(divisor)
    result["mean_latency_ms"] = (
        round(values["latency_ms"] / values["latency_samples"], 6)
        if values["latency_samples"]
        else None
    )
    result["per_question"] = (
        {
            "prompt_tokens": round(values["prompt_tokens"] / divisor, 6),
            "completion_tokens": round(values["completion_tokens"] / divisor, 6),
            "latency_ms": round(values["latency_ms"] / divisor, 6),
            "cost_usd": round(values["cost_usd"] / divisor, 12),
        }
        if divisor
        else None
    )
    return result


def _unit_names(unit: str) -> tuple[str, str]:
    lowered = unit.casefold()
    if "simplemem" in lowered:
        system = "SimpleMem"
    elif "memnexus" in lowered:
        system = "MemNexus"
    else:
        system = "Unknown"
    return system, unit.split("_", 1)[0]


def _error_rows(errors: Counter) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "stage": stage,
            "error_type": error_type,
            "status_code": status_code,
            "count": count,
        }
        for (model, stage, error_type, status_code), count in sorted(
            errors.items(), key=lambda item: (str(item[0]), item[1])
        )
    ]


def build_report(root: Path, questions: int = 0, samples: str = "") -> dict[str, Any]:
    root = Path(root)
    wanted = {item.strip() for item in samples.split(",") if item.strip()}
    attempts = root / "attempts"
    unit_paths = (
        sorted(path for path in attempts.iterdir() if path.is_dir())
        if attempts.exists()
        else []
    )
    per_system_raw: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"audit": _empty_scan(), "accepted": Counter(), "units": 0, "complete": 0}
    )
    all_models = _empty_scan()
    units = []
    per_unit = {}
    error_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    other_prefixes: dict[str, Counter] = defaultdict(Counter)

    for unit_path in unit_paths:
        unit = unit_path.name
        system, sample = _unit_names(unit)
        if wanted and sample not in wanted:
            continue
        ledger = unit_path / "usage.calls.jsonl"
        audit = scan(ledger) if ledger.exists() else _empty_scan()
        result_path = unit_path / "stage_result.json"
        complete = result_path.exists()
        accepted = Counter()
        accepted_status = "missing"
        if complete:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                accepted = _accepted_rows(result.get("rows"))
                accepted_status = "ok"
            except (OSError, json.JSONDecodeError, AttributeError) as error:
                accepted_status = f"unreadable: {type(error).__name__}"
        status = "complete" if complete else (
            "failed" if (unit_path / "failure.json").exists() else "incomplete"
        )
        unit_view = _scan_view(audit)
        unit_view.update(
            {
                "unit": unit,
                "sample": sample,
                "system": system,
                "status": status,
                "complete": complete,
                "ledger_present": ledger.exists(),
                "accepted_status": accepted_status,
                "accepted": _accepted_view(accepted),
                "errors": _error_rows(audit["errors"]),
            }
        )
        if "read_error" in audit:
            unit_view["read_error"] = audit["read_error"]
        per_unit[unit] = unit_view
        units.append(
            {
                "unit": unit,
                "sample": sample,
                "system": system,
                "status": status,
                "complete": complete,
                "ledger_present": ledger.exists(),
                "accepted_status": accepted_status,
            }
        )

        system_raw = per_system_raw[system]
        system_raw["units"] += 1
        system_raw["complete"] += int(complete)
        system_raw["accepted"].update(accepted)
        _merge_scan(system_raw["audit"], audit)
        _merge_scan(all_models, audit)
        for (model, stage, error_type, status_code), count in audit["errors"].items():
            key = (system, model, stage, error_type, status_code)
            group = error_groups.setdefault(
                key,
                {"count": 0, "units": set()},
            )
            group["count"] += count
            group["units"].add(unit)
        for (prefix, model), count in audit["other_prefixes"].items():
            other_prefixes[prefix]["requests"] += count
            other_prefixes[prefix][f"system:{system}"] += count
            other_prefixes[prefix][f"model:{model}"] += count

    per_system = {}
    for system, values in sorted(per_system_raw.items()):
        view = _scan_view(values["audit"])
        view.update(
            {
                "units": values["units"],
                "complete_units": values["complete"],
                "incomplete_units": values["units"] - values["complete"],
                "accepted": _accepted_view(values["accepted"], questions),
            }
        )
        per_system[system] = view

    errors = []
    for (system, model, stage, error_type, status_code), values in sorted(
        error_groups.items(), key=lambda item: str(item[0])
    ):
        errors.append(
            {
                "system": system,
                "model": model,
                "stage": stage,
                "error_type": error_type,
                "status_code": status_code,
                "count": values["count"],
                "units": sorted(values["units"]),
            }
        )
    top_other = []
    for prefix, values in sorted(
        other_prefixes.items(), key=lambda item: (-item[1]["requests"], item[0])
    )[:20]:
        top_other.append(
            {
                "prefix": prefix,
                "requests": values["requests"],
                "systems": {
                    key[7:]: count
                    for key, count in sorted(values.items())
                    if key.startswith("system:")
                },
                "models": {
                    key[6:]: count
                    for key, count in sorted(values.items())
                    if key.startswith("model:")
                },
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "partial_run": any(not unit["complete"] for unit in units),
        "samples": sorted(wanted),
        "questions_override": questions or None,
        "pricing": PRICING,
        "accounting": {
            "gross": "All appended ledger requests, including retries and rebuilt epochs.",
            "accepted": "stage_result.json rows' model_calls only; calculated separately from gross.",
        },
        "units": units,
        "per_system": per_system,
        "per_unit": per_unit,
        "models": _models_view(all_models["models"]),
        "errors": {"total": sum(item["count"] for item in errors), "by": errors},
        "other_prompts": {
            "requests": sum(values["requests"] for values in other_prefixes.values()),
            "unique_prefixes": len(other_prefixes),
            "top20": top_other,
        },
    }


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Token breakdown audit",
        "",
        f"Generated at: `{report['generated_at']}`  ",
        f"Root: `{report['root']}`  ",
        f"Partial run: **{report['partial_run']}**",
        "",
        "## Pricing",
        "",
        "- Currency: USD; uncached input $0.028/1M, cached input $0.0007/1M, output $0.112/1M.",
        f"- Cache policy: {report['pricing']['cached_token_policy']}.",
        f"- Formula: `{report['pricing']['formula']}`.",
        "- Gross and accepted are separate; failed/no-usage/unsettled calls have no guessed usage.",
        "",
        "## Units",
        "",
        "| unit | system | status | ledger | accepted |",
        "|---|---|---|---|---|",
    ]
    for unit in report["units"]:
        lines.append(
            f"| {_md(unit['unit'])} | {unit['system']} | {unit['status']} | "
            f"{unit['ledger_present']} | {_md(unit['accepted_status'])} |"
        )
    lines.extend(["", "## Per system", ""])
    for system, values in report["per_system"].items():
        accepted = values["accepted"]
        per_question = accepted["per_question"]
        lines.extend(
            [
                f"### {system}",
                "",
                (
                    f"Units: {values['complete_units']} complete / {values['units']} total; "
                    f"accepted questions: {accepted['questions']}."
                ),
                "",
                (
                    f"Accepted: prompt={accepted['prompt_tokens']}, "
                    f"completion={accepted['completion_tokens']}, "
                    f"latency_ms={accepted['latency_ms']:.3f}, "
                    f"cost_usd=${accepted['cost_usd']:.8f}."
                ),
                "",
                f"Accepted per question: `{json.dumps(per_question, ensure_ascii=False)}`.",
                "",
                f"Gross cost (not added to accepted): **${values['totals']['cost_usd']:.8f}**.",
                "",
                "| stage | requests | responses | errors | unsettled | with usage | without usage | prompt | cached | completion | total | USD |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for stage, stats in values["stages"].items():
            lines.append(
                f"| {stage} | {stats['requests']} | {stats['responses']} | {stats['errors']} | "
                f"{stats['unsettled']} | {stats['responses_with_usage']} | "
                f"{stats['responses_without_usage']} | {stats['prompt_tokens']} | "
                f"{stats['cached_tokens']} | {stats['completion_tokens']} | "
                f"{stats['total_tokens']} | {stats['cost_usd']:.8f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Per unit",
            "",
            "| unit | status | gross requests | gross responses | gross errors | unsettled | gross USD | accepted QA | accepted USD |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for unit, values in report["per_unit"].items():
        gross = values["totals"]
        accepted = values["accepted"]
        lines.append(
            f"| {_md(unit)} | {values['status']} | {gross['requests']} | {gross['responses']} | "
            f"{gross['errors']} | {gross['unsettled']} | {gross['cost_usd']:.8f} | "
            f"{accepted['questions']} | {accepted['cost_usd']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Models",
            "",
            "| model | stage | requests | responses | errors | unsettled | prompt | cached | completion | USD |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, values in report["models"].items():
        for stage, stats in values["stages"].items():
            lines.append(
                f"| {_md(model)} | {stage} | {stats['requests']} | {stats['responses']} | "
                f"{stats['errors']} | {stats['unsettled']} | {stats['prompt_tokens']} | "
                f"{stats['cached_tokens']} | {stats['completion_tokens']} | {stats['cost_usd']:.8f} |"
            )
    lines.extend(["", "## Errors", "", f"Total: {report['errors']['total']}", ""])
    lines.extend(["| system | model | stage | type | status | count | units |", "|---|---|---|---|---|---:|---|"])
    for error in report["errors"]["by"]:
        lines.append(
            f"| {error['system']} | {_md(error['model'])} | {error['stage']} | "
            f"{_md(error['error_type'])} | {_md(error['status_code'])} | {error['count']} | "
            f"{_md(', '.join(error['units']))} |"
        )
    lines.extend(
        [
            "",
            "## Other prompts (top 20 prefixes)",
            "",
            (
                f"Requests: {report['other_prompts']['requests']}; unique prefixes: "
                f"{report['other_prompts']['unique_prefixes']}. "
                "Other is not automatically allocated to QA."
            ),
            "",
            "| prefix | requests | systems | models |",
            "|---|---:|---|---|",
        ]
    )
    for item in report["other_prompts"]["top20"]:
        lines.append(
            f"| {_md(item['prefix'])} | {item['requests']} | "
            f"{_md(json.dumps(item['systems'], ensure_ascii=False))} | "
            f"{_md(json.dumps(item['models'], ensure_ascii=False))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="run folder with attempts/*/usage.calls.jsonl")
    parser.add_argument("--questions", type=int, default=0, help="override per-system accepted QA divisor")
    parser.add_argument("--samples", default="", help="comma separated sample ids to include")
    parser.add_argument("--json", dest="json_path", help="optional full JSON report path")
    parser.add_argument("--markdown", dest="markdown_path", help="optional Markdown report path")
    args = parser.parse_args()
    if args.questions < 0:
        parser.error("--questions must be non-negative")

    report = build_report(Path(args.root), args.questions, args.samples)
    for system, values in report["per_system"].items():
        accepted = values["accepted"]
        print(
            f"\n{system}: units={values['complete_units']}/{values['units']} complete, "
            f"accepted_QA={accepted['questions']}, gross_USD=${values['totals']['cost_usd']:.8f}, "
            f"accepted_USD=${accepted['cost_usd']:.8f} (separate)"
        )
        print(
            f"{'stage':22s}{'req':>7s}{'resp':>7s}{'err':>6s}{'open':>6s}"
            f"{'prompt':>12s}{'cached':>12s}{'completion':>12s}{'USD':>13s}"
        )
        ordered = sorted(
            values["stages"].items(), key=lambda item: -item[1]["prompt_tokens"]
        )
        for stage, stats in ordered:
            print(
                f"{stage:22s}{stats['requests']:7d}{stats['responses']:7d}"
                f"{stats['errors']:6d}{stats['unsettled']:6d}"
                f"{stats['prompt_tokens']:12d}{stats['cached_tokens']:12d}"
                f"{stats['completion_tokens']:12d}{stats['cost_usd']:13.8f}"
            )
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote JSON: {path}")
    if args.markdown_path:
        path = Path(args.markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_report(report), encoding="utf-8")
        print(f"wrote Markdown: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
