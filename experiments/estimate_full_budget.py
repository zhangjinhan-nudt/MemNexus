"""Estimate complete benchmark cost from official dataset sizes and measured usage.

This script performs no provider calls. It scales the measured English pilot
usage by actual official turn/window/question counts and emits a conservative
range. It is a planning estimate; a paid run must still use the live ledger.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICE = 2.4
OUTPUT_PRICE = 9.6
SAFETY = 1.25


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def turns_in_sample(sample: dict) -> int:
    if "conversation" in sample:
        return sum(
            len(value)
            for key, value in sample["conversation"].items()
            if key.startswith("session_") and isinstance(value, list)
        )
    return sum(
        len(session)
        for session in sample.get("haystack_sessions", [])
        if isinstance(session, list)
    )


def windows(turns: int) -> int:
    return max(1, ((turns - 20) + 4) // 5) if turns > 20 else 1


def cost(input_tokens: float, output_tokens: float) -> float:
    return (input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE) / 1_000_000


def summarize_dataset(name: str, data: list[dict], pilot: dict) -> dict:
    turn_counts = [turns_in_sample(sample) for sample in data]
    window_counts = [windows(turns) for turns in turn_counts]
    questions = sum(len(sample.get("qa", [])) for sample in data)
    if "qa" not in data[0]:
        questions = len(data)

    pilot_turns = pilot["pilot_turns"]
    pilot_windows = pilot["pilot_windows"]
    scaled = {}
    for system, values in pilot["systems"].items():
        build_input_per_window = values["build_input_tokens"] / pilot_windows
        build_output_per_window = values["build_output_tokens"] / pilot_windows
        build_input = build_input_per_window * sum(window_counts)
        build_output = build_output_per_window * sum(window_counts)
        query_input = values["answer_input_tokens"] / pilot["pilot_questions"] * questions
        query_output = values["answer_output_tokens"] / pilot["pilot_questions"] * questions
        scaled[system] = {
            "build_windows": sum(window_counts),
            "build_input_tokens": round(build_input),
            "build_output_tokens": round(build_output),
            "query_questions": questions,
            "query_input_tokens": round(query_input),
            "query_output_tokens": round(query_output),
            "estimated_main_model_cost_rmb": cost(build_input + query_input, build_output + query_output),
        }

    judge_input_per_question = pilot["judge_input_tokens"] / pilot["pilot_questions"]
    judge_output_per_question = pilot["judge_output_tokens"] / pilot["pilot_questions"]
    judge_calls = questions * 2
    judge_cost = cost(judge_input_per_question * judge_calls, judge_output_per_question * judge_calls)
    total = sum(row["estimated_main_model_cost_rmb"] for row in scaled.values()) + judge_cost
    return {
        "dataset": name,
        "samples": len(data),
        "questions": questions,
        "turns_total": sum(turn_counts),
        "windows_total": sum(window_counts),
        "turns_mean": statistics.mean(turn_counts),
        "windows_mean": statistics.mean(window_counts),
        "windows_p95": sorted(window_counts)[max(0, int(len(window_counts) * 0.95) - 1)],
        "systems": scaled,
        "judge_calls": judge_calls,
        "estimated_judge_cost_rmb": judge_cost,
        "estimated_combined_cost_rmb": total,
        "recommended_budget_with_25pct_safety_rmb": total * SAFETY,
        "recommended_budget_three_repeats_with_25pct_safety_rmb": total * SAFETY * 3,
    }


def main() -> None:
    locomo = load(ROOT / "benchmark-data/official/locomo/locomo10.json")
    long_s = load(ROOT / "benchmark-data/official/LongMemEval/longmemeval_s_cleaned.json")
    long_oracle = load(ROOT / "benchmark-data/official/LongMemEval/longmemeval_oracle.json")
    pilot = {
        "pilot_turns": 35,
        "pilot_windows": 5,
        "pilot_questions": 5,
        "judge_input_tokens": 1352,
        "judge_output_tokens": 288,
        "systems": {
            "SimpleMem": {
                "build_input_tokens": 8173,
                "build_output_tokens": 13147,
                "answer_input_tokens": 35421,
                "answer_output_tokens": 2502,
            },
            "MemNexus": {
                "build_input_tokens": 23378,
                "build_output_tokens": 5254,
                "answer_input_tokens": 15657,
                "answer_output_tokens": 1281,
            },
        },
    }
    report = {
        "pricing": {
            "input_rmb_per_million": INPUT_PRICE,
            "output_rmb_per_million": OUTPUT_PRICE,
            "basis": "DeepSeek Flash peak cache-miss USD price converted at 8 RMB/USD",
        },
        "pilot_basis": pilot,
        "datasets": {
            "LoCoMo": summarize_dataset("LoCoMo", locomo, pilot),
            "LongMemEval-oracle": summarize_dataset("LongMemEval-oracle", long_oracle, pilot),
            "LongMemEval-S": summarize_dataset("LongMemEval-S", long_s, pilot),
        },
        "interpretation": {
            "budget_8_rmb": "supports pilot and mechanism tests only",
            "complete_claim": "requires a funded run at the recommended budget and paired confidence intervals",
            "warning": "Scaling assumes the measured prompt/output profile remains stable; live ledger remains authoritative.",
        },
    }
    output = ROOT / "result" / "FULL_BUDGET_ESTIMATE.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
