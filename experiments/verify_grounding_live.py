"""Small real-provider component ablation on controlled synthetic cases.

Both variants use identical manually specified lossy memories, raw utterances,
and deterministic retrieval plans. Only source grounding is toggled. This is
not LoCoMo accuracy, not an extraction benchmark and not an independent judge.
"""

import asyncio
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))

from agent_memory.models import Candidate, Event, Evidence, QueryPlan, Scope, TurnInput
from agent_memory.providers import build_model
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings

CASES = [
    {"id": "year_boundary", "speaker": "Inez", "date": "2025-03-04",
     "source": "I painted the mural last year.", "memory": "Inez painted a mural last year.",
     "subject": "Inez", "question": "In which year did Inez paint the mural?", "expected": ["2024"]},
    {"id": "month_rollover", "speaker": "Salma", "date": "2028-12-18",
     "source": "My troupe is performing at the festival next month.",
     "memory": "Salma's troupe will perform at the festival next month.", "subject": "Salma",
     "question": "In which month and year will Salma's troupe perform at the festival?", "expected": ["january", "2029"]},
    {"id": "weekend_interval", "speaker": "Tariq", "date": "2025-05-21",
     "source": "I paddled through the gorge last weekend.", "memory": "Tariq paddled through the gorge last weekend.",
     "subject": "Tariq", "question": "On which weekend did Tariq paddle through the gorge?", "expected": ["17", "18", "2025"]},
    {"id": "omitted_detail", "speaker": "Ren", "date": "2024-04-07",
     "source": "I bought a basalt paddle for the river race. What do you think?", "memory": "Ren prepared equipment.",
     "subject": "Ren", "question": "What material was Ren's paddle made of?", "expected": ["basalt"]},
    {"id": "speaker_misattribution", "speaker": "Salma", "date": "2024-07-16",
     "source": "I use linen sails.", "memory": "Arun uses linen sails.", "subject": "Arun",
     "question": "Who uses linen sails?", "expected": ["salma"]},
    {"id": "unknown_supplier", "speaker": "Kei", "date": "2024-11-02",
     "source": "My canoe is blue.", "memory": "Kei has a blue canoe.", "subject": "Kei",
     "question": "Which supplier manufactured Kei's canoe?", "expected": ["unknown"]},
]


def check(text, expected):
    text = text.casefold()
    if expected == ["unknown"]:
        return bool(re.search(r"not (?:specified|recorded|mentioned|provided)|unknown|no (?:information|evidence)|cannot|can't|don't|doesn't", text))
    if expected == ["january", "2029"] and "2029-01" in text:
        return True
    return all(value in text for value in expected)


async def main():
    output = ROOT / "result/grounding_live_ablation.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite prior live evidence")
    os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = "0.30"
    os.environ["EXPERIMENT_USAGE_LEDGER_PATH"] = str(ROOT / "result/grounding_live_usage.json")
    provider = build_model(Settings(model_max_tokens=512, model_timeout=90, output_language="en"))
    if provider is None:
        raise RuntimeError("Real model is not configured")
    report = {"status": "running", "kind": "real-provider synthetic component ablation",
              "prior_cumulative_upper_bound_rmb": 17.5110032, "run_cap_rmb": 0.30,
              "global_cap_rmb": 18, "cases": CASES, "rows": []}

    class Model:
        async def plan(self, query, context):
            return QueryPlan(required_info=[query], semantic_queries=[query])

        async def answer(self, prompt):
            return await provider.answer(prompt)

    try:
        for index, case in enumerate(CASES):
            for enabled in ((False, True) if index % 2 == 0 else (True, False)):
                before = dict(provider.usage)
                with tempfile.TemporaryDirectory(prefix="memnexus-grounding-check-") as folder:
                    runtime = MemoryRuntime(Settings(db_path=Path(folder) / "memory.db", context_turns=0,
                                                     acceleration_enabled=False, evidence_grounding_enabled=enabled,
                                                     output_language="en"), model=Model())
                    scope = Scope(tenant_id="synthetic", owner_id=case["id"])
                    try:
                        session = runtime.store.create_session(scope)
                        source = runtime.store.append_turn(scope, session.session_id, TurnInput(request_id="source", events=[
                            Event(role="user", speaker=case["speaker"], content=case["source"],
                                  occurred_at=datetime.fromisoformat(case["date"]).replace(tzinfo=UTC))]))
                        if case["id"] == "speaker_misattribution":
                            runtime.store.append_turn(scope, session.session_id, TurnInput(request_id="other", events=[
                                Event(role="user", speaker="Arun", content="I teach map reading.",
                                      occurred_at=datetime.fromisoformat(case["date"]).replace(tzinfo=UTC))]))
                        runtime.store.add_candidates(scope, session.session_id, [Candidate(
                            subject=case["subject"], predicate="event", value=case["memory"], content=case["memory"],
                            evidence=[Evidence(turn_id=source.turn_id, event_index=0, quote=case["source"])])])
                        answer = await runtime.answer(scope, session.session_id, case["question"],
                                                      prepare=False, accelerate=False, timeout=90)
                        row = {"case_id": case["id"], "grounding_enabled": enabled,
                               "prediction": answer.generated_text, "rule_check_passed": check(answer.generated_text, case["expected"]),
                               "grounding": answer.grounding.model_dump(mode="json"),
                               "sources": [h.model_dump(mode="json") for h in answer.sources],
                               "usage": {k: provider.usage[k] - before.get(k, 0) for k in provider.usage}}
                        report["rows"].append(row)
                        print(json.dumps({"case_id": case["id"], "grounding_enabled": enabled,
                                          "rule_check_passed": row["rule_check_passed"]}), flush=True)
                    finally:
                        await runtime.close()
                report["usage"] = dict(provider.usage)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["usage"] = dict(provider.usage)
        report["cumulative_upper_bound_rmb"] = report["prior_cumulative_upper_bound_rmb"] + provider.usage["cost_rmb"]
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
