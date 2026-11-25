"""One native repair sees independent pre-existing scalar conflicts, without new guards."""
import asyncio
import copy
import json

import pytest

from agent_memory import synthesis
from agent_memory.models import Hit, QueryPlan
from agent_memory.providers import CallableModel
from agent_memory.synthesis import AnswerTable, render_table, validation_feedback

QUOTE = "Assembly took 6 minutes and labeling took 11 minutes. Inspection time was not measured."
SUPPORT = {"source_id": 1, "quote": QUOTE}


def source():
    return Hit(content=QUOTE, score=1, source_file="original", chunk_id="one", metadata={
        "source_events": [{"text": QUOTE, "role": "user", "speaker": "Noel",
                           "occurred_at": "2030-02-10T12:00:00+00:00", "context_only": False,
                           "evidence_grade": "primary"}]})


def bound(identity, amount=None):
    return {"identity": identity, "amount": amount, "statement": identity, "supports": [SUPPORT]}


def table_with(values):
    return AnswerTable.model_validate({"slots": [{"target": "Requested total", "derivation": "Check the requested total.",
        "values": values, "missing": "A required remaining stage is unknown.",
        "missing_kind": "necessary_premise", "coverage": "partial"}]})


def test_one_native_repair_gets_both_independent_errors():
    question = "Requested total"
    first = table_with([{"operation": "sum", "unit": "minutes",
        "value": "17 minutes for the timed stages; the whole total is unknown",
        "inputs": [bound("assembly", "6"), bound("labeling", "11")], "supports": [SUPPORT]}]).model_dump()
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(first)
        data = json.JSONDecoder().raw_decode(prompt[prompt.rindex('{"previous_response":'):])[0]
        assert "A necessary missing premise must be linked" in data["errors"]
        assert "Declared operation='sum' computes '17 minutes'" in data["errors"]
        assert data["errors"].count("Declared operation='sum'") == 1
        assert "Use operation=none" in data["errors"]
        fixed = copy.deepcopy(first)
        fixed["slots"][0]["missing_kind"] = "requested_part"
        fixed["slots"][0]["values"][0].update(operation="none", inputs=[])
        return json.dumps(fixed)

    result = asyncio.run(CallableModel(chat).synthesize(question, QueryPlan(), [source()]))
    assert len(calls) == 2
    assert "17 minutes for the timed stages" in render_table(result)
    assert "whole total is unknown" in render_table(result)
    assert result.slots[0].coverage == "partial"


def test_feedback_skips_uncomputable_blocked_and_none_values_without_mutation():
    values = [
        {"operation": "sum", "value": "99", "inputs": [bound("untimed")]},
        {"operation": "sum", "value": "99", "inputs": [bound("same", "6"), bound("SAME", "11")]},
        {"operation": "sum", "value": "99", "inputs": []},
        {"operation": "count", "value": "99", "inputs": [bound("omitted")], "unresolved_premise": "Not supported"},
        {"operation": "none", "value": "Qualified conclusion", "inputs": [bound("descriptive")]},
        {"operation": "count", "value": "1", "inputs": [bound("same"), bound(" SAME ")]},
        {"operation": "count", "value": "9", "inputs": [bound("one"), bound("two")]},
    ]
    table = table_with(values)
    original = table.model_dump()
    feedback = validation_feedback(table, ValueError("Original contract failure"))
    assert feedback.startswith("Original contract failure\n")
    assert feedback.count("Declared operation=") == 1
    assert "operation='count' computes '2'" in feedback
    assert "99" not in feedback
    assert table.model_dump() == original


def test_primary_operation_conflict_is_not_repeated():
    table = table_with([{"operation": "count", "value": "9", "inputs": [bound("one"), bound("two")]}])
    item = table.slots[0].values[0]
    primary = synthesis._operation_value_mismatch(item, synthesis._computed_value(item))
    assert validation_feedback(table, ValueError(primary)) == primary


def test_successful_synthesis_never_calls_failure_collector(monkeypatch):
    def forbidden(*args):
        pytest.fail("The failure-only collector ran on an accepted response")

    monkeypatch.setattr(synthesis, "validation_feedback", forbidden)
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return json.dumps({"slots": [{"target": "How long was assembly?", "derivation": "The duration is stated.",
            "values": [{"value": "6 minutes", "supports": [SUPPORT]}]}]})

    result = asyncio.run(CallableModel(chat).synthesize("How long was assembly?", QueryPlan(), [source()]))
    assert render_table(result) == "6 minutes"
    assert len(calls) == 1
