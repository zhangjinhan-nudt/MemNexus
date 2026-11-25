"""Isolated response-shape repairs: preserve content and bounded native repair."""

import asyncio
import copy
import json

import httpx
import pytest
from pydantic import ValidationError

from agent_memory.models import Contract, ExtractionResult, Hit, QueryPlan
from agent_memory.providers import CallableModel, ModelOutputError, OpenAICompatibleModel
from agent_memory.synthesis import AnswerTable, BoundInput, Support, render_table, validate_table


def hit(text, role="user", grade="primary"):
    return Hit(content=text, score=1, source_file="fixture", chunk_id=text, metadata={
        "source_events": [{"text": text, "role": role, "speaker": role,
                           "occurred_at": "2034-06-18T12:00:00+00:00", "context_only": False,
                           "evidence_grade": grade}]})


def response(value, quote="A useful note.", **item_fields):
    return {"slots": [{"target": "Original question", "derivation": "Bind the provided premise.",
                       "values": [{"value": value, "supports": [{"source_id": 1, "quote": quote}],
                                   **item_fields}]}]}


def validate(data, sources):
    table = AnswerTable.model_validate(data)
    validate_table(table, sources, ["Original question"], "text")
    return table


@pytest.mark.parametrize("length", [699, 700, 701, 1800])
def test_value_text_is_not_truncated_or_rejected_at_a_per_value_boundary(length):
    value = "A useful note. " + "x" * (length - len("A useful note. "))
    result = validate(response(value), [hit(value)])
    assert render_table(result) == value
    assert len(result.slots[0].values[0].value) == length


def test_long_value_still_requires_valid_citations_and_handles():
    value = "A useful note. " * 60
    with pytest.raises(ValueError, match="does not contain exact quote"):
        validate(response(value, quote="Not in the source."), [hit(value)])
    bad_id = response(value)
    bad_id["slots"][0]["values"][0]["supports"][0]["source_id"] = 2
    with pytest.raises(ValueError, match="Unknown source number"):
        validate(bad_id, [hit(value)])
    with pytest.raises(ValueError, match="Recalled @ identifiers"):
        validate(response(value + "@invented"), [hit(value)])


def test_other_text_and_structural_limits_remain():
    with pytest.raises(ValidationError, match="at most 700"):
        Support(source_id=1, quote="x" * 701)
    with pytest.raises(ValidationError, match="at most 200"):
        BoundInput(identity="x" * 201, statement="A premise", supports=[{"source_id": 1, "quote": "x"}])
    bad = response("A useful note.")
    bad["slots"][0]["derivation"] = "x" * 1001
    with pytest.raises(ValidationError, match="at most 1000"):
        AnswerTable.model_validate(bad)
    bad = response("A useful note.")
    bad["slots"][0]["supports"] = [{"source_id": 1, "quote": "A useful note."}]
    with pytest.raises(ValidationError, match="Extra inputs"):
        AnswerTable.model_validate(bad)


def test_long_value_does_not_override_declared_arithmetic_or_missing_premises():
    data = response("A useful note. " * 60, operation="count", inputs=[{
        "identity": "one", "statement": "One item", "supports": [{"source_id": 1, "quote": "A useful note."}]}])
    with pytest.raises(ValueError, match="Declared operation"):
        validate(data, [hit("A useful note.")])
    data = response("A useful note. " * 60, unresolved_premise="The dependent premise is missing.")
    result = validate(data, [hit("A useful note.")])
    assert not result.slots[0].values
    assert result.slots[0].coverage == "unavailable"


USER = "I prefer quiet indoor activities."
ASSISTANT = "Consider an indoor sketching session."


def misplaced_support():
    data = response("Try indoor sketching.", quote=ASSISTANT, basis="recommendation")
    data["slots"][0]["supports"] = [{"source_id": 2, "quote": USER}]
    return data


def repair_data(prompt):
    return json.JSONDecoder().raw_decode(prompt[prompt.rindex('{"previous_response":'):])[0]


def test_one_native_schema_repair_keeps_and_rehomes_a_relevant_existing_quote():
    initial = misplaced_support()
    raw = json.dumps(initial)
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            assert "Slot-level supports is not a valid field" not in prompt
            return raw
        error = repair_data(prompt)
        assert error["previous_response"] == raw
        assert "slots.0.supports" in error["errors"]
        assert "Preserve relevant, valid source references" in error["errors"]
        assert "Do not copy references to unrelated values" in error["errors"]
        fixed = copy.deepcopy(initial)
        fixed["slots"][0]["values"][0]["supports"].extend(fixed["slots"][0].pop("supports"))
        return json.dumps(fixed)

    result = asyncio.run(CallableModel(chat).synthesize("Original question", QueryPlan(), [
        hit(ASSISTANT, "assistant", "secondary"), hit(USER)]))
    assert len(calls) == 2 and render_table(result) == "Try indoor sketching."
    assert {s.source_id for s in result.slots[0].values[0].supports} == {1, 2}


@pytest.mark.parametrize("mode", ["delete_only", "invalid_quote"])
def test_schema_guidance_never_automatically_accepts_or_adds_support(mode):
    initial, calls = misplaced_support(), []

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(initial)
        fixed = copy.deepcopy(initial)
        selected = fixed["slots"][0].pop("supports")
        if mode == "invalid_quote":
            selected[0]["quote"] = "A fabricated user preference."
            fixed["slots"][0]["values"][0]["supports"].extend(selected)
        return json.dumps(fixed)

    error = "quoted user assertion" if mode == "delete_only" else "does not contain exact quote"
    with pytest.raises(ModelOutputError, match=error):
        asyncio.run(CallableModel(chat).synthesize("Original question", QueryPlan(), [
            hit(ASSISTANT, "assistant", "secondary"), hit(USER)]))
    assert len(calls) == 2


def test_other_answer_schema_errors_do_not_get_support_migration_instructions():
    initial, calls = response("A useful note."), []
    initial["slots"][0]["unexpected"] = "extra"

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(initial)
        assert "Slot-level supports is not a valid field" not in repair_data(prompt)["errors"]
        return json.dumps(response("A useful note."))

    asyncio.run(CallableModel(chat).synthesize("Original question", QueryPlan(), [hit("A useful note.")]))
    assert len(calls) == 2


def test_same_unknown_field_path_in_a_different_schema_does_not_trigger_answer_guidance():
    class Slot(Contract):
        value: str

    class OtherResult(Contract):
        slots: list[Slot]

    calls = []

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return '{"slots":[{"value":"ok","supports":[]}]}'
        assert "Slot-level supports is not a valid field" not in repair_data(prompt)["errors"]
        return '{"slots":[{"value":"ok"}]}'

    result = asyncio.run(CallableModel(chat)._structured("Other contract", OtherResult))
    assert result.slots[0].value == "ok" and len(calls) == 2


@pytest.mark.parametrize("result_type,valid", [
    (QueryPlan, {}),
    (ExtractionResult, {"candidates": [], "summary": "", "state_patch": {}}),
])
def test_planner_and_extraction_schema_failures_keep_their_own_feedback(result_type, valid):
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({**valid, "supports": []})
        errors = repair_data(prompt)["errors"]
        assert "Extra inputs are not permitted" in errors
        assert "Slot-level supports is not a valid field" not in errors
        return json.dumps(valid)

    asyncio.run(CallableModel(chat)._structured("Existing non-answer contract", result_type))
    assert len(calls) == 2


def test_long_valid_answer_still_uses_existing_provider_output_limit(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_COMPONENT_BUDGET_RMB", raising=False)
    value = "A useful note. " * 65
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["max_tokens"] == 4096
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response(value))},
                                                     "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 300}})

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAICompatibleModel(base_url="https://example.invalid/v1", model="fixture", client=client,
                                          api_key="", max_tokens=4096)
            return await model.synthesize("Original question", QueryPlan(), [hit(value)])

    result = asyncio.run(invoke())
    assert len(calls) == 1 and render_table(result) == value.strip()
