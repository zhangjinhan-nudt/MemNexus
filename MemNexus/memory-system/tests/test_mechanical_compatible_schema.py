"""The producer's concise request is distinct from the truthful runtime schema."""

import asyncio
import copy
import json

import pytest

from agent_memory.models import AnswerResponse, ExtractionResult, Hit, QueryPlan
from agent_memory.prompt_codec import compact_schema
from agent_memory.providers import CallableModel
from agent_memory.synthesis import AnswerTable, render_table


def wire_schema(prompt):
    marker = "Omit optional fields when their default applies; never omit required facts or support:\n"
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_public_schema_does_not_promise_a_length_rejected_by_neither_parser_nor_renderer(mode):
    schema = AnswerTable.model_json_schema(mode=mode)
    assert "maxLength" not in schema["$defs"]["SupportedValue"]["properties"]["value"]
    assert schema["$defs"]["Support"]["properties"]["quote"]["maxLength"] == 700
    assert "maxLength" not in AnswerResponse.model_json_schema(mode=mode)["properties"]["generated_text"]


def test_existing_generation_length_target_does_not_mutate_public_schema_or_clip_a_valid_value():
    quote = "A recorded note."
    value = quote + " Supporting text." * 60
    sources = [Hit(content=value, score=1, source_file="fixture", chunk_id="one", metadata={
        "source_events": [{"text": value, "role": "user", "speaker": "Neri", "context_only": False,
                           "occurred_at": "2035-08-04T12:00:00+00:00", "evidence_grade": "primary"}]})]
    public_schema = copy.deepcopy(AnswerTable.model_json_schema())
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        generation = wire_schema(prompt)
        assert generation["$defs"]["SupportedValue"]["properties"]["value"]["maxLength"] == 700
        generation["$defs"]["SupportedValue"]["properties"]["value"].pop("maxLength")
        assert generation == compact_schema(public_schema)
        return json.dumps({"slots": [{"target": "What was recorded?", "derivation": "Use the recorded note.",
            "values": [{"value": value, "supports": [{"source_id": 1, "quote": quote}]}]}]})

    result = asyncio.run(CallableModel(chat).synthesize("What was recorded?", QueryPlan(), sources))
    assert len(value) > 700 and len(calls) == 1 and render_table(result) == value
    assert AnswerTable.model_json_schema() == public_schema


@pytest.mark.parametrize("result_type,result", [
    (QueryPlan, {}),
    (ExtractionResult, {"candidates": [], "summary": "", "state_patch": {}}),
])
def test_non_answer_generation_schema_excludes_only_program_resolved_coordinates(result_type, result):
    expected = compact_schema(result_type.model_json_schema())
    if result_type is ExtractionResult:
        # API callers may supply explicit source positions. Extraction only
        # generates literal quotes; storage resolves these two optional fields.
        props = expected["$defs"]["Evidence"]["properties"]
        props.pop("start")
        props.pop("end")
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        assert wire_schema(prompt) == expected
        return json.dumps(result)

    asyncio.run(CallableModel(chat)._structured("Existing task", result_type))
    assert len(calls) == 1
