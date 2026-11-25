"""Metadata does not create a second citation format or a second repair."""

import asyncio
import json

import pytest
from test_request_semantics_interface import answer, source

from agent_memory.models import QueryPlan
from agent_memory.providers import CallableModel, ModelOutputError
from agent_memory.synthesis import render_table


def malformed_answer():
    result = json.dumps(answer("What color?", "blue", "I prefer blue."))
    return result.replace('"quote": "I prefer blue."', '"quote":"occurred_at":"2024-02-02"')


def test_json_repair_locates_syntax_without_changing_supported_facts():
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return malformed_answer()
        repair_data = json.loads(prompt[prompt.index('{"previous_response":'):])
        errors = repair_data["errors"]
        assert '"json_syntax"' in errors and '"column"' in errors and '"near_error"' in errors
        assert "not metadata key/value pairs" in prompt
        return json.dumps(answer("What color?", "blue", "I prefer blue."))

    table = asyncio.run(CallableModel(chat).synthesize("What color?", QueryPlan(), [source("I prefer blue.")]))
    assert len(calls) == 2 and render_table(table) == "blue"
    assert table.slots[0].values[0].supports[0].quote == "I prefer blue."


def test_repeated_invalid_metadata_quote_is_not_silently_accepted_or_retried():
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        return malformed_answer()

    with pytest.raises(ModelOutputError, match="after one repair"):
        asyncio.run(CallableModel(chat).synthesize("What color?", QueryPlan(), [source("I prefer blue.")]))
    assert len(calls) == 2


def test_source_text_follows_static_schema_and_metadata_has_no_quote_address():
    calls = []

    async def chat(prompt):
        calls.append(prompt)
        assert prompt.index('"$defs"') < prompt.index('"question":')
        assert "not additional quote strings" in prompt
        return json.dumps(answer("What color?", "blue", "I prefer blue."))

    asyncio.run(CallableModel(chat).synthesize("What color?", QueryPlan(), [source("I prefer blue.")]))
    assert len(calls) == 1
