"""Offline mechanism tests; fixtures are not benchmark predictions."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from agent_memory.grounding import _source_event
from agent_memory.models import Event, Hit, QueryPlan, Turn
from agent_memory.providers import CallableModel, ModelOutputError, OpenAICompatibleModel, answer_call_budget
from agent_memory.temporal import resolve_time_references


def source(text, year=2023):
    t = Turn(turn_id="t", request_id="r", session_id="s", sequence=1,
             events=[Event(role="user", speaker="Person", content=text, occurred_at=datetime(year, 7, 16, tzinfo=UTC))])
    return Hit(content=text, score=1, source_file="source", chunk_id="c",
               metadata={"source_events": [_source_event(t, 0, text, 0)]})


@pytest.mark.parametrize("text,precision,start", [
    ("I arrived yesterday.", "day", "2023-07-15"),
    ("I arrived on 2020-02-29.", "day", "2020-02-29"),
    ("I arrived last week.", "week", None),
    ("I will arrive next week.", "week", None),
    ("I arrived last month.", "month", "2023-06-01"),
])
def test_canonical_time_input_preserves_precision(text, precision, start):
    h = source(text)
    references = resolve_time_references(text, source_time=h.metadata["source_events"][0]["occurred_at"])
    assert references
    reference = references[0]
    assert reference.precision == precision
    if start is not None:
        assert reference.start == start


def test_normal_calls_and_one_repair_are_independently_bounded():
    async def scenario():
        replies = iter(['{}', 'bad json', '{}', 'answer'])
        model = CallableModel(lambda _: next(replies))
        with answer_call_budget(True, max_calls=2) as state:
            await model._structured("planner", QueryPlan)
            assert state["calls"] == 0
            state["active"] = True
            await model._structured("answer", QueryPlan)
            assert state["calls"] == 2
            assert state["normal_calls"] == state["repair_calls"] == 1
            await model.answer("bounded update")
            assert state["normal_calls"] == 2
            with pytest.raises(ModelOutputError, match="call limit"):
                await model.answer("extra")
    asyncio.run(scenario())


def test_http_adapter_counts_one_actual_attempt():
    async def scenario():
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            model = OpenAICompatibleModel(base_url="https://example.invalid/v1", model="test", client=client)
            with answer_call_budget(True, max_calls=1) as state:
                state["active"] = True
                assert await model.answer("question") == "ok"
                assert state["calls"] == len(requests) == model.usage["provider_attempts"] == 1
    asyncio.run(scenario())


def test_lightweight_contract_does_not_render_planner_premises():
    async def scenario():
        question = "How long did the delivery take?"
        h = source("Delivery took three weeks.")
        captured = []
        def chat(prompt):
            payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"question":'):])
            captured.append(payload)
            return json.dumps({"slots": [{"target": question, "role": "answer",
                "derivation": "The source states the requested delivery duration directly.", "values": [
                    {"value": "three weeks", "supports": [{"source_id": 1, "quote": h.content}]}]}]})
        plan = QueryPlan(required_info=["order date", "arrival date", "elapsed weeks"])
        result = await CallableModel(chat).synthesize(question, plan, [h])
        assert captured[0]["targets"] == [question]
        assert result.slots[0].values[0].value == "three weeks"
    asyncio.run(scenario())
