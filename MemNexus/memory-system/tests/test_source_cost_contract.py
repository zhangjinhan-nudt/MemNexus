"""The packer's budget must describe the evidence actually sent to the reader."""
import asyncio
import json
from datetime import UTC, datetime

import pytest

from agent_memory.grounding import _source_event
from agent_memory.models import Event, Hit, QueryPlan, Turn
from agent_memory.prompt_codec import (
    SOURCE_CATALOG_OVERHEAD,
    compact_json,
    synthesis_request_context,
    synthesis_source_cost,
)
from agent_memory.providers import CallableModel


class CapturedInput(Exception):
    pass


def source(number, *, distinct_claim=False):
    text = f"I moved on March 5, 2024 and built bench {number}."
    turn = Turn(turn_id=f"turn-{number}", request_id=f"request-{number}", session_id="session",
                sequence=number, events=[Event(role="user", speaker="Inez", content=text,
                    occurred_at=datetime(2025, 1, 1, tzinfo=UTC))])
    return Hit(content=f"Inez built bench {number}." if distinct_claim else text,
               score=1, source_file="session", chunk_id=f"hit-{number}", metadata={
                   "status": "active", "assertion": "stated", "origin": "compressed_memory",
                   "source_support_status": "resolved",
                   "source_events": [_source_event(turn, 0, text, 0)]})


@pytest.mark.parametrize("question,expected_time", [
    ("What did Inez build?", False),
    ("When did Inez move?", True),
])
@pytest.mark.parametrize("count", [1, 10])
@pytest.mark.parametrize("distinct_claim", [False, True])
def test_cost_equals_real_provider_source_catalog(question, expected_time, count, distinct_claim):
    sources = [source(i, distinct_claim=distinct_claim) for i in range(1, count + 1)]
    original = [hit.model_dump() for hit in sources]
    captured = []

    def chat(prompt):
        captured.append(prompt)
        raise CapturedInput

    plan = QueryPlan()
    with pytest.raises(CapturedInput):
        asyncio.run(CallableModel(chat).synthesize(question, plan, sources, as_of=datetime(2025, 1, 1, tzinfo=UTC)))
    assert len(captured) == 1
    payload = json.JSONDecoder().raw_decode(captured[0].split("\n输入数据：\n", 1)[1])[0]
    _, _, _, needs_time = synthesis_request_context(question, plan, datetime(2025, 1, 1, tzinfo=UTC))
    assert needs_time is expected_time
    catalog = {"sources": payload["sources"]}
    accounted = SOURCE_CATALOG_OVERHEAD + sum(
        synthesis_source_cost(hit, source_id, needs_time=needs_time)
        for source_id, hit in enumerate(sources, 1))
    assert accounted == len(compact_json(catalog))
    assert len(compact_json(catalog)) == len(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")))
    for number, entry in enumerate(payload["sources"], 1):
        assert entry["source_id"] == number
        assert entry["attributes"]["assertion"] == "stated"
        assert "provenance" not in entry and "claim" not in entry
        assert ("claim_hint" in entry) is distinct_claim
        event = entry["source_events"][0]
        assert event["text"] == sources[number - 1].metadata["source_events"][0]["text"]
        assert "time_derivations" not in event
        assert ("time_references" in event) is expected_time
    assert [hit.model_dump() for hit in sources] == original


def test_contract_temporal_operator_uses_same_context_without_time_keyword():
    question = "Describe the milestones."
    plan = QueryPlan(answer_contract={"outputs": [{
        "output_id": "period", "question_span": "milestones", "operator": "duration",
        "subjects": ["Inez"], "relation": "milestones",
    }]})
    original = plan.model_dump()
    _, contract, _, needs_time = synthesis_request_context(question, plan, datetime(2025, 1, 1, tzinfo=UTC))
    assert contract.outputs[0].operator == "duration" and needs_time
    assert plan.model_dump() == original


def test_shared_json_preserves_unicode_dates_and_models():
    value = {"label": "原文", "date": datetime(2025, 1, 1, tzinfo=UTC), "plan": QueryPlan()}
    payload = json.loads(compact_json(value))
    assert payload["label"] == "原文" and payload["date"] == "2025-01-01T00:00:00+00:00"
    assert payload["plan"] == value["plan"].model_dump(mode="json")
