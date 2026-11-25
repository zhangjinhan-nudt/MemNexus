"""Visible extraction hints preserve identity links without becoming evidence."""

import json
from copy import deepcopy

from test_collection_source_limit import candidate, query

from agent_memory.composition import collection_sources
from agent_memory.grounding import _cost, _source_event, source_payload
from agent_memory.models import Evidence, FactMember, FactSemantics
from agent_memory.prompt_codec import evidence_catalog
from agent_memory.retrieval import Retriever, _Match


def collection_fixture():
    memory, turn = candidate(1)
    text = "Blair says Alex visited Paris and Rome."
    turn.events[0].speaker = "Blair"
    turn.events[0].content = text
    memory.content = "Alex visited Paris and Rome."
    memory.value = "Paris; Rome"
    memory.evidence[0].quote = text
    memory.semantics = FactSemantics(cardinality="many", event_key="requested-event",
        event_identity="holiday-2024", members=[
            FactMember(value=city, identity=city.casefold(), evidence=[Evidence(
                turn_id=turn.turn_id, event_index=0, quote=city)]) for city in ("Paris", "Rome")])
    return memory, turn


def test_collection_binding_survives_to_catalog_with_original_local_spans():
    memory, turn = collection_fixture()
    sources = collection_sources(query(), [memory], {turn.turn_id: turn})
    catalog = evidence_catalog(sources)
    event = catalog["sources"][0]["source_events"][0]
    hint = event["extraction_hint"]
    assert event["speaker"] == "Blair" and hint["subject"] == "Alex"
    assert hint["predicate"] == "activities"
    assert hint["event_identity"] == "holiday-2024"
    assert [member["identity"] for member in hint["members"]] == ["paris", "rome"]
    for member in hint["members"]:
        assert all(event["text"][start:end] == member["value"] for start, end in member["spans"])
    assert catalog["sources"][0]["source_id"] == 1
    assert "events" not in catalog


def test_retrieval_projection_preserves_semantics_without_mutating_memory():
    memory, turn = collection_fixture()
    before = memory.model_dump(mode="json")
    hit = Retriever._hit(_Match(memory=memory, score=1), set())
    hit.metadata["source_events"] = [_source_event(turn, 0, turn.events[0].content, 0)]
    payload = source_payload(hit)
    assert payload["source_events"][0]["extraction_hint"]["members"][1]["identity"] == "rome"
    assert memory.model_dump(mode="json") == before


def test_invisible_member_does_not_leak_through_extraction_hint():
    memory, turn = collection_fixture()
    hit = Retriever._hit(_Match(memory=memory, score=1), set())
    hit.metadata["source_events"] = [_source_event(turn, 0, "Paris", turn.events[0].content.index("Paris"))]
    event = source_payload(hit)["source_events"][0]
    assert [member["value"] for member in event["extraction_hint"]["members"]] == ["Paris"]
    assert "Rome" not in json.dumps(event)


def test_same_quote_from_another_event_cannot_create_binding():
    memory, turn = collection_fixture()
    hit = Retriever._hit(_Match(memory=memory, score=1), set())
    record = _source_event(turn, 0, "Paris", 0)
    record["turn_id"] = "different-turn"
    hit.metadata["source_events"] = [record]
    assert "extraction_hint" not in source_payload(hit)["source_events"][0]


def test_known_attribution_conflict_preserves_evidence_but_omits_hint():
    memory, turn = candidate(1)
    hit = Retriever._hit(_Match(memory=memory, score=1), set())
    hit.metadata["source_events"] = [_source_event(turn, 0, turn.events[0].content, 0)]
    hit.metadata["attribution_conflict"] = True
    event = source_payload(hit)["source_events"][0]
    assert event["text"] == turn.events[0].content
    assert "extraction_hint" not in event


def test_scalar_binding_and_all_added_fields_are_in_existing_cost():
    memory, turn = candidate(1)
    hit = Retriever._hit(_Match(memory=memory, score=1), set())
    hit.metadata["source_events"] = [_source_event(turn, 0, turn.events[0].content, 0)]
    before = deepcopy(hit.metadata)
    payload = source_payload(hit)
    assert payload["source_events"][0]["extraction_hint"]["value"] == "event 1"
    assert _cost(hit) == len(json.dumps(payload, ensure_ascii=False)) + 16
    assert hit.metadata == before
    assert source_payload(hit, compact=True)["source_events"] == payload["source_events"]


def test_unbound_raw_evidence_does_not_gain_fabricated_subject():
    memory, turn = candidate(1)
    hit = Retriever._hit(_Match(memory=memory, score=1), set())
    hit.metadata = {"source_events": [_source_event(turn, 0, turn.events[0].content, 0)]}
    assert "extraction_hint" not in source_payload(hit)["source_events"][0]
