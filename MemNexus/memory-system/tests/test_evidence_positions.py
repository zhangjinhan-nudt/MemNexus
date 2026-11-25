"""Literal Unicode positions survive writes without inventing ambiguous anchors."""

import pytest

from agent_memory.models import Candidate, Event, Evidence, FactMember, FactSemantics, Scope, TurnInput
from agent_memory.store import SQLiteStore


@pytest.fixture
def storage(tmp_path):
    store = SQLiteStore(tmp_path / "positions.db")
    scope = Scope(tenant_id="test", owner_id="reader")
    session = store.create_session(scope)
    yield store, scope, session
    store.close()


def source(storage, text):
    store, scope, session = storage
    return store.append_turn(scope, session.session_id, TurnInput(
        request_id="r", events=[Event(role="user", content=text)]))


def candidate(turn, quote, **positions):
    return Candidate(content="A recorded preference.", subject="Alex", predicate="likes", value=quote,
                     evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=quote, **positions)])


def test_unique_legacy_quote_gets_unicode_coordinates_on_new_write(storage):
    store, scope, session = storage
    turn = source(storage, "前言🌱：我喜欢蓝色。")
    value = candidate(turn, "蓝色")
    saved = store.add_candidates(scope, session.session_id, [value])[0]
    ref = saved.evidence[0]
    assert ref.start == turn.events[0].content.index("蓝色")
    assert ref.end == ref.start + 2
    assert turn.events[0].content[ref.start:ref.end] == ref.quote
    assert store.memories(scope)[0].evidence[0] == ref


@pytest.mark.parametrize("text,quote", [("blue then blue", "blue"), ("aaa", "aa")])
def test_repeated_legacy_quote_does_not_invent_first_occurrence(storage, text, quote):
    store, scope, session = storage
    turn = source(storage, text)
    saved = store.add_candidates(scope, session.session_id, [candidate(turn, quote)])[0]
    assert saved.evidence[0].start is None
    assert saved.evidence[0].end is None


def test_explicit_second_occurrence_and_first_remain_distinct_when_merged(storage):
    store, scope, session = storage
    turn = source(storage, "blue then blue")
    store.add_candidates(scope, session.session_id, [candidate(turn, "blue", start=0, end=4)])
    saved = store.add_candidates(scope, session.session_id, [candidate(turn, "blue", start=10, end=14)])[0]
    assert {(e.start, e.end) for e in saved.evidence} == {(0, 4), (10, 14)}
    assert len(store.memories(scope)) == 1


def test_wrong_positions_rejected_even_when_quote_occurs_elsewhere(storage):
    store, scope, session = storage
    turn = source(storage, "blue then blue")
    with pytest.raises(ValueError, match="positions do not match"):
        store.add_candidates(scope, session.session_id, [candidate(turn, "blue", start=5, end=9)])
    assert not store.memories(scope)


@pytest.mark.parametrize("positions", [{"start": 1}, {"end": 4}, {"start": 3, "end": 2},
                                      {"start": 0, "end": 3}, {"start": True, "end": 5}])
def test_coordinate_schema_rejects_partial_or_nonliteral_ranges(positions):
    with pytest.raises(ValueError):
        Evidence(turn_id="t", event_index=0, quote="blue", **positions)


def test_member_evidence_uses_the_same_position_validation(storage):
    store, scope, session = storage
    turn = source(storage, "I like blue and green.")
    value = candidate(turn, turn.events[0].content)
    value.semantics = FactSemantics(cardinality="many", members=[FactMember(value="green", evidence=[
        Evidence(turn_id=turn.turn_id, event_index=0, quote="green", start=0, end=5)])])
    with pytest.raises(ValueError, match="positions do not match"):
        store.add_candidates(scope, session.session_id, [value])
    assert not store.memories(scope)
    value.semantics.members[0].evidence[0] = Evidence(turn_id=turn.turn_id, event_index=0, quote="green")
    saved = store.add_candidates(scope, session.session_id, [value])[0]
    ref = saved.semantics.members[0].evidence[0]
    assert (ref.start, ref.end) == (16, 21)


def test_old_serialized_reference_remains_loadable():
    ref = Evidence.model_validate_json('{"turn_id":"old","event_index":0,"quote":"blue"}')
    assert (ref.start, ref.end) == (None, None)
def test_generator_is_not_asked_to_compute_source_offsets():
    import asyncio
    import json

    from agent_memory.models import ExtractionResult
    from agent_memory.providers import CallableModel

    prompts = []
    def chat(prompt):
        prompts.append(prompt)
        return '{"candidates": []}'

    asyncio.run(CallableModel(chat)._structured("Extract facts.", ExtractionResult))
    schema = json.loads(prompts[0].split("support:\n", 1)[1])
    props = schema["$defs"]["Evidence"]["properties"]
    assert "quote" in props and "turn_id" in props
    assert "start" not in props and "end" not in props

