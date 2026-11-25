"""A collection output limit counts eligible evidence, not rejected candidates."""

from datetime import UTC, datetime

import pytest

from agent_memory.composition import collection_sources
from agent_memory.models import CollectionQuery, Event, Evidence, FactMember, FactSemantics, Memory, Turn


def candidate(index):
    when = datetime(2024, 1, index, tzinfo=UTC)
    text = f"Alex attended event {index}."
    turn = Turn(
        turn_id=f"turn-{index}", request_id=f"request-{index}", session_id="session",
        sequence=index, events=[Event(role="user", speaker="Alex", content=text, occurred_at=when)],
    )
    memory = Memory(
        memory_id=f"memory-{index}", session_id="session", scope_id="session",
        subject="Alex", predicate="activities", value=f"event {index}", content=text,
        evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=text)],
        semantics=FactSemantics(event_key="requested-event"), valid_from=when, recorded_at=when,
    )
    return memory, turn


def query():
    return CollectionQuery(subjects=["Alex"], predicate="activities", event_key="requested-event")


def test_collection_projection_preserves_explicit_member_positions():
    from agent_memory.grounding import source_payload

    memory, turn = candidate(1)
    ref = memory.evidence[0]
    ref.start, ref.end = 0, len(ref.quote)
    memory.semantics = FactSemantics(cardinality="many", event_key="requested-event",
        members=[FactMember(value="event 1", evidence=[ref.model_copy()])])
    hit, = collection_sources(query(), [memory], {turn.turn_id: turn})
    event, = hit.metadata["source_events"]
    assert turn.events[0].content[event["start"]:event["end"]] == event["text"]
    assert source_payload(hit)["source_events"][0]["extraction_hint"]["members"][0]["value"] == "event 1"


@pytest.mark.parametrize("rejection", [
    "event", "assertion", "quote", "member_quote", "missing_turn", "assistant", "question",
])
def test_rejected_candidate_does_not_consume_output_limit(rejection, monkeypatch):
    first, turn_a = candidate(1)
    second, turn_b = candidate(2)
    turns = {turn.turn_id: turn for turn in (turn_a, turn_b)}
    if rejection == "event":
        first.semantics.event_key = "different-event"
    elif rejection == "assertion":
        first.assertion = "planned"
    elif rejection == "quote":
        first.evidence[0].quote = "This assertion does not exist in the source."
    elif rejection == "member_quote":
        first.semantics = FactSemantics(
            cardinality="many", event_key="requested-event",
            members=[FactMember(value="event 1", evidence=[Evidence(
                turn_id=turn_a.turn_id, event_index=0, quote="An invented member quote.",
            )])],
        )
    elif rejection == "missing_turn":
        turns.pop(turn_a.turn_id)
    elif rejection == "assistant":
        monkeypatch.setenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "false")
        turn_a.events[0].role = "assistant"
    elif rejection == "question":
        turn_a.events[0].content = "Did Alex attend event 1?"
        first.evidence[0].quote = turn_a.events[0].content

    # The first record must still fail independently. This fix does not weaken
    # event, assertion, role, question, or complete-reference requirements.
    assert collection_sources(query(), [first], turns, limit=1) == []
    result = collection_sources(query(), [first, second], turns, limit=1)
    assert [hit.metadata["memory_id"] for hit in result] == [second.memory_id]
    assert result[0].metadata["source_events"][0]["text"] == turn_b.events[0].content


def test_output_limit_preserves_order_and_inputs_after_filtering():
    pairs = [candidate(index) for index in (1, 2, 3, 4)]
    pairs[0][0].assertion = "planned"
    memories = [memory for memory, _ in reversed(pairs)]
    turns = {turn.turn_id: turn for _, turn in pairs}
    before_memories = [memory.model_dump(mode="json") for memory in memories]
    before_turns = {key: turn.model_dump(mode="json") for key, turn in turns.items()}
    result = collection_sources(query(), memories, turns, limit=2)
    assert [hit.metadata["memory_id"] for hit in result] == ["memory-2", "memory-3"]
    assert [memory.model_dump(mode="json") for memory in memories] == before_memories
    assert {key: turn.model_dump(mode="json") for key, turn in turns.items()} == before_turns


def test_all_rejected_candidates_produce_no_source():
    pairs = [candidate(index) for index in (1, 2)]
    memories = [memory for memory, _ in pairs]
    for memory in memories:
        memory.assertion = "planned"
    assert collection_sources(query(), memories, {turn.turn_id: turn for _, turn in pairs}, limit=1) == []


def test_zero_limit_does_not_read_candidates():
    class UnreadableCandidates:
        def __iter__(self):
            raise AssertionError("A zero output budget must not scan candidates")

    assert collection_sources(query(), UnreadableCandidates(), {}, limit=0) == []


def test_negative_output_limit_is_invalid():
    with pytest.raises(ValueError, match="non-negative"):
        collection_sources(query(), [], {}, limit=-1)
