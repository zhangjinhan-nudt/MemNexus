"""Language/direction invariants of source-bound automatic state evolution."""

import asyncio
from datetime import UTC, datetime

import pytest

from agent_memory.long_term import LongTermMemory
from agent_memory.maintenance import correction_relation
from agent_memory.models import Candidate, Event, Evidence, QueryPlan, Scope, TurnInput
from agent_memory.retrieval import Retriever
from agent_memory.store import SQLiteStore

OWNER = Scope(tenant_id="tenant", owner_id="owner")
OLD_AT = datetime(2025, 1, 1, tzinfo=UTC)
NEW_AT = datetime(2025, 2, 1, tzinfo=UTC)


@pytest.fixture
def facts(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    session = store.create_session(OWNER)

    def add(value, quote, *, at=NEW_AT, subject="Studio", predicate="opening_time", **changes):
        turn = store.append_turn(OWNER, session.session_id, TurnInput(
            request_id=f"turn-{len(store.turns(OWNER, session.session_id))}",
            events=[Event(role="user", content=quote, occurred_at=at)],
        ))
        item = Candidate(
            subject=subject, predicate=predicate, value=value, content=quote,
            durable=True, scope_type="user", semantics={"cardinality": "one"},
            evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=quote)],
            **changes,
        )
        return store.add_candidates(OWNER, session.session_id, [item])[0]

    old = add("9 AM", "Studio opens at 9 AM.", at=OLD_AT, valid_from=OLD_AT)
    new = old.model_copy(update={"value": "10 AM"})
    yield store, session, add, old, new
    store.close()


@pytest.mark.parametrize("quote", [
    "Studio changed its opening time from 9 AM to 10 AM.",
    "Studio has changed its opening time from 9 AM to 10 AM.",
    "Studio's opening time changed from 9 AM to 10 AM.",
    "Studio's opening time was changed from 9 AM to 10 AM.",
    "STUDIO changed its opening time from 9 am to 10 am.",
    "Studio changed its opening time from 9   AM to 10 AM.",
    "Studio changed its opening time from 9 AM to 10 AM. Can you remember this?",
    "Studio 将开放时间从 9 AM 改为 10 AM。",
])
def test_explicit_directional_change_is_language_and_spacing_invariant(facts, quote):
    *_, old, new = facts
    assert correction_relation(new, old, [quote]) == "supersede"


@pytest.mark.parametrize("quote", [
    "Studio changed its opening time from 10 AM to 9 AM.",
    "Studio did not change its opening time from 9 AM to 10 AM.",
    "Studio never changed its opening time from 9 AM to 10 AM.",
    "Studio would have changed its opening time from 9 AM to 10 AM.",
    "Studio may have changed its opening time from 9 AM to 10 AM.",
    "Studio will change its opening time from 9 AM to 10 AM.",
    "Studio plans to change its opening time from 9 AM to 10 AM.",
    "Studio changed its opening time from 9 AM to 10 AM if demand increased.",
    "If demand increased, Studio changed its opening time from 9 AM to 10 AM.",
    "Did Studio change its opening time from 9 AM to 10 AM?",
    "Studio changed its opening time from 9 AM to 10 AM?",
    "Studio opens at 9 AM. Gallery changed its opening time from 9 AM to 10 AM.",
    "Studio said Gallery changed its opening time from 9 AM to 10 AM.",
    "Studio's manager said Gallery changed its opening time from 9 AM to 10 AM.",
    "Studio's closing time changed from 9 AM to 10 AM.",
    "Studio changed its opening time from 19 AM to 10 AM.",
    "Studio changed its opening time from 9 AM to 110 AM.",
    "Studio opens at 10 AM; it used to open at 9 AM.",
])
def test_unknown_negative_or_reversed_transition_does_not_mutate_canonical_state(facts, quote):
    *_, old, new = facts
    assert correction_relation(new, old, [quote]) is None


@pytest.mark.parametrize("subject,old_value,new_value", [
    ("Mira", "Berlin", "Lisbon"), ("Workshop", "blue", "amber"), ("Café A", "08:30", "09:45"),
    ("May", "April", "June"), ("Studio", "available", "not available"),
])
def test_entities_and_value_types_do_not_define_the_transition(facts, subject, old_value, new_value):
    *_, old, new = facts
    old = old.model_copy(update={"subject": subject, "value": old_value})
    new = new.model_copy(update={"subject": subject, "value": new_value})
    quote = f"{subject} changed its setting from {old_value} to {new_value}."
    assert correction_relation(new, old, [quote]) == "supersede"
    assert correction_relation(old, new, [quote]) is None


@pytest.mark.parametrize("change", [
    {"predicate": "closing_time"}, {"scope_id": "someone_else"}, {"subject": "Gallery"},
    {"assertion": "planned"}, {"assertion": "hypothetical"},
])
def test_shared_quote_does_not_bridge_distinct_fact_partitions(facts, change):
    *_, old, new = facts
    new = new.model_copy(update=change)
    assert correction_relation(new, old, ["Studio changed its opening time from 9 AM to 10 AM."]) is None


def test_distinct_events_remain_separate(facts):
    *_, old, new = facts
    new = new.model_copy(update={"semantics": new.semantics.model_copy(update={"event_identity": "summer"})})
    assert correction_relation(new, old, ["Studio changed its opening time from 9 AM to 10 AM."]) is None


@pytest.mark.parametrize("quote,action", [
    ("Studio's opening time is not 9 AM but 10 AM.", "correct"),
    ("Correction: Studio's opening time was not 9 AM, but 10 AM.", "correct"),
    ("Studio's opening time is not 10 AM but 9 AM.", None),
    ("Studio's opening time is not 9 AM but 10 AM if we extend staffing.", None),
])
def test_correction_distinguishes_wrong_old_fact_from_historical_old_state(facts, quote, action):
    *_, old, new = facts
    assert correction_relation(new, old, [quote]) == action


def test_fresh_write_evolution_preserves_historical_interval_and_exact_evidence(facts):
    store, session, add, old, _ = facts
    service = LongTermMemory(store)
    asyncio.run(service.process_extraction(OWNER, [old]))
    new = add("10 AM", "Studio changed its opening time from 9 AM to 10 AM.")
    result = asyncio.run(service.process_extraction(OWNER, [new]))
    assert [item["action"] for item in result["applied"]] == ["supersede"]
    current = store.get_memory(OWNER, old.memory_id)
    previous = next(m for m in store.memories(OWNER) if m.memory_id == old.memory_id and m.version == 1)
    assert (current.value, current.version, current.tier) == ("10 AM", 2, "long")
    assert previous.value == "9 AM" and previous.valid_to == current.valid_from == NEW_AT
    assert current.evidence == new.evidence and previous.evidence == old.evidence
    assert store.get_memory(OWNER, new.memory_id).status == "superseded"
    assert Retriever._eligible(previous, session, QueryPlan(temporal_mode="history"), NEW_AT)
    assert not Retriever._eligible(previous, session, QueryPlan(), NEW_AT)
    assert Retriever._eligible(previous, session, QueryPlan(as_of=OLD_AT), NEW_AT)


def test_unknown_state_update_stays_readable_without_overwriting_old_state(facts):
    store, _, add, old, _ = facts
    service = LongTermMemory(store)
    asyncio.run(service.process_extraction(OWNER, [old]))
    new = add("10 AM", "Studio opens at 10 AM.")
    result = asyncio.run(service.process_extraction(OWNER, [new]))
    assert not result["applied"] and len(result["review"]) == 1
    retained = store.get_memory(OWNER, new.memory_id)
    assert (retained.status, retained.tier, retained.evidence) == ("active", "short", new.evidence)
    assert store.get_memory(OWNER, old.memory_id).value == "9 AM"


def test_fresh_write_correction_retracts_error_instead_of_backdating_a_change(facts):
    store, _, add, old, _ = facts
    service = LongTermMemory(store)
    asyncio.run(service.process_extraction(OWNER, [old]))
    new = add("10 AM", "Studio's opening time is not 9 AM but 10 AM.")
    result = asyncio.run(service.process_extraction(OWNER, [new]))
    assert [item["action"] for item in result["applied"]] == ["correct"]
    previous = next(m for m in store.memories(OWNER) if m.memory_id == old.memory_id and m.version == 1)
    assert previous.status == "retracted"
    assert store.get_memory(OWNER, old.memory_id).value == "10 AM"
