"""Persistence must preserve assertion strength and every original source."""
import asyncio

from agent_memory.long_term import LongTermMemory
from agent_memory.models import Candidate, Event, Evidence, Scope, TurnInput
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.store import SQLiteStore


def add(store, scope, session, index, *, wording=False, assertion="stated"):
    quote = "I plan to study astronomy." if assertion == "planned" else "My favorite subject is astronomy."
    turn = store.append_turn(scope, session.session_id, TurnInput(
        request_id=str(index), events=[Event(role="user", content=quote)]))
    return store.add_candidates(scope, session.session_id, [Candidate(
        subject="user", predicate="subject", value="astronomy", content=quote + (f" ({index})" if wording else ""),
        assertion=assertion, durable=True, scope_type="user", semantics={"cardinality": "one"},
        evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=quote)],
    )])[0]


def test_durable_plan_promotes_without_becoming_completed(tmp_path):
    store = SQLiteStore(tmp_path / "plans.db")
    scope = Scope(tenant_id="t", owner_id="u")
    session = store.create_session(scope)
    memory = add(store, scope, session, 1, assertion="planned")
    asyncio.run(LongTermMemory(store).process_extraction(scope, [memory]))
    retained = store.get_memory(scope, memory.memory_id)
    assert retained.tier == "long"
    assert retained.assertion == "planned" and retained.evidence == memory.evidence
    store.close()


def test_unresolved_conflict_preserves_facts_without_unused_model_call(tmp_path):
    class UnusedModel:
        async def relate(self, *args):
            raise AssertionError("Unconsumed relation proposals must not call a model")

    store = SQLiteStore(tmp_path / "conflict.db")
    scope = Scope(tenant_id="t", owner_id="u")
    session = store.create_session(scope)
    service = LongTermMemory(store, model=UnusedModel())
    first = add(store, scope, session, 1)
    asyncio.run(service.process_extraction(scope, [first]))
    quote = "My favorite subject is botany."
    turn = store.append_turn(scope, session.session_id, TurnInput(request_id="second", events=[Event(role="user",content=quote)]))
    second = store.add_candidates(scope,session.session_id,[Candidate(subject="user",predicate="subject",value="botany",content=quote,
        scope_type="user",durable=True,semantics={"cardinality":"one"},evidence=[Evidence(turn_id=turn.turn_id,event_index=0,quote=quote)])])[0]
    result = asyncio.run(service.process_extraction(scope,[second]))
    assert result["review"] and result["warnings"] == [] and result["applied"] == []
    assert {(m.value,m.tier,m.status) for m in store.memories(scope)} == {
        ("astronomy","long","active"),("botany","short","active")}
    store.close()


def test_direct_dedup_keeps_overflow_sources_as_separate_records(tmp_path):
    store = SQLiteStore(tmp_path / "dedup.db")
    scope = Scope(tenant_id="t", owner_id="u")
    session = store.create_session(scope)
    for index in range(32):
        add(store, scope, session, index)
    active = [m for m in store.memories(scope) if m.status == "active"]
    assert len({e.turn_id for m in active for e in m.evidence}) == 32
    assert all(len(m.evidence) <= 30 for m in active)
    store.close()


def test_long_term_merge_and_maintenance_preserve_overflow_sources(tmp_path):
    store = SQLiteStore(tmp_path / "merge.db")
    scope = Scope(tenant_id="t", owner_id="u")
    session = store.create_session(scope)
    service = LongTermMemory(store)
    for index in range(32):
        memory = add(store, scope, session, index, wording=True)
        report = asyncio.run(service.process_extraction(scope, [memory]))
        assert not report["warnings"]
    asyncio.run(service.maintain(scope))
    active = [m for m in store.memories(scope) if m.status == "active"]
    assert len({e.turn_id for m in active for e in m.evidence}) == 32
    assert all(len(m.evidence) <= 30 for m in active)
    store.close()


def test_references_keep_distinct_past_events_and_plan_state(tmp_path):
    from datetime import UTC, datetime

    from agent_memory.models import EvolutionInput

    async def run():
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "refs.db", maintenance_interval=0))
        scope = Scope(tenant_id="t", owner_id="u")
        session = runtime.store.create_session(scope)
        for index, city in enumerate(("Lisbon", "Rome")):
            quote = f"I visited {city}."
            turn = runtime.store.append_turn(scope, session.session_id, TurnInput(
                request_id=str(index), events=[Event(role="user", content=quote)]))
            memory = runtime.store.add_candidates(scope, session.session_id, [Candidate(
                subject="user", predicate="visited", value=city, content=quote,
                scope_type="user", durable=True,
                valid_to=datetime(2024, 1, 1, tzinfo=UTC),
                semantics={"event_key": city, "event_identity": f"visit to {city}"},
                evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=quote)],
            )])[0]
            runtime.store.evolve(scope, memory.memory_id, EvolutionInput(action="promote", expected_version=1))
        planned = add(runtime.store, scope, session, 10, assertion="planned")
        runtime.store.evolve(scope, planned.memory_id, EvolutionInput(action="promote", expected_version=1))
        current = runtime.store.create_session(scope)
        runtime.store.append_turn(scope, current.session_id, TurnInput(request_id="new", events=[
            Event(role="user", content="I visited Lisbon and Rome; I also discussed my astronomy plan.")]))
        refs = (await runtime.short_term.window(scope, current.session_id)).reference_fields
        assert {r.value for r in refs} >= {"Lisbon", "Rome", "astronomy"}
        assert next(r for r in refs if r.value == "astronomy").assertion == "planned"
        assert all(r.valid_to is not None for r in refs if r.value in {"Lisbon", "Rome"})
        await runtime.close()

    asyncio.run(run())
