"""Mechanism checks, not fabricated benchmark accuracy. Fixtures are explicit test data."""

import asyncio
from datetime import UTC, datetime

import pytest

from agent_memory.composition import collection_sources
from agent_memory.models import (
    Candidate,
    CollectionQuery,
    Event,
    Evidence,
    EvolutionInput,
    ExtractionResult,
    FactMember,
    FactSemantics,
    QueryPlan,
    Scope,
    TurnInput,
    utcnow,
)
from agent_memory.ports import ConflictError
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.store import SQLiteStore
from agent_memory.synthesis import AnswerTable


def add(
    runtime,
    scope,
    session,
    values,
    *,
    subject="Alex",
    predicate="hobbies",
    operation="add",
    day=1,
    durable=False,
    cardinality="many",
    identity=False,
):
    text = f"{subject}: {operation} {', '.join(values)}."
    when = datetime(2024, 1, day, tzinfo=UTC)
    turn = runtime.store.append_turn(
        scope,
        session.session_id,
        TurnInput(
            request_id=f"r-{len(runtime.store.turns(scope, session.session_id))}",
            events=[Event(role="user", speaker=subject, content=text, occurred_at=when)],
        ),
    )
    evidence = [Evidence(turn_id=turn.turn_id, event_index=0, quote=text)]
    semantics = FactSemantics(
        cardinality=cardinality,
        operation=operation if cardinality == "many" else "assert",
        members=[
            FactMember(value=v, evidence=evidence, identity=f"{subject}:{v}" if identity else None)
            for v in values
        ]
        if cardinality == "many"
        else [],
    )
    candidate = Candidate(
        subject=subject,
        predicate=predicate,
        value=", ".join(values) or "empty set",
        content=text,
        evidence=evidence,
        semantics=semantics,
        valid_from=when,
        durable=durable,
        scope_type="user" if durable else "session",
    )
    latest = runtime.store.get_session(scope, session.session_id)
    memories = runtime.store.commit_extraction(
        scope, latest, [turn], ExtractionResult(candidates=[candidate])
    )
    return memories[0], turn


@pytest.fixture
def env(tmp_path):
    runtime = MemoryRuntime(Settings(db_path=tmp_path / "memory.db", maintenance_interval=0))
    scope = Scope(tenant_id="tenant", owner_id="owner")
    session = runtime.store.create_session(scope)
    yield runtime, scope, session
    asyncio.run(runtime.close())


def selected_sources(runtime, scope, session, query, *, as_of=None):
    plan = QueryPlan(collection_query=query, as_of=as_of)
    memories = [memory for memory in runtime.store.collection_memories(scope, query)
                if runtime.retriever._eligible(memory, session, plan, utcnow())]
    references = [ref.turn_id for memory in memories
                  for ref in memory.evidence + [e for member in memory.semantics.members for e in member.evidence]]
    turns = runtime.store.source_turns(scope, references)
    return collection_sources(query, memories, turns)


def test_many_valued_long_term_records_coexist(env):
    runtime, scope, session = env
    first, _ = add(runtime, scope, session, ["swimming"], durable=True)
    second, _ = add(runtime, scope, session, ["chess"], durable=True, day=2)
    asyncio.run(runtime.long_term.process_extraction(scope, [first, second]))
    memories = runtime.store.memories(scope)
    assert {m.memory_id for m in memories} == {first.memory_id, second.memory_id}
    assert all(m.status == "active" and m.tier == "long" for m in memories)
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    assert {hit.metadata["value"] for hit in selected_sources(runtime, scope, session, query)} == {
        "swimming", "chess",
    }


def test_member_operations_retraction_and_historical_source_eligibility(env):
    runtime, scope, session = env
    original, _ = add(runtime, scope, session, ["swimming", "chess"])
    removal, _ = add(runtime, scope, session, ["swimming"], operation="remove", day=2)
    replacement, _ = add(runtime, scope, session, ["drawing"], operation="replace_set", day=3)
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")

    def ids(**kwargs):
        return {hit.metadata["memory_id"] for hit in selected_sources(runtime, scope, session, query, **kwargs)}

    assert ids() == {original.memory_id, removal.memory_id, replacement.memory_id}
    assert ids(as_of=datetime(2024, 1, 1, 12, tzinfo=UTC)) == {original.memory_id}
    assert ids(as_of=datetime(2024, 1, 2, 12, tzinfo=UTC)) == {original.memory_id, removal.memory_id}
    runtime.store.evolve(scope, removal.memory_id, EvolutionInput(action="retract", expected_version=1))
    assert ids() == {original.memory_id, replacement.memory_id}
    runtime.store.evolve(scope, replacement.memory_id, EvolutionInput(action="retract", expected_version=1))
    assert ids() == {original.memory_id}
    stored = {m.memory_id: m for m in runtime.store.memories(scope)}
    assert stored[removal.memory_id].semantics.operation == "remove"
    assert stored[replacement.memory_id].semantics.operation == "replace_set"
    assert stored[removal.memory_id].status == stored[replacement.memory_id].status == "retracted"


def test_explicit_empty_snapshot_retains_its_source_and_canonical_operation(env):
    runtime, scope, session = env
    original, _ = add(runtime, scope, session, ["swimming"])
    empty, turn = add(runtime, scope, session, [], operation="replace_set", day=2)
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    sources = selected_sources(runtime, scope, session, query)
    assert {h.metadata["memory_id"] for h in sources} == {original.memory_id, empty.memory_id}
    source = next(h for h in sources if h.metadata["memory_id"] == empty.memory_id)
    assert source.metadata["source_events"][0]["text"] == turn.events[0].content
    assert empty.semantics.operation == "replace_set" and not empty.semantics.members
    assert not selected_sources(runtime, scope, session, query.model_copy(update={"predicate": "pets"}))


def test_collection_sources_keep_original_order_quotes_and_limit(env):
    runtime, scope, session = env
    first, turn_a = add(runtime, scope, session, ["swimming"])
    second, turn_b = add(runtime, scope, session, ["chess"], day=2)
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    turns = {turn.turn_id: turn for turn in [turn_a, turn_b]}
    sources = collection_sources(query, [second, first], turns, limit=1)
    assert [h.metadata["memory_id"] for h in sources] == [first.memory_id]
    assert sources[0].metadata["source_events"][0]["text"] == turn_a.events[0].content
    assert len(sources[0].metadata["source_events"]) == 1  # repeated member refs retain one original span
    assert not collection_sources(query, [first, second], turns, limit=0)
    assert [h.metadata["memory_id"] for h in collection_sources(
        query, [first, second], turns
    )] == [first.memory_id, second.memory_id]


def test_collection_sources_require_every_reference_and_respect_evidence_policy(env, monkeypatch):
    runtime, scope, session = env
    memory, turn = add(runtime, scope, session, ["swimming"])
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    assert not collection_sources(query, [memory], {})
    corrupted = memory.model_copy(deep=True)
    corrupted.semantics.members[0].evidence[0].quote = "invented quote"
    assert not collection_sources(query, [corrupted], {turn.turn_id: turn})
    assistant = turn.model_copy(deep=True)
    assistant.events[0].role = "assistant"
    monkeypatch.setenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "false")
    assert not collection_sources(query, [memory], {turn.turn_id: assistant})
    monkeypatch.setenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "true")
    assert collection_sources(query, [memory], {turn.turn_id: assistant})
    question = turn.model_copy(deep=True)
    question.events[0].content = "Does Alex like swimming?"
    question_memory = memory.model_copy(deep=True)
    for ref in question_memory.evidence + question_memory.semantics.members[0].evidence:
        ref.quote = question.events[0].content
    assert not collection_sources(query, [question_memory], {turn.turn_id: question})


def test_collection_sources_respect_event_and_assertion_filters(env):
    runtime, scope, session = env
    memory, turn = add(runtime, scope, session, ["swimming"])
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies", event_key="competition-a")
    assert not collection_sources(query, [memory], {turn.turn_id: turn})
    memory.semantics.event_key = "competition-b"
    assert not collection_sources(query, [memory], {turn.turn_id: turn})
    memory.semantics.event_key = "competition-a"
    assert collection_sources(query, [memory], {turn.turn_id: turn})
    memory.assertion = "planned"
    assert not collection_sources(query, [memory], {turn.turn_id: turn})
    assert collection_sources(query.model_copy(update={"assertions": ["planned"]}),
                              [memory], {turn.turn_id: turn})


def test_collection_relation_partition_still_requires_session_authorization(env):
    runtime, scope, session = env
    own, _ = add(runtime, scope, session, ["swimming"])
    other_session = runtime.store.create_session(scope)
    add(runtime, scope, other_session, ["chess"])
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    assert len(runtime.store.collection_memories(scope, query)) == 2
    assert [h.metadata["memory_id"] for h in selected_sources(runtime, scope, session, query)] == [own.memory_id]


def test_unknown_values_do_not_become_scalar_conflicts(env):
    runtime, scope, session = env
    first, _ = add(runtime, scope, session, ["swimming"], durable=True, cardinality="unknown")
    second, _ = add(runtime, scope, session, ["chess"], durable=True, cardinality="unknown", day=2)
    asyncio.run(runtime.long_term.process_extraction(scope, [first, second]))
    assert all(m.status == "active" for m in runtime.store.memories(scope))


def test_projection_is_owner_scoped_rebuilt_on_upgrade_and_deleted(env):
    runtime, scope, session = env
    add(runtime, scope, session, ["swimming"])
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    other = Scope(tenant_id="tenant", owner_id="other")
    assert not runtime.store.collection_memories(other, query)
    runtime.store._db.execute("DELETE FROM fact_projection")
    runtime.store._db.execute("PRAGMA user_version=1")
    runtime.store._db.commit()
    reopened = SQLiteStore(runtime.settings.db_path)
    try:
        assert len(reopened.collection_memories(scope, query)) == 1
        reopened.delete_owner(scope)
        assert not reopened.collection_memories(scope, query)
    finally:
        reopened.close()


def test_collection_member_cannot_smuggle_an_unverified_quote(env):
    runtime, scope, session = env
    memory, _ = add(runtime, scope, session, ["swimming"])
    corrupted = Candidate.model_validate(memory.model_dump(include=set(Candidate.model_fields)))
    corrupted.semantics.members[0].evidence[0].quote = "not present in the source"
    with pytest.raises(ValueError):
        runtime.store.add_candidates(scope, session.session_id, [corrupted])


class CollectionAnswerModel:
    def __init__(self, on_synthesis=None):
        self.on_synthesis = on_synthesis
        self.calls = 0
        self.sources = []

    async def plan(self, question, context):
        return QueryPlan(collection_query=CollectionQuery(subjects=["Alex"], predicate="hobbies"))

    async def synthesize(self, question, plan, sources, **kwargs):
        self.calls += 1
        self.sources = sources
        if self.on_synthesis:
            self.on_synthesis()
        source_id, quote = next((i, event["text"]) for i, source in enumerate(sources, 1)
                                for event in source.metadata["source_events"]
                                if "swimming" in event["text"])
        return AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": question, "values": [{
            "value": "swimming", "supports": [{"source_id": source_id, "quote": quote}]
        }]}]})

    async def compose(self, *args, **kwargs):
        raise AssertionError("The retired collection answer route must not run")

    async def bind_answer(self, *args, **kwargs):
        raise AssertionError("The retired semantic answer route must not run")

    async def answer(self, *args, **kwargs):
        raise AssertionError("A structured adapter must use synthesis")


def test_collection_retrieval_feeds_the_single_synthesis_answer_chain(env, monkeypatch):
    runtime, scope, session = env
    add(runtime, scope, session, ["swimming", "chess"])
    model = CollectionAnswerModel()
    runtime.model = runtime.retriever.model = model

    def reject_unused_graph(*args, **kwargs):
        raise AssertionError("Structured synthesis must not build an unused answer graph")

    monkeypatch.setattr("agent_memory.runtime.EventGraph.from_hits", reject_unused_graph)
    answer = asyncio.run(runtime.answer(scope, session.session_id, "What hobbies does Alex have?",
                                        prepare=False, accelerate=False, top_k=1))
    assert model.calls == answer.model_calls.answer_calls == 1
    assert model.sources and any("chess" in e["text"] for h in model.sources
                                 for e in h.metadata["source_events"])
    assert answer.generated_text == "swimming"
    assert answer.reasoning["answer_path"] == "source_bound_synthesis"
    assert "answer_ir" not in answer.reasoning and "bound_answer" not in answer.reasoning
    assert answer.reasoning["source_count"] == len(answer.sources)
    assert answer.reasoning["source_event_count"] == sum(len(h.metadata["source_events"]) for h in answer.sources)
    assert not {"graph_nodes", "graph_edges", "decided_order", "typed_negative_gate"} & answer.reasoning.keys()
    assert answer.citations
    receipt = runtime.store.get_answer_receipt(scope, session.session_id, answer.answer_id)
    assert receipt.reasoning == answer.reasoning


def test_collection_synthesis_rechecks_snapshot_before_publish(env):
    runtime, scope, session = env
    memory, _ = add(runtime, scope, session, ["swimming"])
    model = CollectionAnswerModel(lambda: runtime.store.evolve(
        scope, memory.memory_id, EvolutionInput(action="retract", expected_version=1)))
    runtime.model = runtime.retriever.model = model
    with pytest.raises(ConflictError, match="Memory changed"):
        asyncio.run(runtime.answer(scope, session.session_id, "What hobbies does Alex have?",
                                   prepare=False, accelerate=False))


def test_answer_only_adapter_remains_compatible_with_collection_plans(env):
    runtime, scope, session = env
    add(runtime, scope, session, ["swimming"])

    class TextAdapter:
        async def plan(self, question, context):
            return QueryPlan(collection_query=CollectionQuery(subjects=["Alex"], predicate="hobbies"))

        async def answer(self, prompt):
            assert "swimming" in prompt
            assert "STRUCTURED EVENT GRAPH" in prompt
            assert "ORDER IS DETERMINED" not in prompt
            return "swimming"

    runtime.model = runtime.retriever.model = TextAdapter()
    answer = asyncio.run(runtime.answer(scope, session.session_id, "What hobbies does Alex have?",
                                        prepare=False, accelerate=False))
    assert answer.generated_text == "swimming"
    assert answer.model_calls.answer_calls == 1
    assert "answer_ir" not in answer.reasoning
    assert answer.reasoning["graph_nodes"] > 0
