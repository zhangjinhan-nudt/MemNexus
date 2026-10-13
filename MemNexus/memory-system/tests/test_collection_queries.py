"""Mechanism checks, not fabricated benchmark accuracy. Fixtures are explicit test data."""

import asyncio
from datetime import UTC, datetime

import pytest

from agent_memory.composition import (
    AnswerAtom,
    AnswerIR,
    CollectionExtraction,
    Coverage,
    adjudicated_answer,
    execute,
    indexed_answer,
    render,
    validate_extraction,
)
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
)
from agent_memory.ports import ConflictError
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.store import SQLiteStore


class QueryModel:
    def __init__(self, query, **plan_fields):
        self.query, self.fields = query, plan_fields

    async def plan(self, question, context):
        return QueryPlan(collection_query=self.query, **self.fields)

    async def answer(self, prompt):
        raise AssertionError("Typed collection answers must not call a free-form writer")


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


def ask(runtime, scope, session, query, **plan_fields):
    runtime.model = runtime.retriever.model = QueryModel(query, **plan_fields)
    return asyncio.run(
        runtime.answer(
            scope, session.session_id, "List the requested items.", prepare=False, accelerate=False, top_k=1
        )
    )


def values(answer):
    return {item["value"] for item in answer.reasoning["answer_ir"]["items"]}


def test_group_preserves_all_members_and_bypasses_top_k_writer(env):
    runtime, scope, session = env
    add(runtime, scope, session, ["swimming", "running", "chess"])
    add(runtime, scope, session, ["drawing", "climbing"], subject="Blair")
    query = CollectionQuery(subjects=["Alex", "Blair"], predicate="hobbies", operation="group")
    answer = ask(runtime, scope, session, query)
    assert values(answer) == {"swimming", "running", "chess", "drawing", "climbing"}
    assert answer.model_calls.answer_calls == 0 and answer.model_calls.plan_calls == 1
    assert "Alex:" in answer.generated_text and "Blair:" in answer.generated_text
    assert answer.reasoning["answer_ir"]["coverage"]["output_complete"]
    assert answer.reasoning["answer_ir"]["coverage"]["extraction_completeness"] == "not_proven"
    receipt = runtime.store.get_answer_receipt(scope, session.session_id, answer.answer_id)
    assert receipt.reasoning == answer.reasoning


def test_intersection_keeps_both_premises_and_duplicate_mentions(env):
    runtime, scope, session = env
    add(runtime, scope, session, ["swimming", "chess"])
    add(runtime, scope, session, ["swimming"], day=2)
    add(runtime, scope, session, ["swimming", "drawing"], subject="Blair")
    answer = ask(
        runtime,
        scope,
        session,
        CollectionQuery(subjects=["Alex", "Blair"], predicate="hobbies", operation="intersection"),
    )
    assert values(answer) == {"swimming"}
    item = answer.reasoning["answer_ir"]["items"][0]
    assert {p["subject"] for p in item["premises"]} == {"Alex", "Blair"}
    assert len(answer.citations) == 3


def test_member_remove_replace_retract_and_historical_query(env):
    runtime, scope, session = env
    add(runtime, scope, session, ["swimming", "chess"])
    removal, _ = add(runtime, scope, session, ["swimming"], operation="remove", day=2)
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    assert values(ask(runtime, scope, session, query)) == {"chess"}
    historical = ask(runtime, scope, session, query, as_of=datetime(2024, 1, 1, 12, tzinfo=UTC))
    assert values(historical) == {"swimming", "chess"}
    runtime.store.evolve(scope, removal.memory_id, EvolutionInput(action="retract", expected_version=1))
    assert values(ask(runtime, scope, session, query)) == {"swimming", "chess"}
    replacement, _ = add(runtime, scope, session, ["drawing"], operation="replace_set", day=3)
    assert values(ask(runtime, scope, session, query)) == {"drawing"}
    runtime.store.evolve(scope, replacement.memory_id, EvolutionInput(action="retract", expected_version=1))
    assert values(ask(runtime, scope, session, query)) == {"swimming", "chess"}


def test_many_valued_long_term_records_coexist(env):
    runtime, scope, session = env
    first, _ = add(runtime, scope, session, ["swimming"], durable=True)
    second, _ = add(runtime, scope, session, ["chess"], durable=True, day=2)
    asyncio.run(runtime.long_term.process_extraction(scope, [first, second]))
    assert all(m.status == "active" and m.tier == "long" for m in runtime.store.memories(scope))
    assert values(ask(runtime, scope, session, CollectionQuery(subjects=["Alex"], predicate="hobbies"))) == {
        "swimming",
        "chess",
    }


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


def test_count_uses_identities_not_mentions(env):
    runtime, scope, session = env
    add(runtime, scope, session, ["Max", "Ned"], predicate="pets", identity=True)
    add(runtime, scope, session, ["Max"], predicate="pets", identity=True, day=2)
    answer = ask(
        runtime,
        scope,
        session,
        CollectionQuery(subjects=["Alex"], predicate="pets", operation="count", count_unit="entity"),
    )
    assert values(answer) == {"2"}


def test_missing_identity_and_budget_do_not_produce_exact_counts(env):
    runtime, scope, session = env
    memory, turn = add(runtime, scope, session, ["Max", "Ned"], predicate="pets")
    query = CollectionQuery(subjects=["Alex"], predicate="pets", operation="count", count_unit="entity")
    ir, _ = indexed_answer(query, [memory], {turn.turn_id: turn}, snapshot=(0, 0), pending=0)
    assert not ir.items and "exact_count_not_established" in ir.coverage.unresolved
    limited, _ = indexed_answer(query, [memory], {turn.turn_id: turn}, snapshot=(0, 0), pending=0, limit=0)
    assert limited.coverage.budget_exhausted and not limited.items


def test_empty_intersection_does_not_assert_absence():
    ir = AnswerIR(
        query=CollectionQuery(subjects=["Alex", "Blair"], predicate="hobbies", operation="intersection"),
        atoms=[AnswerAtom(subject="Alex", value="chess", source_ids=[1])],
    )
    execute(ir)
    assert "does not prove absence" in render(ir)
    assert not ir.coverage.enumeration_complete


def test_source_adjudication_is_complete_only_relative_to_selected_sources(env):
    runtime, scope, session = env
    memory, turn = add(runtime, scope, session, ["relief", "hope", "worry"], predicate="emotions")
    query = CollectionQuery(subjects=["Alex"], predicate="emotions")
    _, sources = indexed_answer(query, [memory], {turn.turn_id: turn}, snapshot=(0, 0), pending=0)
    result = CollectionExtraction(
        sources=[
            {
                "source_id": 1,
                "reason": "three explicitly listed emotions",
                "candidates": [
                    {
                        "subject": "Alex",
                        "value": v,
                        "quote": memory.content,
                        "status": "accepted",
                        "reason": "explicit member",
                    }
                    for v in ["relief", "hope", "worry"]
                ],
            }
        ]
    )
    ir = adjudicated_answer(query, result, sources, snapshot=(0, 0), pending=0)
    assert len(ir.items) == 3 and all(v in render(ir) for v in ["relief", "hope", "worry"])
    assert not ir.coverage.enumeration_complete and ir.coverage.output_complete
    with pytest.raises(ValueError, match="exactly once"):
        validate_extraction(CollectionExtraction(sources=[]), query, sources)
    result.sources[0].candidates[0].quote = "invented quote"
    with pytest.raises(ValueError, match="original"):
        validate_extraction(result, query, sources)


def test_direct_count_statement_does_not_require_enumerating_names():
    ir = AnswerIR(
        query=CollectionQuery(subjects=["Alex"], predicate="pets", operation="count", count_unit="entity"),
        atoms=[AnswerAtom(subject="Alex", value="three pets", quantity=3, source_ids=[1])],
        coverage=Coverage(enumeration_complete=False),
    )
    assert execute(ir).items[0].value == "3"


def test_distinct_same_named_entities_are_not_merged():
    ir = AnswerIR(
        query=CollectionQuery(subjects=["Alex"], predicate="pets", operation="count", count_unit="entity"),
        atoms=[
            AnswerAtom(subject="Alex", value="Max", identity=identity, source_ids=[1])
            for identity in ["pet-1", "pet-2", "pet-1"]
        ],
        coverage=Coverage(enumeration_complete=True),
    )
    assert execute(ir).items[0].value == "2"


def test_event_qualifier_prevents_cross_event_merging():
    query = CollectionQuery(subjects=["Alex"], predicate="emotions", event_key="submission-a")
    ir = AnswerIR(
        query=query,
        atoms=[
            AnswerAtom(subject="Alex", value="hope", event_key="submission-a", source_ids=[1]),
            AnswerAtom(subject="Alex", value="fear", event_key="submission-b", source_ids=[2]),
        ],
    )
    assert {item.value for item in execute(ir).items} == {"hope"}


def test_event_count_uses_occurrence_identity_not_topic_or_label():
    query = CollectionQuery(subjects=["Alex"], predicate="trips", operation="count", count_unit="event")
    ir = AnswerIR(
        query=query,
        coverage=Coverage(enumeration_complete=True),
        atoms=[
            AnswerAtom(
                subject="Alex", value="Paris", event_key="holiday", event_identity=identity, source_ids=[1]
            )
            for identity in ["trip-1", "trip-2", "trip-1"]
        ],
    )
    assert execute(ir).items[0].value == "2"
    for atom in ir.atoms:
        atom.event_identity = None
    ir.coverage.unresolved.clear()
    assert not execute(ir).items


def test_semantic_normalization_keeps_intersection_proof_and_original_values(env):
    runtime, scope, session = env
    first, turn_a = add(runtime, scope, session, ["distributing food"], predicate="service")
    second, turn_b = add(runtime, scope, session, ["food distribution"], predicate="service", subject="Blair")
    query = CollectionQuery(subjects=["Alex", "Blair"], predicate="service", operation="intersection")
    _, sources = indexed_answer(
        query, [first, second], {t.turn_id: t for t in [turn_a, turn_b]}, snapshot=(0, 0), pending=0
    )
    decisions = CollectionExtraction(
        sources=[
            {
                "source_id": i,
                "reason": "fixture equivalent activity labels",
                "candidates": [
                    {
                        "subject": m.subject,
                        "value": m.value,
                        "normalized_value": "food distribution",
                        "quote": m.content,
                        "status": "accepted",
                        "reason": "same activity",
                    }
                ],
            }
            for i, m in enumerate([first, second], 1)
        ]
    )
    ir = adjudicated_answer(query, decisions, sources, snapshot=(0, 0), pending=0)
    assert len(ir.items) == 1 and len(ir.items[0].premises) == 2
    assert ir.normalizations[0]["original_value"] == "distributing food"


def test_source_adjudication_runtime_rechecks_snapshot_before_publish(env):
    runtime, scope, session = env
    memory, _ = add(runtime, scope, session, ["swimming"], cardinality="unknown")
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")

    class MutatingModel(QueryModel):
        async def compose(self, spec, sources, question):
            runtime.store.evolve(
                scope, memory.memory_id, EvolutionInput(action="retract", expected_version=1)
            )
            return CollectionExtraction(
                sources=[
                    {"source_id": i, "reason": "no accepted items", "candidates": []}
                    for i in range(1, len(sources) + 1)
                ]
            )

    runtime.model = runtime.retriever.model = MutatingModel(query)
    with pytest.raises(ConflictError, match="Memory changed"):
        asyncio.run(
            runtime.answer(scope, session.session_id, "What hobbies?", prepare=False, accelerate=False)
        )


def test_collection_member_cannot_smuggle_an_unverified_quote(env):
    runtime, scope, session = env
    memory, _ = add(runtime, scope, session, ["swimming"])
    corrupted = Candidate.model_validate(memory.model_dump(include=set(Candidate.model_fields)))
    corrupted.semantics.members[0].evidence[0].quote = "not present in the source"
    with pytest.raises(ValueError):
        runtime.store.add_candidates(scope, session.session_id, [corrupted])


def test_explicit_empty_snapshot_is_distinct_from_no_retrieval_hit(env):
    runtime, scope, session = env
    add(runtime, scope, session, ["swimming"])
    add(runtime, scope, session, [], operation="replace_set", day=2)
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    answer = ask(runtime, scope, session, query)
    assert values(answer) == {"none"}
    assert answer.reasoning["answer_ir"]["items"][0]["rule"] == "explicit_empty"
    count = ask(runtime, scope, session, query.model_copy(update={"operation": "count"}))
    assert values(count) == {"0"}


def test_conflicting_explicit_counts_are_not_silently_merged():
    ir = AnswerIR(
        query=CollectionQuery(subjects=["Alex"], predicate="pets", operation="count"),
        atoms=[AnswerAtom(subject="Alex", value="pets", quantity=q, source_ids=[1]) for q in [2, 3]],
        coverage=Coverage(enumeration_complete=True),
    )
    assert not execute(ir).items
    assert "conflicting_explicit_totals" in ir.coverage.unresolved


def test_rejected_other_subject_is_a_valid_audit_record_but_not_an_answer(env):
    runtime, scope, session = env
    memory, turn = add(runtime, scope, session, ["swimming"], subject="Blair")
    query = CollectionQuery(subjects=["Alex"], predicate="hobbies")
    _, sources = indexed_answer(query, [memory], {turn.turn_id: turn}, snapshot=(0, 0), pending=0)
    result = CollectionExtraction(sources=[{
        "source_id": 1, "reason": "evidence concerns another person",
        "candidates": [{"subject": "Blair", "value": "swimming", "quote": memory.content,
                        "status": "rejected", "reason": "not Alex's hobby"}],
    }])
    ir = adjudicated_answer(query, result, sources, snapshot=(0, 0), pending=0)
    assert not ir.items and ir.excluded[0]["subject"] == "Blair"
    result.sources[0].candidates[0].status = "accepted"
    with pytest.raises(ValueError, match="subject must belong"):
        validate_extraction(result, query, sources)
