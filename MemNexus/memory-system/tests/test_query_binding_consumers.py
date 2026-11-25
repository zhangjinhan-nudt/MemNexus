"""Final plan bindings survive the existing supplemental and receipt interfaces."""
import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from agent_memory.models import (
    CollectionQuery,
    Event,
    Evidence,
    Memory,
    QueryPlan,
    Scope,
    SearchResponse,
    TurnInput,
)
from agent_memory.providers import CallableModel
from agent_memory.reasoning import apply_route, bind_plan_entities, classify_question, indexed_memory_hits
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.synthesis import AnswerSlot, AnswerTable


def memory(identifier, subject, content, **changes):
    values = dict(memory_id=identifier, session_id="s", scope_id="s", subject=subject,
                  predicate="participation", value=content, content=content,
                  evidence=[Evidence(turn_id=identifier, event_index=0, quote=content)])
    values.update(changes)
    return Memory(**values)


@pytest.mark.parametrize("collection", [False, True])
def test_same_question_constraints_survive_both_plan_shapes(collection):
    query = "Which movie did Morgan watch in March 2023?"
    route = classify_question(query, ["Morgan", "Riley"])
    proposed = QueryPlan(collection_query=CollectionQuery(subjects=["Morgan"], predicate="movie")
                         if collection else None)
    final = apply_route(proposed, route)
    assert final.query_entities == ["Morgan"]
    assert final.query_constraints == list(route.constraints)
    assert {"object_type=movie", "month=march", "year=2023"} <= set(final.query_constraints)


def test_collection_subject_is_not_replaced_by_capitalized_reference_event():
    query = "Which workshops did I attend before the Winter Forum?"
    route = classify_question(query, ["user", "assistant"])
    final = apply_route(QueryPlan(collection_query=CollectionQuery(
        subjects=["user"], predicate="participation")), route)
    assert tuple(final.query_entities) != route.entities
    records = [memory("member", "user", "I attended a pottery workshop."),
               memory("anchor", "Winter Forum", "The Winter Forum has workshop registration.")]
    hits = indexed_memory_hits(query, final, records, [], limit=1)
    assert [hit.metadata["memory_id"] for hit in hits] == ["member"]


def test_existing_explicit_speaker_binding_precedes_collection_consumption():
    question = "Which movie did Morgan watch in 2023?"
    proposed = QueryPlan(collection_query=CollectionQuery(subjects=["Riley"], predicate="movie"))
    bound, changed = bind_plan_entities(question, proposed, ["Morgan", "Riley"])
    final = apply_route(bound, classify_question(question, ["Morgan", "Riley"]))
    assert changed == ["collection_subjects"]
    assert final.query_entities == ["Morgan"]
    assert final.query_constraints == ["object_type=movie", "year=2023"]


def test_no_named_entity_does_not_create_user_subject():
    query = "What preferences are available?"
    final = apply_route(QueryPlan(), classify_question(query, ["Morgan", "Riley"]))
    assert final.query_entities == []


def test_existing_selected_memory_is_not_added_again():
    plan = QueryPlan(query_entities=["Morgan"])
    records = [memory("member", "Morgan", "Morgan attended a pottery workshop.")]
    first = indexed_memory_hits("What workshops did Morgan attend?", plan, records, [])
    assert len(first) == 1
    assert indexed_memory_hits("What workshops did Morgan attend?", plan, records, first) == []


def test_runtime_consumes_final_plan_policy_without_reclassifying_it(tmp_path, monkeypatch):
    async def run():
        calls = []

        class Model:
            async def synthesize(self, question, plan, sources, **kwargs):
                calls.append((question, plan))
                return AnswerTable(slots=[AnswerSlot(target=question, derivation="No stored evidence.",
                    coverage="unavailable", missing="No stored evidence.")])

        runtime = MemoryRuntime(Settings(db_path=tmp_path / "memory.db"), model=Model())
        scope = Scope(tenant_id="test", owner_id="test")
        session = runtime.store.create_session(scope)
        query = "When did I attend the Winter Forum?"
        assert classify_question(query).recall_policy == "focused"
        final = QueryPlan(query_entities=["user"], query_constraints=["year=2023"],
                          query_type="temporal", recall_policy="recall_first")

        async def search(*args, **kwargs):
            return SearchResponse(plan=final, results=[], mode="lexical_baseline",
                                  candidate_count=0, selected_k=0, context_tokens=0)

        def indexed(question, plan, records, selected, **kwargs):
            assert question == query and plan == final
            assert plan.query_entities == ["user"]
            assert kwargs["limit"] == 32
            return []

        from agent_memory.grounding import ground_evidence

        def grounding(question, plan, hits, **kwargs):
            assert plan == final
            # Retrieval top-k is not the original reading-unit budget. The
            # public citation catalog remains bounded and serialized below.
            assert 20 <= kwargs["max_sources"] <= 120
            from agent_memory.prompt_codec import SOURCE_CATALOG_OVERHEAD
            assert kwargs["budget"] + SOURCE_CATALOG_OVERHEAD == 15840
            assert callable(kwargs["source_cost"])
            return ground_evidence(question, plan, hits, **kwargs)

        monkeypatch.setattr(runtime, "search", search)
        monkeypatch.setattr("agent_memory.runtime.indexed_memory_hits", indexed)
        monkeypatch.setattr("agent_memory.runtime.ground_evidence", grounding)
        try:
            answer = await runtime.answer(scope, session.session_id, query, prepare=False)
            assert answer.reasoning["entities"] == final.query_entities
            assert answer.reasoning["constraints"] == final.query_constraints
            assert answer.reasoning["question_type"] == final.query_type
            assert answer.reasoning["recall_policy"] == final.recall_policy
            assert len(calls) == 1
        finally:
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("operator", ["set", "direct", None])
def test_real_search_contract_route_reaches_runtime_budget_and_citations(tmp_path, monkeypatch, operator):
    async def run():
        question = "What did Mira enjoy on holiday?"
        quote = "Mira enjoyed hiking and painting on holiday."
        assert classify_question(question, ["Mira"]).recall_policy == "focused"
        prompts, observed = [], []
        cited = []

        async def chat(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                proposal = {"route": "both", "semantic_queries": [question], "keywords": ["holiday"]}
                if operator is not None:
                    proposal["answer_contract"] = {"outputs": [{
                        "output_id": "activities", "question_span": question, "operator": operator,
                        "subjects": ["Mira"], "relation": "hobbies", "qualifier": "on holiday",
                        "count_unit": "value" if operator == "set" else None,
                    }]}
                if operator == "set":
                    # A valid but unqualified CQ cannot stand for the qualified request.
                    proposal["collection_query"] = {"subjects": ["Mira"], "predicate": "hobbies",
                                                    "count_unit": "value"}
                return json.dumps(proposal)
            assert len(prompts) == 2  # Existing planner and reader; no added repair/call.
            data = json.JSONDecoder().raw_decode(prompt.split("\n输入数据：\n", 1)[1])[0]
            assert data["question"] == question
            source = next(source for source in data["sources"]
                          if any(event["text"] == quote for event in source["source_events"]))
            cited.append(source["source_id"])
            return json.dumps({"slots": [{"target": question, "derivation": "Use the original statement.",
                "values": [{"value": "hiking and painting", "supports": [
                    {"source_id": source["source_id"], "quote": quote}]}]}]})

        settings = Settings(db_path=tmp_path / "integrated.db", planner_mode="model", maintenance_interval=0)
        runtime = MemoryRuntime(settings, model=CallableModel(chat))
        scope = Scope(tenant_id="test", owner_id="test")
        session = runtime.store.create_session(scope)
        turn = runtime.store.append_turn(scope, session.session_id, TurnInput(
            request_id="source", events=[Event(role="user", speaker="Mira", content=quote)]))
        from agent_memory.grounding import ground_evidence

        def grounding(query, plan, hits, **kwargs):
            observed.append((plan.model_copy(deep=True), dict(kwargs)))
            assert query == question and plan.collection_query is None
            assert 7 <= kwargs["max_sources"] <= 120  # top_k constrains memory retrieval, not raw units
            from agent_memory.prompt_codec import SOURCE_CATALOG_OVERHEAD
            # The source catalog wrapper shares the same final envelope.
            assert kwargs["budget"] + SOURCE_CATALOG_OVERHEAD == 15840
            assert callable(kwargs["source_cost"])
            return ground_evidence(query, plan, hits, **kwargs)

        monkeypatch.setattr("agent_memory.runtime.ground_evidence", grounding)
        try:
            answer = await runtime.answer(scope, session.session_id, question, top_k=7, prepare=False)
            plan = observed[0][0]
            assert len(observed) == 1 and len(prompts) == 2
            assert plan.query_type == ("set" if operator == "set" else "fact")
            assert plan.recall_policy == ("recall_first" if operator == "set" else "focused")
            assert answer.reasoning["question_type"] == plan.query_type
            assert answer.reasoning["recall_policy"] == plan.recall_policy
            assert answer.generated_text == "hiking and painting" and answer.citations == cited
            assert len(answer.sources) <= 7
            assert answer.grounding.context_tokens_estimate <= int(settings.complex_prompt_token_limit * .72)
            source = answer.sources[cited[0] - 1]
            assert any(event["turn_id"] == turn.turn_id and event["text"] == quote
                       for event in source.metadata["source_events"])
        finally:
            await runtime.close()

    asyncio.run(run())


def test_runtime_filters_lifecycle_and_scope_before_indexed(tmp_path, monkeypatch):
    class ReachedConsumer(Exception):
        pass

    async def run():
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "scope.db"), model=object())
        scope = Scope(tenant_id="test", owner_id="test")
        session = runtime.store.create_session(scope)
        now = datetime.now(UTC)
        records = [memory("ok", "Morgan", "Morgan attended a workshop.", scope_type="user"),
                   memory("private", "Morgan", "Morgan attended a workshop.",
                          scope_type="project", scope_id="other"),
                   memory("retracted", "Morgan", "Morgan attended a workshop.", status="retracted"),
                   memory("expired", "Morgan", "Morgan attended a workshop.",
                          valid_to=now-timedelta(days=1), scope_type="user")]
        final = QueryPlan(query_entities=["Morgan"])

        async def search(*args, **kwargs):
            return SearchResponse(plan=final, results=[], mode="lexical_baseline",
                                  candidate_count=0, selected_k=0, context_tokens=0)

        def indexed(question, plan, eligible, selected, **kwargs):
            assert [m.memory_id for m in eligible] == ["ok"]
            assert plan == final
            raise ReachedConsumer

        monkeypatch.setattr(runtime, "search", search)
        monkeypatch.setattr(runtime.store, "memories", lambda *a, **kw: records)
        monkeypatch.setattr("agent_memory.runtime.indexed_memory_hits", indexed)
        try:
            with pytest.raises(ReachedConsumer):
                await runtime.answer(scope, session.session_id, "What workshops did Morgan attend?", prepare=False)
        finally:
            await runtime.close()

    asyncio.run(run())


def test_legacy_adapter_uses_final_policy_and_preserves_question_diagnostics(tmp_path, monkeypatch):
    async def run():
        questions = []

        class Model:
            async def answer(self, prompt):
                questions.append(prompt)
                return "No matching fact was supplied."

        runtime = MemoryRuntime(Settings(db_path=tmp_path / "legacy.db"), model=Model())
        scope = Scope(tenant_id="test", owner_id="test")
        session = runtime.store.create_session(scope)
        query = "Choose exactly one DJ mentioned at the Winter Forum."
        original = classify_question(query, [])
        assert original.negative_gate
        final = QueryPlan(query_entities=["Morgan"], query_constraints=["year=2023"],
                          query_type="fact", recall_policy="recall_first")

        async def search(*args, **kwargs):
            return SearchResponse(plan=final, results=[], mode="lexical_baseline",
                                  candidate_count=0, selected_k=0, context_tokens=0)

        from agent_memory.reasoning import typed_negative_gate

        def gate(question, route, graph):
            assert route.entities == ("Morgan",) and route.constraints == ("year=2023",)
            assert route.kind == final.query_type and route.recall_policy == final.recall_policy
            for name in ("requires_composition", "negative_gate"):
                assert getattr(route, name) == getattr(original, name)
            result = typed_negative_gate(question, route, graph)
            assert not result.blocked
            assert result.reason.startswith("not_found_in_examined_evidence")
            return result

        monkeypatch.setattr(runtime, "search", search)
        monkeypatch.setattr("agent_memory.runtime.typed_negative_gate", gate)
        try:
            answer = await runtime.answer(scope, session.session_id, query, prepare=False)
            assert len(questions) == 1  # The diagnostic did not veto or add a generation route.
            assert "incomplete evidence, not proof of absence" in questions[0]
            assert not answer.reasoning["typed_negative_gate"]["blocked"]
            assert answer.reasoning["entities"] == ["Morgan"]
            assert answer.reasoning["question_type"] == final.query_type
            assert answer.reasoning["recall_policy"] == final.recall_policy
        finally:
            await runtime.close()

    asyncio.run(run())
