"""Given history stays visible independently of event-target or reference dates."""
import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agent_memory import retrieval
from agent_memory.composition import collection_sources
from agent_memory.grounding import _source_event, close_evidence_bundles, collect_raw_records, ground_evidence
from agent_memory.models import (
    Candidate,
    CollectionQuery,
    Event,
    Evidence,
    ExtractionResult,
    Hit,
    QueryPlan,
    Scope,
    Turn,
    TurnInput,
)
from agent_memory.runtime import MemoryRuntime, _question_date
from agent_memory.settings import Settings
from agent_memory.synthesis import AnswerTable

OBSERVED = datetime(2024, 3, 2, 10, 30, tzinfo=UTC)
VALID = datetime(2024, 3, 1, 12, tzinfo=UTC)


def turn(identifier="t", *, when=OBSERVED, session="s", text="Alex: I visited the museum yesterday."):
    return Turn(turn_id=identifier, request_id=identifier, session_id=session, sequence=1,
                events=[Event(role="user", speaker="Alex", occurred_at=when, content=text)])


def hit(t):
    return Hit(content=t.events[0].content, source_file="fixture", chunk_id=t.turn_id, score=1,
               metadata={"status": "active", "subject": "Alex", "predicate": "visit",
                         "evidence": [{"turn_id": t.turn_id, "event_index": 0, "quote": t.events[0].content}],
                         "source_events": [_source_event(t, 0, t.events[0].content, 0)]})


def test_question_date_remains_a_day_anchor_for_temporal_reasoning():
    query = "When did Alex visit?\nQuestion date: 2024/03/02 (Sat) 09:29"
    assert _question_date(query) == datetime(2024, 3, 2, 12, tzinfo=UTC)


def test_quoted_date_label_does_not_override_question_metadata():
    from agent_memory.runtime import _reasoning_query
    question = 'What did I do yesterday after writing "Question date: 2021/01/04"?'
    assert _question_date(question) is None
    for suffix in ("2024/03/02 (Sat) 09:29", "2024/03/02"):
        wire = question + "\nQuestion date: " + suffix
        assert _reasoning_query(wire) == question
        assert _question_date(wire) == datetime(2024, 3, 2, 12, tzinfo=UTC)



def test_raw_recall_keeps_later_retellings_but_preserves_scope_and_blocked_events():
    t = turn()
    foreign = turn("foreign", session="other")
    blocked = turn("blocked")
    plan = QueryPlan(as_of=VALID, temporal_mode="history")
    records = collect_raw_records("When did Alex visit?", plan, [t,foreign,blocked], "s", {("blocked",0)})
    assert [record["turn_id"] for record, _ in records] == ["t"]


def test_grounding_does_not_treat_fact_validity_date_as_source_observation_cutoff():
    t = turn()
    plan = QueryPlan(as_of=VALID, temporal_mode="history")
    result = ground_evidence("When did Alex visit the museum?", plan, [hit(t)],
        session_id="s", current_turns=[t], resolved_turns={t.turn_id:t},
        blocked_events=set(), max_sources=20, budget=5000)
    assert any(e["turn_id"] == "t" for h in result.sources for e in h.metadata["source_events"])


def test_closure_keeps_nearby_later_context_but_not_foreign_blocked_or_distant_events():
    t = turn(text="Alex: Public museum statement.")
    adjacent = turn("adjacent", when=OBSERVED+timedelta(minutes=1), text="Alex: Relevant adjacent detail.")
    blocked = turn("blocked", text="Alex: Secret revoked detail.")
    foreign = turn("foreign", session="other", text="Alex: Secret foreign detail.")
    distant = turn("distant", when=OBSERVED+timedelta(minutes=31), text="Alex: Distant detail.")
    result = close_evidence_bundles([hit(t)], [t,adjacent,blocked,foreign,distant], session_id="s",
        blocked_events={("blocked",0)}, budget=8000, query="museum detail", plan=QueryPlan(as_of=VALID))
    assert {e["turn_id"] for h in result for e in h.metadata["source_events"]} == {"t","adjacent"}


class CaptureModel:
    def __init__(self, *, as_of=VALID, collection=False):
        self.as_of=as_of
        self.collection=collection
        self.sources=[]

    async def plan(self, question, context):
        return QueryPlan(route="both", temporal_mode="history", as_of=self.as_of,
            semantic_queries=["Alex museum visit"], keywords=["Alex","museum"],
            collection_query=CollectionQuery(subjects=["Alex"],predicate="visit") if self.collection else None)

    async def synthesize(self, question, plan, sources, **kwargs):
        self.sources=sources
        return AnswerTable.model_validate({"slots":[{"target":question,"derivation":"Offline visibility probe.",
            "values":[],"missing":"Synthetic fixture does not generate an answer.","coverage":"unavailable"}]})


async def runtime_fixture(tmp_path, *, valid_from=VALID, collection=False):
    model=CaptureModel(collection=collection)
    runtime=MemoryRuntime(Settings(db_path=tmp_path/'memory.db',maintenance_interval=0,planner_mode="model"),model=model)
    scope=Scope(tenant_id="fixture",owner_id="owner")
    session=runtime.store.create_session(scope)
    t=runtime.store.append_turn(scope,session.session_id,TurnInput(request_id="r",events=[
        Event(role="user",speaker="Alex",content="Alex: I visited the museum yesterday.",occurred_at=OBSERVED)]))
    evidence=[Evidence(turn_id=t.turn_id,event_index=0,quote=t.events[0].content)]
    memories=runtime.store.commit_extraction(scope,runtime.store.get_session(scope,session.session_id),[t],
        ExtractionResult(candidates=[Candidate(subject="Alex",predicate="visit",value="museum",content=t.events[0].content,
                                             valid_from=valid_from,evidence=evidence)]))
    return runtime,model,scope,session,t,memories[0]


@pytest.mark.parametrize("collection",[False,True])
def test_runtime_reference_timestamp_never_hides_given_history(tmp_path,collection):
    async def scenario():
        runtime,model,scope,session,t,memory=await runtime_fixture(tmp_path,collection=collection)
        try:
            query = "When did Alex visit the museum?"
            await runtime.answer(scope,session.session_id,query,prepare=False)
            original = [h.model_dump() for h in model.sources]
            assert any(t.turn_id==e["turn_id"] for h in model.sources for e in h.metadata["source_events"])
            # Dataset reference and session hours need not be chronological.
            for reference in ["2024/03/02 (Sat) 10:29", "2024/03/02 (Sat) 10:30"]:
                await runtime.answer(scope,session.session_id,query+"\nQuestion date: "+reference,prepare=False)
                assert [h.model_dump() for h in model.sources] == original
        finally:
            await runtime.close()
    asyncio.run(scenario())


def test_valid_time_ineligibility_still_quarantines_raw_events(tmp_path):
    async def scenario():
        runtime,model,scope,session,t,memory=await runtime_fixture(tmp_path,valid_from=OBSERVED)
        try:
            assert not runtime.retriever._eligible(memory,session,QueryPlan(as_of=VALID),OBSERVED)
            await runtime.answer(scope,session.session_id,"When did Alex visit the museum?",prepare=False)
            assert all(t.turn_id!=e["turn_id"] for h in model.sources for e in h.metadata["source_events"])
        finally:
            await runtime.close()
    asyncio.run(scenario())


def test_collection_keeps_later_original_quotes_for_eligible_historical_facts(tmp_path):
    async def scenario():
        runtime,model,scope,session,t,memory=await runtime_fixture(tmp_path)
        try:
            assert runtime.retriever._eligible(memory,session,QueryPlan(as_of=VALID),OBSERVED)
            query=CollectionQuery(subjects=["Alex"],predicate="visit")
            sources=collection_sources(query,[memory],{t.turn_id:t})
            assert sources[0].metadata["source_events"][0]["text"] == t.events[0].content
        finally:
            await runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("entry", ["search", "answer"])
def test_same_question_date_reaches_planner_independent_of_wall_clock(tmp_path, monkeypatch, entry):
    async def scenario():
        runtime, model, scope, session, turn, _ = await runtime_fixture(tmp_path)
        observations = []
        original = model.plan

        async def capture(query, context):
            observations.append((query, context["query_time"]))
            return await original(query, context)

        model.plan = capture
        query = "When did Alex visit the museum?\nQuestion date: 2024/03/02 (Sat) 09:29"
        try:
            for year in (2026, 2028):
                monkeypatch.setattr(retrieval, "utcnow", lambda y=year: datetime(y, 11, 16, tzinfo=UTC))
                if entry == "answer":
                    await runtime.answer(scope, session.session_id, query, prepare=False)
                    assert any(e["turn_id"] == turn.turn_id for h in model.sources for e in h.metadata["source_events"])
                else:
                    result = await runtime.search(scope, session.session_id, query)
                    assert result.plan.as_of == VALID  # Never overwritten with question reference.
            assert observations == [("When did Alex visit the museum?", "2024-03-02T12:00:00+00:00")] * 2
        finally:
            await runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("suffix", ["", "\nQuestion date: 2024/13/02 (Sat) 09:29"])
def test_absent_or_invalid_question_date_keeps_wall_clock_fallback(tmp_path, monkeypatch, suffix):
    async def scenario():
        runtime, model, scope, session, _, _ = await runtime_fixture(tmp_path)
        observed = []
        original = model.plan
        now = datetime(2027, 4, 6, 8, tzinfo=UTC)

        async def capture(query, context):
            observed.append(context["query_time"])
            return await original(query, context)

        model.plan = capture
        monkeypatch.setattr(retrieval, "utcnow", lambda: now)
        try:
            await runtime.answer(scope, session.session_id, "When did Alex visit the museum?" + suffix, prepare=False)
            assert observed == [now.isoformat()]
        finally:
            await runtime.close()
    asyncio.run(scenario())
