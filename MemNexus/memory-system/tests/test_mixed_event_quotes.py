import asyncio
from datetime import UTC, datetime

import pytest

from agent_memory.grounding import _cost, _source_event, close_evidence_bundles
from agent_memory.models import Candidate, Event, Evidence, Hit, QueryPlan, Scope, Turn, TurnInput
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.synthesis import AnswerTable


def turn(identifier,text,session,seq):
    return Turn(turn_id=identifier,request_id=identifier,session_id=session,sequence=seq,events=[
        Event(role="user",content=text,occurred_at=datetime(2025,1,1,tzinfo=UTC))])


def hit(t, quote):
    return Hit(content="public reference",score=1,source_file="fixture",chunk_id=t.turn_id,
               metadata={"evidence":[{"turn_id":t.turn_id,"event_index":0,"quote":quote}],
                         "status":"active","source_events":[_source_event(t,0,quote,t.events[0].content.index(quote))]})


@pytest.mark.parametrize("source_session",["old","current"])
def test_selected_mixed_event_only_returns_active_authorized_quotes(source_session):
    anchor="I received a notice."
    detail="The new room allocation is room 37."
    original=turn("a",anchor+" The restricted passcode is private. "+detail,source_session,1)
    adjacent=turn("b","The next room allocation is room 98.",source_session,2)
    seed=hit(original,anchor)
    args=dict(session_id="current",blocked_events={("a",0)},budget=8000,
              query="What is the room allocation?", plan=QueryPlan(route="both"),
              authorized_quotes={("a",0):[anchor,detail,"room 400 does not appear"],
                                 ("b",0):[adjacent.events[0].content]})
    events=close_evidence_bundles([seed],[original,adjacent],**args)[0].metadata["source_events"]
    assert [e["text"] for e in events]==[anchor,detail]
    assert all(e["text"]==original.events[0].content[e["start"]:e["end"]] for e in events)
    limited=close_evidence_bundles([seed],[original,adjacent],**{**args,"budget":_cost(seed)})[0]
    assert [e["text"] for e in limited.metadata["source_events"]]==[anchor]
    for active in ([],[detail],["absent"],[anchor]):
        changed=close_evidence_bundles([seed],[original,adjacent],**{**args,"authorized_quotes":{("a",0):active}})
        assert [e["text"] for e in changed[0].metadata["source_events"]]==[anchor]


def test_blocked_event_does_not_enter_via_a_neighbour():
    earlier=turn("old","The first public entry.","s",1)
    mixed=turn("mix","A room detail with a revoked prefix.","s",2)
    got=close_evidence_bundles([hit(earlier,earlier.events[0].content)], [earlier,mixed],session_id="s",
                               blocked_events={("mix",0)},budget=10000, query="room detail",
                               authorized_quotes={("mix",0):["A room detail"]})
    assert all(e["turn_id"]=="old" for h in got for e in h.metadata["source_events"])


def test_runtime_never_authorizes_private_revoked_or_historical_only_facts(tmp_path,monkeypatch):
    from agent_memory import runtime as module
    class Model:
        async def plan(self,query,context):
            return QueryPlan(route="both",temporal_mode="history",semantic_queries=["room allocation"])
        async def synthesize(self,question,plan,sources,**kwargs):
            return AnswerTable.model_validate({"slots":[{"target":question,"coverage":"unavailable","missing":"No answer needed.","derivation":"Offline."}]})
    async def run():
        runtime=MemoryRuntime(Settings(db_path=tmp_path/"test.db",maintenance_interval=0,planner_mode="model"),model=Model())
        scope=Scope(tenant_id="tenant",owner_id="owner")
        try:
            old=runtime.store.create_session(scope);current=runtime.store.create_session(scope)
            texts=["The room allocation was updated.","The room allocation moved to the library.",
                   "The previous room allocation was underground.","Private room information.","Retracted room detail."]
            t=runtime.store.append_turn(scope,old.session_id,TurnInput(request_id="test",events=[
                Event(role="user",content=" ".join(texts))]))
            candidates=[Candidate(subject="Office",predicate="allocation",value=str(i),content=q,scope_type="user",
                                  evidence=[Evidence(turn_id=t.turn_id,event_index=0,quote=q)]) for i,q in enumerate(texts)]
            ms=runtime.store.add_candidates(scope,old.session_id,candidates)
            for m,change in zip(ms,[{}, {}, {"status":"superseded"},
                                    {"scope_id":old.session_id,"scope_type":"session"},{"status":"retracted"}]):
                runtime.store._save_memory(scope,m.model_copy(update=change))
            runtime.store._db.commit()
            captured={}
            original_ground = module.ground_evidence
            def observe(*args,**kw):
                captured.update(kw)
                return original_ground(*args,**kw)
            monkeypatch.setattr(module,"ground_evidence",observe)
            await runtime.answer(scope,current.session_id,"What room allocation updates have we discussed?",
                                 prepare=False,accelerate=False)
            key=(t.turn_id,0)
            assert key in captured["blocked_events"]
            refs = captured["authorized_quotes"][key]
            assert {r["quote"] for r in refs} == set(texts[:2])
            assert all(t.events[0].content[r["start"]:r["end"]] == r["quote"] for r in refs)
        finally:
            await runtime.close()
    asyncio.run(run())
