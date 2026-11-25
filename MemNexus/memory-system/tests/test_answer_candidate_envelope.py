"""Internal answer recall must not be limited by the public search preview."""
import asyncio

from agent_memory.models import Candidate, Event, Evidence, Hit, QueryPlan, Scope, SearchResponse, TurnInput
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.synthesis import AnswerSlot, AnswerTable


def test_search_preview_does_not_serialize_internal_candidates():
    hidden = Hit(content="internal source", score=1, source_file="memory", chunk_id="hidden")
    response = SearchResponse(results=[], candidates=[hidden], plan=QueryPlan(), mode="hybrid",
                              candidate_count=1, selected_k=0, context_tokens=0)
    assert response.candidates == [hidden]
    assert "candidates" not in response.model_dump()
    assert "internal source" not in response.model_dump_json()


def test_answer_receives_candidates_outside_the_public_preview(tmp_path, monkeypatch):
    from agent_memory import runtime as module

    class Reader:
        async def synthesize(self, question, plan, sources, **kwargs):
            return AnswerTable(slots=[AnswerSlot(target=question, coverage="unavailable",
                                                missing="Offline contract check", derivation="No generation")])

    async def run():
        service = MemoryRuntime(Settings(db_path=tmp_path / "memory.db", maintenance_interval=0), model=Reader())
        scope = Scope(tenant_id="test", owner_id="reader")
        try:
            session = service.store.create_session(scope)
            text = "The lantern is sealed with clear varnish."
            turn = service.store.append_turn(scope, session.session_id, TurnInput(request_id="source",
                events=[Event(role="user", speaker="Ari", content=text)]))
            memory = service.store.add_candidates(scope, session.session_id, [Candidate(
                subject="lantern", predicate="sealant", value="clear varnish", content=text,
                evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=text)])])[0]
            hit = Hit(content=text, score=1, source_file="memory", chunk_id=memory.memory_id,
                      metadata={**memory.source_metadata(), "retrieval_rank": 1})

            async def search(*args, **kwargs):
                return SearchResponse(results=[], candidates=[hit],
                    plan=QueryPlan(required_info=["lantern sealant"]), mode="hybrid",
                    candidate_count=1, selected_k=0, context_tokens=0)

            seen = []
            original = module.ground_evidence

            def ground(query, plan, hits, **kwargs):
                seen.extend(h.metadata.get("memory_id") for h in hits)
                return original(query, plan, hits, **kwargs)

            monkeypatch.setattr(service.retriever, "search", search)
            monkeypatch.setattr(module, "ground_evidence", ground)
            answer = await service.answer(scope, session.session_id, "What seals the lantern?", prepare=False)
            assert memory.memory_id in seen
            assert any(text in e["text"] for h in answer.sources for e in h.metadata.get("source_events", []))
            assert answer.model_calls.answer_calls == 1
        finally:
            await service.close()

    asyncio.run(run())
