import asyncio

import pytest

from agent_memory.models import (
    Candidate,
    Event,
    Evidence,
    EvolutionInput,
    FactMember,
    FactSemantics,
    QueryPlan,
    Scope,
    TurnInput,
)
from agent_memory.ports import NotFoundError
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.synthesis import AnswerSlot, AnswerTable


class Reader:
    async def plan(self, *args, **kwargs):
        return QueryPlan(required_info=["sealant"], keywords=["vase", "sealant"])

    async def synthesize(self, question, plan, hits, **kwargs):
        return AnswerTable(slots=[AnswerSlot(target=question, coverage="unavailable",
                          derivation="Offline input capture", missing="No model call")])


def test_unindexed_history_requires_explicit_event_access_and_respects_retraction(tmp_path):
    async def run():
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "memory.db"), model=Reader())
        scope = Scope(tenant_id="test", owner_id="alice")
        source = runtime.store.create_session(scope)
        current = runtime.store.create_session(scope)
        foreign = runtime.store.create_session(Scope(tenant_id="test", owner_id="bob"))
        quote = "For the paper vase, use Clearcoat as the sealant."
        try:
            turn = runtime.store.append_turn(scope, source.session_id, TurnInput(request_id="original",
                events=[Event(role="assistant", speaker="Assistant", content=quote)]))
            async def ask(access="quotes", ids=None):
                return await runtime.answer(scope, current.session_id, "Which sealant did you suggest for the vase?",
                    prepare=False, source_session_ids=(source.session_id,) if ids is None else ids,
                    source_read_scope=access)
            def text(answer):
                return " ".join(e["text"] for h in answer.sources for e in h.metadata.get("source_events", []))
            assert quote not in text(await ask())
            assert quote not in text(await ask("events", ()))
            result = await ask("events")
            assert quote in text(result)
            assert result.reasoning["source_read_scope"] == "events"
            assert all(h.metadata.get("origin") == "raw_fallback" for h in result.sources)
            with pytest.raises(NotFoundError):
                await ask("events", (source.session_id, foreign.session_id))
            candidate = Candidate(subject="Assistant", predicate="suggestion", value="Clearcoat", content=quote,
                assertion="inferred", evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote=quote)])
            memory = runtime.store.add_candidates(scope, source.session_id, [candidate])[0]
            await runtime.long_term.evolve(scope, memory.memory_id, EvolutionInput(action="retract", expected_version=1))
            assert quote not in text(await ask("events"))
        finally:
            await runtime.close()
    asyncio.run(run())


def test_unknown_source_access_is_rejected_without_model_work(tmp_path):
    async def run():
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "memory.db"), model=Reader())
        try:
            with pytest.raises(ValueError, match="source_read_scope"):
                await runtime.answer(Scope(tenant_id="t", owner_id="o"), "unknown", "question",
                                     source_read_scope="everything")
        finally:
            await runtime.close()
    asyncio.run(run())


def test_scalar_lookup_reads_collection_member_evidence_without_full_history(tmp_path):
    async def run():
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "member.db"), model=Reader())
        scope = Scope(tenant_id="test", owner_id="alice")
        source = runtime.store.create_session(scope)
        current = runtime.store.create_session(scope)
        intro = "Here are supplies for the craft project."
        detail = "For the paper vase, use Clearcoat as the sealant."
        try:
            t = runtime.store.append_turn(scope, source.session_id, TurnInput(request_id="r",
                events=[Event(role="assistant", content=intro + " " + detail)]))
            runtime.store.add_candidates(scope, source.session_id, [Candidate(
                subject="Assistant", predicate="supplies", value="craft supplies", content=intro,
                assertion="inferred", evidence=[Evidence(turn_id=t.turn_id, event_index=0, quote=intro)],
                semantics=FactSemantics(cardinality="many", members=[FactMember(value="Clearcoat",
                    evidence=[Evidence(turn_id=t.turn_id, event_index=0, quote=detail)])]))])
            answer = await runtime.answer(scope, current.session_id, "Which sealant did you suggest for the vase?",
                prepare=False, source_session_ids=(source.session_id,))
            text = " ".join(e["text"] for h in answer.sources for e in h.metadata.get("source_events", []))
            assert detail in text
            assert answer.reasoning["source_read_scope"] == "quotes"
        finally:
            await runtime.close()
    asyncio.run(run())
