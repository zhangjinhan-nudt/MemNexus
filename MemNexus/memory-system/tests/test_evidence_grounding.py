"""Mechanism and boundary checks; fixture answers are not QA accuracy results."""

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest

from agent_memory.evidence_policy import question_only
from agent_memory.grounding import ground_evidence, source_payload
from agent_memory.models import (
    Event,
    Evidence,
    EvolutionInput,
    Hit,
    QueryPlan,
    Scope,
    Turn,
    TurnInput,
)
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.store import SQLiteStore
from agent_memory.temporal import derive_times


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I painted yesterday. What do you think?", False),
        ("I painted yesterday. what do you think?", False),
        ("Yesterday I painted. Did you swim?", False),
        ("Did you paint yesterday?", True),
        ("昨天我画了画。你觉得如何？", False),
        ("请问，他是谁？", True),
        ("Did Dr. Silva arrive?", True),
        ("Dr. Silva arrived. Is everyone here?", False),
        ("The cost is 3.14 dollars. Is that enough?", False),
    ],
)
def test_mixed_speech_acts(text, expected):
    assert question_only(text) is expected


@pytest.mark.parametrize(
    "text,anchor,start,end,precision",
    [
        ("yesterday", "2024-03-01T10:00:00+00:00", "2024-02-29", "2024-03-01", "day"),
        ("昨天", "2025-01-01T10:00:00+00:00", "2024-12-31", "2025-01-01", "day"),
        ("last year", "2025-03-01T10:00:00+00:00", "2024-01-01", "2025-01-01", "year"),
        ("next month", "2025-12-31T10:00:00+00:00", "2026-01-01", "2026-02-01", "month"),
        ("last Saturday", "2025-05-17T10:00:00+00:00", "2025-05-10", "2025-05-11", "day"),
        ("last weekend", "2025-05-21T10:00:00+00:00", "2025-05-17", "2025-05-19", "weekend"),
        ("three years ago", "2025-05-21T10:00:00+00:00", "2022-01-01", "2023-01-01", "year"),
        ("two months ago", "2025-01-21T10:00:00+00:00", "2024-11-01", "2024-12-01", "month"),
        ("2024-02-29", "2025-01-21T10:00:00+00:00", "2024-02-29", "2024-03-01", "day"),
    ],
)
def test_calendar_intervals(text, anchor, start, end, precision):
    values = derive_times(text, datetime.fromisoformat(anchor))
    assert len(values) == 1
    assert (values[0].start, values[0].end_exclusive, values[0].precision) == (start, end, precision)


def test_vague_duration_and_missing_anchor_do_not_fabricate_dates():
    anchor = datetime(2025, 1, 1, tzinfo=UTC)
    assert derive_times("I have had it for three years", anchor) == []
    assert derive_times("a few days ago", anchor) == []
    assert derive_times("yesterday", None) == []
    assert derive_times("yesterday", datetime(2025, 1, 1)) == []  # noqa: DTZ001 - missing-zone rejection
    assert derive_times('An older letter said "tomorrow".', anchor) == []
    interval = derive_times("two weeks ago", datetime(2025, 5, 21, tzinfo=UTC))[0]
    assert (interval.start, interval.end_exclusive, interval.precision) == (
        "2025-05-05",
        "2025-05-12",
        "week",
    )
    assert derive_times("next year", datetime(9999, 1, 1, tzinfo=UTC)) == []


def turn(identifier, content, speaker="Nora", session_id="s", at=None, role="user"):
    return Turn(
        turn_id=identifier,
        session_id=session_id,
        sequence=1,
        request_id=identifier,
        events=[
            Event(
                content=content,
                role=role,
                speaker=speaker,
                occurred_at=at or datetime(2025, 1, 1, tzinfo=UTC),
            )
        ],
    )


def hit(identifier, quote, subject="Nora"):
    return Hit(
        content=f"{subject} painted last year",
        score=0.8,
        source_file="memory:m",
        chunk_id="m:1",
        metadata={
            "subject": subject,
            "memory_id": "m",
            "status": "active",
            "evidence": [{"turn_id": identifier, "event_index": 0, "quote": quote}],
        },
    )


def test_time_anchor_survives_utc_conversion_and_answer_join():
    source = turn(
        "t", "I painted yesterday.", at=datetime(2025, 1, 1, 0, 30, tzinfo=timezone(timedelta(hours=14)))
    )
    assert source.events[0].occurred_at.date().isoformat() == "2024-12-31"
    result = ground_evidence(
        "When did Nora paint?",
        QueryPlan(required_info=["paint date"]),
        [hit("t", "I painted yesterday.")],
        session_id="s",
        current_turns=[source],
        resolved_turns={"t": source},
        blocked_events=set(),
    )
    times = [
        time for h in result.sources for e in h.metadata["source_events"] for time in e["time_derivations"]
    ]
    assert times and all(t["start"] == "2024-12-31" for t in times)


def test_cross_session_join_reveals_only_authorized_quote():
    source = turn("foreign", "I painted last year. Confidential salary is 9999.", session_id="elsewhere")
    result = ground_evidence(
        "When did Nora paint?",
        QueryPlan(),
        [hit("foreign", "I painted last year.")],
        session_id="s",
        current_turns=[],
        resolved_turns={"foreign": source},
        blocked_events=set(),
    )
    assert result.sources
    assert "9999" not in str(source_payload(result.sources[0]))


def test_compression_subject_conflict_recovers_original_speaker():
    source = turn("a", "I painted last year.", speaker="Nora")
    other = turn("b", "I prefer music.", speaker="Idris")
    result = ground_evidence(
        "When did Nora paint?",
        QueryPlan(),
        [hit("a", "I painted last year.", "Idris")],
        session_id="s",
        current_turns=[source, other],
        resolved_turns={"a": source},
        blocked_events=set(),
    )
    assert result.trace.attribution_conflicts == 1
    assert all(h.metadata.get("subject") != "Idris" for h in result.sources)
    assert any(e["speaker"] == "Nora" for h in result.sources for e in h.metadata["source_events"])


def test_raw_fallback_recovers_omitted_fact_without_model_or_gold():
    turns = [turn(str(i), f"I visited a cafe number {i}.") for i in range(20)]
    omitted = turn("needed", "I use a titanium splint for rowing practice.")
    turns.insert(2, omitted)
    result = ground_evidence(
        "What splint does Nora use?",
        QueryPlan(required_info=["splint material"]),
        [],
        session_id="s",
        current_turns=turns,
        resolved_turns={},
        blocked_events=set(),
        raw_limit=2,
        max_sources=2,
        budget=2000,
    )
    assert result.trace.supplemental_rounds == 1
    assert len(result.sources) <= 2
    assert any("titanium" in h.content for h in result.sources)
    assert result.trace.context_tokens_estimate <= 2000


def test_denied_raw_event_is_not_resurrected_and_no_other_session_is_scanned():
    revoked = turn("revoked", "Nora's code is 123456.")
    other = turn("private", "Nora's code is 654321.", session_id="other")
    result = ground_evidence(
        "Nora code",
        QueryPlan(),
        [],
        session_id="s",
        current_turns=[revoked, other],
        resolved_turns={},
        blocked_events={("revoked", 0)},
    )
    assert result.sources == []


def test_source_turn_lookup_enforces_owner_boundary(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    alice, bob = Scope(tenant_id="t", owner_id="a"), Scope(tenant_id="t", owner_id="b")
    try:
        session = store.create_session(alice)
        source = store.append_turn(
            alice,
            session.session_id,
            TurnInput(request_id="x", events=[Event(role="user", content="private")]),
        )
        assert store.source_turns(bob, [source.turn_id]) == {}
        assert list(store.source_turns(alice, [source.turn_id])) == [source.turn_id]
    finally:
        store.close()


def test_runtime_passes_source_and_derivation_to_answer_and_retains_trace(tmp_path):
    class Model:
        def __init__(self):
            self.prompt = ""
            self.calls = 0

        async def plan(self, query, context):
            return QueryPlan(required_info=["painting date"], keywords=["painted"])

        async def answer(self, prompt):
            self.prompt = prompt
            self.calls += 1
            return "2024【来源1】"

    async def run():
        model = Model()
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "runtime.db"), model=model)
        scope = Scope(tenant_id="test", owner_id="test")
        try:
            session = runtime.store.create_session(scope)
            await runtime.append_turn(
                scope,
                session.session_id,
                TurnInput(
                    request_id="one",
                    events=[
                        Event(
                            role="user",
                            speaker="Inez",
                            content="I painted last year. What do you think?",
                            occurred_at=datetime(2025, 3, 4, tzinfo=UTC),
                        )
                    ],
                ),
            )
            response = await runtime.answer(
                scope, session.session_id, "When did Inez paint?", prepare=False, accelerate=False
            )
            assert model.calls == 1
            assert "2024-01-01" in model.prompt and "2025-03-04" in model.prompt
            assert '"speaker": "Inez"' in model.prompt
            assert "lexical missing_info" in model.prompt
            assert response.grounding.enabled and response.grounding.temporal_derivations >= 1
        finally:
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("language", ["en", "zh"])
def test_long_plan_reserves_header_before_source_selection(tmp_path, language):
    """A valid long plan must reduce evidence capacity, not overflow the prompt."""

    class Model:
        prompt = ""

        async def plan(self, query, context):
            return QueryPlan(
                keywords=["painted"],
                required_info=["painting date"],
                information_gathering=["Inspect original painting evidence. " * 7] * 5,
                integration_steps=["Keep calendar precision and cite the original speaker. " * 5] * 5,
            )

        async def answer(self, prompt):
            self.prompt = prompt
            return "2024【来源1】"

    async def run():
        model = Model()
        settings = Settings(db_path=tmp_path / "bounded.db", output_language=language)
        runtime = MemoryRuntime(settings, model=model)
        scope = Scope(tenant_id="budget", owner_id="budget")
        try:
            session = runtime.store.create_session(scope)
            for index in range(12):
                await runtime.append_turn(
                    scope,
                    session.session_id,
                    TurnInput(
                        request_id=str(index),
                        events=[
                            Event(
                                role="user",
                                speaker="Inez",
                                content=f"I painted last year in gallery {index}. " + "painting detail " * 25,
                                occurred_at=datetime(2025, 3, 4, tzinfo=UTC),
                            )
                        ],
                    ),
                )
            result = await runtime.answer(
                scope,
                session.session_id,
                "When did Inez paint?",
                prepare=False,
                accelerate=False,
            )
            assert model.prompt and len(model.prompt) <= settings.prompt_token_limit - 998
            assert result.sources and result.grounding.temporal_derivations
            assert result.dynamic_k.token_limit < settings.prompt_token_limit // 2
            assert result.grounding.context_tokens_estimate <= result.dynamic_k.token_limit
        finally:
            await runtime.close()

    asyncio.run(run())


def test_retracted_event_cannot_return_via_raw_fallback_or_recent_context(tmp_path):
    class Model:
        async def plan(self, query, context):
            return QueryPlan(required_info=["access token"])

        async def answer(self, prompt):
            assert "DO_NOT_REINTRODUCE" not in prompt
            return "Unknown"

    async def run():
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "retracted.db"), model=Model())
        scope = Scope(tenant_id="t", owner_id="o")
        try:
            session = runtime.store.create_session(scope)
            stored = runtime.store.append_turn(
                scope,
                session.session_id,
                TurnInput(
                    request_id="s", events=[Event(role="user", content="Access token: DO_NOT_REINTRODUCE")]
                ),
            )
            from agent_memory.models import Candidate

            memory = runtime.store.add_candidates(
                scope,
                session.session_id,
                [
                    Candidate(
                        subject="access",
                        predicate="token",
                        value="DO_NOT_REINTRODUCE",
                        content="Access token: DO_NOT_REINTRODUCE",
                        evidence=[
                            Evidence(turn_id=stored.turn_id, event_index=0, quote="DO_NOT_REINTRODUCE")
                        ],
                    )
                ],
            )[0]
            runtime.store.evolve(
                scope, memory.memory_id, EvolutionInput(action="retract", expected_version=memory.version)
            )
            answer = await runtime.answer(
                scope, session.session_id, "What is my access token?", prepare=False, accelerate=False
            )
            assert not answer.sources
        finally:
            await runtime.close()

    asyncio.run(run())


def test_recall_first_index_respects_owner_and_retraction(tmp_path):
    from agent_memory.models import Candidate

    class Model:
        prompt = ""

        async def plan(self, query, context):
            return QueryPlan(required_info=["Alice's keys"])

        async def answer(self, prompt):
            self.prompt = prompt
            return "PUBLIC_KEY【来源1】"

    async def run():
        model = Model()
        runtime = MemoryRuntime(Settings(db_path=tmp_path / "scope.db"), model=model)
        alice = Scope(tenant_id="tenant", owner_id="alice")
        bob = Scope(tenant_id="tenant", owner_id="bob")
        try:
            sessions = {
                alice.owner_id: runtime.store.create_session(alice),
                bob.owner_id: runtime.store.create_session(bob),
            }
            for scope, text, retract in (
                (alice, "Alice's key is RETRACTED_SECRET", True),
                (alice, "Alice's key is PUBLIC_KEY", False),
                (bob, "Alice's key is BOB_PRIVATE_KEY", False),
            ):
                session = sessions[scope.owner_id]
                stored = runtime.store.append_turn(
                    scope,
                    session.session_id,
                    TurnInput(request_id=text, events=[Event(role="user", speaker="Alice", content=text)]),
                )
                memory = runtime.store.add_candidates(
                    scope,
                    session.session_id,
                    [Candidate(
                        subject="Alice",
                        predicate="key",
                        value=text,
                        content=text,
                        evidence=[Evidence(turn_id=stored.turn_id, event_index=0, quote=text)],
                    )],
                )[0]
                if retract:
                    runtime.store.evolve(
                        scope, memory.memory_id,
                        EvolutionInput(action="retract", expected_version=memory.version),
                    )
            answer = await runtime.answer(
                alice,
                sessions[alice.owner_id].session_id,
                "What are Alice's keys?",
                prepare=False,
                accelerate=False,
            )
            assert answer.plan.recall_policy == "recall_first"
            assert "PUBLIC_KEY" in model.prompt
            assert "RETRACTED_SECRET" not in model.prompt
            assert "BOB_PRIVATE_KEY" not in model.prompt
        finally:
            await runtime.close()

    asyncio.run(run())


def test_optional_dense_raw_projection_reuses_document_vectors(tmp_path):
    class Embedder:
        endpoint = "https://embedding.invalid"
        model = "fixture"

        async def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class Model:
        async def plan(self, query, context):
            return QueryPlan(required_info=["instrument"])

        async def answer(self, prompt):
            return "A hurdy-gurdy【来源1】"

    async def run():
        runtime = MemoryRuntime(
            Settings(db_path=tmp_path / "vectors.db", context_turns=0, grounding_semantic_enabled=True),
            model=Model(),
            embedder=Embedder(),
        )
        scope = Scope(tenant_id="t", owner_id="o")
        try:
            session = runtime.store.create_session(scope)
            await runtime.append_turn(
                scope,
                session.session_id,
                TurnInput(
                    request_id="one",
                    events=[Event(role="user", speaker="Nora", content="I practice a hurdy-gurdy.")],
                ),
            )
            first = await runtime.answer(
                scope, session.session_id, "Which instrument does Nora use?", prepare=False, accelerate=False
            )
            second = await runtime.answer(
                scope, session.session_id, "Which instrument does Nora use?", prepare=False, accelerate=False
            )
            assert first.grounding.semantic_fallback_used and first.grounding.raw_vector_inputs == 1
            assert second.grounding.raw_vector_reused == 1
        finally:
            await runtime.close()

    asyncio.run(run())


def test_historical_assistant_response_remains_context_not_user_fact():
    source = turn("advice", "I recommended a paper map.", speaker="assistant", role="assistant")
    result = ground_evidence(
        "What map did the assistant recommend?",
        QueryPlan(),
        [],
        session_id="s",
        current_turns=[source],
        resolved_turns={},
        blocked_events=set(),
    )
    assert result.sources and result.sources[0].metadata["source_events"][0]["context_only"]
    assert result.sources[0].metadata["assertion"] == "inferred"


def test_delete_owner_fences_inflight_raw_vector_projection(tmp_path):
    from agent_memory.ports import ConflictError, NotFoundError

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()

        class Embedder:
            endpoint = "https://embedding.invalid"
            model = "fixture"

            async def encode(self, texts):
                started.set()
                await finish.wait()
                return [[1.0, 0.0] for _ in texts]

        class Model:
            async def plan(self, query, context):
                return QueryPlan(required_info=["instrument"])

            async def answer(self, prompt):
                raise AssertionError("Deleted source must not reach answer generation")

        runtime = MemoryRuntime(
            Settings(db_path=tmp_path / "deleted.db", context_turns=0, grounding_semantic_enabled=True),
            model=Model(),
            embedder=Embedder(),
        )
        scope = Scope(tenant_id="t", owner_id="o")
        try:
            session = runtime.store.create_session(scope)
            await runtime.append_turn(
                scope,
                session.session_id,
                TurnInput(request_id="one", events=[Event(role="user", content="I practice a hurdy-gurdy.")]),
            )
            task = asyncio.create_task(
                runtime.answer(scope, session.session_id, "Which instrument?", prepare=False)
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            await runtime.delete_owner(scope)
            finish.set()
            with pytest.raises((ConflictError, NotFoundError)):
                await task
            count = runtime.retriever.projection._connection.execute(
                "SELECT COUNT(*) FROM projection_vectors"
            ).fetchone()[0]
            assert count == 0
        finally:
            await runtime.close()

    asyncio.run(run())
