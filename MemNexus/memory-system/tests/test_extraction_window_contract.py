"""Budget, source continuity and persisted-state contracts, independent of QA gold."""

import asyncio
import json
from datetime import UTC, datetime

import pytest

from agent_memory.models import (
    AnswerResponse,
    Event,
    ExtractionResult,
    Scope,
    Session,
    StatePatch,
    Turn,
    TurnInput,
)
from agent_memory.ports import ConflictError
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.short_term import ShortTermMemory

STAMP = datetime(2024, 6, 1, tzinfo=UTC)
SCOPE = Scope(tenant_id="t", owner_id="o")


def turn(sequence, content, role="user"):
    return Turn(
        turn_id=f"t{sequence}", request_id=f"r{sequence}", session_id="s", sequence=sequence,
        events=[Event(role=role, content=content, occurred_at=STAMP)],
    )


def cost(item):
    return len(json.dumps(item.prompt_dump(), ensure_ascii=False))


class WindowStore:
    def __init__(self, session, turns):
        self.session, self.turns = session, turns

    def session_snapshot(self, scope, session_id):
        assert scope == SCOPE and session_id == "s"
        return self.session, self.turns

    def memories(self, *args):
        return []


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_large_recent_context_preserves_contiguous_context_boundary(role):
    session = Session(session_id="s", processed_sequence=2)
    previous = turn(1, "The parcel is labelled ORCHID.", role)
    large = turn(2, "Long unrelated passage. " * 180, "assistant")
    new = turn(3, "Please check the labelled parcel.")
    budget = len(session.model_dump_json()) + 1000 + cost(new) + cost(previous) + 8
    settings = Settings(window_token_limit=budget)
    window = asyncio.run(ShortTermMemory(WindowStore(session, [previous, large, new]), settings).window(SCOPE, "s"))
    assert window.new_turns == [new]
    # The optional sparse-selection policy was withdrawn after fresh evaluation.
    # Preserve this same three-role scenario, now checking the original break:
    # an over-budget recent turn must not make a non-contiguous older turn enter.
    assert window.context_turns == []
    assert (previous.sequence, large.sequence, window.new_turns[0].sequence) == (1, 2, 3)
    assert previous.events[0].occurred_at == STAMP
    assert previous.events[0].content == "The parcel is labelled ORCHID."
    assert session.processed_sequence == 2
    assert sum(cost(t) for t in window.new_turns + window.context_turns) <= budget - 1000 - len(session.model_dump_json())


def test_context_horizon_and_chronological_order_remain_bounded():
    session = Session(session_id="s", processed_sequence=4)
    rows = [turn(i, f"Observation {i}.") for i in range(1, 6)]
    settings = Settings(window_token_limit=8000, context_turns=2)
    window = asyncio.run(ShortTermMemory(WindowStore(session, rows), settings).window(SCOPE, "s"))
    assert [t.sequence for t in window.context_turns] == [3, 4]
    assert [t.sequence for t in window.new_turns] == [5]


def test_oversized_single_turn_is_isolated_without_extra_context():
    session = Session(session_id="s", processed_sequence=1)
    old, new, pending = turn(1, "Earlier context."), turn(2, "x" * 2200), turn(3, "Next input.")
    budget = len(session.model_dump_json()) + 1000 + cost(new) - 1
    settings = Settings(window_token_limit=budget, max_turn_chars=6000)
    window = asyncio.run(ShortTermMemory(WindowStore(session, [old, new, pending]), settings).window(SCOPE, "s"))
    assert window.new_turns == [new]
    assert window.context_turns == []
    assert window.reference_fields == []
    assert session.processed_sequence == 1


@pytest.mark.parametrize("oversized", ["summary", "state"])
def test_clamped_state_is_what_is_committed_and_reported(tmp_path, oversized):
    async def scenario():
        original = ExtractionResult(
            summary="s" * 2800 if oversized == "summary" else "Short summary.",
            state_patch=StatePatch(goal="g" * 9000, plan=["p" * 200], pending_tasks=["t" * 200])
            if oversized == "state" else StatePatch(goal="Keep the delivery record."),
        )

        class Model:
            async def extract(self, window):
                return original

        runtime = MemoryRuntime(Settings(db_path=tmp_path / "state.db", state_clamp_enabled=True), model=Model())
        try:
            session = runtime.store.create_session(SCOPE, topic="Delivery project")
            runtime.store.append_turn(SCOPE, session.session_id, TurnInput(
                request_id="first", events=[Event(role="user", content="A package arrived.")],
            ))
            before = original.model_dump_json()
            result = await runtime.short_term.extract(SCOPE, session.session_id)
            saved = runtime.store.get_session(SCOPE, session.session_id)
            assert len(saved.model_dump_json()) <= runtime.settings.window_token_limit // 3
            assert saved.summary == result["summary"]
            assert saved.topic == "Delivery project"
            assert saved.processed_sequence == 1
            assert original.model_dump_json() == before  # Raw model response remains auditable.
            runtime.store.append_turn(SCOPE, session.session_id, TurnInput(
                request_id="second", events=[Event(role="user", content="The package was signed for.")],
            ))
            window = await runtime.short_term.window(SCOPE, session.session_id)
            assert [t.sequence for t in window.new_turns] == [2]
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_disabled_clamp_rejects_before_advancing_watermark(tmp_path):
    async def scenario():
        class Model:
            async def extract(self, window):
                return ExtractionResult(summary="s" * 2800)

        runtime = MemoryRuntime(Settings(db_path=tmp_path / "strict.db", state_clamp_enabled=False), model=Model())
        try:
            session = runtime.store.create_session(SCOPE)
            runtime.store.append_turn(SCOPE, session.session_id, TurnInput(
                request_id="first", events=[Event(role="user", content="A package arrived.")],
            ))
            with pytest.raises(ValueError, match="state exceeds budget"):
                await runtime.short_term.extract(SCOPE, session.session_id)
            saved = runtime.store.get_session(SCOPE, session.session_id)
            assert saved.processed_sequence == 0 and saved.summary == ""
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_clamp_cannot_erase_caller_metadata_to_claim_a_valid_state(tmp_path):
    async def scenario():
        class Model:
            async def extract(self, window):
                return ExtractionResult(summary="s" * 2800)

        runtime = MemoryRuntime(Settings(db_path=tmp_path / "metadata.db", state_clamp_enabled=True), model=Model())
        try:
            session = runtime.store.create_session(SCOPE, topic="Caller topic " * 180)
            runtime.store.append_turn(SCOPE, session.session_id, TurnInput(
                request_id="first", events=[Event(role="user", content="An observation.")],
            ))
            with pytest.raises(ValueError, match="Session metadata"):
                await runtime.short_term.extract(SCOPE, session.session_id)
            saved = runtime.store.get_session(SCOPE, session.session_id)
            assert saved.topic == session.topic
            assert saved.processed_sequence == 0 and saved.summary == ""
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_clamped_result_retains_the_transaction_revision_fence(tmp_path):
    async def scenario():
        class Model:
            async def extract(self, window):
                runtime.store.append_turn(SCOPE, window.session.session_id, TurnInput(
                    request_id="concurrent", events=[Event(role="user", content="A concurrent observation.")],
                ))
                return ExtractionResult(summary="s" * 2800)

        runtime = MemoryRuntime(Settings(db_path=tmp_path / "concurrent.db", state_clamp_enabled=True), model=Model())
        try:
            session = runtime.store.create_session(SCOPE)
            runtime.store.append_turn(SCOPE, session.session_id, TurnInput(
                request_id="first", events=[Event(role="user", content="An observation.")],
            ))
            with pytest.raises(ConflictError, match="Session changed"):
                await runtime.short_term.extract(SCOPE, session.session_id)
            saved = runtime.store.get_session(SCOPE, session.session_id)
            assert saved.processed_sequence == 0 and saved.summary == ""
            assert len(runtime.store.turns(SCOPE, session.session_id)) == 2
        finally:
            await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("pending_sequence,trailing_receipts,initial_microsecond", [
    (9, 0, 123456), (10, 0, 123456), (99, 0, 123456), (100, 0, 123456),
    (9, 1, 123456), (99, 1, 123456), (1, 0, 0),
])
def test_clamp_sizes_metadata_at_decimal_boundaries(tmp_path, monkeypatch, pending_sequence, trailing_receipts, initial_microsecond):
    # 9/99 exercise revision growth; 10/100 exercise watermark growth. The
    # former clamp checked old metadata and could persist limit+1 characters.
    monkeypatch.setattr("agent_memory.store.utcnow", lambda: STAMP.replace(microsecond=initial_microsecond))

    async def scenario():
        original = None

        class Model:
            async def extract(self, window):
                nonlocal original
                empty = window.session.model_copy(update={"summary": "", "goal": "", "plan": []})
                # Dropping this plan previously landed exactly on the old-state
                # limit, leaving no room for commit's metadata digit changes.
                goal = "g" * (2000 - len(empty.model_dump_json()))
                original = ExtractionResult(summary="", state_patch=StatePatch(goal=goal, plan=["x"]))
                monkeypatch.setattr("agent_memory.store.utcnow", lambda: STAMP.replace(microsecond=123456))
                return original

        runtime = MemoryRuntime(Settings(db_path=tmp_path / "boundary.db", state_clamp_enabled=True, pending_turns=1), model=Model())
        try:
            session = runtime.store.create_session(SCOPE, topic="Caller-owned topic")
            for index in range(pending_sequence):
                runtime.store.append_turn(SCOPE, session.session_id, TurnInput(
                    request_id=str(index), events=[Event(role="user", content="An observation.")],
                ))
            if trailing_receipts:
                answer = AnswerResponse(generated_text="Recorded reply.", citations=[], sources=[], retrieval_count=0, elapsed_ms=0)
                receipt = runtime.store.save_answer(SCOPE, session.session_id, answer, [])
                runtime.store.append_turn(SCOPE, session.session_id, TurnInput(
                    request_id="receipt", events=[Event(role="assistant", content=answer.generated_text, answer_id=receipt)],
                ))
            session = runtime.store.get_session(SCOPE, session.session_id)
            session.processed_sequence = pending_sequence - 1
            with runtime.store.transaction():
                runtime.store._save_session(SCOPE, session)
            before_revision = session.revision
            result = await runtime.short_term.extract(SCOPE, session.session_id)
            saved = runtime.store.get_session(SCOPE, session.session_id)
            assert len(saved.model_dump_json()) <= 2000
            assert saved.processed_sequence == pending_sequence + trailing_receipts
            assert result["processed_sequence"] == pending_sequence  # Existing response contract is unchanged.
            assert saved.revision == before_revision + 1
            assert saved.topic == "Caller-owned topic"
            assert saved.updated_at.microsecond == 123456
            assert saved.summary == result["summary"]
            assert original.state_patch.plan == ["x"]
        finally:
            await runtime.close()

    asyncio.run(scenario())
