"""Automatic LT admission obeys the existing ablation switch, without changing scope."""

import asyncio
import shutil
from datetime import UTC, datetime

import pytest

from agent_memory.long_term import LongTermMemory
from agent_memory.models import Candidate, Event, Evidence, EvolutionInput, ExtractionResult, Scope, TurnInput
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings
from agent_memory.store import SQLiteStore

OWNER = Scope(tenant_id="tenant", owner_id="owner")
QUOTE = "I attended the Aster expedition in 2025."
QUESTION = "What happened at the Aster expedition?"


class FrozenExtractor:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def extract(self, window):
        self.calls += 1
        result = ExtractionResult.model_validate_json(self.response)
        assert {ref.turn_id for item in result.candidates for ref in item.evidence} <= {
            turn.turn_id for turn in window.new_turns
        }
        return result


def seed(tmp_path, *, scope_type="user", durable=True, role="user"):
    path = tmp_path / "seed.db"
    store = SQLiteStore(path)
    session = store.create_session(OWNER, project_id="project-a")
    turn = store.append_turn(OWNER, session.session_id, TurnInput(
        request_id="history", events=[
            Event(role="user", content="请将以下信息保存到长期记忆。"),
            Event(role=role, speaker="Mira", content=QUOTE, occurred_at=datetime(2025, 6, 1, tzinfo=UTC)),
        ],
    ))
    candidate = Candidate(
        content=QUOTE, kind="event", subject="Mira", predicate="participation", value="Aster expedition in 2025",
        durable=durable, scope_type=scope_type,
        semantics={"cardinality": "one", "event_key": "aster_2025", "event_identity": "Aster expedition 2025"},
        evidence=[Evidence(turn_id=turn.turn_id, event_index=1, quote=QUOTE)],
    )
    response = ExtractionResult(candidates=[candidate], summary="Recorded the stated expedition.").model_dump_json()
    assert store.memories(OWNER) == []
    store.close()  # Checkpoint before copying the identical empty-memory input to each arm.
    return path, session.session_id, response


@pytest.mark.parametrize("scope_type", ["user", "project"])
def test_same_frozen_extraction_obeys_promotion_switch_and_authorized_cross_session_recall(tmp_path, scope_type):
    path, session_id, response = seed(tmp_path, scope_type=scope_type)

    async def run():
        observed = {}
        for enabled in (True, False):
            arm_path = tmp_path / f"arm-{enabled}.db"
            shutil.copy2(path, arm_path)
            model = FrozenExtractor(response)
            runtime = MemoryRuntime(Settings(
                db_path=arm_path, promotion_enabled=enabled, planner_mode="rule", maintenance_interval=0,
            ), model=model)
            try:
                assert runtime.store.memories(OWNER) == []
                result = await runtime.extract(OWNER, session_id)
                memory = result["memories"][0]
                assert model.calls == 1
                assert memory.tier == ("long" if enabled else "short")
                assert memory.scope_type == scope_type and memory.durable is True
                assert memory.scope_id == (OWNER.owner_id if scope_type == "user" else "project-a")
                original = ExtractionResult.model_validate_json(response).candidates[0]
                stored = {key: memory.model_dump()[key] for key in Candidate.model_fields}
                # Provenance coordinates are resolved by storage, not generated
                # by extraction. Promotion must preserve all extracted meaning.
                for ref in stored["evidence"]:
                    assert ref["start"] is not None and ref["end"] - ref["start"] == len(ref["quote"])
                    ref.pop("start")
                    ref.pop("end")
                expected = original.model_dump(exclude={"evidence": {"__all__": {"start", "end"}}})
                assert stored == expected
                observed[enabled] = memory.evidence
                assert bool(runtime.store.memories(OWNER, tier="long")) is enabled

                same_project = runtime.store.create_session(OWNER, project_id="project-a")
                found = await runtime.search(OWNER, same_project.session_id, QUESTION)
                assert bool(found.results) is enabled
                if enabled:
                    assert found.results[0].metadata["tier"] == "long"
                    assert found.results[0].metadata["evidence"][0]["quote"] == QUOTE
                else:
                    local = await runtime.search(OWNER, session_id, QUESTION)
                    assert local.results  # Off keeps the original short memory; it does not erase evidence.

                other_project = runtime.store.create_session(OWNER, project_id="project-b")
                found = await runtime.search(OWNER, other_project.session_id, QUESTION)
                assert bool(found.results) is (enabled and scope_type == "user")
                for stranger in (Scope(tenant_id="tenant", owner_id="other"),
                                 Scope(tenant_id="other", owner_id="owner")):
                    unrelated = runtime.store.create_session(stranger, project_id="project-a")
                    assert not (await runtime.search(stranger, unrelated.session_id, QUESTION)).results
            finally:
                await runtime.close()
        assert observed[True] == observed[False]  # Both arms used identical source IDs and quotes.

    asyncio.run(run())


def test_disabled_admission_does_not_remove_existing_long_memory_and_default_store_commit_is_compatible(tmp_path):
    path, session_id, response = seed(tmp_path)
    store = SQLiteStore(path)
    session = store.get_session(OWNER, session_id)
    existing = store.commit_extraction(OWNER, session, store.turns(OWNER, session_id),
                                       ExtractionResult.model_validate_json(response))[0]
    assert existing.tier == "long"  # Existing direct callers keep the default admission behavior.
    store.close()

    async def run():
        model = FrozenExtractor(response)
        runtime = MemoryRuntime(Settings(db_path=path, promotion_enabled=False, planner_mode="rule",
                                         maintenance_interval=0), model=model)
        try:
            await runtime.extract(OWNER, session_id)
            retained = runtime.store.get_memory(OWNER, existing.memory_id)
            assert retained.tier == "long" and retained.evidence == existing.evidence
            assert model.calls == 0
            new_session = runtime.store.create_session(OWNER)
            assert (await runtime.search(OWNER, new_session.session_id, QUESTION)).results
        finally:
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_request_does_not_override_session_scope(tmp_path, enabled):
    path, session_id, response = seed(tmp_path, scope_type="session", durable=False)

    async def run():
        runtime = MemoryRuntime(Settings(db_path=path, promotion_enabled=enabled, planner_mode="rule",
                                         maintenance_interval=0), model=FrozenExtractor(response))
        try:
            result = await runtime.extract(OWNER, session_id)
            memory = result["memories"][0]
            assert memory.tier == "short" and memory.scope_type == "session" and not memory.durable
            assert memory.scope_id == session_id
            other = runtime.store.create_session(OWNER)
            assert not (await runtime.search(OWNER, other.session_id, QUESTION)).results
        finally:
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
def test_admission_switch_does_not_bypass_original_speaker_evidence_check(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "false")
    path, session_id, response = seed(tmp_path, role="assistant")

    async def run():
        runtime = MemoryRuntime(Settings(db_path=path, promotion_enabled=enabled, maintenance_interval=0),
                                model=FrozenExtractor(response))
        try:
            with pytest.raises(ValueError, match="Assistant output alone"):
                await runtime.extract(OWNER, session_id)
            assert runtime.store.memories(OWNER) == []
        finally:
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("new_position", [None, 10])
def test_long_merge_retains_legacy_null_and_distinct_literal_positions(tmp_path, new_position):
    store = SQLiteStore(tmp_path / "legacy-position.db")
    try:
        session = store.create_session(OWNER)
        turn = store.append_turn(OWNER, session.session_id, TurnInput(
            request_id="source", events=[Event(role="user", content="blue then blue")],
        ))
        base = Candidate(content="Nora prefers blue.", subject="Nora", predicate="preference", value="blue",
                         scope_type="user", durable=True,
                         evidence=[Evidence(turn_id=turn.turn_id, event_index=0, quote="blue")])
        [target] = store.add_candidates(OWNER, session.session_id, [base])
        target = store.evolve(OWNER, target.memory_id, EvolutionInput(action="promote", expected_version=1))
        assert target.evidence[0].start is None  # Repeated legacy quote stays ambiguous.
        source_ref = Evidence(turn_id=turn.turn_id, event_index=0, quote="blue", start=new_position,
                              end=None if new_position is None else new_position + 4)
        [source] = store.add_candidates(OWNER, session.session_id,
                                        [base.model_copy(update={"evidence": [source_ref]})])
        report = asyncio.run(LongTermMemory(store).process_extraction(OWNER, [source]))
        assert [item["action"] for item in report["applied"]] == ["merge"]
        saved = store.get_memory(OWNER, target.memory_id)
        expected = {(None, None)} if new_position is None else {(None, None), (10, 14)}
        assert {(ref.start, ref.end) for ref in saved.evidence} == expected
        assert saved.content == base.content and saved.tier == "long"
    finally:
        store.close()


def test_long_merge_capacity_counts_repeated_quotes_at_distinct_positions(tmp_path):
    store = SQLiteStore(tmp_path / "position-capacity.db")
    try:
        session = store.create_session(OWNER)
        turn = store.append_turn(OWNER, session.session_id, TurnInput(
            request_id="source", events=[Event(role="user", content="blue " * 31)],
        ))
        refs = [Evidence(turn_id=turn.turn_id, event_index=0, quote="blue", start=i * 5, end=i * 5 + 4)
                for i in range(31)]
        base = Candidate(content="Nora prefers blue.", subject="Nora", predicate="preference", value="blue",
                         scope_type="user", durable=True, evidence=refs[:30])
        [target] = store.add_candidates(OWNER, session.session_id, [base])
        target = store.evolve(OWNER, target.memory_id, EvolutionInput(action="promote", expected_version=1))
        [source] = store.add_candidates(OWNER, session.session_id,
                                        [base.model_copy(update={"evidence": refs[30:]})])
        report = asyncio.run(LongTermMemory(store).process_extraction(OWNER, [source]))
        assert report["applied"] == []
        assert report["review"] == [{"memory_id": source.memory_id, "reason": "source_capacity_keep_separate"}]
        # Capacity must be handled before store.evolve, without losing a source
        # or repeatedly failing the automatic promotion job.
        retained = store.get_memory(OWNER, source.memory_id)
        assert retained.status == "active" and retained.tier == "short"
        assert retained.evidence == refs[30:]
        assert store.get_memory(OWNER, target.memory_id).evidence == refs[:30]
    finally:
        store.close()
