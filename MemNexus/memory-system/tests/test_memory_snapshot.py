"""Frozen memory bases: a replayed base must be byte-for-byte the built one.

Why this exists: rebuilding the same conversation changes the evidence set on
150/150 questions and flips 14-18 verdicts between identical runs. Freezing the
build stage is the only way a before/after comparison on this benchmark is
readable, so the freeze itself has to be trustworthy -- if a snapshot silently
loses a WAL page or a projection row, every number downstream is wrong in a way
no test would otherwise catch.
"""

from __future__ import annotations

import json
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_memory import snapshot  # noqa: E402
from agent_memory.models import Event, Evidence, Memory, Scope, TurnInput  # noqa: E402
from agent_memory.store import SQLiteStore  # noqa: E402


def make_store(path: pathlib.Path) -> tuple[SQLiteStore, Scope]:
    store = SQLiteStore(path)
    scope = Scope(tenant_id="t", owner_id="o")
    session = store.create_session(scope, "topic")
    store.append_turn(
        scope,
        session.session_id,
        TurnInput(
            request_id="r1",
            events=[
                Event(role="user", speaker="Caroline", content="Caroline: I went to the LGBTQ support group."),
                Event(role="user", speaker="Melanie", content="Melanie: That sounds great."),
            ],
        ),
    )
    with store.transaction():
        store._save_memory(
            scope,
            Memory(
                memory_id="m1",
                session_id=session.session_id,
                scope_id=session.session_id,
                subject="Caroline",
                predicate="attended",
                value="LGBTQ support group",
                content="Caroline attended the LGBTQ support group.",
                evidence=[Evidence(turn_id="x", event_index=0, quote="LGBTQ support group")],
            ),
        )
    return store, scope


def test_freeze_then_thaw_preserves_counts_and_content(tmp_path):
    store, scope = make_store(tmp_path / "live.sqlite3")
    try:
        manifest = snapshot.freeze(store, tmp_path / "snaps", "sample-1")
        assert manifest["counts"]["memories"] == 1
        assert manifest["counts"]["turns"] == 1
        assert len(manifest["sha256"]) == 64
    finally:
        store.close()

    restored = tmp_path / "replayed.sqlite3"
    snapshot.thaw(tmp_path / "snaps", "sample-1", restored)
    replay = SQLiteStore(restored)
    try:
        # A replayed base must be the same experimental condition, not a
        # similar-looking one.
        assert replay.base_counts() == manifest["counts"]
        assert [m.memory_id for m in replay.memories(scope)] == ["m1"]
        assert [t.request_id for t in replay.turns(scope, replay.sessions(scope)[0].session_id)] == ["r1"]
    finally:
        replay.close()


def test_freeze_is_atomic_and_leaves_no_partial_snapshot(tmp_path):
    store, _ = make_store(tmp_path / "live.sqlite3")
    try:
        snapshot.freeze(store, tmp_path / "snaps", "sample-1")
        database, manifest_path = snapshot.snapshot_paths(tmp_path / "snaps", "sample-1")
        assert database.is_file() and manifest_path.is_file()
        # Staging happens in a temp directory: a crash mid-copy must not leave
        # a file a later run would load as frozen.
        leftovers = [p.name for p in (tmp_path / "snaps").iterdir() if p.name.startswith(".freeze-")]
        assert leftovers == []
    finally:
        store.close()


def test_thaw_detects_a_corrupted_snapshot(tmp_path):
    store, _ = make_store(tmp_path / "live.sqlite3")
    try:
        snapshot.freeze(store, tmp_path / "snaps", "sample-1")
    finally:
        store.close()
    database, _ = snapshot.snapshot_paths(tmp_path / "snaps", "sample-1")
    database.write_bytes(b"not a database")
    with pytest.raises(ValueError, match="checksum"):
        snapshot.thaw(tmp_path / "snaps", "sample-1", tmp_path / "out.sqlite3")


def test_available_reports_only_completed_snapshots(tmp_path):
    assert snapshot.available(tmp_path / "snaps", "nope") is False
    store, _ = make_store(tmp_path / "live.sqlite3")
    try:
        snapshot.freeze(store, tmp_path / "snaps", "sample/with spaces")
    finally:
        store.close()
    # Sample ids are filename-sanitised, and lookup must apply the same rule or
    # a replay run silently rebuilds.
    assert snapshot.available(tmp_path / "snaps", "sample/with spaces") is True


def test_manifest_records_what_the_base_was_built_from(tmp_path):
    store, _ = make_store(tmp_path / "live.sqlite3")
    try:
        snapshot.freeze(
            store,
            tmp_path / "snaps",
            "sample-1",
            metadata={"raw_turns": 2, "model": "deepseek-v4-flash", "builder": "MemNexus"},
        )
    finally:
        store.close()
    _, manifest_path = snapshot.snapshot_paths(tmp_path / "snaps", "sample-1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Provenance travels with the file: a snapshot whose builder or model is
    # unknown cannot be used as an experimental condition.
    assert manifest["builder"] == "MemNexus"
    assert manifest["model"] == "deepseek-v4-flash"
    assert manifest["raw_turns"] == 2
    assert manifest["format_version"] == snapshot.SNAPSHOT_FORMAT_VERSION
