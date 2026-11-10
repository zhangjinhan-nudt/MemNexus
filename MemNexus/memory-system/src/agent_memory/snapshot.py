"""Freeze a built memory base to disk so QA runs are reproducible and cheap.

Measured on three 150-question runs of the *same* conversation set, rebuilding
memory from scratch produced a different evidence set every single time:
SimpleMem's own unchanged code matched 0/150 questions run-to-run, and MemNexus
flipped 14-18 question verdicts between identical runs.  At temperature 0.
The build stage -- not the answering stage -- is the noise source, and its
magnitude (+-14~18 questions) is 4-9x the effect any optimisation is trying to
measure (+-2~4).  Freezing the built base and replaying it removes that noise
and, as a side effect, removes the build cost (65% of a 150-question run).

What is frozen: the SQLite memory base (turns, memories, session state, vector
projection) exactly as the build stage left it.
What is not frozen: planning, retrieval, grounding and answering.  Those still
run live, so architecture changes remain measurable -- the freeze *controls* a
nuisance variable, it does not remove the treatment.

Three snapshots per conversation are meant to be built (see
``snapshot_manifest``); the paper reports one as the main condition and the
other two as a memory-base robustness check, which answers "would your result
survive a different draw?" directly instead of hiding it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_FORMAT_VERSION = 1


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_paths(directory: Path | str, sample_id: str) -> tuple[Path, Path]:
    """Return ``(database, manifest)`` for one conversation's frozen base."""
    folder = Path(directory)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(sample_id))
    return folder / f"{safe}.sqlite3", folder / f"{safe}.json"


def freeze(store, directory: Path | str, sample_id: str, *, metadata: dict | None = None) -> dict:
    """Write a consistent copy of ``store`` to ``directory`` and return its manifest.

    Uses the SQLite online-backup API rather than a file copy: the store runs in
    WAL mode, so copying ``memory.sqlite3`` alone would miss committed pages that
    still live in the ``-wal`` file and could freeze a truncated base.
    """
    database, manifest_path = snapshot_paths(directory, sample_id)
    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=database.parent, prefix=".freeze-") as staging:
        staged = Path(staging) / "memory.sqlite3"
        # store.backup_to takes the store lock and runs SQLite's own integrity
        # check, so a corrupt base can never be recorded as a frozen snapshot.
        store.backup_to(staged)
        digest = sha256_of(staged)
        # Replace atomically so an interrupted run can never leave a half
        # written base that a later run would happily load as "frozen".
        os.replace(staged, database)
    manifest = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "sample_id": str(sample_id),
        "created_at": _utcnow(),
        "sha256": digest,
        "bytes": database.stat().st_size,
        "counts": store.base_counts(),
        **(metadata or {}),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def available(directory: Path | str, sample_id: str) -> bool:
    database, _ = snapshot_paths(directory, sample_id)
    return database.is_file()


def thaw(directory: Path | str, sample_id: str, destination: Path | str, *, verify: bool = True) -> dict:
    """Copy a frozen base into ``destination`` and return its manifest.

    ``verify`` re-hashes the copy, so a corrupted or partially written snapshot
    fails loudly instead of silently changing every downstream number.
    """
    database, manifest_path = snapshot_paths(directory, sample_id)
    if not database.is_file():
        raise FileNotFoundError(f"No memory snapshot for {sample_id!r} in {directory}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(database, destination)
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify and manifest.get("sha256"):
        actual = sha256_of(destination)
        if actual != manifest["sha256"]:
            raise ValueError(
                f"Memory snapshot for {sample_id!r} failed its checksum "
                f"({actual[:12]} != {manifest['sha256'][:12]})"
            )
    return manifest
