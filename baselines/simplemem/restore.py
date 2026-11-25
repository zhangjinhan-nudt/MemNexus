"""Verify or restore the pinned SimpleMem benchmark adapter (no API calls)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def canonical_bytes(relative: str, content: bytes) -> bytes:
    """Ignore platform line endings and credentials in example configurations."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    text = text.replace("\r\n", "\n")
    if relative in {"config.py.example", "config.paper.py.example"}:
        text = re.sub(
            r'(?m)^(\s*(?:OPENAI|JUDGE)_API_KEY\s*=\s*)[^\n]*$',
            lambda match: match.group(1) + '""',
            text,
        )
    return text.encode("utf-8")


def digest(relative: str, content: bytes) -> str:
    return hashlib.sha256(canonical_bytes(relative, content)).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def check_files(repo: Path, expected: dict[str, str]) -> list[str]:
    return [
        relative for relative, sha in expected.items()
        if not (repo / relative).is_file()
        or digest(relative, (repo / relative).read_bytes()) != sha
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2] / "SimpleMem")
    parser.add_argument("--apply", action="store_true", help="Apply only to the exact upstream file state")
    parser.add_argument("--check", action="store_true", help="Verify only (the default)")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    repo = args.repo.resolve()
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    if head != manifest["upstream_commit"]:
        raise SystemExit(f"Wrong upstream revision: {head}; expected {manifest['upstream_commit']}")
    patch = directory / "adapter.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise SystemExit("Adapter patch hash mismatch")
    mismatches = check_files(repo, manifest["files"])
    if not mismatches:
        print(f"SimpleMem adapter verified: {len(manifest['files'])} files; no API calls.")
        return
    if not args.apply:
        raise SystemExit("Adapter not restored, or files differ: " + ", ".join(mismatches))
    for relative in manifest["changed_files"]:
        path = repo / relative
        expected = manifest["base_hashes"][relative]
        if expected is None:
            if path.exists():
                raise SystemExit(f"Refusing to overwrite existing new adapter file: {relative}")
        elif not path.is_file() or digest(relative, path.read_bytes()) != expected:
            raise SystemExit(f"Refusing to overwrite local changes: {relative}")
    unrelated = set(mismatches) - set(manifest["changed_files"])
    if unrelated:
        raise SystemExit("Unrelated upstream files differ: " + ", ".join(sorted(unrelated)))
    git(repo, "apply", "--check", str(patch))
    git(repo, "apply", str(patch))
    remaining = check_files(repo, manifest["files"])
    if remaining:
        raise SystemExit("Post-apply verification failed: " + ", ".join(remaining))
    print(f"SimpleMem adapter restored and verified: {len(manifest['files'])} files; no API calls.")


if __name__ == "__main__":
    main()
