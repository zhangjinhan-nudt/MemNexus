"""Where does each system spend its LLM calls? Built from the real call ledger.

The ledger (`<run>.calls.jsonl`) records every provider request with its full
body, so the call sites can be grouped by the system prompt that produced them
instead of by guessing from the source.  Files from a smoke run contain both
systems, so requests are attributed with the timestamp order of the two stages.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarise(path: Path, label: str) -> None:
    calls = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    requests = [c for c in calls if c.get("event") == "request"]
    print("=" * 76)
    print(f"{label}  ({path.name})")
    print("=" * 76)
    print(f"  provider requests: {len(requests)}")

    groups = collections.Counter()
    chars = collections.defaultdict(int)
    for call in requests:
        body = call.get("body") or {}
        messages = body.get("messages") or []
        system = ""
        for message in messages:
            if message.get("role") == "system":
                system = str(message.get("content") or "")
                break
        key = re.sub(r"\s+", " ", system).strip()[:70] or "(no system prompt)"
        groups[key] += 1
        chars[key] += sum(len(str(m.get("content") or "")) for m in messages)

    for key, count in groups.most_common():
        print(f"\n  [{count:3d} calls, {chars[key]:7d} chars] {key}")


def main() -> None:
    for path in sorted(ROOT.glob("result/**/*.calls.jsonl")):
        summarise(path, path.parent.name)


if __name__ == "__main__":
    main()
