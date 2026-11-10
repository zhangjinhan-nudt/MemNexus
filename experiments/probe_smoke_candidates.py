"""List candidate smoke questions from the first official LoCoMo conversation.

Used to choose the 3-question smoke set before any model call, so the selection
cannot be tuned to a result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
POLAR = re.compile(
    r"^(?:does|do|did|is|are|was|were|has|have|had|can|could|will|would|should|am)\b",
    re.IGNORECASE,
)
RELATIVE = re.compile(
    r"\b(?:\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:days?|weeks?|months?|years?)\s+(?:before|after|prior to|earlier|later)\b",
    re.IGNORECASE,
)


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    sample = data[0]
    print("sample_id =", sample.get("sample_id"))
    for category in sorted(NAMES):
        rows = [qa for qa in sample["qa"] if qa["category"] == category]
        print(f"\n=== {NAMES[category]} (n={len(rows)}) ===")
        for index, qa in enumerate(rows[:6]):
            flags = []
            if POLAR.match(qa["question"].strip()):
                flags.append("POLAR")
            if RELATIVE.search(qa["question"]):
                flags.append("RELATIVE")
            print(f"  [{index}] {'/'.join(flags) or '-':14s} {qa['question'][:110]}")
            print(f"        gold = {str(qa.get('answer'))[:90]}")


if __name__ == "__main__":
    main()
