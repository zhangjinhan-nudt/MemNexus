"""Print one paired question with both answers and the losing side's evidence.

Used for the manual pass over the worst questions: the numbers say who lost,
this says what each system actually saw and wrote.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="result/bench260/rows.json")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--qa", type=int, required=True)
    parser.add_argument("--evidence", type=int, default=8)
    args = parser.parse_args()

    rows = [row for row in read(ROOT / args.rows)
            if row["sample_id"] == args.sample and row["qa_index"] == args.qa]
    if not rows:
        print("no such row")
        return 1
    for row in rows:
        print("=" * 100)
        print(f"{row['system']} | {row.get('category')} | {row['question']}")
        print(f"  gold: {row.get('gold_answer')}")
        print(f"  pred: {row.get('prediction')}")
        calls = row.get("model_calls") or {}
        print(f"  tokens in/out: {calls.get('input_tokens')}/{calls.get('output_tokens')}")
        sources = row.get("retrieved_sources") or []
        if row["system"] == "MemNexus":
            print(f"  sources: {len(sources)}")
            pattern = re.compile(r"[a-z0-9]{4,}")
            gold_terms = set(pattern.findall(str(row.get("gold_answer", "")).casefold()))
            for source in sources[: args.evidence]:
                metadata = source.get("metadata") or {}
                print(f"    - {metadata.get('subject')} | {metadata.get('predicate')} | "
                      f"{str(metadata.get('value'))[:70]}")
            hits = [source for source in sources
                    if gold_terms & set(pattern.findall(json.dumps(source, ensure_ascii=False).casefold()))]
            print(f"  sources containing a gold term: {len(hits)}")
        else:
            print(f"  contexts: {len(sources)}")
            for context in sources[: args.evidence]:
                print(f"    - {str(context)[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
