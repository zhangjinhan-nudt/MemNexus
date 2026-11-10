"""Probe the retriever against a frozen memory snapshot without any model calls.

Answers two questions that a benchmark run cannot answer after the fact:

* at which rank does a gold memory sit for a given question
* how many sources fit under different evidence budgets

Only the local embedding service is used, so a sweep over budgets is free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus" / "memory-system" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_memory import snapshot as memory_snapshot  # noqa: E402
from agent_memory.models import Scope  # noqa: E402
from agent_memory.runtime import MemoryRuntime  # noqa: E402
from agent_memory.settings import Settings  # noqa: E402


def build_runtime(database: Path) -> MemoryRuntime:
    return MemoryRuntime.from_settings(
        Settings(
            db_path=database,
            pending_turns=5,
            context_turns=15,
            model_timeout=180,
            planner_mode="model",
            acceleration_enabled=False,
            output_language="en",
            evidence_grounding_enabled=True,
            grounding_semantic_enabled=False,
            grounding_max_raw_sources=3,
        )
    )


async def probe(snapshot_dir: Path, questions: list[dict], top_k: int, limit: int):
    out = []
    by_sample: dict[str, list[dict]] = {}
    for question in questions:
        by_sample.setdefault(question["sample_id"], []).append(question)
    for sample_id, items in sorted(by_sample.items()):
        folder = Path("result/tmp/probe") / sample_id
        folder.mkdir(parents=True, exist_ok=True)
        database = folder / "memory.sqlite3"
        if database.exists():
            database.unlink()
        memory_snapshot.thaw(snapshot_dir, sample_id, database)
        runtime = build_runtime(database)
        scope = Scope(tenant_id="locomo-pilot", owner_id=sample_id)
        sessions = runtime.store.sessions(scope)
        if not sessions:
            await runtime.close()
            continue
        session = sessions[0]
        for question in items[:limit]:
            result = await runtime.search(scope, session.session_id, question["question"], top_k=top_k)
            out.append(
                {
                    "sample_id": sample_id,
                    "qa_index": question["qa_index"],
                    "category": question["category"],
                    "question": question["question"],
                    "selected": len(result.results),
                    "context_tokens": result.context_tokens,
                    "items": [
                        {
                            "rank": index,
                            "score": round(float(hit.score), 4),
                            "subject": hit.metadata.get("subject"),
                            "predicate": hit.metadata.get("predicate"),
                            "value": str(hit.metadata.get("value"))[:80],
                            "content": hit.content[:120],
                        }
                        for index, hit in enumerate(result.results, 1)
                    ],
                }
            )
        await runtime.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--questions", required=True, help="questions.json from a small_ab phase")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    rows = asyncio.run(probe(Path(args.snapshots), questions, args.top_k, args.limit))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    for row in rows:
        print(f"{row['sample_id']}/{row['qa_index']} sel={row['selected']} tok={row['context_tokens']} {row['question'][:70]}")
        for item in row["items"][:6]:
            print(f"   {item['rank']:2d} {item['score']:.3f} {item['subject']} | {item['predicate']} | {item['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
