"""Small real-provider smoke test for the semantic execution path.

Uses immutable memories from the completed 100-question run. It is a mechanism
check, not a benchmark result; no gold is sent to the model and no labels are
generated here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

from run_synthesis_replay import restore

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "MemNexus/memory-system/src"
import sys
sys.path.insert(0, str(sys_path))

from agent_memory.models import Scope
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings


CASES = [
    ("conv-47", 8),   # count must not be detached from the member set
    ("conv-47", 9),   # names and count should share entity evidence
    ("conv-47", 79),  # adopted puppy subject/event constraint
    ("conv-48", 17),  # pronoun/entity and relative purchase time
    ("conv-48", 105), # milestone followed by concrete explanation
    ("conv-49", 25),  # inverse duration to start year
]


async def main(output: Path):
    os.environ["MEMORY_SEMANTIC_EXECUTION"] = "true"
    os.environ["MEMORY_PREMISE_RETRIEVAL"] = "true"
    source = ROOT / "result/locomo_100_collection_20261013_v3.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    rows = {(r["sample_id"], r["qa_index"]): r for r in original["rows"] if r["system"] == "MemNexus"}
    ledger_dir = source.with_name(source.stem + "_ledgers")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "semantic execution smoke; frozen memories; no judge", "rows": [], "status": "running"}
    for sample_id, qa_index in CASES:
        old = rows[(sample_id, qa_index)]
        snapshot = json.loads((ledger_dir / f"{sample_id}_memnexus.memory.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="memnexus-semantic-smoke-") as folder:
            runtime = MemoryRuntime.from_settings(Settings(
                db_path=Path(folder) / "memory.sqlite3", model_timeout=180, model_max_tokens=4096,
                planner_mode="model", acceleration_enabled=False, output_language="en",
                grounding_semantic_enabled=False, answer_synthesis_enabled=True,
                semantic_execution_enabled=True, premise_retrieval_enabled=True,
            ))
            scope = Scope(tenant_id="locomo-pilot", owner_id=sample_id)
            try:
                session = restore(runtime.store, scope, snapshot)
                before = dict(runtime.model.usage)
                answer = await runtime.answer(scope, session.session_id, old["evaluation_query"],
                                              top_k=20, timeout=180, accelerate=False, prepare=False)
                report["rows"].append({
                    "sample_id": sample_id, "qa_index": qa_index, "question": old["question"],
                    "gold_for_review": old["gold_answer"], "old_prediction": old["prediction"],
                    "new_prediction": answer.generated_text,
                    "usage": {k: runtime.model.usage[k] - before.get(k, 0) for k in runtime.model.usage},
                    "reasoning": answer.reasoning,
                    "retrieved_sources": [h.model_dump(mode="json") for h in answer.sources],
                })
                print(json.dumps(report["rows"][-1], ensure_ascii=False), flush=True)
            finally:
                await runtime.close()
    report["status"] = "completed"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args().output))
