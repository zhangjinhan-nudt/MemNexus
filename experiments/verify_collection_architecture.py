"""Small live, fixed-evidence development check. Gold is never sent to the model.

This isolates query compilation/composition on saved real LoCoMo evidence. It is
not a fresh retrieval evaluation, an independent test set, or a SimpleMem rerun.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))

from agent_memory.composition import adjudicated_answer, render
from agent_memory.models import (
    Event,
    ExtractionWindow,
    Hit,
    Scope,
    TurnInput,
)
from agent_memory.providers import build_model
from agent_memory.settings import Settings
from agent_memory.store import SQLiteStore


async def run(source, output):
    if output.exists():
        raise FileExistsError(output)
    original = json.loads(source.read_text(encoding="utf-8"))
    selected = {("conv-41", 3), ("conv-42", 10), ("conv-47", 5), ("conv-47", 8), ("conv-50", 5)}
    rows = [r for r in original["rows"] if (r["sample_id"], r["qa_index"]) in selected]
    assert len(rows) == 5 and all(r["retrieved_sources"] for r in rows)
    model = build_model(Settings(model_timeout=90, model_max_tokens=6500))
    if model is None:
        raise RuntimeError("A configured live model is required")
    if model.model != "deepseek-v4-flash":
        await model.aclose()
        raise ValueError("This verification requires the user's deepseek-v4-flash configuration")
    model.ledger_path = str(output.with_suffix(".usage.json"))
    code_files = list((ROOT / "MemNexus/memory-system/src/agent_memory").glob("*.py"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output.with_suffix(".source.zip"), "x", zipfile.ZIP_DEFLATED) as archive:
        for path in [*code_files, Path(__file__)]:
            archive.write(path, path.relative_to(ROOT))
    report = {
        "status": "running", "started_at_utc": datetime.now(UTC).isoformat(),
        "model": model.model, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in code_files},
        "protocol": "5 selected development questions; frozen evidence; no gold in model input; no judge",
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Real source text, no question or answer: exercise new extraction schema and transactional indexing.
        emotion = next(r for r in rows if r["sample_id"] == "conv-42")
        event = emotion["retrieved_sources"][0]["metadata"]["source_events"][0]
        with tempfile.TemporaryDirectory(prefix="memnexus-composition-smoke-") as folder:
            store = SQLiteStore(Path(folder) / "memory.sqlite3")
            try:
                scope = Scope(tenant_id="smoke", owner_id="smoke")
                session = store.create_session(scope)
                turn = store.append_turn(scope, session.session_id, TurnInput(request_id="source-1", events=[
                    Event(role="user", speaker=event["speaker"], content=event["text"],
                          occurred_at=datetime.fromisoformat(event["occurred_at"]))
                ]))
                session = store.get_session(scope, session.session_id)
                extraction = await model.extract(ExtractionWindow(session=session, new_turns=[turn], context_turns=[]))
                memories = store.commit_extraction(scope, session, [turn], extraction)
                report["ingestion_check"] = {
                    "source_event": event,
                    "memories": [m.model_dump(mode="json") for m in memories],
                    "typed_records": sum(m.semantics.cardinality != "unknown" for m in memories),
                    "atomic_members": sum(len(m.semantics.members) for m in memories),
                    "indexed_rows": store._db.execute("SELECT COUNT(*) FROM fact_projection").fetchone()[0],
                }
                if not memories:
                    raise RuntimeError("Live extraction produced no memories for the selected factual source")
            finally:
                store.close()
        for old in rows:
            sources = [Hit.model_validate(s) for s in old["retrieved_sources"]]
            catalog = list(dict.fromkeys((s.metadata.get("subject"), s.metadata.get("predicate"), "unknown", None)
                                        for s in sources if s.metadata.get("subject") and s.metadata.get("predicate")))
            before = dict(model.usage)
            plan = await model.plan(old["evaluation_query"], {"relation_catalog": catalog})
            if plan.collection_query is None:
                raise RuntimeError(f"Collection query not compiled for {old['sample_id']}/{old['qa_index']}")
            result = await model.compose(plan.collection_query, sources, old["evaluation_query"])
            ir = adjudicated_answer(plan.collection_query, result, sources, snapshot=(0, 0), pending=0)
            prediction = render(ir)
            row = {
                "sample_id": old["sample_id"], "qa_index": old["qa_index"], "question": old["question"],
                "gold_answer_for_posthoc_review_only": old["gold_answer"],
                "previous_prediction": old["prediction"], "prediction": prediction,
                "plan": plan.model_dump(mode="json"), "answer_ir": ir.model_dump(mode="json"),
                "source_decisions": result.model_dump(mode="json"),
                "usage": {k: model.usage[k] - before.get(k, 0) for k in model.usage},
            }
            report["rows"].append(row)
            report["usage"] = model.usage
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"sample_id": row["sample_id"], "qa_index": row["qa_index"],
                              "prediction": prediction, "usage": row["usage"]}, ensure_ascii=False), flush=True)
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["usage"] = model.usage
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        await model.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "result/locomo_multihop_retest_v2_20261013.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.source, args.output))
