"""Real-provider, frozen-memory development comparison; never a fresh build claim.

Restore original turns/memories without gold-dependent selection. Re-run all
MemNexus queries, reuse immutable SimpleMem predictions, and rejudge both in a
seeded shuffled order. Charge original build usage when comparing system costs.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from run_locomo_pilot import judge_rows, usage_delta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))
from agent_memory.models import Memory, Scope, Session, Turn
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def restore(store, scope, snapshot):
    turns = [Turn.model_validate(t) for t in snapshot["turns"]]
    assert turns and snapshot["memories"]
    session = Session(session_id=turns[0].session_id, topic="LoCoMo official pilot",
                      processed_sequence=max(t.sequence for t in turns), revision=len(turns))
    with store.transaction():
        store._db.execute("INSERT INTO sessions VALUES (?,?,?,?)",
                          (*store._who(scope), session.session_id, session.model_dump_json()))
        for turn in turns:
            assert turn.session_id == session.session_id
            store._db.execute("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?)",
                              (*store._who(scope), session.session_id, turn.turn_id, turn.sequence,
                               turn.request_id, "snapshot-read-only", turn.model_dump_json()))
        for body in snapshot["memories"]:
            store._save_memory(scope, Memory.model_validate(body))
    assert len(store.turns(scope, session.session_id)) == len(turns)
    assert [m.model_dump(mode="json") for m in store.memories(scope)] == snapshot["memories"]
    return session


async def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    original = json.loads(args.source.read_text(encoding="utf-8"))
    assert original["status"] == "completed"
    folder = args.source.with_name(args.source.stem + "_ledgers")
    ledgers = args.output.with_name(args.output.stem + "_ledgers")
    ledgers.mkdir(parents=True, exist_ok=False)
    simple_cost = sum(s["total_usage"]["cost_rmb"] for s in original["stages"] if s["system"] == "SimpleMem")
    build_cost = sum(s["build_usage"]["cost_rmb"] for s in original["stages"] if s["system"] == "MemNexus")
    budget = simple_cost - build_cost - args.prior_spend
    groups = defaultdict(list)
    for row in original["rows"]:
        if row["system"] == "MemNexus":
            groups[row["sample_id"]].append(row)
    code = [*(ROOT / "MemNexus/memory-system/src/agent_memory").glob("*.py"), Path(__file__)]
    report = {"status": "running", "protocol": "frozen-memory development replay; not independent held-out or fresh ingestion",
              "source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
              "limitations": ["Session summary/state not included in archived memory; reconstructed session has empty summary",
                              "SimpleMem answers are reused, not regenerated; both systems are rejudged blind",
                              "Development questions have influenced design; no independent generalization claim"],
              "simplemem_main_cost_rmb": simple_cost, "memnexus_original_build_cost_rmb": build_cost,
              "new_qa_cap_rmb": budget, "prior_development_spend_rmb": args.prior_spend,
              "rows": [], "qa_usage": [], "judge_usage": [], "snapshot_sha256": {}}
    with zipfile.ZipFile(args.output.with_suffix(".source.zip"), "x", zipfile.ZIP_DEFLATED) as z:
        for p in code:
            z.write(p, p.relative_to(ROOT))
    save(args.output, report)
    consumed = 0.0
    if args.resume:
        previous = json.loads(args.resume.read_text(encoding="utf-8"))
        assert previous["status"] == "aborted" and previous["source_sha256"] == report["source_sha256"]
        assert previous["prior_development_spend_rmb"] == args.prior_spend
        for p,digest in previous["code_sha256"].items():
            if Path(p).name not in {"synthesis.py", Path(__file__).name}:
                assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest, p
        report["resumed_from"] = {"path":str(args.resume),"code_sha256":previous["code_sha256"],
                                  "policy":"Add checked inverse year-duration arithmetic; retain all previously validated answers, not selected by score"}
        for field in ("rows","qa_usage","judge_usage","snapshot_sha256"):
            report[field] = deepcopy(previous[field])
        completed_samples = {j["sample_id"] for j in previous["judge_usage"]}
        for p in args.resume.with_name(args.resume.stem+"_ledgers").iterdir():
            # A judge ledger may contain a partial conversation and cannot be
            # appended as if it were a complete usage record. QA/build ledgers
            # remain reusable because their rows are checkpointed separately.
            if p.name.endswith("_judge.json") or p.name.endswith("_judge.calls.jsonl"):
                sample = p.name.split("_judge", 1)[0]
                if sample not in completed_samples:
                    continue
            shutil.copyfile(p,ledgers/p.name)
        consumed = sum(json.loads(p.read_text(encoding="utf-8"))["cost_rmb"] for p in ledgers.glob("*_memnexus.json"))
        save(args.output,report)
    try:
        for sid, rows in groups.items():
            if any(j["sample_id"] == sid for j in report["judge_usage"]):
                continue
            existing_ledger = ledgers / f"{sid}_memnexus.json"
            previous_usage = json.loads(existing_ledger.read_text(encoding="utf-8")) if existing_ledger.exists() else None
            existing_cost = previous_usage["cost_rmb"] if previous_usage else 0
            os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"] = str(budget - consumed + existing_cost)
            os.environ["EXPERIMENT_USAGE_LEDGER_PATH"] = str(ledgers / f"{sid}_memnexus.json")
            snapshot_path = folder / f"{sid}_memnexus.memory.json"
            report["snapshot_sha256"][sid] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory(prefix="memnexus-synthesis-replay-") as tmp:
                runtime = MemoryRuntime.from_settings(Settings(
                    db_path=Path(tmp) / "memory.sqlite3", model_timeout=180, model_max_tokens=4096,
                    planner_mode="model", acceleration_enabled=False, output_language="en",
                    grounding_semantic_enabled=False, answer_synthesis_enabled=True))
                scope = Scope(tenant_id="locomo-pilot", owner_id=sid)
                try:
                    if previous_usage:
                        runtime.model.usage = previous_usage
                    assert runtime.model.model == "deepseek-v4-flash" and runtime.retriever.embedder is not None
                    session = restore(runtime.store, scope, snapshot)
                    for old in rows:
                        if any(r["system"] == "MemNexus" and r["sample_id"] == sid and r["qa_index"] == old["qa_index"] for r in report["rows"]):
                            continue
                        before = dict(runtime.model.usage)
                        started = time.perf_counter()
                        answer = await runtime.answer(scope, session.session_id, old["evaluation_query"],
                                                      top_k=20, timeout=180, accelerate=False, prepare=False)
                        result = {k: old[k] for k in ("sample_id", "qa_index", "category_id", "question", "evaluation_query", "gold_answer")}
                        result.update(system="MemNexus", prediction=answer.generated_text,
                                      model_calls=usage_delta(runtime.model.usage, before),
                                      latency_ms=(time.perf_counter()-started)*1000,
                                      plan=answer.plan.model_dump(mode="json"), reasoning=answer.reasoning,
                                      retrieved_sources=[h.model_dump(mode="json") for h in answer.sources],
                                      judge_result=None)
                        assert result["prediction"].strip()
                        report["rows"].append(result)
                        report["current_qa_usage"] = dict(runtime.model.usage)
                        save(args.output, report)
                        print(json.dumps({"answered": len([r for r in report['rows'] if r['system']=='MemNexus']),
                                          "id": f"{sid}/{old['qa_index']}", "prediction": result['prediction']}, ensure_ascii=False), flush=True)
                    report["qa_usage"].append({"sample_id": sid, **runtime.model.usage})
                    consumed += runtime.model.usage["cost_rmb"] - existing_cost
                finally:
                    await runtime.close()
            paired = [r for r in report["rows"] if r["sample_id"] == sid]
            for old in original["rows"]:
                if old["system"] == "SimpleMem" and old["sample_id"] == sid:
                    if any(r["system"] == "SimpleMem" and r["sample_id"] == sid and
                           r["qa_index"] == old["qa_index"] for r in report["rows"]):
                        continue
                    copy = deepcopy(old)
                    copy["original_judge_result"] = copy["judge_result"]
                    copy["judge_result"] = None
                    report["rows"].append(copy)
                    paired.append(copy)
            random.Random(20261013 + len(report["qa_usage"])).shuffle(paired)
            usage = await asyncio.to_thread(judge_rows, paired, 1.0, usage_ledger=ledgers / f"{sid}_judge.json")
            report["judge_usage"].append({"sample_id": sid, **usage})
            save(args.output, report)
        assert len(report["rows"]) == 200
        assert all(r["judge_result"] in {"CORRECT", "WRONG"} for r in report["rows"])
        assert consumed + build_cost + args.prior_spend <= simple_cost
        assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p,h in report["code_sha256"].items())
        report["status"] = "completed"
        report["memnexus_main_cost_rmb"] = consumed + build_cost
    except BaseException as exc:
        report["status"] = "aborted"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "result/locomo_100_collection_20261013_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-spend", type=float, default=0)
    parser.add_argument("--resume",type=Path)
    asyncio.run(run(parser.parse_args()))
