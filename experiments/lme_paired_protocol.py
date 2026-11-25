"""Frozen, budgeted paired evaluation using the two native benchmark adapters."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

from run_parallel_bench import ROOT, cohort_questions, read, read_selected_samples, run_unit, write

ACCOUNT = ROOT / "result/remaining_balance_20261118"
CONFIG = ROOT / "result/ltm_synergy_20261118"
MODEL = "deepseek-v4.1-flash"
SYSTEMS = ("MemNexus", "SimpleMem")
CAPS = {"MemNexus": 8.0, "SimpleMem": 10.0, "judge": 0.08}
MEMNEXUS_SOURCE_READ_SCOPE = "events"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_freeze(path, dataset, cohort_path):
    manifest = read(path)
    files = {**manifest["source"], **manifest["external_pinned_by_hash"], **manifest["test_sources"]}
    required = [Path(__file__), Path(__file__).with_name("run_longmemeval_bench.py"),
                Path(__file__).with_name("run_parallel_bench.py"), Path(__file__).with_name("run_locomo_pilot.py"),
                Path(__file__).with_name("paper_protocol.py"), CONFIG / "study.py", ACCOUNT / "budget_account.py",
                ROOT / "SimpleMem/config.py", ROOT / "SimpleMem/main.py",
                ROOT / "MemNexus/memory-system/src/agent_memory/runtime.py", dataset, cohort_path]
    pinned = {(ROOT / name).resolve() for name in files}
    for file in required:
        if file.resolve() not in pinned:
            raise ValueError(f"Unpinned benchmark dependency: {file.name}")
    for name, digest in files.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen source/data changed: {name}")
    if (platform.python_version() != manifest["python"] or
            any(importlib.metadata.version(p) != v for p,v in manifest["installed_versions"].items())):
        raise ValueError("Frozen interpreter or package version mismatch")
    return manifest


def runtime_config():
    # The previous accepted paired study is the common config owner. Do not
    # trust inherited overrides for the experiment switches or silently load a
    # historical agent_memory checkout via PYTHONPATH.
    sys.path[:0] = [str(CONFIG), str(ROOT / "MemNexus/memory-system/src")]
    import agent_memory.settings  # Load local dotenv before isolating method overrides.
    for key in tuple(os.environ):
        if key.startswith(("MEMORY_", "SIMPLEMEM_", "EXPERIMENT_")):
            del os.environ[key]
    from study import configure
    config = configure()
    os.environ.update(MEMORY_EMBEDDING_ENCODING="query_prompt",
                      MEMORY_EMBEDDING_REVISION="qwen3-0.6b-eager-fp32-20261117",
                      MEMORY_EXPERIMENT_DROP_QUESTION_CANDIDATES="1")
    import agent_memory
    from run_locomo_pilot import RemoteQwenEmbedding
    from simplemem.core.settings import settings as native_sm
    if Path(config.__file__).resolve() != ROOT/"SimpleMem/config.py" or native_sm._user_config is not config:
        raise RuntimeError("SimpleMem is not using the pinned native config")
    if Path(agent_memory.__file__).resolve().parent != ROOT / "MemNexus/memory-system/src/agent_memory":
        raise RuntimeError("MemNexus loaded from a different checkout")
    expected = (MODEL, MODEL, MODEL, 32768, 0., False, True, 20, 15, True, True)
    actual = (config.LLM_MODEL, config.JUDGE_MODEL, os.environ["MEMORY_MODEL_NAME"], config.MAX_TOKENS,
              config.TEMPERATURE, config.ENABLE_THINKING, config.DISABLE_THINKING, config.WINDOW_SIZE,
              config.OVERLAP_SIZE, config.ENABLE_PLANNING, config.ENABLE_REFLECTION)
    if actual != expected or not config.OPENAI_API_KEY or not config.OPENAI_BASE_URL:
        raise RuntimeError("Backbone, generator or SimpleMem config differs from frozen protocol")
    if (os.environ["MEMORY_EMBEDDING_BASE_URL"].rstrip("/") + "/embeddings" != RemoteQwenEmbedding().endpoint
            or os.environ["MEMORY_EMBEDDING_NAME"] != RemoteQwenEmbedding.model_name
            or config.EMBEDDING_MODEL != RemoteQwenEmbedding.model_name):
        raise RuntimeError("The two systems must use the same local encoder")
    return config


def effective_method(config):
    from run_locomo_pilot import benchmark_memory_settings
    from simplemem.core.settings import settings as sm
    mn = benchmark_memory_settings(Path("unused"))
    method = {
        "MemNexus": {f.name: getattr(mn, f.name) for f in fields(mn)
                     if f.name not in {"api_key","db_path","local_tenant","local_owner"}},
        "SimpleMem": {k:getattr(sm,k) for k in
                      ("LLM_MODEL","JUDGE_MODEL","TEMPERATURE","MAX_TOKENS","ENABLE_THINKING","DISABLE_THINKING",
                       "USE_STREAMING","USE_JSON_FORMAT","WINDOW_SIZE","OVERLAP_SIZE","ENABLE_PLANNING","ENABLE_REFLECTION",
                       "MAX_REFLECTION_ROUNDS","EMBEDDING_MODEL","EMBEDDING_DIMENSION","EMBEDDING_QUERY_PROMPT",
                       "SEMANTIC_TOP_K","KEYWORD_TOP_K","STRUCTURED_TOP_K","ENABLE_PARALLEL_PROCESSING",
                       "ENABLE_PARALLEL_RETRIEVAL","INPUT_PRICE_RMB_PER_MILLION_TOKENS","OUTPUT_PRICE_RMB_PER_MILLION_TOKENS")},
        "native_controls": {k:os.environ[k] for k in
                            ("MEMORY_EXPERIMENT_DROP_QUESTION_CANDIDATES","MEMORY_EMBEDDING_ENCODING",
                             "MEMORY_EMBEDDING_REVISION","EXPERIMENT_TRUNCATION_MAX_TOKENS","MEMORY_MODEL_DISABLE_THINKING")},
        "qa_history_access": {"MemNexus": {"source_read_scope": MEMNEXUS_SOURCE_READ_SCOPE,
                              "grant": "all supplied history sessions; no gold evidence selection"}},
    }
    if not (mn.promotion_enabled and mn.evidence_grounding_enabled and not mn.grounding_semantic_enabled
            and mn.planner_mode=="model" and sm.ENABLE_PLANNING and sm.ENABLE_REFLECTION
            and sm.MAX_TOKENS==config.MAX_TOKENS==32768):
        raise ValueError("Ablated or otherwise incompatible native settings")
    return method


def encoder_check():
    import httpx
    import numpy as np
    from run_locomo_pilot import RemoteQwenEmbedding
    adapter = RemoteQwenEmbedding()
    with httpx.Client(trust_env=False, timeout=45) as client:
        response = client.get(adapter.endpoint.split("/v1/")[0] + "/health")
        response.raise_for_status()
        health = response.json()
    expected = {"status": "ok", "model": adapter.model_name, "dimension": 1024,
                "dtype": "torch.float32", "attention_implementation": "eager"}
    if any(health.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Local embedding service differs from frozen encoder")
    for is_query in (True, False):
        vec = adapter.encode_single("paired protocol readiness", is_query=is_query)
        if not np.isfinite(vec).all() or not np.isclose(np.linalg.norm(vec), 1., atol=.001):
            raise RuntimeError("Invalid encoder query/document vector")
    return health


def inspect_inputs(questions, samples):
    from run_locomo_pilot import evaluation_query, raw_turns
    details = {}
    for q in questions:
        sid = q["sample_id"]
        s = samples[q["sample_index"]]
        if (sid in details or Path(sid).name != sid or s["sample_id"] != sid
                or q["setting"] != "S" or s["setting"] != "S" or s["dataset"] != "LongMemEval-S"
                or s["question_type"] != q["category"] or int(q["category_id"]) != 0
                or q["qa_index"] != 0 or len(s["qa"]) != 1):
            raise ValueError(f"Invalid or duplicate LME-S cohort entry: {sid}")
        qa = cohort_questions(s, [q])[0]
        if (qa["question"] != q["question"] or qa["dataset"] != "LongMemEval-S" or not qa["question_date"]
                or evaluation_query(qa)[0] != q["question"] + "\nQuestion date: " + qa["question_date"]):
            raise ValueError(f"Question/date mismatch: {sid}")
        sessions = [k for k,v in s["conversation"].items() if isinstance(v, list)]
        maximum = max(int(k.removeprefix("session_")) for k in sessions)
        turns = raw_turns(s, maximum)
        expected = sum(bool(t["text"].strip()) for k in sessions for t in s["conversation"][k])
        active = sum(any(t["text"].strip() for t in s["conversation"][k]) for k in sessions)
        if len(turns) != expected or len({t["source_session"] for t in turns}) != active:
            raise ValueError(f"Session or raw turn coverage mismatch: {sid}")
        s["expected_raw_turns"] = expected
        details[sid] = dict(max_session=maximum, nonempty_sessions=active, raw_turns=expected)
    return details


@contextmanager
def coordinator_lock(path):
    # OS releases this lock if a coordinator crashes. The persistent admission
    # and provider ledgers independently retain all money already reserved.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"1")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def audit_traces(folder, arm):
    usage = read(folder / "usage.json")
    trace = folder / "usage.calls.jsonl"
    requests = responses = 0
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") in ("request", "response"):
                body = event["body"]
                if body.get("model") != MODEL:
                    raise ValueError("Wire request/response model differs from registered model")
                if event["event"] == "request":
                    thinking = body.get("thinking", (body.get("extra_body") or {}).get("thinking"))
                    if (body.get("temperature") != 0 or thinking != {"type": "disabled"}
                            or not 0 < body.get("max_tokens", 0) <= (512 if arm == "judge" else 32768)):
                        raise ValueError("Wire decoding settings differ from registered protocol")
                requests += event["event"] == "request"
                responses += event["event"] == "response"
    if requests != usage["provider_attempts"] or not responses or not usage["input_tokens"] or usage["reserved_rmb"]:
        raise ValueError("Payment ledger/trace mismatch or unresolved request")


def verified_result(folder, arm, question, info):
    result = read(folder / "stage_result.json")
    audit_traces(folder, arm)
    rows = result["rows"]
    if (len(rows) != (2 if arm == "judge" else 1) or
            any(r["sample_id"] != question["sample_id"] or r["question"] != question["question"] for r in rows)):
        raise ValueError("Result belongs to a different question or has missing rows")
    if arm == "judge":
        if {r["system"] for r in rows} != set(SYSTEMS) or any(r.get("judge_result") not in ("CORRECT", "WRONG") for r in rows):
            raise ValueError("Missing paired blind judgment")
    else:
        stage = result["stage"]
        if rows[0]["system"] != arm or not rows[0]["prediction"].strip() or stage["raw_turns"] != info["raw_turns"]:
            raise ValueError("Missing or incomplete system result")
        if arm == "MemNexus" and (stage["memory_base_replayed"] or stage["session_layout"] != "dataset_sessions"
                                   or stage["internal_session_count"] != info["nonempty_sessions"]):
            raise ValueError("Full, freshly constructed session history required")
        if arm == "MemNexus":
            from run_locomo_pilot import validate_completed_qa_history
            validate_completed_qa_history(result, MEMNEXUS_SOURCE_READ_SCOPE)
    return result


def charged(folder, cap):
    if not (folder / "completion.json").is_file():
        return cap
    u = read(folder / "usage.json")
    amount = u["cost_rmb"] + u["reserved_rmb"]
    if not math.isfinite(amount) or not 0 <= amount <= cap + 1e-8:
        raise ValueError("Invalid component charge")
    return amount


async def paired(args):
    cohort_path, freeze_path, out = (ROOT / p for p in (args.cohort, args.freeze_manifest, args.out))
    cohort = read(cohort_path)
    total = args.total_budget_rmb
    if not math.isfinite(total) or total < 0 or (args.paired_run and total < sum(CAPS.values())):
        raise ValueError("Supply a finite total budget; paid runs need at least one full pair reservation")
    candidates = cohort["questions"]
    if any(q["setting"] != "S" for q in candidates):
        raise ValueError("Paired runs require a registered S-only cohort")
    if not 1 <= args.limit_pairs <= len(candidates):
        raise ValueError("Invalid --limit-pairs")
    questions = candidates[:args.limit_pairs]
    dataset = ROOT / cohort["dataset"]
    manifest = verify_freeze(freeze_path, dataset, cohort_path)
    if manifest["external_pinned_by_hash"].get(dataset.relative_to(ROOT).as_posix()) != cohort["dataset_sha256"]:
        raise ValueError("Cohort and frozen dataset digest differ")
    config = runtime_config()
    method = effective_method(config)
    health = encoder_check()
    samples = read_selected_samples(dataset, {q["sample_index"] for q in questions})
    info = inspect_inputs(questions, samples)
    protocol = {"freeze_version": manifest["version"], "freeze_sha256": sha(freeze_path),
                "cohort_sha256": sha(cohort_path), "dataset_sha256": cohort["dataset_sha256"],
                "ids": list(info), "component_caps_rmb": CAPS, "total_budget_rmb": total,
                "model": MODEL, "judge": "SimpleMem Appendix A.4; DeepSeek substitute",
                "memnexus_source_read_scope": MEMNEXUS_SOURCE_READ_SCOPE,
                "effective_method": method, "method_sha256": hashlib.sha256(json.dumps(method,sort_keys=True).encode()).hexdigest(),
                "order": "registered prefix; alternate systems and judge ordering",
                "metrics": "retain A.4 labels, count recorded system errors separately and as incorrect for operational accuracy",
                "cost": "normalized RMB; full fresh builds, QA and judge charged separately"}
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    stored = out / "protocol.json"
    preflight = dict(protocol_id=protocol_id, questions=len(questions), categories=dict(Counter(q["category"] for q in questions)),
                     history=info, encoder=health, paid_calls=0, ledger_balance_snapshot=read(ACCOUNT / "budget.json"))
    with coordinator_lock(ACCOUNT / "paired-coordinator.lock"):
        if stored.exists() and read(stored) != protocol:
            raise ValueError("This output directory belongs to a different frozen experiment")
        if not stored.exists() and out.exists() and any(out.iterdir()):
            raise ValueError("An unregistered output directory must be empty")
        write(stored, protocol)
        write(out / "preflight.json", preflight)
        if not args.paired_run:
            print(json.dumps({"status": "preflight_passed", "paid_calls": 0, "pairs": len(info), "out": str(out)}))
            return 0
        return await run_registered(out, protocol_id, questions, samples, info, total, config)


async def run_registered(out, run_id, questions, samples, info, total, config):
    sys.path.insert(0, str(ACCOUNT))
    import budget_account
    from run_locomo_pilot import judge_rows
    prefix = "paired_" + run_id + "_"
    results = []
    for index, q in enumerate(questions):
        sid = q["sample_id"]
        admissions, global_spent, global_reserved = budget_account.reconcile()
        own = {(a["arm"], a["sample_id"]): a for a in admissions if a["arm"].startswith(prefix)}
        if len(own) != sum(a["arm"].startswith(prefix) for a in admissions):
            raise ValueError("Duplicate budget admissions for this run")
        if any(not (ACCOUNT / "components" / a["arm"] / a["sample_id"] / "completion.json").exists()
               for a in admissions if not a["arm"].startswith(prefix)):
            raise RuntimeError("Another Forward100 experiment has outstanding reservations")
        spent = sum(charged(ACCOUNT / "components" / arm / sample_id, a["cap"]) for (arm,sample_id),a in own.items())
        missing = sum(cap for arm,cap in CAPS.items() if (prefix+arm,sid) not in own)
        allowance = read(ACCOUNT / "authorization.json")["forward_allowance_rmb"]
        if spent + missing > total + 1e-8 or global_spent + global_reserved + missing > allowance + 1e-8:
            write(out / "status.json", dict(status="budget_stop", planned=len(questions), completed_pairs=len(results)//2,
                  next_id=sid, spent_or_reserved_rmb=spent, next_reservation_rmb=missing))
            return 2
        order = SYSTEMS if index % 2 == 0 else SYSTEMS[::-1]
        pair = []
        for arm in (*order, "judge"):
            folder = ACCOUNT / "components" / (prefix+arm) / sid
            cap = CAPS[arm]
            key = (prefix+arm, sid)
            if key in own and own[key]["cap"] != cap:
                raise ValueError("Previously admitted component cap differs")
            if key not in own:
                if folder.exists():
                    raise ValueError("Unregistered component artifacts present")
                # Register atomically in the existing Forward100 account. If
                # interrupted before started.json, no request was dispatched.
                entries, forward_spent, forward_reserved = budget_account.reconcile()
                if forward_spent + forward_reserved + cap > allowance + 1e-8:
                    raise RuntimeError("Remaining Forward100 authorization is insufficient")
                write(ACCOUNT / "admissions.json", [*entries, dict(arm=key[0],sample_id=sid,cap=cap,
                      time=time.time())])
            marker = {"cap": cap, "protocol_id": run_id}
            if (folder / "started.json").exists() and read(folder / "started.json") != marker:
                raise ValueError("Component provenance differs")
            if (folder / "stage_result.json").is_file():
                if not (folder / "started.json").is_file():
                    raise ValueError("Result lacks a matching admission marker")
                result = verified_result(folder, arm, q, info[sid])
            else:
                if folder.exists() and any(p.name != "started.json" for p in folder.iterdir()):
                    write(out / "status.json", dict(status="unresolved_component", arm=arm, id=sid,
                          component=str(folder), completed_pairs=len(results)//2, reserved_rmb=cap))
                    raise RuntimeError(f"Unfinished component {arm}/{sid}; preserved, never reissued automatically")
                write(folder / "started.json", marker)
                try:
                    if arm == "judge":
                        config.MAX_TOKENS = 512
                        os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "0"
                        try:
                            usage = judge_rows(pair, cap, usage_ledger=folder/"usage.json")
                        finally:
                            config.MAX_TOKENS = 32768
                            os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "32768"
                        write(folder / "stage_result.json", {"stage": {"total_usage": usage}, "rows": pair})
                    else:
                        sample = samples[q["sample_index"]]
                        await run_unit(arm, sample, cohort_questions(sample,[q]), [q], folder, cap,
                                       info[sid]["max_session"], folder/"snapshots" if arm=="MemNexus" else None)
                    result = verified_result(folder, arm, q, info[sid])
                except BaseException as error:
                    write(out / "status.json", dict(status="unresolved_component", arm=arm, id=sid, component=str(folder),
                          completed_pairs=len(results)//2, reserved_rmb=cap, error=type(error).__name__))
                    budget_account.reconcile()
                    raise
            if arm == "judge":
                for row in result["rows"]:
                    source = ACCOUNT / "components" / (prefix + row["system"]) / sid
                    stage = read(source / "stage_result.json")["stage"]
                    if row["system"] == "MemNexus":
                        from run_locomo_pilot import validate_qa_history_scope
                        validate_qa_history_scope(row, tuple(sorted(stage["authorized_source_sessions"])),
                                                  MEMNEXUS_SOURCE_READ_SCOPE)
                    results.append({**row, "operational_correct": bool(row["judge_result"] == "CORRECT" and not row.get("error")),
                                    "build_usage": stage["build_usage"], "build_ms": stage["build_ms"],
                                    "total_usage": stage["total_usage"], "component_dir": str(source),
                                    "paired_judge_dir": str(folder)})
            else:
                pair.extend(result["rows"])
            write(folder / "completion.json", {"status": "complete", "protocol_id": run_id})
        write(out / "rows_judged.json", results)
        write(out / "status.json", dict(status="running", completed_pairs=index+1, planned=len(questions)))
    _, spent, reserved = budget_account.reconcile()
    write(out / "status.json", dict(status="complete", completed_pairs=len(questions), planned=len(questions),
          forward_spent_rmb=spent, forward_reserved_rmb=reserved, component_prefix=prefix))
    return 0
