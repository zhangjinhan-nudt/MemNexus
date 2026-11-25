"""Independent 100 RMB validation using existing native benchmark functions.

prepare is offline. run dispatches paid workers only after immutable registration.
No production methods are replaced; old memories and new builds are separate arms.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "SimpleMem")]
LIMIT = 100.0
CAPS = {"replay": 1.0, "fresh": 8.0, "replay_judge": .2, "fresh_judge": .2}
PHASES = tuple(CAPS)
MODEL = "deepseek-v4.1-flash"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def component(phase, sid, home=HERE):
    if phase not in PHASES or Path(sid).name != sid:
        raise ValueError("Invalid phase/sample identity")
    return home / "components" / phase / sid


def package_versions():
    names = ("numpy", "pydantic", "httpx", "openai", "torch", "transformers",
             "sentence-transformers", "lancedb", "rank-bm25")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def balance(home=HERE):
    """Unfinished/failed dispatches retain their full admitted maximum."""
    spent = held = metered = unknown = 0.0
    counts = Counter()
    for marker in sorted((home / "components").glob("*/*/admission.json")):
        folder = marker.parent
        admitted = read(marker)
        cap = float(admitted["cap_rmb"])
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("Invalid admitted amount")
        if (folder / "canceled.json").exists():
            if (folder / "started.json").exists() or (folder / "usage.json").exists():
                raise ValueError("Dispatched components cannot be canceled")
            counts["canceled_before_dispatch"] += 1
            continue
        usage = read(folder / "usage.json") if (folder / "usage.json").exists() else {}
        paid = float(usage.get("cost_rmb", 0))
        reserve = float(usage.get("reserved_rmb", 0))
        uncertain = float(usage.get("unknown_usage_upper_bound_rmb", 0))
        if not all(math.isfinite(v) and v >= 0 for v in (paid, reserve, uncertain)) or paid + reserve > cap + 1e-7:
            raise ValueError("Native component ledger violates admission")
        known = 0.0
        trace = folder / "usage.calls.jsonl"
        if trace.exists():
            with trace.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        break  # A live write retains the entire admission cap.
                    event = json.loads(line)
                    tokens = event.get("body", {}).get("usage") or {}
                    inp, out = tokens.get("prompt_tokens"), tokens.get("completion_tokens")
                    if event.get("event") == "response" and type(inp) is int and type(out) is int and min(inp,out) >= 0:
                        known += (inp * 2.4 + out * 9.6) / 1_000_000
        metered += known
        unknown += max(uncertain, paid - known, 0.0)
        if (folder / "completion.json").exists():
            if reserve:
                raise ValueError("Completed component has an outstanding request")
            spent += paid
            counts["completed"] += 1
        else:
            held += cap
            counts["failed_cap_retained" if (folder / "ended.json").exists() else "reserved"] += 1
    if spent + held > LIMIT + 1e-7:
        raise ValueError("Global study budget exceeded")
    return dict(limit_rmb=LIMIT, settled_rmb=round(spent, 8), reserved_rmb=round(held, 8),
                available_rmb=round(LIMIT - spent - held, 8),
                native_metered_normalized_rmb=round(metered, 8),
                native_explicit_unknown_upper_rmb=round(unknown, 8), counts=dict(counts))


def settle(home=HERE):
    state = balance(home)
    write(home / "budget.json", state)
    return state


def admit_pair(phase, sid, protocol, home=HERE):
    paths = [component(phase, sid, home), component(phase + "_judge", sid, home)]
    missing = [(p, CAPS[p.parent.name]) for p in paths if not (p / "admission.json").exists()]
    if sum(cap for _, cap in missing) > balance(home)["available_rmb"] + 1e-8:
        return False
    for folder, cap in missing:
        write(folder / "admission.json", dict(cap_rmb=cap, protocol_sha256=protocol,
                                             time_unix=time.time(), phase=folder.parent.name, sample_id=sid))
    settle(home)
    return True


def cancel_unused_judge(phase, sid, home=HERE):
    folder = component(phase + "_judge", sid, home)
    if (folder / "admission.json").exists() and not (folder / "started.json").exists():
        write(folder / "canceled.json", dict(reason="QA did not complete; no judge request sent", time_unix=time.time()))


def blocked_by_provider(folder):
    path = folder / "usage.calls.jsonl"
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == "error" and event.get("status_code") in (400, 401, 403):
                return True
    return False


def prepare():
    from run_parallel_bench import read_selected_samples
    from lme_paired_protocol import inspect_inputs, runtime_config, effective_method
    method = effective_method(runtime_config())  # Local configuration only; no provider call.
    cohort = read(HERE / "cohort.json")
    questions = cohort["questions"]
    ids = [q["sample_id"] for q in questions]
    if len(set(ids)) != len(ids) or not questions:
        raise ValueError("The cohort must have unique sample IDs")
    dataset = resolve(cohort["dataset"])
    if sha(dataset) != cohort["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    source = ROOT / "MemNexus/memory-system/src"
    release = read(ROOT / "result/simplemem_transfer_20261123/release.json")
    for name, digest in release["hashes"].items():
        if sha(source / name) != digest:
            raise ValueError("Current source differs from accepted release: " + name)
    samples = read_selected_samples(dataset, {q["sample_index"] for q in questions})
    info = inspect_inputs(questions, samples)
    snapshots = {}
    for q in questions:
        sid = q["sample_id"]
        folder = resolve(q["snapshot_dir"])
        db, manifest = folder / (sid + ".sqlite3"), folder / (sid + ".json")
        meta = read(manifest)
        if sha(db) != meta["sha256"] or meta.get("session_layout") != "dataset_sessions":
            raise ValueError("Invalid session-preserving historical snapshot: " + sid)
        for key, system in (("old_mn_row", "MemNexus"), ("old_sm_row", "SimpleMem")):
            row = q[key]
            if row["sample_id"] != sid or row["question"] != q["question"] or row["system"] != system or not row["prediction"].strip():
                raise ValueError("Historical result belongs to a different item/system")
        snapshots[sid] = dict(database=str(db), manifest=str(manifest), database_sha256=sha(db), manifest_sha256=sha(manifest))
    files = set(source.rglob("*.py")) | set((ROOT / "SimpleMem/simplemem").rglob("*.py"))
    files.update(ROOT / name for name in (
        "SimpleMem/config.py", "SimpleMem/main.py", "experiments/run_locomo_pilot.py",
        "experiments/run_parallel_bench.py", "experiments/run_longmemeval_bench.py", "experiments/locomo_cohort.py",
        "experiments/lme_paired_protocol.py", "experiments/paper_protocol.py", "result/ltm_synergy_20261118/study.py"))
    files.update((Path(__file__), HERE / "cohort.json", ROOT / "result/simplemem_transfer_20261123/release.json"))
    protocol = dict(schema=1, budget_rmb=LIMIT, caps=CAPS, cohort_sha256=sha(HERE / "cohort.json"),
        dataset=str(dataset), dataset_sha256=cohort["dataset_sha256"], ids=ids, info=info, snapshots=snapshots,
        release_bundle_sha256=release["bundle_sha256"], source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
        python=platform.python_version(), packages=package_versions(), model=MODEL,
        effective_method=method,
        generator=dict(temperature=0, max_tokens=32768, thinking="disabled"),
        judge=dict(model=MODEL, prompt="SimpleMem Appendix A.4", substitution="DeepSeek for GPT-4.1-mini", max_tokens=512,
                   order="seeded shuffle of unlabelled answers; no system name in prompt"),
        pricing=dict(normalized_input_rmb_per_million=2.4, normalized_output_rmb_per_million=9.6,
                     public_source="https://help.aliyun.com/zh/model-studio/deepseek-v4-1-flash",
                     public_beijing_input=[2,1], public_beijing_output=[8,4], use_discount_for_admission=False),
        phases=["all old-snapshot native QA plus old/new/SM rejudgment", "assistant fresh build and native QA plus blind judgment"],
        accounting="100 RMB conservative normalized hard envelope; all dispatched failures retain full caps; build/QA/judge separate",
        interpretation="Old snapshot replays measure reading changes. Fresh builds measure current full MN. Historical SM answers are reused, never regenerated or relabelled as current fresh runs.")
    path = HERE / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("Existing immutable protocol differs")
    write(path, protocol)
    write(HERE / "preflight.json", dict(status="offline_passed", paid_requests=0, questions=len(questions),
                                        fresh_assistant_questions=sum(bool(q.get("fresh_build")) for q in questions),
                                        protocol_sha256=sha(path), release_bundle_sha256=release["bundle_sha256"]))
    settle()
    print(json.dumps(dict(status="offline_prepared", questions=len(questions), paid_requests=0)), flush=True)


def verify():
    protocol = read(HERE / "protocol.json")
    if protocol["budget_rmb"] != LIMIT or protocol["caps"] != CAPS:
        raise ValueError("Budget protocol changed")
    if platform.python_version() != protocol["python"] or package_versions() != protocol["packages"]:
        raise ValueError("Runtime dependency versions changed")
    if sha(protocol["dataset"]) != protocol["dataset_sha256"]:
        raise ValueError("Registered dataset changed")
    for name, digest in protocol["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Registered source changed: " + name)
    return protocol


async def worker(phase, sid):
    from lme_paired_protocol import runtime_config, effective_method, audit_traces
    from run_parallel_bench import read_selected_samples, cohort_questions, run_unit
    from run_locomo_pilot import judge_rows, validate_completed_qa_history
    folder = component(phase, sid)
    protocol = verify()
    admission = read(folder / "admission.json")
    if admission["protocol_sha256"] != sha(HERE / "protocol.json") or admission["cap_rmb"] != CAPS[phase]:
        raise ValueError("Admission mismatch")
    if (folder / "started.json").exists() or (folder / "canceled.json").exists():
        raise ValueError("Refusing to reissue an interrupted or canceled component")
    write(folder / "started.json", dict(time_unix=time.time(), pid=os.getpid(), phase=phase, id=sid))
    try:
        config = runtime_config()
        if effective_method(config) != protocol["effective_method"]:
            raise ValueError("Registered runtime method changed")
        q = next(q for q in read(HERE / "cohort.json")["questions"] if q["sample_id"] == sid)
        if phase.endswith("_judge"):
            parent = phase.removesuffix("_judge")
            new = read(component(parent, sid) / "stage_result.json")["rows"][0]
            rows = [dict(deepcopy(new), validation_arm=parent)]
            if phase == "replay_judge":
                rows += [dict(deepcopy(q["old_mn_row"]), validation_arm="old_mn"),
                         dict(deepcopy(q["old_sm_row"]), validation_arm="old_sm")]
            for row in rows:
                row["historical_judge_result"] = row.get("judge_result")
                row["judge_result"] = None
            random.Random(20261124 + int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)).shuffle(rows)
            config.MAX_TOKENS = 512
            os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "0"
            usage = judge_rows(rows, CAPS[phase], usage_ledger=folder / "usage.json")
            result = {"stage": {"total_usage":usage}, "rows":rows}
            write(folder / "stage_result.json", result)
            audit_traces(folder, "judge")
        else:
            sample = read_selected_samples(Path(protocol["dataset"]), {q["sample_index"]})[q["sample_index"]]
            sample["expected_raw_turns"] = protocol["info"][sid]["raw_turns"]
            if phase == "replay":
                snap = protocol["snapshots"][sid]
                if sha(snap["database"]) != snap["database_sha256"] or sha(snap["manifest"]) != snap["manifest_sha256"]:
                    raise ValueError("Historical snapshot changed")
                snapshots = Path(snap["database"]).parent
            else:
                snapshots = folder / "snapshots"
            result = await run_unit("MemNexus", sample, cohort_questions(sample,[q]), [q], folder,
                                    CAPS[phase], protocol["info"][sid]["max_session"], snapshots, phase == "replay")
            validate_completed_qa_history(result, "events")
            if result["stage"]["memory_base_replayed"] != (phase == "replay"):
                raise ValueError("Fresh and replay memory bases mixed")
            audit_traces(folder, "MemNexus")
        verify()
        write(folder / "completion.json", dict(status="complete", time_unix=time.time()))
        write(folder / "ended.json", dict(status="complete", time_unix=time.time()))
    except BaseException as error:
        # Logs retain provider status without printing the configured key or HTTP headers.
        write(folder / "ended.json", dict(status="failed", error_type=type(error).__name__,
                                           time_unix=time.time(), cap_retained=True))
        raise


def launch(phase, sid):
    folder = component(phase, sid)
    env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    with (folder / "worker.log").open("a", encoding="utf-8") as log:
        return subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "worker", phase, sid],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def phase_run(phase, questions, workers, protocol_digest):
    """One coordinator owns admission; child processes own disjoint ledgers."""
    active = {}
    pending = list(questions)
    outcome = "complete"
    try:
        while pending or active:
            while pending and len(active) < workers:
                q = pending.pop(0)
                sid = q["sample_id"]
                folder = component(phase, sid)
                if (folder / "completion.json").exists() or (folder / "ended.json").exists():
                    if blocked_by_provider(folder):
                        return "provider_blocked"
                    continue
                if (folder / "started.json").exists():
                    raise RuntimeError("Unsettled worker exists; inspect it before resuming: " + sid)
                if phase.endswith("_judge"):
                    parent = phase.removesuffix("_judge")
                    if not (component(parent, sid) / "completion.json").exists():
                        cancel_unused_judge(parent, sid)
                        continue
                    if (folder / "canceled.json").exists():
                        continue
                elif not admit_pair(phase, sid, protocol_digest):
                    outcome = "budget_stop"
                    continue
                active[sid] = launch(phase, sid)
            for sid, child in list(active.items()):
                if child.poll() is None:
                    continue
                active.pop(sid)
                folder = component(phase, sid)
                if not (folder / "ended.json").exists():
                    write(folder / "ended.json", dict(status="process_failed", exit_code=child.returncode, cap_retained=True))
                if not phase.endswith("_judge") and not (folder / "completion.json").exists():
                    cancel_unused_judge(phase, sid)
                state = settle()
                print(json.dumps(dict(phase=phase,id=sid,exit_code=child.returncode,**state)), flush=True)
                if blocked_by_provider(folder):
                    return "provider_blocked"
            if active:
                time.sleep(1)
        return outcome
    finally:
        # A coordinator interruption must not leave undisclosed paid jobs running.
        for child in active.values():
            child.terminate()
        for child in active.values():
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        settle()


def collect(status):
    rows = []
    for path in sorted((HERE / "components").glob("*/*/completion.json")):
        folder = path.parent
        result = read(folder / "stage_result.json")
        rows.extend(dict(row, validation_phase=folder.parent.name, component_dir=str(folder)) for row in result["rows"])
    write(HERE / "results.json", dict(status=status, rows=rows, budget=balance()))
    write(HERE / "status.json", dict(status=status, time_unix=time.time(), budget=balance()))


def run(workers, requested_phase="all"):
    from lme_paired_protocol import coordinator_lock
    with coordinator_lock(HERE / "coordinator.lock"):
        verify()
        questions = read(HERE / "cohort.json")["questions"]
        digest = sha(HERE / "protocol.json")
        collect("running")
        # Historical service state included AccessDenied.Unpurchased. Probe exactly
        # one ordinary paid QA before permitting any other admission.
        status = phase_run("replay", questions[:1], 1, digest)
        collect(status)
        if status == "provider_blocked" or not (component("replay", questions[0]["sample_id"]) / "completion.json").exists():
            collect("provider_blocked" if status == "provider_blocked" else "pilot_failed")
            return
        if requested_phase in ("replay", "all"):
            status = phase_run("replay", questions[1:], 1, digest)
            collect(status)
            if status == "provider_blocked":
                return
            status = phase_run("replay_judge", questions, 1, digest)
            collect("replay_complete" if status == "complete" else status)
            if status == "provider_blocked" or requested_phase == "replay":
                return
        if requested_phase == "fresh" and any(
            (component("replay", q["sample_id"]) / "completion.json").exists()
            and not (component("replay_judge", q["sample_id"]) / "ended.json").exists() for q in questions
        ):
            raise RuntimeError("Finish replay judgments before fresh builds")
        fresh = [q for q in questions if q.get("fresh_build")]
        status = phase_run("fresh", fresh, workers, digest)
        collect(status)
        if status == "provider_blocked":
            return
        judged = phase_run("fresh_judge", fresh, 1, digest)
        failures = balance()["counts"].get("failed_cap_retained", 0)
        collect("provider_blocked" if judged == "provider_blocked" else "budget_stop" if status == "budget_stop"
                else "complete_with_technical_failures" if failures else "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    execute = sub.add_parser("run")
    execute.add_argument("--workers", type=int, choices=(1,2), default=2)
    execute.add_argument("--phase", choices=("replay", "fresh", "all"), default="all")
    item = sub.add_parser("worker")
    item.add_argument("phase", choices=PHASES)
    item.add_argument("sample_id")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        asyncio.run(worker(args.phase, args.sample_id))
    else:
        run(args.workers, args.phase)
