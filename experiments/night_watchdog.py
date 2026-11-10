"""Watchdog for the overnight double run (LoCoMo 1000 + LongMemEval 1000).

Every 5 minutes:
  - quarantine poisoned checkpoints by rename (completed but all-empty
    predictions, a signature of a 429/403 storm) so they re-run without
    deleting prior results or snapshots
  - log done counts + gateway status
  - restart a bench if its process died with work left, but only when the
    gateway actually answers (avoids burning retries during an outage)
  - restart the embedding service if it dies
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\zhangjinhan\Desktop\10-agent")
LOG = ROOT / "result" / "night1000_watchdog.log"
PYTHON = r"C:\Users\zhangjinhan\AppData\Local\Programs\Python\Python313\python.exe"
BENCHES = {
    "locomo": {
        "cmd": [PYTHON, "experiments/run_parallel_bench.py",
                "--cohort", "result/locomo_1000_cohort_20261023.json",
                "--out", "result/bench1000_locomo_20261023",
                "--snapshots", "result/memnexus_snapshots/full30",
                "--concurrency", "2", "--stagger", "30",
                "--budget-per-conversation", "60"],
        "out": ROOT / "result" / "bench1000_locomo_20261023",
        "cohort": ROOT / "result" / "locomo_1000_cohort_20261023.json",
        "marker": "run_parallel_bench",
        "log": "result/bench1000_locomo.log",
    },
    "longmemeval": {
        "enabled": False,
        "cmd": [PYTHON, "experiments/run_longmemeval_bench.py",
                "--cohort", "result/longmemeval_1000_cohort_20261023.json",
                "--out", "result/bench1000_longmemeval_20261023",
                "--snapshots", "result/memnexus_snapshots/lme1000",
                "--concurrency", "1", "--stagger", "30",
                "--budget-per-question", "8"],
        "out": ROOT / "result" / "bench1000_longmemeval_20261023",
        "cohort": ROOT / "result" / "longmemeval_1000_cohort_20261023.json",
        "marker": "run_longmemeval_bench",
        "log": "result/bench1000_longmemeval.log",
    },
}


def child_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}}
    env.update({
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "NIGHT_TEMPORAL_RESCUE": "0",
    })
    return env


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")
        handle.flush()


def benchmark_processes(marker: str):
    """Return only Python benchmark processes, not diagnostic shell commands."""
    import psutil
    matches = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid == os.getpid() or "python" not in (proc.info.get("name") or "").lower():
                continue
            cmd = proc.info.get("cmdline") or []
            if any(part.endswith(".py") and marker in part for part in cmd):
                matches.append(proc)
        except Exception:
            continue
    return matches


def process_running(marker: str) -> bool:
    return bool(benchmark_processes(marker))


def stop_benchmark(marker: str) -> None:
    for proc in benchmark_processes(marker):
        try:
            proc.terminate()
        except Exception:
            continue
    time.sleep(2)
    for proc in benchmark_processes(marker):
        try:
            proc.kill()
        except Exception:
            continue


def gateway_healthy() -> bool:
    import httpx
    sys.path.insert(0, str(ROOT / "SimpleMem"))
    import config
    try:
        base_attr = [a for a in dir(config) if a.endswith("_BASE_URL") and "JUDGE" not in a][0]
        key_attr = [a for a in dir(config) if a.endswith("_API_KEY") and "JUDGE" not in a][0]
        base, key = getattr(config, base_attr), getattr(config, key_attr)
        # The gateway drops ~1/3 handshakes right now; a single probe would
        # spuriously report "busy" and stall bench launches. 3 tries, 1 pass.
        for attempt in range(3):
            try:
                r = httpx.post(base.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + key},
                    json={"model": config.LLM_MODEL,
                          "messages": [{"role": "user", "content": "Reply exactly: OK"}],
                          "temperature": 0, "thinking": {"type": "disabled"}, "max_tokens": 8},
                    timeout=30, trust_env=False)
                if r.status_code == 200:
                    content = (r.json().get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                    if str(content).strip():
                        return True
            except Exception:
                pass
            if attempt < 2:
                time.sleep((2 ** attempt) + random.random())
        return False
    except Exception:
        return False


def gateway_listing() -> str:
    try:
        import httpx
        sys.path.insert(0, str(ROOT / "SimpleMem"))
        import config
        base_attr = [a for a in dir(config) if a.endswith("_BASE_URL") and "JUDGE" not in a][0]
        key_attr = [a for a in dir(config) if a.endswith("_API_KEY") and "JUDGE" not in a][0]
        base, key = getattr(config, base_attr), getattr(config, key_attr)
        r = httpx.get(base.rstrip("/") + "/models",
                      headers={"Authorization": "Bearer " + key}, timeout=30, trust_env=False)
        return f"{len(r.json().get('data', []))} models listed"
    except Exception:
        return "listing failed"


def embedding_healthy() -> bool:
    import httpx
    try:
        r = httpx.post("http://127.0.0.1:8099/v1/embeddings",
                       json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": ["ping"]},
                       timeout=20, trust_env=False)
        return r.status_code == 200
    except Exception:
        return False


def ensure_embedding(wait_seconds: float = 180.0) -> bool:
    if embedding_healthy():
        return True
    log("embedding service unhealthy; starting and waiting for readiness")
    with (ROOT / "result" / "embedding_server.log").open("a", encoding="utf-8") as handle:
        subprocess.Popen(
            [PYTHON, str(ROOT / "MemNexus" / "memory-system" / "scripts" / "qwen_embedding_server.py"),
             "--host", "127.0.0.1", "--port", "8099"],
            cwd=str(ROOT / "MemNexus" / "memory-system"),
            stdout=handle, stderr=subprocess.STDOUT,
            env={**child_environment(), "HF_HUB_OFFLINE": "1"})
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if embedding_healthy():
            log("embedding service healthy")
            return True
        time.sleep(5)
    log("embedding service did not become ready within 180 seconds")
    return False


def done_count(out: Path) -> int:
    """Count completed conversation/question x system units, not dataset rows."""
    return len(list((out / "attempts").glob("*/stage_result.json")))


def expected_unit_count(cohort_path: Path) -> int:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    questions = cohort["questions"]
    if "setting" in questions[0]:
        return 2 * len(questions)
    return 2 * len({question.get("sample_index", 0) for question in questions})


def completed_row_count(out: Path) -> int:
    total = 0
    for result in out.glob("attempts/*/stage_result.json"):
        try:
            total += len(json.loads(result.read_text(encoding="utf-8")).get("rows", []))
        except (OSError, json.JSONDecodeError):
            continue
    return total


def checkpoint_progress(out: Path) -> tuple[int, float]:
    """Count resumable non-empty system answers and latest provider activity."""
    valid = 0
    latest_activity = 0.0
    for path in (out / "attempts").glob("*/usage.rows.jsonl"):
        latest_activity = max(latest_activity, path.stat().st_mtime)
        completed = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prediction = str(row.get("prediction") or "").strip()
            if not prediction or prediction == "Failed to generate answer" or row.get("error"):
                continue
            key = (str(row.get("sample_id", "")), str(row.get("qa_index", "")),
                   str(row.get("question", "")), str(row.get("evaluation_query", "")),
                   str(row.get("gold_answer", "")))
            completed[key] = True
        valid += len(completed)
    for path in (out / "attempts").glob("*/usage.json"):
        try:
            latest_activity = max(latest_activity, path.stat().st_mtime)
        except OSError:
            continue
    return valid, latest_activity


def preserve_poisoned(out: Path) -> int:
    """Quarantine all-empty checkpoints by rename, preserving every prior file."""
    preserved = 0
    for result in out.glob("attempts/*/stage_result.json"):
        unit = result.parent
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))
            rows = payload.get("rows", [])
            if rows and all(not (r.get("prediction") or "").strip() for r in rows):
                stamp = f"{datetime.now():%Y%m%dT%H%M%S}_{os.getpid()}"
                invalid_result = unit / f"invalid_stage_result_{stamp}.json"
                result.rename(invalid_result)
                ok = unit / "ok.json"
                if ok.exists():
                    ok.rename(unit / f"invalid_ok_{stamp}.json")
                sample_id = unit.name.rsplit("_", 1)[0]
                snap_dir = ROOT / "result" / "memnexus_snapshots"
                for pattern in (f"{sample_id}.json", f"{sample_id}.sqlite3"):
                    for snap in snap_dir.rglob(pattern):
                        snap.rename(snap.with_name(f"{snap.name}.invalid_{stamp}"))
                preserved += 1
                log(f"preserved poisoned unit {unit.name}; invalid checkpoints quarantined by rename")
        except Exception as error:
            log(f"poisoned-checkpoint preservation failed for {unit.name}: {error}")
    return preserved


def main() -> None:
    max_hours = float(os.getenv("WATCHDOG_MAX_HOURS", "168"))
    max_cycles = max(1, int(max_hours * 3600 / 300))
    active = {name: spec for name, spec in BENCHES.items() if spec.get("enabled", True)}
    expected = {name: expected_unit_count(spec["cohort"]) for name, spec in active.items()}
    log(f"watchdog start; max_hours={max_hours} expected_units={expected}")
    for name in BENCHES.keys() - active.keys():
        log(f"{name}: paused by configuration; checkpoints preserved")
    for cycle in range(max_cycles):
        try:
            if not ensure_embedding():
                log("deferring both benches until embedding is ready")
                if cycle + 1 < max_cycles:
                    time.sleep(300)
                continue
            for name, spec in active.items():
                preserve_poisoned(spec["out"])
                done = done_count(spec["out"])
                rows = completed_row_count(spec["out"])
                valid, last_activity = checkpoint_progress(spec["out"])
                activity_age = int(time.time() - last_activity) if last_activity else -1
                alive = process_running(spec["marker"])
                log(f"{name}: units={done}/{expected[name]} rows={rows}/1000 "
                    f"valid={valid}/2000 activity_age_s={activity_age} alive={alive} {gateway_listing()}")
                stall_seconds = int(os.getenv("WATCHDOG_STALL_SECONDS", "2700"))
                if alive and done < expected[name] and 0 <= activity_age <= stall_seconds:
                    pass
                elif alive and done < expected[name] and activity_age > stall_seconds:
                    log(f"{name}: no checkpoint/provider activity for {activity_age}s; restarting stalled bench")
                    stop_benchmark(spec["marker"])
                    alive = False
                if not alive and done < expected[name]:
                    if not gateway_healthy():
                        log(f"{name}: work left but gateway busy; deferring launch")
                        continue
                    log(f"{name}: bench process dead with work left; restarting")
                    with (ROOT / spec["log"]).open("a", encoding="utf-8") as handle:
                        subprocess.Popen(spec["cmd"], cwd=str(ROOT), env=child_environment(),
                                         stdout=handle, stderr=subprocess.STDOUT)
        except Exception as error:
            log(f"watchdog error: {type(error).__name__}: {error}")
        if cycle + 1 < max_cycles:
            time.sleep(300)
    log(f"watchdog stopped after {max_cycles} cycles; inspect incomplete units before extending")


if __name__ == "__main__":
    main()
