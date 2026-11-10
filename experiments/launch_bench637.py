"""Detached launcher for the conv-48/49/50 benchmark (637 questions).

Why this exists: background shells started from the agent session are children
of that session and are killed when the session ends. A Windows scheduled task
owns its own process tree, so the run survives closing WorkBuddy. It does NOT
survive machine shutdown - nothing local does.

Sequence: ensure the Qwen embedding service answers on :8099 (start it and wait
for cold start if not), then run the benchmark. The embedding service is left
running afterwards; it is cheap and the next run will reuse it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\zhangjinhan\AppData\Local\Programs\Python\Python313\python.exe")
SERVER = ROOT / "MemNexus" / "memory-system" / "scripts" / "qwen_embedding_server.py"
LOG = ROOT / "result" / "bench637_launcher.log"
COHORT = os.getenv("BENCH_COHORT", "result/locomo_150_cohort_conv48-50_20261023.json")
OUT = os.getenv("BENCH_OUT", "result/bench150_conv48-50_20261023")
SNAPSHOTS = "result/memnexus_snapshots/full30"


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def embedding_ready(timeout: float = 3.0) -> bool:
    payload = json.dumps({"input": ["ping"]}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8099/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    # The user session may carry a system proxy; a loopback probe must never
    # be routed through it (it returns an upstream error, never 200).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ensure_embedding_server(wait_seconds: float = 180.0) -> bool:
    if embedding_ready():
        log("embedding service already healthy")
        return True
    log("starting embedding service")
    environment = dict(os.environ)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    server_log = (ROOT / "result" / "embedding_server.log").open("a", encoding="utf-8")
    subprocess.Popen(
        [str(PYTHON), str(SERVER), "--host", "127.0.0.1", "--port", "8099"],
        cwd=str(ROOT / "MemNexus" / "memory-system"),
        env=environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "DETACHED_PROCESS", 0),
    )
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if embedding_ready():
            log("embedding service healthy")
            return True
        time.sleep(5)
    log("embedding service FAILED to become healthy")
    return False


def main() -> int:
    log("launcher start")
    if not ensure_embedding_server():
        return 2
    environment = dict(os.environ)
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["PYTHONUNBUFFERED"] = "1"
    # The console is GBK on this machine; a status line containing any other
    # character must not be able to kill the run.
    environment["PYTHONIOENCODING"] = "utf-8"
    command = [
        str(PYTHON),
        "experiments/run_parallel_bench.py",
        "--cohort", COHORT,
        "--out", OUT,
        "--snapshots", SNAPSHOTS,
        "--concurrency", os.getenv("BENCH_CONCURRENCY", "3"),
        "--stagger", os.getenv("BENCH_STAGGER", "8"),
        "--budget-per-conversation", "60",
    ]
    log("bench command: " + " ".join(command))
    bench_log = (ROOT / "result" / "bench637_run.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=str(ROOT), env=environment,
        stdout=bench_log, stderr=subprocess.STDOUT,
    )
    code = process.wait()
    log(f"bench finished with exit code {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
