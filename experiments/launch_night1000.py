"""Overnight orchestrator: LoCoMo 1000 + LongMemEval 1000 in parallel.

Sequence:
  1. Ensure the Qwen embedding service answers on :8099 (start + retry).
  2. Wait for the provider gateway to become healthy. The probe iterates every
     model listed by /models; the first live model is pinned into
     SimpleMem/config.py and MemNexus/.env so the benches use a live channel.
     Preference: keep the currently configured model if it is alive, else the
     fastest live channel.
  3. Launch both benches concurrently (concurrency 2 each). Both are
     resumable and write per-unit checkpoints.
  4. Blind-judge both cohorts once both finish.
Everything logs to result/night1000.log.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
PYTHON = Path(r"C:\Users\zhangjinhan\AppData\Local\Programs\Python\Python313\python.exe")
LOG = ROOT / "result" / "night1000.log"
CONFIG = ROOT / "SimpleMem" / "config.py"
ENV_FILE = ROOT / "MemNexus" / "memory-system" / ".env"
LOCOMO_COHORT = "result/locomo_1000_cohort_20261023.json"
LOCOMO_OUT = "result/bench1000_locomo_20261023"
LME_COHORT = "result/longmemeval_1000_cohort_20261023.json"
LME_OUT = "result/bench1000_longmemeval_20261023"
LME_SNAPSHOTS = "result/memnexus_snapshots/lme1000"


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        handle.flush()


def gateway_credentials() -> tuple[str, str, str]:
    """(base_url, api_key, currently configured model) without literals."""
    sys.path.insert(0, str(ROOT / "SimpleMem"))
    import config
    base_attr = [a for a in dir(config) if a.endswith("_BASE_URL") and "JUDGE" not in a][0]
    key_attr = [a for a in dir(config) if a.endswith("_API_KEY") and "JUDGE" not in a][0]
    return getattr(config, base_attr), getattr(config, key_attr), config.LLM_MODEL


def probe_models() -> dict[str, float]:
    """Return {live model id: latency_s}."""
    import httpx
    base, key, _ = gateway_credentials()
    try:
        listing = httpx.get(base.rstrip("/") + "/models",
                            headers={"Authorization": "Bearer " + key}, timeout=30)
        ids = [m["id"] for m in listing.json().get("data", [])]
    except Exception:
        return {}
    latencies: dict[str, float] = {}
    for mid in ids:
        started = time.time()
        try:
            r = httpx.post(base.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + key},
                json={"model": mid,
                      "messages": [{"role": "user", "content": "Reply exactly: OK"}],
                      "max_tokens": 32},
                timeout=60)
            if r.status_code != 200:
                continue
            content = (r.json().get("choices", [{}])[0].get("message", {}) or {}).get("content")
            if content and content.strip():
                latencies[mid] = time.time() - started
        except Exception:
            continue
    return latencies


def pin_model(model: str) -> None:
    """Point both systems' configs at the live model."""
    text = CONFIG.read_text(encoding="utf-8")
    text = re.sub(r'(?m)^LLM_MODEL = ".*"', f'LLM_MODEL = "{model}"', text)
    text = re.sub(r'(?m)^JUDGE_MODEL = ".*"', f'JUDGE_MODEL = "{model}"', text)
    CONFIG.write_text(text, encoding="utf-8")
    env = ENV_FILE.read_text(encoding="utf-8")
    env = re.sub(r'(?m)^MEMORY_MODEL_NAME=.*', f"MEMORY_MODEL_NAME={model}", env)
    env = re.sub(r'(?m)^MEMORY_JUDGE_MODEL_NAME=.*', f"MEMORY_JUDGE_MODEL_NAME={model}", env)
    ENV_FILE.write_text(env, encoding="utf-8")
    log(f"pinned model {model!r} into config.py and .env")


def wait_for_api(timeout_hours: float = 6.0) -> str:
    """Require 3 consecutive rounds with at least one live model."""
    deadline = time.time() + timeout_hours * 3600
    consecutive = 0
    while time.time() < deadline:
        live = probe_models()
        if live:
            _, _, configured = gateway_credentials()
            best = configured if configured in live else min(live, key=live.get)
            consecutive += 1
            log(f"api probe ok ({consecutive}/3), {len(live)} live models, best={best}")
            if consecutive >= 3:
                pin_model(best)
                return best
        else:
            consecutive = 0
            log("api probe failed: no live model")
        time.sleep(60)
    return ""


def ensure_embedding_server() -> bool:
    from launch_bench637 import ensure_embedding_server
    for attempt in range(4):
        if ensure_embedding_server():
            return True
        log(f"embedding server attempt {attempt + 1} failed; retrying")
        time.sleep(15)
    return False


def main() -> int:
    log("night1000 orchestrator start")
    if not ensure_embedding_server():
        log("embedding service failed to start; aborting")
        return 2
    log("waiting for provider gateway health")
    model = wait_for_api()
    if not model:
        log("gateway never became healthy within budget; aborting without spend")
        return 3
    log(f"gateway healthy with model {model!r}; launching benches")
    environment = dict(os.environ)
    environment.update({
        "NO_PROXY": "127.0.0.1,localhost",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "NIGHT_TEMPORAL_RESCUE": "0",
    })
    locomo_cmd = [
        str(PYTHON), "experiments/run_parallel_bench.py",
        "--cohort", LOCOMO_COHORT, "--out", LOCOMO_OUT,
        "--snapshots", "result/memnexus_snapshots/full30",
        "--concurrency", "2", "--stagger", "20",
        "--budget-per-conversation", "60",
    ]
    lme_cmd = [
        str(PYTHON), "experiments/run_longmemeval_bench.py",
        "--cohort", LME_COHORT, "--out", LME_OUT,
        "--snapshots", LME_SNAPSHOTS,
        "--concurrency", "2", "--stagger", "20",
        "--budget-per-question", "8",
    ]
    log("launching LoCoMo bench: " + " ".join(locomo_cmd))
    log("launching LongMemEval bench: " + " ".join(lme_cmd))
    locomo_log = (ROOT / "result" / "bench1000_locomo.log").open("a", encoding="utf-8")
    lme_log = (ROOT / "result" / "bench1000_longmemeval.log").open("a", encoding="utf-8")
    locomo = subprocess.Popen(locomo_cmd, cwd=str(ROOT), env=environment,
                              stdout=locomo_log, stderr=subprocess.STDOUT)
    lme = subprocess.Popen(lme_cmd, cwd=str(ROOT), env=environment,
                           stdout=lme_log, stderr=subprocess.STDOUT)
    locomo_code = locomo.wait()
    log(f"LoCoMo bench finished with exit code {locomo_code}")
    lme_code = lme.wait()
    log(f"LongMemEval bench finished with exit code {lme_code}")

    if locomo_code == 0:
        log("judging LoCoMo rows")
        judge = subprocess.run(
            [str(PYTHON), "experiments/run_parallel_bench.py",
             "--cohort", LOCOMO_COHORT, "--out", LOCOMO_OUT, "--judge"],
            cwd=str(ROOT), env=environment, stdout=locomo_log, stderr=subprocess.STDOUT)
        log(f"LoCoMo judging finished with exit code {judge.returncode}")
    if lme_code == 0:
        log("judging LongMemEval rows")
        judge = subprocess.run(
            [str(PYTHON), "experiments/run_longmemeval_bench.py",
             "--cohort", LME_COHORT, "--out", LME_OUT, "--judge"],
            cwd=str(ROOT), env=environment, stdout=lme_log, stderr=subprocess.STDOUT)
        log(f"LongMemEval judging finished with exit code {judge.returncode}")
    log("night1000 orchestrator done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
