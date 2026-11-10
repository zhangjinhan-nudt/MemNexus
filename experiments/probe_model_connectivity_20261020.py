"""Probe the configured LLM endpoint for both systems.

Checks, per candidate model:
  1. plain chat
  2. thinking disabled honoured (content non-empty under a tiny budget)
  3. response_format=json_object honoured
  4. usage / cache fields present (needed by the cost ledger)
  5. latency on a realistic prompt size

No file outside this script is written; cost is a few fen.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "MemNexus" / "memory-system" / ".env"
CANDIDATES = ["deepseek-flash", "deepseek-v4-pro", "deepseek-v4.1-flash"]


def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def post(base: str, key: str, body: dict) -> tuple[int, dict | str, float]:
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return 200, payload, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")[:300], (time.perf_counter() - started) * 1000
    except Exception as error:  # noqa: BLE001 - probe script reports any failure
        return -1, f"{type(error).__name__}: {error}", (time.perf_counter() - started) * 1000


def content_of(payload: dict) -> str:
    choices = payload.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    return message.get("content") or ""


def reasoning_of(payload: dict) -> str:
    choices = payload.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    return message.get("reasoning_content") or message.get("reasoning") or ""


def main() -> int:
    env = read_env()
    base = env["MEMORY_MODEL_BASE_URL"]
    key = env["MEMORY_MODEL_API_KEY"]
    configured = env["MEMORY_MODEL_NAME"]
    judge = env.get("MEMORY_JUDGE_MODEL_NAME", "")

    print(f"base_url        : {base}")
    print(f"configured model: {configured!r}")
    print(f"judge model     : {judge!r}")
    print()

    print("== 1. 配置中的模型名 ==")
    status, body, ms = post(base, key, {
        "model": configured,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0,
    })
    print(f"  {configured!r}: HTTP {status} ({ms:.0f}ms) {body if status != 200 else content_of(body)!r}")
    print()

    rows = []
    for model in CANDIDATES:
        print(f"== {model} ==")
        row = {"model": model}

        # 2. thinking disabled, tiny budget: content must not be empty
        status, body, ms = post(base, key, {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 8,
            "temperature": 0,
            "thinking": {"type": "disabled"},
        })
        if status == 200:
            text, think = content_of(body), reasoning_of(body)
            row["thinking_disabled"] = "OK" if text.strip() else f"EMPTY(content={text!r}, reasoning={think[:40]!r})"
            row["finish"] = (body.get("choices") or [{}])[0].get("finish_reason")
        else:
            row["thinking_disabled"] = f"HTTP {status} {str(body)[:120]}"
        row["t_disabled_ms"] = round(ms)

        # 3. no thinking directive: does it think by default?
        status, body, ms = post(base, key, {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 8,
            "temperature": 0,
        })
        if status == 200:
            text, think = content_of(body), reasoning_of(body)
            row["no_directive"] = "content" if text.strip() else ("thinking_only" if think else "empty")
            usage = body.get("usage") or {}
            row["completion_tokens"] = usage.get("completion_tokens")
        else:
            row["no_directive"] = f"HTTP {status} {str(body)[:120]}"

        # 4. json_object
        status, body, ms = post(base, key, {
            "model": model,
            "messages": [{"role": "user", "content": 'Return only JSON: {"ok": true, "n": 3}'}],
            "max_tokens": 64,
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        })
        if status == 200:
            text = content_of(body)
            try:
                parsed = json.loads(text)
                row["json_object"] = f"OK {parsed}"
            except json.JSONDecodeError:
                row["json_object"] = f"NOT JSON: {text[:80]!r}"
        else:
            row["json_object"] = f"HTTP {status} {str(body)[:120]}"

        # 5. usage envelope completeness for the cost ledger
        status, body, ms = post(base, key, {
            "model": model,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 8,
            "temperature": 0,
            "thinking": {"type": "disabled"},
        })
        if status == 200:
            usage = body.get("usage") or {}
            row["usage_keys"] = sorted(usage)
            row["cached_field"] = any("cache" in k for k in usage)
        rows.append(row)
        print(f"  thinking=disabled : {row['thinking_disabled']}  finish={row.get('finish')}  {row['t_disabled_ms']}ms")
        print(f"  no directive      : {row['no_directive']}  completion_tokens={row.get('completion_tokens')}")
        print(f"  json_object       : {row['json_object']}")
        print(f"  usage keys        : {row.get('usage_keys')}  has_cache={row.get('cached_field')}")
        print()

    print("== 汇总 ==")
    header = f"{'model':24s} {'think_off':10s} {'default':14s} {'json':10s}"
    print(header)
    for row in rows:
        json_ok = "OK" if str(row.get("json_object", "")).startswith("OK") else "FAIL"
        print(f"{row['model']:24s} {str(row.get('thinking_disabled'))[:10]:10s} "
              f"{str(row.get('no_directive')):14s} {json_ok:10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
