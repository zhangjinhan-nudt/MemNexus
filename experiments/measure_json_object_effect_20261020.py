"""Does response_format=json_object actually help on this gateway?

Runs the same real extraction prompt N times with and without the directive and
reports what changed: envelope validity, fence leakage, length, tokens, latency.

Also checks the one case where json_object can hurt: a workload whose payload is
a top-level JSON array (SimpleMem extraction), which the "object" mode forbids.

Cost: a few fen. Writes nothing outside stdout and an optional JSON report.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "MemNexus" / "memory-system" / ".env"
RUNS = 5


def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def load_real_prompt() -> str:
    """Reuse the exact prompt that failed before the thinking fix."""
    candidates = sorted((ROOT / "result").glob("connectivity_*/usage.calls.jsonl"))
    for path in reversed(candidates):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("event") == "request":
                return record["body"]["messages"][0]["content"]
    raise SystemExit("no recorded real prompt found under result/connectivity_*")


def post(base: str, key: str, body: dict) -> tuple[int, dict | str, float]:
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return 200, json.loads(response.read().decode("utf-8")), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")[:250], (time.perf_counter() - started) * 1000
    except Exception as error:  # noqa: BLE001 - probe reports any failure verbatim
        return -1, f"{type(error).__name__}: {error}", (time.perf_counter() - started) * 1000


def inspect(content: str) -> dict:
    text = content.strip()
    fenced = text.startswith("```")
    stripped = text
    if fenced:
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    valid = False
    shape = "invalid"
    try:
        parsed = json.loads(stripped)
        valid = True
        shape = "object" if isinstance(parsed, dict) else ("array" if isinstance(parsed, list) else type(parsed).__name__)
    except json.JSONDecodeError:
        pass
    return {"valid": valid, "shape": shape, "fenced": fenced, "chars": len(stripped)}


def run_block(base: str, key: str, model: str, prompt: str, label: str, **extra) -> list[dict]:
    results = []
    for index in range(RUNS):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0,
            "thinking": {"type": "disabled"},
        }
        body.update(extra)
        status, payload, ms = post(base, key, body)
        if status != 200:
            results.append({"status": status, "error": str(payload)[:160], "ms": ms})
            continue
        choice = (payload.get("choices") or [{}])[0] or {}
        content = (choice.get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        detail = inspect(content)
        detail |= {
            "status": 200,
            "ms": ms,
            "finish": choice.get("finish_reason"),
            "tokens": usage.get("completion_tokens"),
            "chars": detail["chars"],
        }
        results.append(detail)
        print(f"  {label} #{index + 1}: {ms:6.0f}ms finish={detail['finish']:8s} "
              f"valid={detail['valid']} shape={detail['shape']:7s} fenced={detail['fenced']} "
              f"chars={detail['chars']:5d} tok={detail['tokens']}")
    return results


def summarize(label: str, results: list[dict]) -> None:
    ok = [r for r in results if r.get("status") == 200]
    if not ok:
        print(f"  {label}: 全部失败 {results[0].get('error')}")
        return
    valid = sum(1 for r in ok if r["valid"])
    fenced = sum(1 for r in ok if r["fenced"])
    lengths = [r["chars"] for r in ok]
    tokens = [r["tokens"] for r in ok if r["tokens"] is not None]
    latencies = [r["ms"] for r in ok]
    print(f"  {label}: 合法JSON {valid}/{len(ok)}  围栏 {fenced}/{len(ok)}  "
          f"字数 {statistics.mean(lengths):.0f}±{statistics.pstdev(lengths):.0f}  "
          f"token {statistics.mean(tokens):.0f}  "
          f"延迟 {statistics.mean(latencies):.0f}ms")


def main() -> int:
    env = read_env()
    base, key, model = env["MEMORY_MODEL_BASE_URL"], env["MEMORY_MODEL_API_KEY"], env["MEMORY_MODEL_NAME"]
    prompt = load_real_prompt()
    print(f"model={model}  prompt={len(prompt)} 字符  runs={RUNS}\n")

    print("== A. 不发送 response_format ==")
    without = run_block(base, key, model, prompt, "no-json")
    summarize("不发送", without)

    print("\n== B. 发送 response_format=json_object ==")
    with_json = run_block(base, key, model, prompt, "json-obj", response_format={"type": "json_object"})
    summarize("json_object", with_json)

    print("\n== C. 顶层数组负载 + json_object（SimpleMem 抽取形状）==")
    array_prompt = (
        "Extract every atomic fact from the dialogue below. "
        "Return a top-level JSON array. Each element has keys: content, speakers, timestamp.\n\n"
        "Caroline: I adopted a beagle named Comet last week.\n"
        "Melanie: I started pottery class last November.\n"
        "Caroline: My sister Dana visits in June.\n"
    )
    array_results = run_block(base, key, model, array_prompt, "array+json", response_format={"type": "json_object"})
    summarize("数组+json_object", array_results)

    payload = {
        "model": model,
        "prompt_chars": len(prompt),
        "runs": RUNS,
        "without_json_object": without,
        "with_json_object": with_json,
        "array_with_json_object": array_results,
    }
    out = ROOT / "result" / "json_object_effect_20261020.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
