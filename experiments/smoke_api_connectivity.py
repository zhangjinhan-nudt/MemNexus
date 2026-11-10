"""Connectivity smoke test for the overnight 1000-question run."""
import sys, os, httpx
sys.path.insert(0, r"C:/Users/zhangjinhan/Desktop/10-agent/SimpleMem")
import config

def test(name, base, key, model):
    try:
        r = httpx.post(base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                  "max_tokens": 20},
            timeout=60)
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        print(f"{name}: HTTP {r.status_code} reply={text!r} usage="
              f"{ {k: usage.get(k) for k in ('prompt_tokens', 'completion_tokens')} }")
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__}: {e}")

test("SimpleMem LLM", getattr(config, "Moonshot AI_BASE_URL"),
     getattr(config, "Moonshot AI_API_KEY"), config.LLM_MODEL)
test("Judge", config.JUDGE_BASE_URL, config.JUDGE_API_KEY, config.JUDGE_MODEL)

from dotenv import load_dotenv
load_dotenv(r"C:/Users/zhangjinhan/Desktop/10-agent/MemNexus/memory-system/.env")
test("MemNexus LLM", os.getenv("MEMORY_MODEL_BASE_URL"), os.getenv("MEMORY_MODEL_API_KEY"),
     os.getenv("MEMORY_MODEL_NAME"))
