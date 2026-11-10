"""Production-equivalent chat test: thinking=disabled, check content non-empty."""
import sys, time, httpx
sys.path.insert(0, r"C:/Users/zhangjinhan/Desktop/10-agent/SimpleMem")
import config

base = getattr(config, [a for a in dir(config) if a.endswith("_BASE_URL") and "JUDGE" not in a][0])
key = getattr(config, [a for a in dir(config) if a.endswith("_API_KEY") and "JUDGE" not in a][0])

# exact production payload: thinking disabled + temp 0
payload = {
    "model": config.LLM_MODEL,
    "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
    "max_tokens": 256,
    "temperature": 0,
    "thinking": {"type": "disabled"},
}
t0 = time.time()
r = httpx.post(base.rstrip("/") + "/chat/completions",
    headers={"Authorization": "Bearer " + key},
    json=payload, timeout=90)
d = r.json() if r.status_code == 200 else {}
msg = (d.get("choices", [{}])[0].get("message", {}) or {}) if r.status_code == 200 else {}
print("status:", r.status_code, f"{time.time()-t0:.1f}s")
print("content:", repr(msg.get("content"))[:100])
print("reasoning_content len:", len(msg.get("reasoning_content") or ""))
print("finish_reason:", (d.get("choices", [{}])[0] or {}).get("finish_reason") if r.status_code == 200 else None)
print("usage:", d.get("usage") if r.status_code == 200 else r.text[:200])
