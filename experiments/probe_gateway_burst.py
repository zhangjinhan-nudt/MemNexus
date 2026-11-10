"""Burst probe the new gateway: 5 rapid chat calls with 1s gaps."""
import sys, time, httpx
sys.path.insert(0, r"C:/Users/zhangjinhan/Desktop/10-agent/SimpleMem")
import config

base = getattr(config, [a for a in dir(config) if a.endswith("_BASE_URL") and "JUDGE" not in a][0])
key = getattr(config, [a for a in dir(config) if a.endswith("_API_KEY") and "JUDGE" not in a][0])
for i in range(5):
    t0 = time.time()
    try:
        r = httpx.post(base.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={"model": config.LLM_MODEL, "messages": [{"role": "user", "content": "OK?"}],
                  "max_tokens": 8},
            timeout=60)
        print(i, r.status_code, f"{time.time()-t0:.1f}s", r.text[:130])
    except Exception as e:
        print(i, "ERR", type(e).__name__, str(e)[:130])
    time.sleep(1)
