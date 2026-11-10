"""Sequential gateway stability probe: 8 calls, 3s apart."""
import sys, time, httpx
sys.path.insert(0, r"C:/Users/zhangjinhan/Desktop/10-agent/SimpleMem")
import config

base = getattr(config, [a for a in dir(config) if a.endswith("_BASE_URL") and "JUDGE" not in a][0])
key = getattr(config, [a for a in dir(config) if a.endswith("_API_KEY") and "JUDGE" not in a][0])
ok = 0
for i in range(8):
    t0 = time.time()
    try:
        r = httpx.post(base.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={"model": config.LLM_MODEL, "messages": [{"role": "user", "content": "OK?"}],
                  "max_tokens": 16, "thinking": {"type": "disabled"}},
            timeout=60)
        content = ""
        if r.status_code == 200:
            d = r.json()
            content = ((d.get("choices", [{}])[0].get("message", {}) or {}).get("content") or "").strip()
            ok += 1
        print(i, r.status_code, f"{time.time()-t0:.1f}s", repr(content)[:30], r.text[:80] if r.status_code != 200 else "")
    except Exception as e:
        print(i, "ERR", type(e).__name__, str(e)[:80])
    time.sleep(3)
print(f"SUMMARY ok={ok}/8")