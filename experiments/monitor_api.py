"""Probe the gateway every 30s and log channel health to result/api_monitor.log.

Purpose: the overnight run must not start while channels are down, and we want a
timestamped record of the outage window. Run in background.
"""
import sys, time, httpx

sys.path.insert(0, r"C:/Users/zhangjinhan/Desktop/10-agent/SimpleMem")
import config

BASE = getattr(config, [a for a in dir(config) if a.endswith("_BASE_URL") and "JUDGE" not in a][0])
KEY = getattr(config, [a for a in dir(config) if a.endswith("_API_KEY") and "JUDGE" not in a][0])
MODELS = ["9g-Kimi", "cctoken-Kimi", "ddtoken-Kimi",
          "Kimi-claude"]
LOG = r"C:/Users/zhangjinhan/Desktop/10-agent/result/api_monitor.log"

with open(LOG, "a", encoding="utf-8") as handle:
    handle.write(f"monitor start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    while True:
        line = []
        for mid in MODELS:
            try:
                r = httpx.post(BASE.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + KEY},
                    json={"model": mid, "messages": [{"role": "user", "content": "Reply exactly: OK"}],
                          "max_tokens": 64},
                    timeout=45)
                d = r.json() if r.status_code == 200 else {}
                content = (d.get("choices", [{}])[0].get("message", {}) or {}).get("content")
                ok = r.status_code == 200 and content and content.strip()
                line.append(f"{mid.split('-')[0]}={'OK' if ok else r.status_code}")
            except Exception as e:
                line.append(f"{mid.split('-')[0]}={type(e).__name__}")
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        handle.write(f"[{stamp}] " + " ".join(line) + "\n")
        handle.flush()
        time.sleep(30)
