"""One-line status snapshot for the overnight double run."""
import json, subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:/Users/zhangjinhan/Desktop/10-agent/result")
NOW = datetime.now().strftime("%H:%M:%S")
CUTOFF = "2026-10-23 20:19:00"

def count(pattern_dir, pattern):
    return len(list(pattern_dir.glob(pattern)))

for name in ("bench1000_locomo_20261023", "bench1000_longmemeval_20261023"):
    d = ROOT / name / "attempts"
    if not d.exists():
        print(f"{NOW} {name}: no dir")
        continue
    done = count(d, "*/stage_result.json")
    fails = sum(1 for p in d.glob("*/failure.json")
                if not (p.parent / "stage_result.json").exists()
                and p.stat().st_mtime > datetime.strptime(CUTOFF, "%Y-%m-%d %H:%M:%S").timestamp())
    mn = sum(1 for p in d.glob("*/stage_result.json") if "memnexus" in p.parent.name)
    sm = sum(1 for p in d.glob("*/stage_result.json") if "simplemem" in p.parent.name)
    log = ROOT / ("bench1000_locomo.log" if "locomo" in name else "bench1000_longmemeval.log")
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    last = lines[-1][:110] if lines else ""
    print(f"{NOW} {name}: done={done} (MN={mn} SM={sm}) new_failures={fails} last={last}")
