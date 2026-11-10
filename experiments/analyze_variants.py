"""Paired comparison of MemNexus baseline vs the three LLM variants.

Each variant directory holds a full 150-question run of MemNexus under one
flag; SimpleMem rows are shared with the baseline.  This script computes, per
configuration:

  * paper-protocol F1 / BLEU-1 / adversarial success (same as the main table)
  * paired per-question delta against the MemNexus baseline run
  * cost: provider attempts and tokens per question
  * mechanism telemetry: answer_path distribution, forced abstentions, and the
    gate's false-abstain rate (abstaining on a question that had an answer)

Usage:
    python experiments/analyze_variants.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from paper_protocol import adversarial_success, bleu1_of, f1_of  # noqa: E402

BASE = ROOT / "result" / "bench150_conv48-50_20261023" / "rows_baseline.json"
VARIANTS = ["bridge", "gate", "execution"]
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def key_of(row: dict) -> tuple:
    return (row.get("conversation_id"), row.get("qa_index") or row.get("question_id"))


def summarize(rows: list[dict], system: str) -> dict:
    sub = [r for r in rows if r["system"] == system]
    f1 = [float(r["f1"]) for r in sub if r.get("category_id") != 5]
    bleu = [float(r["bleu1"]) for r in sub if r.get("category_id") != 5]
    adv = [r for r in sub if r.get("category_id") == 5]
    adv_ok = sum(1 for r in adv if adversarial_success(r))
    calls = [(r.get("model_calls") or {}).get("provider_attempts", 0) for r in sub]
    toks = [(r.get("model_calls") or {}).get("input_tokens", 0)
            + (r.get("model_calls") or {}).get("output_tokens", 0) for r in sub]
    paths = Counter((r.get("reasoning") or {}).get("answer_path") or "none" for r in sub)
    return {
        "n": len(sub),
        "f1": statistics.mean(f1) if f1 else 0.0,
        "bleu": statistics.mean(bleu) if bleu else 0.0,
        "adv_rate": adv_ok / len(adv) if adv else 0.0,
        "adv_n": len(adv),
        "adv_ok": adv_ok,
        "calls_per_q": statistics.mean(calls) if calls else 0.0,
        "tok_per_q": statistics.mean(toks) if toks else 0.0,
        "paths": dict(paths),
        "forced_abstain": sum(1 for r in sub if r.get("forced_abstain")),
    }


def per_category(rows: list[dict], system: str) -> dict:
    sub = [r for r in rows if r["system"] == system and r.get("category_id") != 5]
    out = {}
    for cat, name in CATEGORY.items():
        if cat == 5:
            continue
        vals = [float(r["f1"]) for r in sub if r.get("category_id") == cat]
        out[name] = (round(statistics.mean(vals), 2) if vals else None, len(vals))
    return out


def paired_delta(base_rows: list[dict], var_rows: list[dict]) -> dict:
    b = {key_of(r): r for r in base_rows if r["system"] == "MemNexus"}
    v = {key_of(r): r for r in var_rows if r["system"] == "MemNexus"}
    shared = sorted((k for k in set(b) & set(v)), key=lambda k: (str(k[0]), str(k[1])))
    diffs = []
    better = worse = same = 0
    for k in shared:
        fb = float(b[k]["f1"])
        fv = float(v[k]["f1"])
        d = fv - fb
        diffs.append(d)
        if d > 1e-9:
            better += 1
        elif d < -1e-9:
            worse += 1
        else:
            same += 1
    n = len(diffs)
    mean = statistics.mean(diffs) if n else 0.0
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n else 0.0
    return {"n": n, "mean_delta": round(mean, 3), "se": round(se, 3),
            "ci": (round(mean - 1.96 * se, 3), round(mean + 1.96 * se, 3)),
            "better": better, "same": same, "worse": worse}


def false_abstain(var_rows: list[dict]) -> tuple[int, int]:
    """Abstentions on questions that actually have an answer (category != 5)."""
    sub = [r for r in var_rows if r["system"] == "MemNexus" and r.get("category_id") != 5]
    bad = sum(1 for r in sub
              if "not mentioned" in (r["prediction"] or "").lower()
              or r.get("forced_abstain"))
    return bad, len(sub)


def main() -> int:
    base_rows = load(BASE)
    configs = {"baseline": base_rows}
    for name in VARIANTS:
        path = ROOT / "result" / "bench150_variants" / name / "rows.json"
        configs[name] = load(path)

    print(f"{'config':<10} {'F1':>6} {'BLEU':>6} {'advSR':>7} {'calls/q':>8} {'tok/q':>7} "
          f"{'fAbst':>6}  answer_path")
    summary = {}
    for name, rows in configs.items():
        s = summarize(rows, "MemNexus")
        summary[name] = s
        fa, nfa = false_abstain(rows)
        print(f"{name:<10} {s['f1']:>6.2f} {s['bleu']:>6.2f} "
              f"{s['adv_ok']}/{s['adv_n']} ({s['adv_rate']*100:>4.1f}%) "
              f"{s['calls_per_q']:>8.2f} {s['tok_per_q']:>7.0f} {fa:>4}/{nfa}  {s['paths']}")

    print("\nper-category F1 (MemNexus, adversarial excluded):")
    for name, rows in configs.items():
        pc = per_category(rows, "MemNexus")
        print(f"  {name:<10} " + "  ".join(f"{k}={v[0]}(n={v[1]})" for k, v in pc.items()))

    print("\npaired delta vs MemNexus baseline (paper protocol):")
    for name in VARIANTS:
        d = paired_delta(base_rows, configs[name])
        print(f"  {name:<10} mean={d['mean_delta']:+.3f}  se={d['se']:.3f}  "
              f"95% CI [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  "
              f"better={d['better']} same={d['same']} worse={d['worse']}  n={d['n']}")

    print("\nSimpleMem (shared reference):")
    sm = summarize(base_rows, "SimpleMem")
    print(f"  F1={sm['f1']:.2f}  BLEU={sm['bleu']:.2f}  "
          f"advSR={sm['adv_ok']}/{sm['adv_n']} ({sm['adv_rate']*100:.1f}%)  "
          f"calls/q={sm['calls_per_q']:.2f}  tok/q={sm['tok_per_q']:.0f}")

    out = {"summary": summary, "simplemem": sm,
           "paired": {n: paired_delta(base_rows, configs[n]) for n in VARIANTS}}
    (ROOT / "result" / "variant_comparison_20261023.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nwrote result/variant_comparison_20261023.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
