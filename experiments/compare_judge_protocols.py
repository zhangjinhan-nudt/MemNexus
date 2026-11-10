"""对比 strict 与 paper 两套判分口径的结果差异。

旧判决在 `attempts/judge_*/judgment.json`（strict），
新判决在 `result/judge_paper_20261020/*.json`（SimpleMem Appendix A.4，generous）。

用法：
    python experiments/compare_judge_protocols.py result/benchmark100/20261020_161439_654687
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import statistics

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "result" / "judge_paper_20261020"


def load_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((run_dir / "attempts").iterdir()):
        if not path.is_dir() or path.name.startswith("judge_"):
            continue
        stage = path / "stage_result.json"
        if not stage.exists():
            continue
        rows.extend(json.loads(stage.read_text(encoding="utf-8"))["rows"])
    return rows


def old_verdicts(run_dir: Path) -> dict[tuple, str]:
    out: dict[tuple, str] = {}
    for path in (run_dir / "attempts").glob("judge_*/judgment.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        out[(value["sample_id"], value["qa_index"], value["system"])] = value["judge_result"]
    return out


def new_verdicts() -> dict[tuple, str]:
    out: dict[tuple, str] = {}
    for path in PAPER_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        out[(value["sample_id"], value["qa_index"], value["system"])] = value["judge_result"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.run)
    old = old_verdicts(args.run)
    new = new_verdicts()

    pairs = []
    for row in rows:
        key = (row["sample_id"], row["qa_index"], row["system"])
        if key in old and key in new:
            pairs.append((row, old[key], new[key]))

    print(f"两套口径都有判决的题：{len(pairs)}")
    if not pairs:
        return 1

    print("\n=== 判决迁移矩阵（行=strict，列=paper）===")
    print(f"{'':12s}{'CORRECT':>10s}{'WRONG':>10s}")
    for a in ("CORRECT", "WRONG"):
        cells = [sum(1 for _, o, n in pairs if o == a and n == b) for b in ("CORRECT", "WRONG")]
        print(f"{a:12s}{cells[0]:10d}{cells[1]:10d}")

    up = [(r, o, n) for r, o, n in pairs if o == "WRONG" and n == "CORRECT"]
    down = [(r, o, n) for r, o, n in pairs if o == "CORRECT" and n == "WRONG"]
    print(f"\nstrict WRONG -> paper CORRECT : {len(up)} 条（放宽）")
    print(f"strict CORRECT -> paper WRONG : {len(down)} 条（收紧，理论上不该有）")

    print("\n=== 两套口径下的判断比例 ===")
    for system in ("MemNexus", "SimpleMem"):
        sub = [(r, o, n) for r, o, n in pairs if r["system"] == system]
        if not sub:
            continue
        oc = sum(1 for _, o, _ in sub if o == "CORRECT")
        nc = sum(1 for _, _, n in sub if n == "CORRECT")
        print(f"{system:11s} strict {oc:3d}/{len(sub):3d} = {oc/len(sub):6.1%}"
              f"    paper {nc:3d}/{len(sub):3d} = {nc/len(sub):6.1%}")

    # 同题配对
    by_key: dict[tuple, dict] = {}
    for r, o, n in pairs:
        by_key.setdefault((r["sample_id"], r["qa_index"]), {})[r["system"]] = (r, o, n)
    paired = {k: v for k, v in by_key.items() if len(v) == 2}
    for label, idx in (("strict", 1), ("paper", 2)):
        c = Counter()
        for v in paired.values():
            m, s = v["MemNexus"][idx], v["SimpleMem"][idx]
            c[(m == "CORRECT", s == "CORRECT")] += 1
        both, mo, so, none = c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]
        print(f"\n[{label}] 双方都判 {len(paired)} 题：都对 {both}  MN独对 {mo}  SM独对 {so}  都错 {none}"
              f"  → 净胜 {mo - so:+d}")

    print("\n=== 论文主指标（Average F1 / BLEU-1 / Adversarial SR / Token）===")
    print(f"{'系统':11s}{'F1':>9s}{'BLEU-1':>9s}{'AdvSR':>9s}{'in/题':>8s}{'out/题':>8s}{'¥/题':>9s}")
    AB = re.compile(
        r"^\s*(?:A\.\s*)?not mentioned|no information|cannot be determined|not stated|"
        r"does not (?:mention|provide|specify)", re.I)
    for system in ("MemNexus", "SimpleMem"):
        sub = [r for r in rows if r["system"] == system]
        if not sub:
            continue
        f1 = statistics.mean(r["f1"] for r in sub)
        bl = statistics.mean(r["bleu1"] for r in sub)
        adv = [r for r in sub if r["category"] == "adversarial"]
        sr = sum(1 for r in adv if AB.search(str(r["prediction"]))) / max(len(adv), 1)
        tin = sum((r.get("model_calls") or {}).get("input_tokens") or 0 for r in sub) / len(sub)
        tout = sum((r.get("model_calls") or {}).get("output_tokens") or 0 for r in sub) / len(sub)
        cost = sum((r.get("model_calls") or {}).get("cost_rmb") or 0 for r in sub) / len(sub)
        print(f"{system:11s}{f1:9.4f}{bl:9.4f}{sr:9.1%}{tin:8.0f}{tout:8.0f}{cost:9.4f}")

    print("\n=== 分题型 Average F1 ===")
    print(f"{'题型':13s}{'MN':>9s}{'SM':>9s}{'差':>9s}")
    for cat in ("multi-hop", "temporal", "open-domain", "single-hop", "adversarial"):
        a = [r["f1"] for r in rows if r["system"] == "MemNexus" and r["category"] == cat]
        b = [r["f1"] for r in rows if r["system"] == "SimpleMem" and r["category"] == cat]
        if a and b:
            print(f"{cat:13s}{statistics.mean(a):9.4f}{statistics.mean(b):9.4f}"
                  f"{statistics.mean(a)-statistics.mean(b):+9.4f}")

    if up:
        print("\n=== 放宽后转正的题（strict WRONG -> paper CORRECT）===")
        for r, _, _ in sorted(up, key=lambda x: (x[0]["system"], x[0]["sample_id"])):
            print(f"  [{r['system']:9s}][{r['category']:12s}] {r['question'][:64]}")
            print(f"      gold: {str(r['gold_answer'])[:80]}")
            print(f"      pred: {str(r['prediction'])[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
