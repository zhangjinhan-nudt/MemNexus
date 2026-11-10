"""Full overnight analysis: LoCoMo 1000 + LongMemEval 1000.

Produces result/NIGHT1000_ANALYSIS_20261024.md with:
  - LoCoMo paper-protocol table (F1/BLEU-1 per reasoning type, adversarial SR)
  - LoCoMo paired win/loss and per-category breakdown
  - LongMemEval accuracy by setting (S / oracle) and question type
  - cost & token efficiency, call counts
  - per-question error mining: top failure patterns for MemNexus
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

LOCOMO_OUT = ROOT / "result" / "bench1000_locomo_20261023"
LME_OUT = ROOT / "result" / "bench1000_longmemeval_20261023"
REPORT = ROOT / "result" / "NIGHT1000_ANALYSIS_20261024.md"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def score(row: dict) -> float:
    return 1.0 if row.get("judge_result") == "CORRECT" else 0.0


def fmt(x: float) -> str:
    return f"{x:.4f}"


def locomo_section(lines: list[str]) -> dict | None:
    judged = LOCOMO_OUT / "rows_judged.json"
    if not judged.exists():
        lines.append("## LoCoMo 1000\n\n未完成（无 rows_judged.json）\n")
        return None
    rows = read(judged)
    try:
        from paper_protocol import paper_table
        paper = paper_table(rows)
    except Exception as error:
        paper = None
        lines.append(f"(paper_table failed: {error})\n")
    lines.append("## LoCoMo 1000（按 paper 协议盲评）\n")
    for system in ("MemNexus", "SimpleMem"):
        sel = [r for r in rows if r["system"] == system]
        if not sel:
            continue
        acc = sum(score(r) for r in sel) / len(sel)
        f1 = sum(r["f1"] for r in sel) / len(sel)
        in_tok = sum(r["model_calls"].get("input_tokens", 0) for r in sel)
        out_tok = sum(r["model_calls"].get("output_tokens", 0) for r in sel)
        cost = sum(r["model_calls"].get("cost_rmb", 0) for r in sel)
        lines.append(f"- **{system}**: n={len(sel)}  accuracy={fmt(acc)}  F1={fmt(f1)}  "
                     f"in={in_tok}  out={out_tok}  cost=RMB {cost:.2f}")
    if paper:
        lines.append("\n### Paper 协议分类 F1\n")
        lines.append("| system | multi-hop | temporal | open-domain | single-hop | adversarial | average | adv-SR |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for system, table in paper["systems"].items():
            f1t = table.get("f1_by_type", {})
            lines.append(
                f"| {system} | {fmt(f1t.get('multi-hop', 0))} | {fmt(f1t.get('temporal', 0))} | "
                f"{fmt(f1t.get('open-domain', 0))} | {fmt(f1t.get('single-hop', 0))} | "
                f"{fmt(f1t.get('adversarial', 0))} | {fmt(table.get('average_f1', 0))} | "
                f"{fmt(table.get('adversarial_sr', 0))} |")
    # paired analysis
    key = lambda r: (r["sample_id"], r["qa_index"])
    mn = {key(r): r for r in rows if r["system"] == "MemNexus"}
    sm = {key(r): r for r in rows if r["system"] == "SimpleMem"}
    common = sorted(set(mn) & set(sm))
    wins = sum(1 for k in common if score(mn[k]) > score(sm[k]))
    losses = sum(1 for k in common if score(mn[k]) < score(sm[k]))
    delta = sum(score(mn[k]) - score(sm[k]) for k in common)
    lines.append(f"\n### 配对比较（n={len(common)}）\n")
    lines.append(f"- MemNexus 胜 {wins} / 负 {losses} / 平 {len(common)-wins-losses}")
    lines.append(f"- Δaccuracy（MN−SM）= {fmt(delta/len(common)) if common else 'n/a'}")
    by_cat = defaultdict(lambda: [0, 0, 0.0])
    for k in common:
        c = mn[k]["category"]
        by_cat[c][0] += score(mn[k])
        by_cat[c][1] += 1
        by_cat[c][2] += score(mn[k]) - score(sm[k])
    lines.append("\n| category | n | MN acc | Δacc |")
    lines.append("|---|---|---|---|")
    for c, (w, n, d) in sorted(by_cat.items()):
        lines.append(f"| {c} | {n} | {fmt(w/n)} | {fmt(d/n)} |")
    return rows


def lme_section(lines: list[str]) -> None:
    judged = LME_OUT / "rows_judged.json"
    if not judged.exists():
        lines.append("\n## LongMemEval 1000\n\n未完成（无 rows_judged.json）\n")
        return
    rows = read(judged)
    valid = [r for r in rows if r.get("judge_result") in ("CORRECT", "WRONG")]
    errors = [r for r in rows if r.get("judge_result") == "error"]
    lines.append(f"\n## LongMemEval（有效 {len(valid)} / 判定错误 {len(errors)}）\n")
    for setting in ("O", "S"):
        sel = [r for r in valid if r.get("setting") == setting]
        if not sel:
            continue
        lines.append(f"\n### setting={setting}（n={len(sel)}）\n")
        lines.append("| system | accuracy | mean F1 | in tokens | out tokens | cost RMB |")
        lines.append("|---|---|---|---|---|---|")
        for system in ("MemNexus", "SimpleMem"):
            s = [r for r in sel if r["system"] == system]
            if not s:
                continue
            acc = sum(score(r) for r in s) / len(s)
            f1 = sum(r["f1"] for r in s) / len(s)
            it = sum(r["model_calls"].get("input_tokens", 0) for r in s)
            ot = sum(r["model_calls"].get("output_tokens", 0) for r in s)
            cost = sum(r["model_calls"].get("cost_rmb", 0) for r in s)
            lines.append(f"| {system} | {fmt(acc)} | {fmt(f1)} | {it} | {ot} | {cost:.2f} |")
        by_type = defaultdict(lambda: {"MemNexus": [0, 0], "SimpleMem": [0, 0]})
        for r in sel:
            t = r["category"]
            by_type[t][r["system"]][0] += score(r)
            by_type[t][r["system"]][1] += 1
        lines.append("\n| question type | MN acc (n) | SM acc (n) | Δ |")
        lines.append("|---|---|---|---|")
        for t, d in sorted(by_type.items()):
            mn, sm = d["MemNexus"], d["SimpleMem"]
            if mn[1] and sm[1]:
                lines.append(f"| {t} | {fmt(mn[0]/mn[1])} ({mn[1]}) | "
                             f"{fmt(sm[0]/sm[1])} ({sm[1]}) | {fmt(mn[0]/mn[1]-sm[0]/sm[1])} |")


def error_mining(lines: list[str], locomo_rows: list[dict] | None) -> None:
    lines.append("\n## 错误归因（MemNexus judge=WRONG）\n")
    for name, rows in (("LoCoMo", locomo_rows), ("LongMemEval", None)):
        if rows is None:
            if not (LME_OUT / "rows_judged.json").exists():
                continue
            rows = read(LME_OUT / "rows_judged.json")
        wrong = [r for r in rows if r["system"] == "MemNexus" and r.get("judge_result") == "WRONG"]
        right = [r for r in rows if r["system"] == "SimpleMem" and r.get("judge_result") == "CORRECT"]
        mn_keys = {(r["sample_id"], r["qa_index"]) for r in wrong}
        sm_only = [r for r in rows if r["system"] == "SimpleMem"
                   and r.get("judge_result") == "CORRECT"
                   and (r["sample_id"], r["qa_index"]) in mn_keys]
        lines.append(f"\n### {name}: MN 错 {len(wrong)} 题；其中 SM 对 {len(sm_only)} 题（MN 独有失败）\n")
        forced = sum(1 for r in wrong if r.get("forced_abstain"))
        empty = sum(1 for r in wrong if not (r.get("prediction") or "").strip())
        abstain = sum(1 for r in wrong if "not mentioned" in (r.get("prediction", "").casefold()))
        lines.append(f"- forced_abstain/超时: {forced}")
        lines.append(f"- 空预测: {empty}")
        lines.append(f"- 回答 Not mentioned（可能在对抗题是正确弃答）: {abstain}")
        # warnings frequency
        warn = Counter()
        for r in wrong:
            for w in r.get("warnings", []) or []:
                warn[w.split(":")[0]] += 1
        if warn:
            lines.append(f"- warnings 分布: {dict(warn.most_common(8))}")
        # sample dump for manual reading
        sample_path = ROOT / "result" / f"NIGHT1000_{name.lower()}_mn_wrong_dump.md"
        with sample_path.open("w", encoding="utf-8") as handle:
            for r in wrong[:400]:
                handle.write(f"## {r['sample_id']} qa{r['qa_index']} [{r['category']}]\n")
                handle.write(f"Q: {r['question']}\nGold: {r['gold_answer']}\n")
                handle.write(f"MN: {r['prediction']}\n")
                handle.write(f"reason: {r.get('judge_reason', '')}\n\n")
        lines.append(f"- 逐题 dump: {sample_path}")


def main() -> int:
    lines = ["# Night1000 全量分析（LoCoMo 1000 + LongMemEval 1000）\n",
             f"生成时间: 2026-10-24\n"]
    # progress snapshot first
    for name, out in (("LoCoMo", LOCOMO_OUT), ("LongMemEval", LME_OUT)):
        attempts = out / "attempts"
        if attempts.exists():
            units = sorted(p.name for p in attempts.glob("*/stage_result.json"))
            failures = [p.parent.name for p in attempts.glob("*/failure.json")
                        if not (p.parent / "stage_result.json").exists()]
            lines.append(f"- {name}: 完成 {len(units)} 单元；未恢复失败 {len(failures)}")
    lines.append("")
    locomo_rows = locomo_section(lines)
    lme_section(lines)
    error_mining(lines, locomo_rows)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
