"""Single source of truth for the SimpleMem paper's LoCoMo evaluation protocol.

Why this module exists
----------------------
Before 2026-10-20 this project aggregated LoCoMo with mean-over-questions F1
*including* adversarial items, and treated an LLM judge as the primary score.
The SimpleMem paper does neither.  §3.1 states:

    For LoCoMo, we report F1 and BLEU-1 (accuracy), Adversarial Success Rate
    (robustness to distractors), and Token Cost (retrieval efficiency).
    For LongMemEval-S, we use its standard accuracy-style metric.

and Tables 1/3 have exactly four reasoning-type columns (MultiHop, Temporal,
OpenDomain, SingleHop) plus an Average column whose value is the unweighted
mean of the four category scores.  Adversarial items are *not* part of that
Average; they form a separate success rate.  The LLM judge appears only for
LongMemEval-S (Appendix A.4), never for LoCoMo.

Everything downstream (report, dashboards, A/B decisions) must read the numbers
from here so the three "same data, three conclusions" episode cannot recur.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Appendix A.4, verbatim.  Used only for LongMemEval-S (and as a diagnostic on
# LoCoMo).  Do not "tighten" this text: the paper deliberately asks for a
# generous grader so that long answers touching the same topic still count.
# ---------------------------------------------------------------------------
JUDGE_INSTRUCTION = (
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.\n"
    "You will be given the following data:\n"
    "(1) a question (posed by one user to another user),\n"
    "(2) a 'gold' (ground truth) answer,\n"
    "(3) a generated answer\n"
    "which you will score as CORRECT/WRONG.\n"
    "\n"
    "The point of the question is to ask about something one user should know about the other\n"
    "user based on their prior conversations.\n"
    "The gold answer will usually be a concise and short answer that includes the referenced\n"
    "topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading -\n"
    "as long as it touches on the same topic as the gold answer, it should be counted as\n"
    "CORRECT.\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The\n"
    "generated answer might be much longer or use relative time references (like \"last\n"
    "Tuesday\" or \"next month\"), but you should be generous with your grading - as long as\n"
    "it refers to the same date or time period as the gold answer, it should be counted as\n"
    "CORRECT. Even if the format differs (e.g., \"May 7th\" vs \"7 May\"), consider it CORRECT\n"
    "if it's the same date.\n"
    "Now it's time for the real question:\n"
    "Question: {question}\n"
    "Gold answer: {gold_answer}\n"
    "Generated answer: {generated_answer}\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with\n"
    "CORRECT or WRONG.\n"
    "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation\n"
    "script.\n"
    "Just return the label CORRECT or WRONG in a json format with the key as \"label\"."
)

# Table 1 column order.  Adversarial is deliberately absent.
REASONING_TYPES = OrderedDict(
    [(1, "MultiHop"), (2, "Temporal"), (3, "OpenDomain"), (4, "SingleHop")]
)
ADVERSARIAL_CATEGORY = 5

# Abstain phrasings accepted as a successful rejection on adversarial items.
# SimpleMem's own answer contract is "I do not have enough information in my
# memory.", so the phrases below all count as declining the distractor.
_ABSTAIN_MARKERS = (
    "not mentioned",
    "no information available",
    "do not have enough information",
    "don't have enough information",
)


def _official():
    """LoCoMo's official F1 normalisation, sourced from one implementation."""
    from analyze_optimized_comparison import official_f1, official_normalize

    return official_f1, official_normalize


def official_bleu1(prediction: str, gold: str) -> float:
    """Sentence BLEU-1 over LoCoMo-normalised tokens (clipped precision + brevity)."""
    _, normalize = _official()
    pred = normalize(prediction).split()
    ref = normalize(gold).split()
    if not pred or not ref:
        return 0.0
    overlap = 0
    pool = list(ref)
    for token in pred:
        if token in pool:
            pool.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    brevity = 1.0 if len(pred) >= len(ref) else _exp(1 - len(ref) / len(pred))
    return brevity * precision


def _exp(value: float) -> float:
    import math

    return math.exp(value)


def _scores(row: dict) -> tuple[float, float]:
    """(F1, BLEU-1) for one row under official LoCoMo answer-pair rules.

    Multi-hop gold is a comma separated list: the official scorer takes, for
    each gold item, the best matching predicted item, then averages over gold
    items.  Open-domain gold keeps only the segment before the first semicolon.
    Adversarial items are scored as a 0/1 rejection, matching locomo_score.
    """
    if row.get("category_id") == ADVERSARIAL_CATEGORY:
        hit = float(adversarial_success(row))
        return hit, hit
    official_f1, _ = _official()
    prediction = str(row.get("prediction") or "")
    gold = str(row.get("gold_answer") or "")
    if row.get("category_id") == 1:
        preds = [p.strip() for p in prediction.split(",") if p.strip()] or [prediction]
        golds = [g.strip() for g in gold.split(",") if g.strip()] or [gold]
        return (
            statistics.mean(max(official_f1(p, g) for p in preds) for g in golds),
            statistics.mean(max(official_bleu1(p, g) for p in preds) for g in golds),
        )
    if row.get("category_id") == 3:
        gold = gold.split(";")[0].strip()
    return official_f1(prediction, gold), official_bleu1(prediction, gold)


def f1_of(row: dict) -> float:
    """Official-style LoCoMo F1 for one row."""
    return _scores(row)[0]


def bleu1_of(row: dict) -> float:
    """Sentence BLEU-1 under the same answer-pair rules as F1."""
    return _scores(row)[1]


def adversarial_success(row: dict) -> bool:
    """True when the system declined the distractor instead of picking it."""
    prediction = str(row.get("prediction") or "").lower()
    return any(marker in prediction for marker in _ABSTAIN_MARKERS)


def _tokens(row: dict) -> tuple[int, int]:
    calls = row.get("model_calls") or {}
    return (
        int(calls.get("input_tokens") or 0),
        int(calls.get("output_tokens") or 0),
    )


def paper_table(rows: list[dict], systems: tuple[str, ...] = ("SimpleMem", "MemNexus")) -> dict:
    """Table-1 shaped metrics per system, plus the separate adversarial rate.

    Returns
    -------
    {
      "protocol": "...",
      "systems": {
        "<system>": {
          "n_by_type": {...},
          "f1_by_type": {...}, "bleu1_by_type": {...},
          "average_f1": float, "average_bleu1": float,   # equal weight over 4 types
          "adversarial_sr": float, "adversarial_n": int,
          "token_cost_per_question": float,              # QA tokens, all items
          "build_tokens": int, "cost_rmb": float,
          "judge_accuracy": float | None,                # auxiliary only
        }
      }
    }
    """
    out = {
        "protocol": (
            "SimpleMem paper §3.1 + Tables 1/3: LoCoMo reports F1, BLEU-1, "
            "Adversarial Success Rate and Token Cost. Average is the unweighted "
            "mean of the four reasoning types; adversarial is reported separately."
        ),
        "systems": OrderedDict(),
    }
    for system in systems:
        subset = [r for r in rows if r.get("system") == system]
        if not subset:
            continue
        f1_by_type: dict[str, float] = OrderedDict()
        bleu_by_type: dict[str, float] = OrderedDict()
        n_by_type: dict[str, int] = OrderedDict()
        for cid, name in REASONING_TYPES.items():
            group = [r for r in subset if r.get("category_id") == cid]
            if not group:
                continue
            f1_by_type[name] = statistics.mean(f1_of(r) for r in group)
            bleu_by_type[name] = statistics.mean(bleu1_of(r) for r in group)
            n_by_type[name] = len(group)
        adversarial = [r for r in subset if r.get("category_id") == ADVERSARIAL_CATEGORY]
        qa_in = sum(_tokens(r)[0] for r in subset)
        qa_out = sum(_tokens(r)[1] for r in subset)
        judged = [r for r in subset if r.get("judge_result") in ("CORRECT", "WRONG")]
        out["systems"][system] = {
            "n": len(subset),
            "n_by_type": n_by_type,
            "f1_by_type": f1_by_type,
            "bleu1_by_type": bleu_by_type,
            "average_f1": statistics.mean(f1_by_type.values()) if f1_by_type else None,
            "average_bleu1": statistics.mean(bleu_by_type.values()) if bleu_by_type else None,
            "adversarial_sr": (
                statistics.mean(float(adversarial_success(r)) for r in adversarial)
                if adversarial
                else None
            ),
            "adversarial_n": len(adversarial),
            "token_cost_per_question": (qa_in + qa_out) / len(subset),
            "qa_input_tokens": qa_in,
            "qa_output_tokens": qa_out,
            "cost_rmb": sum(float((r.get("model_calls") or {}).get("cost_rmb") or 0) for r in subset),
            "judge_accuracy": (
                sum(1 for r in judged if r["judge_result"] == "CORRECT") / len(judged)
                if judged
                else None
            ),
            "judge_n": len(judged),
        }
    return out


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}"


def render_markdown(table: dict, systems: tuple[str, ...] = ("SimpleMem", "MemNexus")) -> list[str]:
    """Table-1 shaped markdown block, used by the run report."""
    present = [s for s in systems if s in table["systems"]]
    if not present:
        return ["（无可用记录）"]
    names = [n for n in REASONING_TYPES.values() if n in table["systems"][present[0]]["f1_by_type"]]
    lines = [
        "## 主指标（SimpleMem 论文 §3.1 / Table 1 口径）",
        "",
        table["protocol"],
        "",
        "| | " + " | ".join(f"{n} F1 | {n} BLEU-1" for n in names)
        + " | **Average F1** | **Average BLEU-1** | 对抗成功率 | 每题tokens |",
        "|---|" + "---:|" * (2 * len(names) + 4),
    ]
    for system in present:
        s = table["systems"][system]
        cells: list[str] = []
        for name in names:
            cells += [_pct(s["f1_by_type"][name]), _pct(s["bleu1_by_type"][name])]
        cells += [
            f"**{_pct(s['average_f1'])}**",
            f"**{_pct(s['average_bleu1'])}**",
            "N/A" if s["adversarial_sr"] is None else f"{100 * s['adversarial_sr']:.1f}%",
            f"{s['token_cost_per_question']:.0f}",
        ]
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Average 为四类推理题的**等权算术平均**，不含对抗题；对抗题按论文 §3.1 单独以成功率报告。",
        "",
        "F1/BLEU-1 使用 LoCoMo 官方规范化（去冠词与标点、Porter 词干）；多跳按逗号拆分后对每个 gold 项取最佳预测项再平均，"
        "开放域只取第一个分号前的片段，与 `analyze_optimized_comparison.locomo_score` 逐行一致（本仓库已交叉校验）。",
        "",
        "每题 tokens 为问答阶段的输入＋输出，建库消耗单独列示。该列与论文的 Token Cost 口径不同（论文为各系统自身提示脚手架下的检索开销），"
        "横向只在本仓库两次运行之间可比。",
        "",
        "**LoCoMo 不使用 LLM 裁判**：论文中裁判只用于 LongMemEval-S。裁判结果仅作错因诊断，不作为 LoCoMo 成绩。",
        "",
    ]
    return lines


def format_table(table: dict) -> str:
    """Render the paper table the way Tables 1/3 lay it out (percent units)."""
    systems = list(table["systems"])
    if not systems:
        return "(no rows)"
    names = [n for n in REASONING_TYPES.values() if n in table["systems"][systems[0]]["f1_by_type"]]
    header = f"{'':14s}" + "".join(f"{n:>12s}{'':>6s}" for n in names) + f"{'Average':>10s}{'':>4s}{'AdvSR':>8s}{'Tok/Q':>9s}"
    lines = ["=" * len(header), header, "-" * len(header)]
    for system in systems:
        s = table["systems"][system]
        cells = f"{system:14s}"
        for name in names:
            cells += f"{100 * s['f1_by_type'][name]:11.2f} {100 * s['bleu1_by_type'][name]:6.2f}"
        cells += f"{100 * (s['average_f1'] or 0):10.2f} {100 * (s['average_bleu1'] or 0):4.2f}"
        sr = f"{100 * s['adversarial_sr']:7.1f}" if s["adversarial_sr"] is not None else "     --"
        cells += f"{sr:>8s}{s['token_cost_per_question']:9.0f}"
        lines.append(cells)
    lines.append("=" * len(header))
    lines.append("每列格式：F1 / BLEU-1（百分数）。Average = 四类等权平均，不含对抗。")
    lines.append("AdvSR = 对抗题成功率（拒答诱饵）；Tok/Q = 每题问答阶段 tokens，不含建库。")
    return "\n".join(lines)


def common_subset(rows: list[dict], systems: tuple[str, ...] = ("SimpleMem", "MemNexus")) -> list[dict]:
    """Rows for questions that *every* requested system answered.

    A crashed or resumed run leaves the systems with different question counts.
    Comparing their averages over different item sets is not a paired comparison,
    so paper metrics are computed on the intersection by default.
    """
    keys_by_system: dict[str, set] = {}
    for row in rows:
        system = row.get("system")
        if system not in systems:
            continue
        keys_by_system.setdefault(system, set()).add((row.get("sample_id"), row.get("qa_index")))
    if len(keys_by_system) < 2:
        return rows
    common = set.intersection(*keys_by_system.values())
    return [
        row
        for row in rows
        if row.get("system") in systems
        and (row.get("sample_id"), row.get("qa_index")) in common
    ]


def load_run_rows(folder: Path) -> list[dict]:
    """Read every completed unit's rows from a benchmark run directory."""
    folder = Path(folder)
    rows: list[dict] = []
    attempts = folder / "attempts"
    if not attempts.exists():
        raise SystemExit(f"{folder} 下没有 attempts 目录")
    for path in sorted(attempts.iterdir()):
        if not path.is_dir() or path.name.startswith("judge_"):
            continue
        stage = path / "stage_result.json"
        if not stage.exists():
            continue
        rows.extend(json.loads(stage.read_text(encoding="utf-8"))["rows"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="benchmark run directory")
    parser.add_argument("--json", action="store_true", help="also print raw JSON")
    parser.add_argument("--markdown", type=Path, default=None, help="write a markdown report here")
    args = parser.parse_args()
    rows = load_run_rows(args.run)
    subset = common_subset(rows)
    table = paper_table(subset)
    print(format_table(table))
    print()
    print(f"配对子集：各系统 {len(subset)//2} 题；已完成行 {len(rows)} 条。")
    if args.json:
        print(json.dumps(table, ensure_ascii=False, indent=1))
    target = args.run / "paper_metrics.json"
    target.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {target}")
    if args.markdown:
        unpaired = paper_table(rows)
        parts = ["# LoCoMo 结果（SimpleMem 论文口径）", ""]
        parts += render_markdown(table)
        parts += [
            f"配对子集为两系统都完成的题（各 {len(subset) // 2} 题）；共有 {len(rows)} 条已完成记录。",
            "",
        ]
        if unpaired != table:
            parts += ["## 非配对口径（含仅单侧完成的题，仅作覆盖参考）", ""]
            parts += render_markdown(unpaired)
        args.markdown.write_text("\n".join(parts), encoding="utf-8")
        print(f"写入 {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
