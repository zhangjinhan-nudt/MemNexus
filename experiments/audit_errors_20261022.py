"""Item-level audit of the latest paired run (result/bench260).

For every paired question it recomputes the official LoCoMo F1 (SimpleMem paper
protocol, ``experiments/paper_protocol.py``) and, for every item where MemNexus
is behind SimpleMem or where both systems fail, it walks the recorded pipeline
trace:

    question -> route -> plan -> retrieval -> evidence -> validation -> render

Goal: assign ONE root cause per item, at the pipeline stage where the loss is
created, not at the stage where it becomes visible.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paper_protocol as pp  # noqa: E402

ROOT = HERE.parent
DEFAULT_ROWS = ROOT / "result" / "bench260" / "rows.json"

ADVERSARIAL = 5
STOP = {
    "the", "is", "a", "an", "what", "was", "were", "of", "my", "me", "please",
    "and", "to", "in", "on", "at", "for", "did", "does", "do", "are", "he", "she",
    "they", "it", "his", "her", "their", "that", "this", "with", "how", "many",
    "when", "which", "who", "whose", "had", "has", "have", "been", "be",
}
ABSTAIN_MARKERS = ("not mentioned", "insufficient evidence", "no information available")


def content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).casefold()) if t not in STOP and len(t) > 1}


def evidence_text(row: dict) -> str:
    parts = []
    for src in row.get("retrieved_sources") or []:
        parts.append(json.dumps(src, ensure_ascii=False))
    return " ".join(parts)


def gold_coverage(row: dict) -> float:
    """Share of gold content tokens that appear anywhere in the retrieved evidence."""
    gold = content_tokens(row.get("gold_answer") or "")
    if not gold:
        return 1.0
    ev = content_tokens(evidence_text(row))
    return len(gold & ev) / len(gold)


def pred_hit_rate(row: dict) -> float:
    """Share of gold content tokens that appear in the prediction."""
    gold = content_tokens(row.get("gold_answer") or "")
    if not gold:
        return 1.0
    pred = content_tokens(row.get("prediction") or "")
    return len(gold & pred) / len(gold)


def pred_noise(row: dict) -> float:
    """Share of the prediction's content tokens that are NOT in the gold answer."""
    pred = content_tokens(row.get("prediction") or "")
    if not pred:
        return 0.0
    gold = content_tokens(row.get("gold_answer") or "")
    return len(pred - gold) / len(pred)


def classify(row: dict, gold_cov: float) -> str:
    """One root cause per item, assigned at the stage where the loss is created."""
    pred = str(row.get("prediction") or "").strip()
    cid = row.get("category_id")
    hit = pred_hit_rate(row)
    noise = pred_noise(row)
    low = pred.casefold()

    if cid == ADVERSARIAL:
        return "A-对抗：未识别诱饵并作答"
    if any(marker in low for marker in ABSTAIN_MARKERS):
        if gold_cov >= 0.99:
            return "B-拒答但证据在库（判定过度）"
        return "C-拒答且证据确实缺失"
    if hit >= 0.99 and noise >= 0.5:
        return "D-答案含多余条目（颗粒度失控）"
    if hit == 0.0:
        if gold_cov >= 0.99:
            return "E-证据在手却取错取值（绑定错误）"
        return "F-证据缺失（召回/表示层未覆盖）"
    if noise > hit:
        return "G-同义表述为主（词面不匹配）"
    return "H-部分覆盖（列举不完整）"


STAGE_OF = {
    "A-对抗：未识别诱饵并作答": "L3/L6 路由+否定门",
    "B-拒答但证据在库（判定过度）": "L6 判定层",
    "C-拒答且证据确实缺失": "L1/L4 表示+召回",
    "D-答案含多余条目（颗粒度失控）": "L7 产出层",
    "E-证据在手却取错取值（绑定错误）": "L7 绑定",
    "F-证据缺失（召回/表示层未覆盖）": "L1/L4 表示+召回",
    "G-同义表述为主（词面不匹配）": "L7 渲染/规范化",
    "H-部分覆盖（列举不完整）": "L5/L7 义务+枚举",
}


def pair_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    by_key: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("sample_id"), row.get("qa_index"))
        by_key.setdefault(key, {})[row.get("system")] = row
    pairs = []
    for key, by_sys in by_key.items():
        mn, sm = by_sys.get("MemNexus"), by_sys.get("SimpleMem")
        if mn is None or sm is None:
            continue
        if not str(mn.get("prediction") or "").strip():
            continue
        if not str(sm.get("prediction") or "").strip():
            continue
        pairs.append((mn, sm))
    pairs.sort(key=lambda p: (str(p[0].get("sample_id")), int(p[0].get("qa_index") or 0)))
    return pairs


def fmt(value: str | None, limit: int = 78) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def row_block(mn: dict, sm: dict, cause: str) -> list[str]:
    plan = mn.get("plan") or {}
    reasoning = mn.get("reasoning") or {}
    gate = reasoning.get("typed_negative_gate") or {}
    table = reasoning.get("answer_table") or {}
    slots = table.get("slots") or []
    calls = mn.get("model_calls") or {}
    keys = (
        "category",
        "route",
        "query_type",
        "recall_policy",
        "depth",
        "subject",
        "predicate",
        "temporal_mode",
        "response_intent",
    )
    lines = [
        f"- **{mn.get('sample_id')}/{mn.get('qa_index')}** · {mn.get('category')}"
        f" · MN F1 {mn['f1']:.2f} vs SM F1 {sm['f1']:.2f}（差 {mn['f1'] - sm['f1']:+.2f}）· 归因 **{cause}**",
        f"  - 问题：{fmt(mn.get('question'))}",
        f"  - 金标：{fmt(mn.get('gold_answer'))}",
        f"  - MN 输出：{fmt(mn.get('prediction'))}",
        f"  - SM 输出：{fmt(sm.get('prediction'))}",
        f"  - 计划：{', '.join(f'{k}={plan.get(k)}' for k in keys if plan.get(k) is not None)}",
        f"  - 形态识别：kind={reasoning.get('question_type')} · recall={reasoning.get('recall_policy')}"
        f" · 实体={reasoning.get('entities')} · 约束={reasoning.get('constraints')}",
        f"  - 图：nodes={reasoning.get('graph_nodes')} edges={reasoning.get('graph_edges')}"
        f" derivations={reasoning.get('deterministic_derivations')}"
        f" · 否定门 blocked={gate.get('blocked')} reason={gate.get('reason')}",
        f"  - 答案表：{len(slots)} 槽 · "
        + " | ".join(
            f"[{s.get('role')}] " + fmt(", ".join(v.get("value", "") for v in (s.get("values") or [])) or (s.get("missing") or "∅"), 60)
            for s in slots
        ),
        f"  - 证据：retrieved={mn.get('retrieval_count')} · citations={mn.get('citations')} · 路径={reasoning.get('answer_path')}",
        f"  - 开销：in={calls.get('input_tokens')} out={calls.get('output_tokens')} ¥{calls.get('cost_rmb')} · {mn.get('latency_ms')} ms",
    ]
    warns = [w for w in (mn.get("warnings") or []) if not w.startswith("question_route")]
    if warns:
        lines.append(f"  - 警告：{fmt(' · '.join(warns), 160)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=str(DEFAULT_ROWS))
    parser.add_argument("--out", default=str(ROOT / "result" / "ERROR_AUDIT_20261022.md"))
    args = parser.parse_args()

    rows = json.loads(pathlib.Path(args.rows).read_text(encoding="utf-8"))
    pairs = pair_rows(rows)

    out: list[str] = []
    add = out.append

    # ---- global numbers -----------------------------------------------------
    # NOTE: the paper's ``Average`` is the UNWEIGHTED MEAN OF THE FOUR CATEGORY
    # MEANS, not the pooled mean over items.  Pooling over-weights single-hop
    # (n=76) and inflates the number by ~6 points.  Never pool here.
    per_cat: dict[str, tuple[list, list]] = {}
    for mn, sm in pairs:
        mn["f1"] = pp.f1_of(mn)
        sm["f1"] = pp.f1_of(sm)
        if mn.get("category_id") == ADVERSARIAL:
            continue
        per_cat.setdefault(mn.get("category"), ([], []))
        per_cat[mn.get("category")][0].append(mn["f1"])
        per_cat[mn.get("category")][1].append(sm["f1"])
    cat_order = ("multi-hop", "temporal", "open-domain", "single-hop")
    mn_means = [statistics.mean(per_cat[c][0]) for c in cat_order if c in per_cat]
    sm_means = [statistics.mean(per_cat[c][1]) for c in cat_order if c in per_cat]

    add("# 最近一次配对评测 · 逐题审计与全流程追踪")
    add("")
    add(f"数据源：`{pathlib.Path(args.rows).as_posix()}`，配对成功 **{len(pairs)}** 对（7 段对话）。")
    add("官方 F1 由 `experiments/paper_protocol.py` 逐题重算，与报告口径一致。")
    add("")

    add("## 1. 总览")
    add("")
    add("| 题型 | 题数 | MN F1 | SM F1 | 差 |")
    add("|---|---:|---:|---:|---:|")
    for cat in cat_order:
        if cat not in per_cat:
            continue
        mn_v, sm_v = per_cat[cat]
        add(f"| {cat} | {len(mn_v)} | {100 * statistics.mean(mn_v):.2f} | "
            f"{100 * statistics.mean(sm_v):.2f} | {100 * (statistics.mean(mn_v) - statistics.mean(sm_v)):+.2f} |")
    add(f"| **Average（四类等权）** | {len(pairs) - sum(1 for m, _ in pairs if m.get('category_id') == ADVERSARIAL)} | "
        f"**{100 * statistics.mean(mn_means):.2f}** | {100 * statistics.mean(sm_means):.2f} | "
        f"**{100 * (statistics.mean(mn_means) - statistics.mean(sm_means)):+.2f}** |")
    adv = [(mn, sm) for mn, sm in pairs if mn.get("category_id") == ADVERSARIAL]
    adv_mn = sum(1 for mn, _ in adv if pp.adversarial_success(mn))
    adv_sm = sum(1 for _, sm in adv if pp.adversarial_success(sm))
    add("")
    add(f"对抗题（不计入 Average）：MN {adv_mn}/{len(adv)} vs SM {adv_sm}/{len(adv)}")
    add("")

    # ---- item classification ------------------------------------------------
    losses, both_bad, wins = [], [], []
    for mn, sm in pairs:
        cov = gold_coverage(mn)
        delta = mn["f1"] - sm["f1"]
        if delta < -1e-9:
            losses.append((mn, sm, cov))
        elif delta > 1e-9:
            wins.append((mn, sm, cov))
        elif mn["f1"] < 0.999:
            both_bad.append((mn, sm, cov))

    def tally(records):
        counts = Counter(classify(mn, cov) for mn, _, cov in records)
        return counts

    add("## 2. 归因总表（按流水线阶段）")
    add("")
    add("| 归因 | MN 落后 | 双方同错 | 合计 | 出错阶段 |")
    add("|---|---:|---:|---:|---|")
    lc, bc = tally(losses), tally(both_bad)
    for cause in sorted(set(lc) | set(bc), key=lambda c: (-(lc[c] + bc[c]), c)):
        add(f"| {cause} | {lc.get(cause, 0)} | {bc.get(cause, 0)} | {lc.get(cause, 0) + bc.get(cause, 0)} | {STAGE_OF.get(cause, '')} |")
    add("")
    add(f"MN 更好 **{len(wins)}** 题 · 持平 **{len(pairs) - len(wins) - len(losses)}** 题 · MN 落后 **{len(losses)}** 题")
    add("")

    # ---- per-item traces ----------------------------------------------------
    add("## 3. MN 落后的题目：全流程追踪")
    add("")
    add("每条给出：问题 → 计划 → 召回 → 证据引用 → 校验产物（答案表）→ 渲染 → 开销。")
    add("")
    grouped: dict[str, list] = {}
    for mn, sm, cov in losses:
        grouped.setdefault(classify(mn, cov), []).append((mn, sm, cov))
    for cause in sorted(grouped, key=lambda c: -len(grouped[c])):
        add(f"### {cause}（{len(grouped[cause])} 题）")
        add("")
        for mn, sm, cov in grouped[cause]:
            out.extend(row_block(mn, sm, cause))
        add("")

    add("## 4. 双方都错的题目：共同缺口")
    add("")
    grouped2: dict[str, list] = {}
    for mn, sm, cov in both_bad:
        grouped2.setdefault(classify(mn, cov), []).append((mn, sm, cov))
    for cause in sorted(grouped2, key=lambda c: -len(grouped2[c])):
        add(f"### {cause}（{len(grouped2[cause])} 题）")
        add("")
        for mn, sm, cov in grouped2[cause]:
            out.extend(row_block(mn, sm, cause))
        add("")

    path = pathlib.Path(args.out)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(out)} lines)")
    print(f"pairs={len(pairs)} losses={len(losses)} wins={len(wins)} both_bad={len(both_bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
