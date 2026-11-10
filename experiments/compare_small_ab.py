"""Per-question comparison of the small paired A/B.

Reads the two phase folders written by ``small_ab.py``, pairs questions by
(sample_id, qa_index), and reports official F1 movement plus whether a
mechanism actually fired. No judge call, no paper-table claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from paper_protocol import adversarial_success, f1_of  # noqa: E402

ABSTAIN = ("not mentioned", "insufficient evidence")

# Official LoCoMo ids. The stored ``category`` label is unreliable in early
# phase files because the pilot module carried a stale map.
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def category_of(row):
    return CATEGORY.get(row.get("category_id"), row.get("category", "?"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(phase_dir):
    rows = {}
    for path in sorted(Path(phase_dir).glob("conv-*.json")):
        if "_usage" in path.name:
            continue
        payload = read(path)
        if "rows" not in payload:
            continue
        questions = payload.get("questions") or []
        for index, row in enumerate(payload["rows"]):
            row["_stage"] = payload["stage"]
            # Early runs predate the explicit qa_index; recover it from the
            # saved question order instead of guessing from the text.
            if "qa_index" not in row and index < len(questions):
                row["qa_index"] = questions[index]["qa_index"]
                row["sample_id"] = questions[index]["sample_id"]
            rows[(row["sample_id"], row["qa_index"])] = row
    return rows


def is_abstain(text):
    return text.strip().lower().startswith(ABSTAIN)


def tokens(row):
    usage = row.get("model_calls") or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "result" / "small_ab_20261020"))
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    base = load(Path(args.root) / "baseline")
    opt = load(Path(args.root) / "optimized")
    keys = sorted(set(base) & set(opt), key=lambda k: (k[0], k[1]))
    if not keys:
        raise SystemExit("no paired questions: run both phases first")

    lines = [
        "# 小规模配对 A/B：表述规范化 + 主体定向召回",
        "",
        f"配对题数：{len(keys)}（baseline ∩ optimized，逐题配对）",
        "",
    ]

    by_cat = defaultdict(list)
    fired = 0
    abstain_fixed = 0
    regressions = []
    for key in keys:
        b, o = base[key], opt[key]
        fb, fo = f1_of(b), f1_of(o)
        grounding = o.get("grounding") or {}
        added = int(grounding.get("subject_recall_added") or 0)
        fired += added > 0
        by_cat[category_of(b)].append((fb, fo))
        b_abs, o_abs = is_abstain(b["prediction"]), is_abstain(o["prediction"])
        abstain_fixed += b_abs and not o_abs
        if fo + 1e-9 < fb:
            regressions.append((key, fb, fo))

    for category, pairs in sorted(by_cat.items()):
        mb = sum(p[0] for p in pairs) / len(pairs)
        mo = sum(p[1] for p in pairs) / len(pairs)
        lines.append(f"| {category} | {len(pairs)} | {mb:.2f} | {mo:.2f} | {mo - mb:+.2f} |")
    lines.insert(4, "| 题型 | 题数 | baseline F1 | optimized F1 | 变化 |")
    lines.insert(5, "|---|---:|---:|---:|---:|")

    overall_b = sum(p[0] for pairs in by_cat.values() for p in pairs) / len(keys)
    overall_o = sum(p[1] for pairs in by_cat.values() for p in pairs) / len(keys)
    lines += [
        "",
        f"逐题均值（含对抗）：baseline {overall_b:.4f} → optimized {overall_o:.4f} "
        f"（{overall_o - overall_b:+.4f}）",
        f"机制触发题数（subject_recall_added>0）：{fired}/{len(keys)}",
        f"拒答转有答：{abstain_fixed} 题",
        "",
        "## 逐题明细",
        "",
        "| 题 | 题型 | F1 前 | F1 后 | Δ | 补召回 | pred 前 | pred 后 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for key in keys:
        b, o = base[key], opt[key]
        fb, fo = f1_of(b), f1_of(o)
        added = int((o.get("grounding") or {}).get("subject_recall_added") or 0)
        mark = "" if abs(fo - fb) < 1e-9 else ("**↑**" if fo > fb else "↓")
        lines.append(
            f"| {key[0]}/{key[1]} | {category_of(b)} | {fb:.2f} | {fo:.2f} | "
            f"{fo - fb:+.2f} {mark} | {added} | {b['prediction'][:48]} | {o['prediction'][:48]} |"
        )

    adv_base = [adversarial_success(base[k]) for k in keys if category_of(base[k]) == "adversarial"]
    adv_opt = [adversarial_success(opt[k]) for k in keys if category_of(opt[k]) == "adversarial"]
    if adv_base:
        lines += [
            "",
            f"对抗成功率：baseline {sum(adv_base)}/{len(adv_base)} → "
            f"optimized {sum(adv_opt)}/{len(adv_opt)}",
        ]

    tok_b = [tokens(base[k]) for k in keys]
    tok_o = [tokens(opt[k]) for k in keys]
    lines += [
        "",
        f"输入 token/题：{sum(t[0] for t in tok_b) / len(tok_b):.0f} → "
        f"{sum(t[0] for t in tok_o) / len(tok_o):.0f}",
        f"输出 token/题：{sum(t[1] for t in tok_b) / len(tok_b):.0f} → "
        f"{sum(t[1] for t in tok_o) / len(tok_o):.0f}",
    ]
    if regressions:
        lines += ["", "## 下降题（须逐题复核）", ""]
        for key, fb, fo in regressions:
            lines.append(f"- {key[0]}/{key[1]}：{fb:.2f} → {fo:.2f}")

    report = "\n".join(lines)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
