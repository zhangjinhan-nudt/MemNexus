"""Report generator for the overnight A/B.

Reads ``<root>/<cell>_r<n>/conv-*.json`` and writes:

  REPORT.md            tables + per-question flips
  figs/*.png           four figures
  report.html          self-contained dashboard (links the figures)

No judge call: F1 / BLEU-1 / adversarial success come from paper_protocol, and
the point of the run is per-question movement plus repeatability, not a paper
table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from paper_protocol import adversarial_success, bleu1_of, f1_of  # noqa: E402

CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ABSTAIN = ("not mentioned", "insufficient evidence")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(folder: Path):
    rows = {}
    for path in sorted(folder.glob("conv-*.json")):
        if "_usage" in path.name:
            continue
        payload = read(path)
        if "rows" not in payload:
            continue
        questions = payload.get("questions") or []
        for index, row in enumerate(payload["rows"]):
            if "qa_index" not in row and index < len(questions):
                row["qa_index"] = questions[index]["qa_index"]
                row["sample_id"] = questions[index]["sample_id"]
            rows[(row["sample_id"], row["qa_index"])] = row
    return rows


def runs(root: Path):
    found = defaultdict(list)
    for folder in sorted(root.iterdir() if root.exists() else []):
        if not folder.is_dir() or "_r" not in folder.name:
            continue
        cell, _, rep = folder.name.rpartition("_r")
        rows = load_run(folder)
        if rows:
            found[cell].append((int(rep), rows))
    for cell in found:
        found[cell].sort()
    return dict(found)


def category_of(row):
    return CATEGORY.get(row.get("category_id"), row.get("category", "?"))


def is_abstain(text):
    return str(text).strip().lower().startswith(ABSTAIN)


def tokens(row):
    usage = row.get("model_calls") or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def mcnemar_exact(a_wins: int, b_wins: int) -> float:
    """Two-sided exact McNemar p on discordant counts."""
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def summarise(per_rep):
    """Aggregate one cell: mean per-question F1, per-category, cost, tokens."""
    by_cat = defaultdict(list)
    f1s, bleus, ins, outs, abstains, adv = [], [], [], [], 0, []
    keys = sorted(per_rep[0][1])
    for key in keys:
        values = [rows[key] for _, rows in per_rep if key in rows]
        f1 = mean(f1_of(r) for r in values)
        f1s.append(f1)
        bleus.append(mean(bleu1_of(r) for r in values))
        tok = [tokens(r) for r in values]
        ins.append(mean(t[0] for t in tok))
        outs.append(mean(t[1] for t in tok))
        if all(is_abstain(r.get("prediction", "")) for r in values):
            abstains += 1
        by_cat[category_of(values[0])].append(f1)
        if category_of(values[0]) == "adversarial":
            adv.append(mean(1.0 if adversarial_success(r) else 0.0 for r in values))
    return {
        "keys": keys,
        "key_f1": dict(zip(keys, f1s, strict=True)),
        "key_cat": {k: category_of(per_rep[0][1][k]) for k in keys},
        "f1": f1s,
        "f1_mean": mean(f1s),
        "bleu": mean(bleus),
        "in_tok": mean(ins),
        "out_tok": mean(outs),
        "abstains": abstains,
        "adv_success": mean(adv),
        "adv_n": len(adv),
        "by_cat": {k: mean(v) for k, v in by_cat.items()},
        "cat_n": {k: len(v) for k, v in by_cat.items()},
        "reps": len(per_rep),
    }


def agreement(per_rep):
    """Per-question stability across repeats: same prediction, same sources."""
    if len(per_rep) < 2:
        return None
    keys = sorted(per_rep[0][1])
    same_pred = same_src = 0
    for key in keys:
        values = [rows[key] for _, rows in per_rep if key in rows]
        preds = {str(r.get("prediction", "")).strip().lower() for r in values}
        if len(preds) == 1:
            same_pred += 1
        sources = []
        for r in values:
            hits = r.get("retrieved_sources") or []
            ids = []
            for i, hit in enumerate(hits):
                if isinstance(hit, dict):
                    ids.append(str(hit.get("chunk_id") or hit.get("source_file") or i))
                else:
                    ids.append(str(hit))
            sources.append(tuple(sorted(ids)))
        if len(set(sources)) == 1:
            same_src += 1
    return {"n": len(keys), "same_pred": same_pred, "same_src": same_src}


def flipping(base_cell, other_cell, base, other):
    paired = [k for k in base["keys"] if k in set(other["keys"])]
    index_b = {k: i for i, k in enumerate(base["keys"])}
    index_o = {k: i for i, k in enumerate(other["keys"])}
    up = down = tie = 0
    a_only = b_only = 0
    for key in paired:
        fb, fo = base["f1"][index_b[key]], other["f1"][index_o[key]]
        up += fo > fb + 1e-9
        down += fo < fb - 1e-9
        tie += abs(fo - fb) <= 1e-9
        cb, co = fb >= 0.5, fo >= 0.5
        a_only += cb and not co
        b_only += co and not cb
    return {"n": len(paired), "up": up, "down": down, "tie": tie,
            "base_only_correct": a_only, "other_only_correct": b_only,
            "p": mcnemar_exact(a_only, b_only)}


def figures(root: Path, cells, summaries):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # noqa: BLE001
        print(f"[report] matplotlib unavailable: {error}")
        return []

    folder = root / "figs"
    folder.mkdir(parents=True, exist_ok=True)
    made = []

    labeled = [c for c in cells if summaries.get(c)]
    categories = ["multi-hop", "temporal", "open-domain", "single-hop", "adversarial"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / max(1, len(labeled))
    for i, cell in enumerate(labeled):
        values = [summaries[cell]["by_cat"].get(c, 0.0) for c in categories]
        ax.bar([x + i * width for x in range(len(categories))], values, width, label=cell)
    ax.set_xticks([x + width * (len(labeled) - 1) / 2 for x in range(len(categories))])
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("F1 (mean over repeats)")
    ax.set_title("Category F1 by cell")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = folder / "category_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cell in labeled:
        data = summaries[cell]["f1"]
        if data:
            ax.hist(data, bins=20, alpha=0.5, label=cell)
    ax.set_xlabel("per-question F1 (mean over repeats)")
    ax.set_ylabel("questions")
    ax.set_title("F1 distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = folder / "f1_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    if len(labeled) >= 2:
        base = labeled[0]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for cell in labeled[1:]:
            flip = flipping(base, cell, summaries[base], summaries[cell])
            ax.bar([f"{base}\nvs\n{cell}"], [flip["up"]], color="#c0392b", label="improved")
            ax.bar([f"{base}\nvs\n{cell}"], [flip["down"]], bottom=[flip["up"]],
                   color="#27ae60", label="regressed")
            ax.bar([f"{base}\nvs\n{cell}"], [flip["tie"]], bottom=[flip["up"] + flip["down"]],
                   color="#bdc3c7", label="unchanged")
        ax.set_ylabel("questions")
        ax.set_title("Per-question movement vs reference")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = folder / "flips.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cell in labeled:
        ax.scatter(summaries[cell]["in_tok"], summaries[cell]["f1_mean"], s=60, label=cell)
    ax.set_xlabel("input tokens per question")
    ax.set_ylabel("F1 (mean)")
    ax.set_title("Cost vs quality")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = folder / "cost_quality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)
    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "result" / "overnight_20261021"))
    args = parser.parse_args()
    root = Path(args.root)

    cells = runs(root)
    if not cells:
        raise SystemExit(f"no runs found under {root}")

    order = [c for c in ("default", "no_grounding", "no_normalize") if c in cells]
    order += [c for c in cells if c not in order]
    summaries = {cell: summarise(cells[cell]) for cell in order}

    lines = ["# 夜间 150 题 A/B 报告", ""]
    lines += [
        f"运行目录：`{root}`",
        f"格：{len(order)} 个，每格重复次数："
        + "、".join(f"{c}×{summaries[c]['reps']}" for c in order),
        f"生成时间：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 1. 主表（逐题均值，含对抗题）",
        "",
        "| 格 | 题数 | F1 | BLEU-1 | 对抗成功率 | 全拒答题数 | 输入 tok/题 | 输出 tok/题 | 重复次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in order:
        s = summaries[cell]
        lines.append(
            f"| {cell} | {len(s['keys'])} | {s['f1_mean']:.4f} | {s['bleu']:.4f} | "
            f"{s['adv_success']*100:.1f}% ({s['adv_n']}) | {s['abstains']} | "
            f"{s['in_tok']:.0f} | {s['out_tok']:.0f} | {s['reps']} |"
        )

    lines += ["", "## 2. 分题型 F1", "", "| 题型 | 题数 | " + " | ".join(order) + " |",
              "|---|---:|" + "---:|" * len(order)]
    common = sorted(set.intersection(*[set(summaries[c]["keys"]) for c in order])) if order else []
    categories = ["multi-hop", "temporal", "open-domain", "single-hop", "adversarial"]
    for category in categories:
        keys = [k for k in common if summaries[order[0]]["key_cat"][k] == category]
        if not keys:
            continue
        row = " | ".join(f"{mean(summaries[c]['key_f1'][k] for k in keys):.3f}" for c in order)
        lines.append(f"| {category} | {len(keys)} | {row} |")

    if "default" in summaries:
        lines += ["", "## 3. 与 default 的配对比较", "",
                  "判定口径：单题 F1 ≥ 0.5 记为答对；p 为精确 McNemar（双侧）。", "",
                  "| 对照 | 配对题数 | 涨 | 跌 | 平 | 仅 default 对 | 仅该格对 | McNemar p |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for cell in order:
            if cell == "default":
                continue
            flip = flipping("default", cell, summaries["default"], summaries[cell])
            lines.append(
                f"| {cell} | {flip['n']} | {flip['up']} | {flip['down']} | {flip['tie']} | "
                f"{flip['base_only_correct']} | {flip['other_only_correct']} | {flip['p']:.4f} |"
            )

    lines += ["", "## 4. 重复一致性（噪声地板）", "",
              "同一格多次运行，题目拿到相同预测 / 相同证据集的比例。这两个数决定了多小的差值可以被判读。", "",
              "| 格 | 重复次数 | 题数 | 预测完全一致 | 证据集完全一致 |",
              "|---|---:|---:|---:|---:|"]
    for cell in order:
        stats = agreement(cells[cell])
        if not stats:
            lines.append(f"| {cell} | {summaries[cell]['reps']} | {len(summaries[cell]['keys'])} | — | — |")
            continue
        lines.append(
            f"| {cell} | {summaries[cell]['reps']} | {stats['n']} | "
            f"{stats['same_pred']}/{stats['n']} ({stats['same_pred']/stats['n']*100:.1f}%) | "
            f"{stats['same_src']}/{stats['n']} ({stats['same_src']/stats['n']*100:.1f}%) |"
        )

    lines += ["", "## 5. 逐题明细（default vs 第一个对照格）", ""]
    if len(order) >= 2:
        other = order[1]
        base, alt = summaries["default"], summaries[other]
        index_b = {k: i for i, k in enumerate(base["keys"])}
        index_o = {k: i for i, k in enumerate(alt["keys"])}
        moved = [k for k in base["keys"] if k in index_o
                 and abs(base["f1"][index_b[k]] - alt["f1"][index_o[k]]) > 1e-9]
        lines += ["| 题 | 题型 | default F1 | " + other + " F1 | Δ |", "|---|---|---:|---:|---:|"]
        for key in moved:
            fb, fo = base["f1"][index_b[key]], alt["f1"][index_o[key]]
            lines.append(f"| {key[0]}/{key[1]} | {category_of(cells['default'][0][1][key])} | "
                         f"{fb:.2f} | {fo:.2f} | {fo-fb:+.2f} |")
        lines += ["", f"共 {len(moved)} 题发生变化（配对子集 {len(common)} 题）。"]
    else:
        lines.append("只有一个格，跳过逐题对照。")

    report = "\n".join(lines)
    (root / "REPORT.md").write_text(report + "\n", encoding="utf-8")

    made = figures(root, order, summaries)
    html = [
        "<!doctype html><meta charset='utf-8'><title>Overnight A/B</title>",
        "<style>body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:980px;"
        "margin:32px auto;padding:0 20px;color:#222;line-height:1.65}"
        "table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}"
        "th,td{border:1px solid #ddd;padding:6px 9px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "th{background:#f6f6f4}h1{font-size:22px}h2{font-size:16px;margin-top:28px}"
        "img{max-width:100%;margin:10px 0;border:1px solid #eee}</style>",
        f"<h1>夜间 150 题 A/B</h1><p>{root}</p>",
        "<pre style='background:#fafafa;padding:12px;overflow:auto;font-size:12px'>"
        + report.replace("<", "&lt;") + "</pre>",
    ]
    for path in made:
        html.append(f"<h2>{path.stem}</h2><img src='figs/{path.name}'>")
    (root / "report.html").write_text("\n".join(html), encoding="utf-8")

    print(report)
    print(f"\n[report] wrote {root/'REPORT.md'}, {root/'report.html'}, {len(made)} figures")


if __name__ == "__main__":
    main()
