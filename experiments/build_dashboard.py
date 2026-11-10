"""Build a self-contained HTML report from the paired run artifacts.

Inputs are the files the experiment already writes: ``rows.json`` (every
answer), ``paired_analysis.json`` (per-question verdicts and loss classes) and
the call ledgers for the token breakdown. Nothing is recomputed from a summary
that could drift, and no external asset is loaded, so the page works offline.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from paper_protocol import adversarial_success, bleu1_of, f1_of  # noqa: E402

CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ORDER = ["multi-hop", "temporal", "open-domain", "single-hop", "adversarial"]
COLOR = {"MemNexus": "#2f6fed", "SimpleMem": "#e8833a"}
CLASS_LABEL = {
    "abstained": "拒答（证据在库）",
    "no_evidence": "证据缺失",
    "partial": "条目漏答",
    "extra": "多余条目",
    "phrasing": "同义表述",
    "wrong": "答错",
}


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bar_chart(rows: list[tuple[str, float, float]], width: int = 660, height: int = 210) -> str:
    """Grouped bars, MemNexus next to SimpleMem, one group per category."""
    if not rows:
        return ""
    left, right, top, bottom = 46, 8, 12, 34
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(rows)
    bar_w = group_w * 0.32
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="axis" text-anchor="end">{tick:.2f}</text>')
    for index, (name, mn, sm) in enumerate(rows):
        base_x = left + group_w * index + group_w * 0.18
        for offset, value, system in ((0, mn, "MemNexus"), (bar_w + 4, sm, "SimpleMem")):
            h = plot_h * max(0.0, min(1.0, value))
            x = base_x + offset
            parts.append(
                f'<rect x="{x:.1f}" y="{top + plot_h - h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{COLOR[system]}" rx="2"><title>{system} {name}: {value:.3f}</title></rect>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h - h - 5:.1f}" class="value" '
                f'text-anchor="middle">{value:.2f}</text>'
            )
        parts.append(
            f'<text x="{base_x + bar_w + 2:.1f}" y="{height - 12}" class="axis" text-anchor="middle">{name}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def histogram(values: list[float], width: int = 660, height: int = 170) -> str:
    """Distribution of per-question F1 differences, binned in 0.1 steps."""
    bins = [0] * 21
    for value in values:
        index = int(round((value + 1.0) * 10))
        bins[max(0, min(20, index))] += 1
    top, bottom, left = 10, 26, 30
    plot_w, plot_h = width - left - 12, height - top - bottom
    peak = max(bins) or 1
    slot = plot_w / len(bins)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for index, count in enumerate(bins):
        h = plot_h * count / peak
        x = left + slot * index
        centre = -1.0 + index / 10
        fill = "#2f6fed" if centre > 0.02 else "#c9ccd4" if abs(centre) <= 0.02 else "#e8833a"
        parts.append(
            f'<rect x="{x + 1:.1f}" y="{top + plot_h - h:.1f}" width="{slot - 2:.1f}" height="{h:.1f}" '
            f'fill="{fill}" rx="2"><title>Δ≈{centre:+.1f}: {count} questions</title></rect>'
        )
        if count and count == peak:
            parts.append(
                f'<text x="{x + slot / 2:.1f}" y="{top + plot_h - h - 4:.1f}" class="value" '
                f'text-anchor="middle">{count}</text>'
            )
    for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
        x = left + (tick + 1.0) * plot_w / 2
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" class="axis" text-anchor="middle">{tick:+.1f}</text>')
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" class="axis" text-anchor="middle"></text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def stacked(classes: Counter, width: int = 660, height: int = 46) -> str:
    total = sum(classes.values()) or 1
    palette = {"abstained": "#c2410c", "no_evidence": "#b45309", "partial": "#a16207",
               "extra": "#4d7c0f", "phrasing": "#0e7490", "wrong": "#6b7280"}
    x = 0.0
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for name, count in classes.most_common():
        w = width * count / total
        parts.append(f'<rect x="{x:.1f}" y="0" width="{max(1.0, w - 1):.1f}" height="26" '
                     f'fill="{palette.get(name, "#9ca3af")}" rx="2">'
                     f'<title>{CLASS_LABEL.get(name, name)}: {count}</title></rect>')
        if w > 54:
            parts.append(f'<text x="{x + w / 2:.1f}" y="18" class="value-light" text-anchor="middle">'
                         f'{CLASS_LABEL.get(name, name)} {count}</text>')
        x += w
    parts.append("</svg>")
    return "".join(parts)


def table(headers: list[str], rows: list[list[str]], *, numeric_from: int = 1) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            cls = ' class="num"' if index >= numeric_from else ""
            cells.append(f"<td{cls}>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="result/bench260/rows.json")
    parser.add_argument("--paired", default="result/paired_260.json")
    parser.add_argument("--out", default="result/DASHBOARD_260.html")
    parser.add_argument("--title", default="MemNexus vs SimpleMem · LoCoMo")
    args = parser.parse_args()

    rows = read(ROOT / args.rows)
    pivot: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        pivot[(row["sample_id"], row["qa_index"])][row["system"]] = row
    paired = {key: value for key, value in pivot.items()
              if len(value) == 2
              and all((row.get("prediction") or "").strip() for row in value.values())}

    per_category: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    diffs, records = [], []
    for key, pair in sorted(paired.items()):
        mn, sm = pair["MemNexus"], pair["SimpleMem"]
        category = CATEGORY.get(int(mn.get("category_id") or 0), "?")
        mn_f1, sm_f1 = f1_of(mn), f1_of(sm)
        per_category[category]["mn"].append(mn_f1)
        per_category[category]["sm"].append(sm_f1)
        per_category[category]["mn_tok"].append((mn.get("model_calls") or {}).get("input_tokens", 0))
        per_category[category]["sm_tok"].append((sm.get("model_calls") or {}).get("input_tokens", 0))
        per_category[category]["mn_bleu"].append(bleu1_of(mn))
        per_category[category]["sm_bleu"].append(bleu1_of(sm))
        diffs.append(mn_f1 - sm_f1)
        records.append({"key": key, "category": category, "mn": mn, "sm": sm,
                        "mn_f1": mn_f1, "sm_f1": sm_f1, "diff": mn_f1 - sm_f1})

    chart_rows = [(name, statistics.mean(per_category[name]["mn"]), statistics.mean(per_category[name]["sm"]))
                  for name in ORDER if name in per_category]
    metric_rows = []
    for name in ORDER:
        bucket = per_category.get(name)
        if not bucket:
            continue
        n = len(bucket["mn"])
        adv_mn = sum(adversarial_success(r["mn"]) for r in records if r["category"] == name) if name == "adversarial" else None
        adv_sm = sum(adversarial_success(r["sm"]) for r in records if r["category"] == name) if name == "adversarial" else None
        extra = (f"{adv_mn}/{n} vs {adv_sm}/{n}" if adv_mn is not None
                 else f"{statistics.mean(bucket['mn_bleu']):.3f} vs {statistics.mean(bucket['sm_bleu']):.3f}")
        metric_rows.append([
            name, str(n),
            f"{statistics.mean(bucket['mn']):.3f}", f"{statistics.mean(bucket['sm']):.3f}",
            f"{statistics.mean(bucket['mn']) - statistics.mean(bucket['sm']):+.3f}",
            extra,
            f"{statistics.mean(bucket['mn_tok']):.0f}", f"{statistics.mean(bucket['sm_tok']):.0f}",
        ])

    wins = sum(1 for d in diffs if d > 0.02)
    ties = sum(1 for d in diffs if abs(d) <= 0.02)
    losses = sum(1 for d in diffs if d < -0.02)
    # The paper's Average is the unweighted mean of the four reasoning types,
    # not the mean over questions: categories have very different sizes here.
    def paper_average(system: str) -> float:
        means = [
            statistics.mean(per_category[name][system])
            for name in ORDER
            if name != "adversarial" and per_category.get(name) and per_category[name][system]
        ]
        return statistics.mean(means)

    overall = {
        "n": len(paired),
        "mn_avg": paper_average("mn"),
        "sm_avg": paper_average("sm"),
        "mn": statistics.mean(f1_of(p["MemNexus"]) for p in paired.values()),
        "sm": statistics.mean(f1_of(p["SimpleMem"]) for p in paired.values()),
        "mn_tok": statistics.mean((p["MemNexus"].get("model_calls") or {}).get("input_tokens", 0)
                                  for p in paired.values()),
        "sm_tok": statistics.mean((p["SimpleMem"].get("model_calls") or {}).get("input_tokens", 0)
                                  for p in paired.values()),
    }
    equal_weight = statistics.mean(
        statistics.mean(per_category[name]["mn"]) - statistics.mean(per_category[name]["sm"])
        for name in per_category if per_category[name]["mn"]
    )

    classes = Counter()
    paired_path = ROOT / args.paired
    loss_rows = []
    if paired_path.exists():
        payload = read(paired_path)
        for record in payload.get("records", []):
            if record.get("verdict") == "sm":
                classes[record.get("class", "?")] += 1
                loss_rows.append(record)
    loss_rows.sort(key=lambda r: (r["sm_f1"] - r["mn_f1"]), reverse=True)

    examples = []
    for record in loss_rows[:40]:
        examples.append([
            f"{record['sample_id']}/{record['qa_index']}<br><span class='muted'>{html.escape(CLASS_LABEL.get(record['class'], record['class']))}</span>",
            html.escape(str(record["question"])[:150]),
            html.escape(str(record["gold"])[:90]),
            html.escape(str(record["sm"])[:90]),
            html.escape(str(record["mn"])[:90]),
            f"{record['sm_f1']:.2f} → {record['mn_f1']:.2f}",
        ])

    document = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>{html.escape(args.title)}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; padding: 32px 40px 64px; font: 15px/1.6 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
        color: #1b1e24; background: #f6f7f9; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 34px 0 10px; border-left: 3px solid #2f6fed; padding-left: 10px; }}
p.lead {{ color: #5b6270; margin: 0 0 20px; }}
.cards {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 18px 0 6px; }}
.card {{ background: #fff; border: 1px solid #e3e6ec; border-radius: 10px; padding: 14px 18px; min-width: 150px; }}
.card .k {{ color: #6b7280; font-size: 12px; letter-spacing: .04em; text-transform: uppercase; }}
.card .v {{ font-size: 22px; font-weight: 600; margin-top: 2px; }}
.card .s {{ color: #6b7280; font-size: 12px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e3e6ec; border-radius: 10px; overflow: hidden; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #eef0f4; text-align: left; vertical-align: top; }}
th {{ background: #f0f2f6; font-weight: 600; font-size: 13px; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr:last-child td {{ border-bottom: none; }}
.chart {{ width: 100%; height: auto; background: #fff; border: 1px solid #e3e6ec; border-radius: 10px; }}
.grid {{ stroke: #eceef2; stroke-width: 1; }}
.axis {{ fill: #6b7280; font-size: 11px; }}
.value {{ fill: #1b1e24; font-size: 11px; font-weight: 600; }}
.value-light {{ fill: #fff; font-size: 11px; font-weight: 600; }}
.muted {{ color: #8b93a1; font-size: 12px; }}
.legend {{ display: flex; gap: 18px; font-size: 13px; color: #4b5563; margin: 6px 0 0; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; }}
.note {{ color: #6b7280; font-size: 13px; }}
</style></head><body>
<h1>{html.escape(args.title)}</h1>
<p class="lead">官方 LoCoMo 口径：F1/BLEU-1 按类别计算后等权平均；对抗题单列为成功率；token 为每题输入 token。
配对样本 {overall['n']} 题（两系统同题同库同模型）。</p>

<div class="cards">
  <div class="card"><div class="k">MemNexus Average F1</div><div class="v">{overall['mn_avg']:.3f}</div><div class="s">四类等权（论文表 1）</div></div>
  <div class="card"><div class="k">SimpleMem Average F1</div><div class="v">{overall['sm_avg']:.3f}</div><div class="s">四类等权（论文表 1）</div></div>
  <div class="card"><div class="k">ΔAverage</div><div class="v">{overall['mn_avg'] - overall['sm_avg']:+.3f}</div><div class="s">逐题均值 {overall['mn']:.3f} vs {overall['sm']:.3f}</div></div>
  <div class="card"><div class="k">胜/平/负</div><div class="v">{wins}/{ties}/{losses}</div><div class="s">逐题 F1</div></div>
  <div class="card"><div class="k">输入 token/题</div><div class="v">{overall['mn_tok']:.0f}</div><div class="s">SimpleMem {overall['sm_tok']:.0f}（{(overall['mn_tok'] / max(1, overall['sm_tok']) - 1) * 100:+.0f}%），仅输入</div></div>
</div>

<h2>分题型指标</h2>
<div class="legend"><span><span class="dot" style="background:{COLOR['MemNexus']}"></span>MemNexus</span>
<span><span class="dot" style="background:{COLOR['SimpleMem']}"></span>SimpleMem</span></div>
{bar_chart(chart_rows)}
{table(["类别", "题数", "MN F1", "SM F1", "ΔF1", "BLEU / 对抗", "MN tok", "SM tok"], metric_rows, numeric_from=1)}
<p class="note">对抗行给出“拒答正确数”，其余行给出 BLEU-1。Δ 为 MemNexus 减 SimpleMem。</p>

<h2>逐题差值分布</h2>
{histogram(diffs)}
<p class="note">横轴为单题 F1 差（MemNexus − SimpleMem），蓝为胜、橙为负、灰为持平（|Δ|≤0.02）。</p>

<h2>MemNexus 落后的题目归因</h2>
{stacked(classes) if classes else '<p class="note">本次运行未提供配对归因文件。</p>'}
<p class="note">归因顺序：先看是否拒答；再看胜方要点是否出现在 MemNexus 证据中（缺失即检索/抽取丢失）；再看是多值漏项、多余项，还是同义表述。</p>

<h2>逐题对照（按差距排序，前 40 题）</h2>
{table(["题目", "问题", "gold", "SimpleMem", "MemNexus", "F1"], examples, numeric_from=5)
       if examples else '<p class="note">无落后题目。</p>'}

<p class="note">生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')} ·
数据：{html.escape(args.rows)}</p>
</body></html>"""
    Path(ROOT / args.out).write_text(document, encoding="utf-8")
    print(f"wrote {args.out} ({len(document)} bytes) for {len(paired)} paired questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
