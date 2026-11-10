"""Render the MemNexus variant A/B dashboard as a self-contained HTML page.

Reads result/variant_comparison_20261023.json (produced by analyze_variants.py)
plus the raw rows to draw per-category breakdowns.  No external assets, so the
page can be opened straight from disk.

Usage: python experiments/build_variant_dashboard.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORY = {1: "MultiHop", 2: "Temporal", 3: "OpenDomain", 4: "SingleHop", 5: "Adversarial"}
PALETTE = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#BB3333", "#888888"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bar_chart(series: list[tuple[str, float, str]], ymax: float = 1.0,
              fmt: str = "{:.2f}", width: int = 300, height: int = 170) -> str:
    """Vertical bars. series = [(label, value, colour)]."""
    n = len(series)
    bw = (width - 20) / n * 0.62
    gap = (width - 20) / n
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = height - 18 - (height - 30) * frac
        out.append(f'<line x1="10" y1="{y:.1f}" x2="{width-4}" y2="{y:.1f}" class="grid"/>')
    for i, (label, value, colour) in enumerate(series):
        x = 10 + gap * i + (gap - bw) / 2
        h = max(0.0, min(1.0, value / ymax)) * (height - 30)
        y = height - 18 - h
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                   f'fill="{colour}" rx="2"/>')
        out.append(f'<text x="{x + bw/2:.1f}" y="{y - 4:.1f}" class="val">{fmt.format(value)}</text>')
        out.append(f'<text x="{x + bw/2:.1f}" y="{height - 5}" class="lab">{label}</text>')
    out.append("</svg>")
    return "".join(out)


def forest_plot(paired: dict, width: int = 420) -> str:
    """Forest plot of paired mean delta with 95% CI vs the MemNexus baseline."""
    rows = [(name, d["mean_delta"], d["ci"][0], d["ci"][1], d["n"])
            for name, d in paired.items()]
    height = 40 + 34 * len(rows)
    lo = min(r[2] for r in rows) - 0.02
    hi = max(r[3] for r in rows) + 0.02

    def xpos(v: float) -> float:
        return 90 + (v - lo) / (hi - lo) * (width - 130)

    out = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    out.append(f'<line x1="{xpos(0):.1f}" y1="14" x2="{xpos(0):.1f}" y2="{height-12}" class="zero"/>')
    for i, (name, mean, c0, c1, n) in enumerate(rows):
        y = 30 + 34 * i
        out.append(f'<text x="6" y="{y+4}" class="lab">{name} (n={n})</text>')
        colour = "#228833" if c0 > 0 else ("#BB3333" if c1 < 0 else "#CCBB44")
        out.append(f'<line x1="{xpos(c0):.1f}" y1="{y}" x2="{xpos(c1):.1f}" y2="{y}" '
                   f'stroke="{colour}" stroke-width="2.5"/>')
        for v in (c0, c1):
            out.append(f'<line x1="{xpos(v):.1f}" y1="{y-4}" x2="{xpos(v):.1f}" y2="{y+4}" '
                       f'stroke="{colour}" stroke-width="2.5"/>')
        out.append(f'<circle cx="{xpos(mean):.1f}" cy="{y}" r="5" fill="{colour}"/>')
        out.append(f'<text x="{width-2}" y="{y+4}" class="val" text-anchor="end">'
                   f'{mean:+.3f} [{c0:+.3f},{c1:+.3f}]</text>')
    out.append(f'<text x="{xpos(0):.1f}" y="{height-2}" class="lab" text-anchor="middle">0</text>')
    out.append("</svg>")
    return "".join(out)


def grouped_bars(per_config: dict[str, dict[str, tuple[float, int]]]) -> str:
    """Per-category F1, one cluster per configuration."""
    configs = list(per_config)
    cats = list(per_config[configs[0]])
    width, height = 640, 220
    cw = (width - 50) / len(cats)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = height - 26 - (height - 34) * frac
        out.append(f'<line x1="40" y1="{y:.1f}" x2="{width-4}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="34" y="{y+4:.1f}" class="lab" text-anchor="end">{frac:.2f}</text>')
    for ci, cat in enumerate(cats):
        bw = (cw - 14) / len(configs)
        for si, cfg in enumerate(configs):
            value, n = per_config[cfg][cat]
            x = 50 + cw * ci + 7 + bw * si
            h = max(0.0, min(1.0, value)) * (height - 34)
            y = height - 26 - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                       f'fill="{PALETTE[si % len(PALETTE)]}" rx="2"><title>{cfg} / {cat}: '
                       f'{value:.2f} (n={n})</title></rect>')
        out.append(f'<text x="{50 + cw*ci + cw/2:.1f}" y="{height-8}" class="lab" '
                   f'text-anchor="middle">{cat}</text>')
    out.append("</svg>")
    legend = " ".join(
        f'<span class="key"><i style="background:{PALETTE[i % len(PALETTE)]}"></i>{c}</span>'
        for i, c in enumerate(configs))
    return "".join(out) + f'<div class="legend">{legend}</div>'


def main() -> int:
    data = load(ROOT / "result" / "variant_comparison_20261023.json")
    base_rows = load(ROOT / "result" / "bench150_conv48-50_20261023" / "rows_baseline.json")
    variants = ["bridge", "gate", "execution"]
    rows_by_cfg = {"baseline": base_rows}
    for name in variants:
        rows_by_cfg[name] = load(ROOT / "result" / "bench150_variants" / name / "rows.json")

    summary, sm = data["summary"], data["simplemem"]
    order = ["baseline", "gate", "execution", "bridge"]

    # per-category F1 for the grouped chart
    per_config = {}
    for name in order:
        sub = [r for r in rows_by_cfg[name] if r["system"] == "MemNexus"
               and r.get("category_id") != 5]
        per_config[name] = {}
        for cat, cname in CATEGORY.items():
            if cat == 5:
                continue
            vals = [float(r["f1"]) for r in sub if r.get("category_id") == cat]
            per_config[name][cname] = (round(statistics.mean(vals), 3) if vals else 0.0, len(vals))

    f1_series = [(n, summary[n]["f1"], PALETTE[i]) for i, n in enumerate(order)]
    f1_series.append(("SimpleMem", sm["f1"], "#888888"))
    adv_series = [(n, summary[n]["adv_rate"] * 100, PALETTE[i]) for i, n in enumerate(order)]
    adv_series.append(("SimpleMem", sm["adv_rate"] * 100, "#888888"))
    fa_series = []
    for n in order:
        sub = [r for r in rows_by_cfg[n] if r["system"] == "MemNexus"
               and r.get("category_id") != 5]
        bad = sum(1 for r in sub if "not mentioned" in (r["prediction"] or "").lower()
                  or r.get("forced_abstain"))
        fa_series.append((n, bad / len(sub) * 100, PALETTE[order.index(n)]))
    call_series = [(n, summary[n]["calls_per_q"], PALETTE[i]) for i, n in enumerate(order)]
    call_series.append(("SimpleMem", sm["calls_per_q"], "#888888"))

    table_rows = "".join(
        f"<tr><td>{n}</td><td>{summary[n]['f1']:.3f}</td><td>{summary[n]['bleu']:.3f}</td>"
        f"<td>{summary[n]['adv_ok']}/{summary[n]['adv_n']} "
        f"({summary[n]['adv_rate']*100:.1f}%)</td>"
        f"<td>{summary[n]['calls_per_q']:.2f}</td>"
        f"<td>{summary[n]['tok_per_q']:.0f}</td>"
        f"<td>{', '.join(f'{k}:{v}' for k, v in summary[n]['paths'].items())}</td></tr>"
        for n in order)
    table_rows += (f"<tr class='ref'><td>SimpleMem</td><td>{sm['f1']:.3f}</td>"
                   f"<td>{sm['bleu']:.3f}</td><td>{sm['adv_ok']}/{sm['adv_n']} "
                   f"({sm['adv_rate']*100:.1f}%)</td><td>{sm['calls_per_q']:.2f}</td>"
                   f"<td>{sm['tok_per_q']:.0f}</td><td>reference system</td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>MemNexus Variant A/B Dashboard (LoCoMo 150)</title>
<style>
 body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #1a1a1a; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin-top: 28px; }}
 .chart {{ width: 100%; max-width: 680px; }}
 .grid {{ stroke: #ddd; stroke-width: 1; }} .zero {{ stroke: #999; stroke-dasharray: 4 3; }}
 .val {{ font-size: 11px; text-anchor: middle; }}
 .lab {{ font-size: 11px; fill: #333; }}
 .legend span.key {{ margin-right: 12px; font-size: 12px; }}
 .legend i {{ display: inline-block; width: 10px; height: 10px; margin-right: 4px; }}
 table {{ border-collapse: collapse; font-size: 12px; max-width: 900px; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 tr.ref td {{ color: #666; font-style: italic; }}
 .grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 24px; max-width: 960px; }}
 .note {{ font-size: 12px; color: #555; max-width: 900px; }}
 .badge {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; }}
 .bad {{ background: #fdecea; color: #a02218; }} .ok {{ background: #e8f5e9; color: #1b5e20; }}
</style></head><body>
<h1>MemNexus LLM 变体 A/B · LoCoMo 150 题（conv-48/49/50）</h1>
<p class="note">论文口径 F1/BLEU-1（对抗题不进 F1 均值）；对抗成功 = 拒答诱导选项；误拒 = 非对抗题上被拒答的比例。
SimpleMem 行为四个配置共用的参照。数据源：<code>result/variant_comparison_20261023.json</code>。</p>

<h2>主表</h2>
<table><tr><th>配置</th><th>F1</th><th>BLEU-1</th><th>对抗 SR</th>
<th>调用/题</th><th>token/题</th><th>answer_path</th></tr>{table_rows}</table>

<h2>F1（论文口径，对抗题除外）</h2>
{bar_chart(f1_series, ymax=0.8, fmt="{:.3f}")}
<div class="legend"><span class="key"><i style="background:#4477AA"></i>MemNexus 配置</span>
<span class="key"><i style="background:#888888"></i>SimpleMem</span></div>

<h2>分题型 F1</h2>
{grouped_bars(per_config)}

<h2>对抗成功率（%，越高越好）</h2>
{bar_chart(adv_series, ymax=100, fmt="{:.0f}")}

<h2>误拒率（%，越低越好；非对抗题被拒答）</h2>
{bar_chart(fa_series, ymax=100, fmt="{:.1f}")}

<h2>每配置配对 ΔF1 vs MemNexus 基线（95% CI）</h2>
{forest_plot(data["paired"])}
<p class="note">绿 = CI 不含 0 的正增益；红 = 不含 0 的负增益；黄 = CI 跨 0（不显著）。
三个变体的 CI 均不显著或显著为负——没有任何 LLM-增强变体超过当前主线配置。</p>

<h2>结论</h2>
<p class="note">
<span class="badge ok">baseline</span> F1 0.56 / 对抗 90.6% / 2.39 调用每题，仍为主线。
<span class="badge bad">bridge</span> 修复契约超限 bug 后重跑：ΔF1 −0.076（CI [−0.151, −0.001]，不含 0 的显著负增益），调用增至 3.71 次/题——检索词改写无净收益。
<span class="badge bad">gate</span> 全库扫描 130 次但 hard_abstain 0 次触发，零收益。
<span class="badge bad">execution</span> 语义执行路径命中 60 题但 F1 反降。
</p>
</body></html>"""

    out = ROOT / "result" / "variant_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
