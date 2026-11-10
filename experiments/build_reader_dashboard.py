"""Reader A/B dashboard: baseline vs the two LLM reader variants.

Reads result/reader_ab_20261023.json (produced by analyze_reader_ab.py) plus
the 150-question fallback run when present.  Self-contained HTML, no assets.

Usage: python experiments/build_reader_dashboard.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ["#4477AA", "#228833", "#CCBB44", "#BB3333", "#66CCEE"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def forest(rows: list[tuple[str, float, float, float]], width: int = 460) -> str:
    """rows = [(label, mean, ci_lo, ci_hi)]"""
    height = 40 + 40 * len(rows)
    lo = min(r[2] for r in rows) - 0.02
    hi = max(r[3] for r in rows) + 0.02

    def x(v: float) -> float:
        return 110 + (v - lo) / (hi - lo) * (width - 150)

    out = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    out.append(f'<line x1="{x(0):.1f}" y1="12" x2="{x(0):.1f}" y2="{height-14}" class="zero"/>')
    for i, (label, mean, c0, c1) in enumerate(rows):
        y = 32 + 40 * i
        colour = "#228833" if c0 > 0 else ("#BB3333" if c1 < 0 else "#CCBB44")
        out.append(f'<text x="6" y="{y+4}" class="lab">{label}</text>')
        out.append(f'<line x1="{x(c0):.1f}" y1="{y}" x2="{x(c1):.1f}" y2="{y}" '
                   f'stroke="{colour}" stroke-width="3"/>')
        for v in (c0, c1):
            out.append(f'<line x1="{x(v):.1f}" y1="{y-5}" x2="{x(v):.1f}" y2="{y+5}" '
                       f'stroke="{colour}" stroke-width="3"/>')
        out.append(f'<circle cx="{x(mean):.1f}" cy="{y}" r="5.5" fill="{colour}"/>')
        out.append(f'<text x="{width-2}" y="{y+4}" class="val" text-anchor="end">'
                   f'{mean:+.3f} [{c0:+.3f}, {c1:+.3f}]</text>')
    out.append("</svg>")
    return "".join(out)


def grouped(cats: list[str], series: list[tuple[str, list[float]]],
            width: int = 640, height: int = 220) -> str:
    cw = (width - 60) / len(cats)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    lo = min(min(v) for _, v in series)
    hi = max(max(v) for _, v in series)
    span = max(hi - lo, 0.05)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        val = lo + span * frac
        y = height - 30 - (height - 38) * frac
        out.append(f'<line x1="50" y1="{y:.1f}" x2="{width-4}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="44" y="{y+4:.1f}" class="lab" text-anchor="end">{val:+.2f}</text>')
    out.append(f'<line x1="{50 + (0 - lo) / span * (width - 60):.1f}" y1="10" '
               f'x2="{50 + (0 - lo) / span * (width - 60):.1f}" y2="{height-30}" class="zero"/>')
    bw = (cw - 12) / len(series)
    for ci, cat in enumerate(cats):
        for si, (name, vals) in enumerate(series):
            v = vals[ci]
            x = 60 + cw * ci + 6 + bw * si
            h = abs(v) / span * (height - 38)
            y = height - 30 - (h if v > 0 else 0)
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                       f'fill="{PALETTE[si]}" rx="2"><title>{name}/{cat}: {v:+.3f}</title></rect>')
        out.append(f'<text x="{60 + cw*ci + cw/2:.1f}" y="{height-8}" class="lab" '
                   f'text-anchor="middle">{cat}</text>')
    out.append("</svg>")
    legend = " ".join(f'<span class="key"><i style="background:{PALETTE[i]}"></i>{n}</span>'
                      for i, (n, _) in enumerate(series))
    return "".join(out) + f'<div class="legend">{legend}</div>'


def main() -> int:
    ab = load(ROOT / "result" / "reader_ab_20261023.json")
    variants = ab["variants"]
    names = ["fallback", "hint"]
    table = "".join(
        f"<tr><td>{n}</td><td>{variants[n]['mean_delta']:+.3f}</td>"
        f"<td>{variants[n]['ci'][0]:+.3f} ~ {variants[n]['ci'][1]:+.3f}</td>"
        f"<td>{variants[n]['better']}/{variants[n]['same']}/{variants[n]['worse']}</td>"
        f"<td>{variants[n]['fired']}/{variants[n]['n']}</td>"
        f"<td>{variants[n]['verified']}</td><td>{variants[n]['calls_per_q']:.2f}</td></tr>"
        for n in names if n in variants)

    cats = ["multi-hop", "temporal", "open-domain", "single-hop"]
    series = [(n, [variants[n]["by_category"].get(c, 0.0) for c in cats])
              for n in names if n in variants]
    forest_rows = [(f"{n} (n={variants[n]['n']})", variants[n]["mean_delta"],
                    variants[n]["ci"][0], variants[n]["ci"][1])
                   for n in names if n in variants]

    full_note = ""
    full = ROOT / "result" / "bench150_reader_fallback" / "rows.json"
    if full.exists():
        base = {(r["sample_id"], r["qa_index"]): float(r["f1"]) for r in
                load(ROOT / "result" / "bench150_conv48-50_20261023" / "rows_baseline.json")
                if r["system"] == "MemNexus"}
        rows = {(r["sample_id"], r["qa_index"]): r for r in load(full) if r["system"] == "MemNexus"}
        keys = sorted(set(base) & set(rows))
        diffs = [float(rows[k]["f1"]) - base[k] for k in keys]
        n = len(diffs)
        mean = statistics.mean(diffs)
        se = statistics.stdev(diffs) / n ** 0.5 if n > 1 else 0.0
        calls = statistics.mean((rows[k].get("model_calls") or {}).get("provider_attempts", 0)
                                for k in keys)
        full_note = (f"<p>全量 150 题复验（fallback）：ΔF1 {mean:+.3f}，95% CI "
                     f"[{mean-1.96*se:+.3f}, {mean+1.96*se:+.3f}]，n={n}，"
                     f"调用 {calls:.2f}/题（基线 2.39）。</p>")
        forest_rows.append((f"fallback full-150 (n={n})", mean, mean - 1.96 * se, mean + 1.96 * se))

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>MemNexus Reader A/B (42)</title>
<style>
 body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #1a1a1a; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin-top: 28px; }}
 .chart {{ width: 100%; max-width: 680px; }}
 .grid {{ stroke: #ddd; }} .zero {{ stroke: #999; stroke-dasharray: 4 3; }}
 .val, .lab {{ font-size: 11px; }} .val {{ text-anchor: middle; }}
 .legend span.key {{ margin-right: 12px; font-size: 12px; }}
 .legend i {{ display: inline-block; width: 10px; height: 10px; margin-right: 4px; }}
 table {{ border-collapse: collapse; font-size: 12px; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 .note {{ font-size: 12px; color: #555; max-width: 760px; }}
</style></head><body>
<h1>LLM Reader 变体 A/B · 42 题瓶颈子集（conv-48/49/50，非对抗）</h1>
<p class="note">基线 F1 在这 42 题上为 {ab['baseline_f1']:.3f}。
fallback = 契约链路与有界推理之后仍弃答时，一次自由阅读兜底（引文逐字验证）；
hint = 合成前一次自由阅读，已验证引文作为提示注入契约（append-only）。</p>

<h2>汇总</h2>
<table><tr><th>变体</th><th>ΔF1</th><th>95% CI</th><th>胜/平/负</th>
<th>reader 触发</th><th>引文验证</th><th>调用/题</th></tr>{table}</table>

<h2>配对 ΔF1 vs 基线（95% CI）</h2>
{forest(forest_rows)}
<p class="note">绿 = CI 不含 0 的正增益；红 = 不含 0 的负增益；黄 = 跨 0（不显著）。</p>

<h2>分题型 ΔF1</h2>
{grouped(cats, series)}
{full_note}
<h2>结论</h2>
<p class="note">
fallback 在 42 题子集为正号（+0.045），但全量 150 题复验：原始 ΔF1 −0.024（CI 含 0）；剔除两侧供应商失败题后（去噪 n=124）ΔF1 +0.011（CI [−0.033, +0.054]，仍含 0）。触发仅 8 题，其中 3 胜 3 负（净 −0.30）。判定：42 题正信号为小样本噪声，按准入判据 3 否决，旗标保持默认关闭。
hint 中性（ΔF1 −0.008）却多花 1.32 调用/题——契约绑定不消费自由阅读提示，按成本否决。
按准入判据 3（CI 必须不含 0），两者均否决：fallback 全量复验不显著（去噪后 +0.011，CI 含 0），hint 中性且贵 55% 调用。至此四轮 LLM 干预探索全部定稿为否决——「2.39 调用/题的契约主线」为最终配置。
</p>
</body></html>"""

    out = ROOT / "result" / "reader_ab_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
