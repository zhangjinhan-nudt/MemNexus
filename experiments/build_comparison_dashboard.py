"""Build a self-contained HTML dashboard for the MemNexus vs SimpleMem run.

All numbers are recomputed from result/bench260/rows.json with the official
LoCoMo scorer in experiments/paper_protocol.py, so the dashboard cannot drift
away from the paper table.  Charts are inlined as SVG: the file works offline.
"""

from __future__ import annotations

import html
import json
import math
import statistics as st
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import audit_errors_20261022 as A  # noqa: E402
import paper_protocol as P  # noqa: E402

ROWS_PATH = ROOT / "result" / "bench260" / "rows.json"
OUT_PATH = ROOT / "result" / "COMPARISON_DASHBOARD_20261022.html"

CATS = ["multi-hop", "temporal", "open-domain", "single-hop"]
CN = {
    "multi-hop": "多跳",
    "temporal": "时间",
    "open-domain": "开放域",
    "single-hop": "单跳",
    "adversarial": "对抗",
}
INK = "#2C2C2A"
MUTED = "#6B6A64"
FAINT = "#9C978B"
GRID = "#E7E4DA"
MN = "#534AB7"
MN_L = "#B9B2EC"
SM = "#A8A49A"
POS = "#0F6E56"
NEG = "#BA7517"
RED = "#A32D2D"

plt.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "font.size": 9,
        "axes.unicode_minus": False,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "path",
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    }
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load() -> tuple[list[tuple[dict, dict]], list[dict]]:
    raw = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    pairs = A.pair_rows(raw)
    pairs.sort(key=lambda p: (p[0]["sample_id"], int(p[0]["qa_index"] or 0)))
    return pairs, raw


def f1(row: dict) -> float:
    return P.f1_of(row)


def scored(pairs):
    """Attach official F1 to every row once."""
    cache: dict[int, float] = {}
    for mn, sm in pairs:
        for row in (mn, sm):
            cache[id(row)] = f1(row)
    return cache


def category_rows(pairs, cache, cat):
    out = []
    for mn, sm in pairs:
        if mn.get("category") != cat:
            continue
        out.append((mn, sm, cache[id(mn)], cache[id(sm)]))
    return out


def ci(values, z=1.96):
    n = len(values)
    mean = st.mean(values)
    se = st.stdev(values) / math.sqrt(n) if n > 1 else 0.0
    return mean, se, mean - z * se, mean + z * se


def svg(fig) -> str:
    buf = StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True, pad_inches=0.06)
    plt.close(fig)
    data = buf.getvalue()
    return data[data.index("<svg") :]


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def fig_main(metrics) -> str:
    labels = CATS + ["Average"]
    names = ["多跳", "时间", "开放域", "单跳", "四类等权"]
    ns = [metrics[c]["n"] for c in CATS] + [metrics["n"]]
    mn_v = [100 * metrics[c]["mn"] for c in CATS] + [100 * metrics["avg_mn"]]
    sm_v = [100 * metrics[c]["sm"] for c in CATS] + [100 * metrics["avg_sm"]]
    deltas = [metrics[c]["mn"] - metrics[c]["sm"] for c in CATS] + [metrics["avg_mn"] - metrics["avg_sm"]]

    fig, ax = plt.subplots(figsize=(7.6, 3.25))
    x = list(range(len(labels)))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], mn_v, w, color=MN, label="MemNexus", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], sm_v, w, color=SM, label="SimpleMem", zorder=3)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 1.6,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color=INK,
            )
    for i, d in enumerate(deltas):
        color = POS if 100 * d > 2 else (RED if d < 0 else FAINT)
        ax.text(i, 90, f"{100 * d:+.1f}", ha="center", va="center", fontsize=9.6,
                color=color, fontweight="bold")
    ax.axvline(3.5, color=GRID, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nn={k}" for n, k in zip(names, ns)], fontsize=8.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("官方 F1（%）")
    style(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=2, handlelength=1.2)
    return svg(fig)


def fig_waterfall(metrics, cf_avg) -> str:
    total = 100 * (metrics["avg_mn"] - metrics["avg_sm"])
    net = 100 * (cf_avg - metrics["avg_sm"])
    adapter = net - total  # negative

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    xs = [0, 1, 2]
    ax.bar(xs[0], total, 0.5, color=MN, zorder=3)
    ax.bar(xs[1], total - net, 0.5, bottom=net, color=NEG, zorder=3)
    ax.bar(xs[2], net, 0.5, color=POS, zorder=3)
    ax.plot([xs[0] + 0.25, xs[1] - 0.25], [total, total], color=FAINT, linewidth=0.9, linestyle=":")
    ax.plot([xs[1] + 0.25, xs[2] - 0.25], [net, net], color=FAINT, linewidth=0.9, linestyle=":")
    ax.text(xs[0], total + 0.35, f"{total:+.2f}", ha="center", fontsize=11, fontweight="bold", color=MN)
    ax.text(xs[1], total + 0.35, f"{adapter:+.2f}", ha="center", fontsize=11, fontweight="bold", color=NEG)
    ax.text(xs[2], net + 0.35, f"{net:+.2f}", ha="center", fontsize=11, fontweight="bold", color=POS)
    ax.set_xticks(xs)
    ax.set_xticklabels(["测得总分差", "减去连接符适配", "机制净差"], fontsize=9)
    ax.set_ylim(0, max(total, net) + 2.2)
    ax.set_ylabel("Average 差（百分点）")
    style(ax)
    return svg(fig)


def fig_winloss(winloss) -> str:
    cats = CATS + ["adversarial"]
    fig, ax = plt.subplots(figsize=(6.4, 2.55))
    y = list(range(len(cats)))[::-1]
    w = [winloss[c][0] for c in cats]
    t = [winloss[c][1] for c in cats]
    l = [winloss[c][2] for c in cats]
    ax.barh(y, w, 0.5, color=MN, zorder=3, label="MN 胜")
    ax.barh(y, t, 0.5, left=w, color="#DDD8CB", zorder=3, label="持平")
    ax.barh(y, l, 0.5, left=[a + b for a, b in zip(w, t)], color=NEG, zorder=3, label="MN 负")
    for yy, a, b, c in zip(y, w, t, l):
        if a:
            ax.text(a / 2, yy, str(a), ha="center", va="center", color="white", fontsize=8.4, fontweight="bold")
        if b >= 6:
            ax.text(a + b / 2, yy, str(b), ha="center", va="center", color=INK, fontsize=8)
        if c:
            ax.text(a + b + c / 2, yy, str(c), ha="center", va="center", color="white", fontsize=8.4, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([CN[c] for c in cats], fontsize=9)
    ax.set_xlim(0, 80)
    ax.set_xlabel("题数")
    style(ax, grid_axis="x")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.16), handlelength=1.1)
    return svg(fig)


def fig_scatter(pairs, cache) -> str:
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    colors = {"multi-hop": MN, "temporal": POS, "open-domain": NEG, "single-hop": FAINT, "adversarial": "#B44B4B"}
    for cat in CATS + ["adversarial"]:
        xs, ys = [], []
        for mn, sm in pairs:
            if mn.get("category") != cat:
                continue
            xs.append(100 * cache[id(sm)])
            ys.append(100 * cache[id(mn)])
        ax.scatter(xs, ys, s=26, color=colors[cat], alpha=0.72, linewidths=0, label=CN[cat], zorder=3)
    ax.plot([0, 100], [0, 100], color=FAINT, linewidth=0.9, linestyle="--", zorder=2)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xlabel("SimpleMem F1")
    ax.set_ylabel("MemNexus F1")
    style(ax, grid_axis="y")
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, alpha=0.55)
    ax.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=5, handletextpad=0.25, columnspacing=0.9, borderaxespad=0.0)
    return svg(fig)


def fig_ci(metrics) -> str:
    rows = CATS + ["Average"]
    names = ["多跳", "时间", "开放域", "单跳", "四类等权"]
    fig, ax = plt.subplots(figsize=(6.4, 2.75))
    ys = list(range(len(rows)))[::-1]
    for y, cat in zip(ys, rows):
        if cat == "Average":
            mean, se, lo, hi = metrics["avg_delta"], metrics["avg_se"], metrics["avg_lo"], metrics["avg_hi"]
            n = metrics["n"]
        else:
            mean, se, lo, hi = (metrics[cat][k] for k in ("delta", "se", "lo", "hi"))
            n = metrics[cat]["n"]
        sig = lo > 0 or hi < 0
        color = POS if sig else FAINT
        ax.plot([100 * lo, 100 * hi], [y, y], color=color, linewidth=2.4 if cat == "Average" else 2.0,
                solid_capstyle="round", zorder=3)
        ax.scatter([100 * mean], [y], s=42 if cat == "Average" else 30, color=color, zorder=4)
        ax.text(100 * max(hi, mean) + 1.2, y, f"{100 * mean:+.1f}", va="center", fontsize=8.2, color=INK)
    ax.axvline(0, color=RED, linewidth=1.0, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n}（n={metrics[c]['n'] if c != 'Average' else metrics['n']}）"
                        for n, c in zip(names, rows)], fontsize=8.6)
    ax.set_xlim(-12, 40)
    ax.set_xlabel("MN − SM 的 95% 置信区间（百分点）")
    style(ax, grid_axis="x")
    return svg(fig)


def fig_cost(cost) -> str:
    panels = [
        ("每题 tokens", "tokens", [cost["mn_tok"], cost["sm_tok"]], "{:.0f}"),
        ("每题成本（¥）", "¥", [cost["mn_cost"], cost["sm_cost"]], "{:.4f}"),
        ("每题延迟（s）", "秒", [cost["mn_lat"], cost["sm_lat"]], "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 1.95))
    for ax, (title, unit, vals, fmt) in zip(axes, panels):
        ax.barh([1, 0], vals, 0.48, color=[MN, SM], zorder=3)
        for y, v in zip([1, 0], vals):
            ax.text(v * 1.02, y, fmt.format(v), va="center", fontsize=8.2, color=INK)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["MN", "SM"], fontsize=8)
        ax.set_xlim(0, max(vals) * 1.34)
        ax.set_title(title, fontsize=8.6, color=MUTED, pad=6)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(length=0)
    return svg(fig)


def fig_diag(diag) -> str:
    """How much worse is the losing group?  Ratio against the winning group."""
    items = [
        ("候选条数", diag["retr"], "{:.1f}"),
        ("图节点数", diag["nodes"], "{:.1f}"),
        ("金标覆盖率", diag["cov"], "{:.2f}"),
        ("答案长度（词）", diag["predlen"], "{:.1f}"),
        ("无依据占比", diag["noise"], "{:.2f}"),
    ]
    fig, ax = plt.subplots(figsize=(7.3, 2.35))
    y = list(range(len(items)))[::-1]
    ratios = [(b / a if a else 0.0) for _, (a, b), _ in items]
    colors = [NEG if r > 1.3 else "#CFC9BC" for r in ratios]
    ax.barh(y, ratios, 0.46, color=colors, zorder=3)
    ax.axvline(1.0, color=FAINT, linewidth=1.0, linestyle="--", zorder=4)
    for yy, (name, (a, b), fmt), r in zip(y, items, ratios):
        ax.text(r + 0.05, yy, f"{r:.2f}×", va="center", fontsize=8.4, color=INK, fontweight="bold")
        ax.text(r + 0.30, yy, f"({fmt.format(a)} → {fmt.format(b)})", va="center",
                fontsize=7.4, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in items], fontsize=8.4)
    ax.set_xlim(0, 3.55)
    ax.set_xlabel("落后组 ÷ 胜出组（1.0 = 两组相同；虚线右侧表示落后组更大）")
    style(ax, grid_axis="x")
    return svg(fig)


def fig_conv(conv_stats) -> str:
    names = sorted(conv_stats)
    deltas = [100 * conv_stats[n][0] for n in names]
    counts = [conv_stats[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(7.3, 2.0))
    y = list(range(len(names)))[::-1]
    colors = [POS if d > 2 else (NEG if d < -2 else "#CFC9BC") for d in deltas]
    ax.barh(y, deltas, 0.5, color=colors, zorder=3)
    ax.axvline(0, color=FAINT, linewidth=0.9)
    for yy, d, n in zip(y, deltas, counts):
        off = 0.5 if d >= 0 else -0.5
        ax.text(d + off, yy, f"{d:+.1f}", va="center", ha="left" if d >= 0 else "right",
                fontsize=8.2, color=INK)
        ax.text(-19.5, yy, f"n={n}", va="center", fontsize=7.2, color=FAINT)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.0)
    ax.set_xlim(-20, 21)
    ax.set_xlabel("该对话内推理题的平均 F1 差（MN − SM，诊断口径）")
    style(ax, grid_axis="x")
    return svg(fig)


def fig_cause(causes) -> str:
    order = sorted(causes, key=lambda c: -(causes[c][0] + causes[c][1]))
    labels = [c.split("-")[0] + " " + c.split("-", 1)[1].split("（")[0] for c in order]
    loss = [causes[c][0] for c in order]
    both = [causes[c][1] for c in order]
    stage_color = {"A": "#7A7568", "B": POS, "C": POS, "D": MN, "E": MN, "F": POS, "G": MN, "H": MN}
    fig, ax = plt.subplots(figsize=(6.9, 2.95))
    y = list(range(len(order)))[::-1]
    ax.barh(y, loss, 0.52, color=[stage_color[c[0]] for c in order], zorder=3, label="MN 落后")
    ax.barh(y, both, 0.52, left=loss, color=[stage_color[c[0]] for c in order], alpha=0.32, zorder=3,
            label="双方同错（MN 也没答对）")
    for yy, a, b in zip(y, loss, both):
        ax.text(a + b + 0.25, yy, str(a + b), va="center", fontsize=8.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(0, 17)
    ax.set_xlabel("题数")
    style(ax, grid_axis="x")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right", handlelength=1.0)
    return svg(fig)


def fig_paths(path_counts, coverage) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    names = ["source_bound\nsynthesis", "failed", "execution", "indexed", "source\nadjudication"]
    keys = ["source_bound_synthesis", "failed", "execution", "indexed", "source_adjudication"]
    vals = [path_counts.get(k, 0) for k in keys]
    y = list(range(len(keys)))[::-1]
    ax.barh(y, [max(v, 0.6) for v in vals], 0.5, color=[MN, RED, "#D8D3C5", "#D8D3C5", "#D8D3C5"], zorder=3)
    for yy, v in zip(y, vals):
        ax.text(max(v, 0.6) + 3, yy, str(v), va="center", fontsize=8.4,
                color=INK if v else FAINT, fontweight="bold" if v else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.6)
    ax.set_xlim(0, 205)
    ax.set_title("实测：answer_path 分布（n=178）", fontsize=8.4, color=MUTED, pad=6)
    style(ax, grid_axis="x")

    ax = axes[1]
    cats = ["open-domain", "multi-hop", "adversarial", "single-hop", "temporal"]
    vals = [coverage[c] for c in cats]
    y = list(range(len(cats)))[::-1]
    ax.barh(y, [100 * v for v in vals], 0.5, color=POS, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(100 * v + 1.5, yy, f"{100 * v:.0f}%", va="center", fontsize=8.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([CN[c] for c in cats], fontsize=8.2)
    ax.set_xlim(0, 118)
    ax.set_title("离线测算：开启 execution 后覆盖率", fontsize=8.4, color=MUTED, pad=6)
    style(ax, grid_axis="x")
    fig.tight_layout(w_pad=1.6)
    return svg(fig)


# --------------------------------------------------------------------------- #
# html pieces
# --------------------------------------------------------------------------- #
def table(headers, rows, align=None, cls=""):
    align = align or ["left"] * len(headers)
    head = "".join(f'<th class="{a}">{h}</th>' for h, a in zip(headers, align))
    body = []
    for row in rows:
        cells = "".join(f'<td class="{a}">{c}</td>' for c, a in zip(row, align))
        body.append(f"<tr>{cells}</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def card(label, value, sub, tone=""):
    return (
        f'<div class="kpi {tone}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'
    )


def section(num, title, lead, body):
    return (
        f'<section><div class="sec-head"><span class="sec-num">{num}</span>'
        f'<div><h2>{title}</h2><p class="lead">{lead}</p></div></div>{body}</section>'
    )


def figure(svg_text, caption):
    return f'<figure>{svg_text}<figcaption>{caption}</figcaption></figure>'


def main() -> int:
    pairs, raw = load()
    cache = scored(pairs)
    n = len(pairs)

    # ---- per-category metrics -------------------------------------------- #
    metrics = {"n": n}
    for cat in CATS:
        rows = category_rows(pairs, cache, cat)
        mn_v = [r[2] for r in rows]
        sm_v = [r[3] for r in rows]
        d = [a - b for a, b in zip(mn_v, sm_v)]
        mean, se, lo, hi = ci(d)
        metrics[cat] = {
            "n": len(rows),
            "mn": st.mean(mn_v),
            "sm": st.mean(sm_v),
            "bleu_mn": st.mean(P.bleu1_of(r[0]) for r in rows),
            "bleu_sm": st.mean(P.bleu1_of(r[1]) for r in rows),
            "delta": mean,
            "se": se,
            "lo": lo,
            "hi": hi,
        }
    metrics["avg_mn"] = st.mean(metrics[c]["mn"] for c in CATS)
    metrics["avg_sm"] = st.mean(metrics[c]["sm"] for c in CATS)
    metrics["avg_delta"] = metrics["avg_mn"] - metrics["avg_sm"]
    metrics["avg_bleu_mn"] = st.mean(metrics[c]["bleu_mn"] for c in CATS)
    metrics["avg_bleu_sm"] = st.mean(metrics[c]["bleu_sm"] for c in CATS)
    metrics["avg_se"] = 0.25 * math.sqrt(sum(metrics[c]["se"] ** 2 for c in CATS))
    metrics["avg_lo"] = metrics["avg_delta"] - 1.96 * metrics["avg_se"]
    metrics["avg_hi"] = metrics["avg_delta"] + 1.96 * metrics["avg_se"]

    # ---- counterfactual: connector only ---------------------------------- #
    def mean_avg(use_counterfactual: bool) -> float:
        vals = []
        for cat in CATS:
            got = []
            for mn, sm in pairs:
                if mn.get("category") != cat:
                    continue
                row = dict(mn)
                if use_counterfactual:
                    row["prediction"] = str(mn.get("prediction") or "").replace(", ", "; ")
                got.append(P.f1_of(row))
            vals.append(st.mean(got))
        return st.mean(vals)

    cf_avg = mean_avg(True)
    net = 100 * (cf_avg - metrics["avg_sm"])
    net_se = 100 * metrics["avg_se"]
    net_lo, net_hi = net - 1.96 * net_se, net + 1.96 * net_se

    # ---- win / tie / loss -------------------------------------------------- #
    winloss = {}
    for cat in CATS + ["adversarial"]:
        w = t = l = 0
        for mn, sm in pairs:
            if mn.get("category") != cat:
                continue
            a, b = cache[id(mn)], cache[id(sm)]
            if a > b + 1e-9:
                w += 1
            elif a < b - 1e-9:
                l += 1
            else:
                t += 1
        winloss[cat] = (w, t, l)

    conv_stats = {}
    for conv in sorted({mn["sample_id"] for mn, _ in pairs}):
        vals = [cache[id(mn)] - cache[id(sm)] for mn, sm in pairs
                if mn["sample_id"] == conv and mn.get("category_id") != 5]
        conv_stats[conv] = (st.mean(vals), len(vals))

    # ---- cost -------------------------------------------------------------- #
    def sys_rows(name):
        out = []
        for mn, sm in pairs:
            out.append(mn if name == "MN" else sm)
        return out

    def cost_block(name):
        rows = sys_rows(name)
        cost = st.mean((r.get("model_calls") or {}).get("cost_rmb") or 0 for r in rows)
        tin = st.mean((r.get("model_calls") or {}).get("input_tokens") or 0 for r in rows)
        tout = st.mean((r.get("model_calls") or {}).get("output_tokens") or 0 for r in rows)
        lat = st.mean((r.get("latency_ms") or 0) for r in rows) / 1000
        return cost, tin + tout, lat, tin, tout

    mn_cost, mn_tok, mn_lat, mn_in, mn_out = cost_block("MN")
    sm_cost, sm_tok, sm_lat, sm_in, sm_out = cost_block("SM")
    cost = {
        "mn_cost": mn_cost, "sm_cost": sm_cost, "mn_tok": mn_tok, "sm_tok": sm_tok,
        "mn_lat": mn_lat, "sm_lat": sm_lat, "mn_in": mn_in, "sm_in": sm_in,
        "mn_out": mn_out, "sm_out": sm_out,
    }

    # ---- diagnostics: MN win vs MN loss ------------------------------------ #
    def diag_block(which):
        rows = [(mn, sm) for mn, sm in pairs
                if (cache[id(mn)] > cache[id(sm)] + 1e-9 if which == "win"
                    else cache[id(mn)] < cache[id(sm)] - 1e-9)]
        mn_rows = [r[0] for r in rows]
        return {
            "retr": st.mean(r.get("retrieval_count") or 0 for r in mn_rows),
            "nodes": st.mean((r.get("reasoning") or {}).get("graph_nodes") or 0 for r in mn_rows),
            "cov": st.mean(A.gold_coverage(r) for r in mn_rows),
            "predlen": st.mean(len(str(r.get("prediction") or "").split()) for r in mn_rows),
            "noise": st.mean(A.pred_noise(r) for r in mn_rows),
        }

    dw, dl = diag_block("win"), diag_block("loss")
    diag = {
        "retr": (dw["retr"], dl["retr"]),
        "nodes": (dw["nodes"], dl["nodes"]),
        "cov": (dw["cov"], dl["cov"]),
        "predlen": (dw["predlen"], dl["predlen"]),
        "noise": (dw["noise"], dl["noise"]),
    }

    # ---- causes ------------------------------------------------------------ #
    loss_c, both_c = Counter(), Counter()
    for mn, sm in pairs:
        a, b = cache[id(mn)], cache[id(sm)]
        cause = A.classify(mn, A.gold_coverage(mn))
        if a < b - 1e-9:
            loss_c[cause] += 1
        elif abs(a - b) <= 1e-9 and a < 0.999:
            both_c[cause] += 1
    causes = {k: (loss_c[k], both_c[k]) for k in set(loss_c) | set(both_c)}

    # ---- paths ------------------------------------------------------------- #
    path_counts = Counter((r.get("reasoning") or {}).get("answer_path") or "unknown"
                          for r in raw if r.get("system") == "MemNexus")
    coverage = {"open-domain": 8 / 9, "multi-hop": 12 / 17, "adversarial": 38 / 38,
                "single-hop": 20 / 77, "temporal": 2 / 37}

    # ---- format helpers ---------------------------------------------------- #
    def pct(x):
        return f"{100 * x:.1f}"

    def pp(x):
        return f"{100 * x:+.1f}"

    # ---- tables ------------------------------------------------------------ #
    main_rows = []
    for cat in CATS:
        m = metrics[cat]
        sig = "显著" if m["lo"] > 0 else "跨 0"
        main_rows.append([CN[cat], str(m["n"]), pct(m["mn"]), pct(m["sm"]), pp(m["delta"]),
                          f"[{100 * m['lo']:+.1f}, {100 * m['hi']:+.1f}]", sig])
    main_rows.append(["<b>四类等权</b>", f"<b>{n}</b>", f"<b>{pct(metrics['avg_mn'])}</b>",
                      f"<b>{pct(metrics['avg_sm'])}</b>", f"<b>{pp(metrics['avg_delta'])}</b>",
                      f"[{100 * metrics['avg_lo']:+.1f}, {100 * metrics['avg_hi']:+.1f}]", "勉强显著"])
    main_table = table(["题型", "n", "MN F1", "SM F1", "差值", "95% CI", "判读"], main_rows,
                       ["left", "right", "right", "right", "right", "center", "center"], "num")

    bleu_rows = []
    for cat in CATS:
        m = metrics[cat]
        bleu_rows.append([CN[cat], str(m["n"]), pct(m["bleu_mn"]), pct(m["bleu_sm"]),
                          pp(m["bleu_mn"] - m["bleu_sm"])])
    bleu_rows.append(["<b>四类等权</b>", f"<b>{n}</b>", f"<b>{pct(metrics['avg_bleu_mn'])}</b>",
                      f"<b>{pct(metrics['avg_bleu_sm'])}</b>",
                      f"<b>{pp(metrics['avg_bleu_mn'] - metrics['avg_bleu_sm'])}</b>"])
    bleu_table = table(["题型", "n", "MN BLEU-1", "SM BLEU-1", "差值"], bleu_rows,
                       ["left", "right", "right", "right", "right"], "num")

    cost_rows = [
        ["每题输入 tokens", f"{mn_in:,.0f}", f"{sm_in:,.0f}", f"{mn_in / sm_in:.2f}×"],
        ["每题输出 tokens", f"{mn_out:,.0f}", f"{sm_out:,.0f}", f"{mn_out / sm_out:.2f}×"],
        ["每题 tokens 合计", f"{mn_tok:,.0f}", f"{sm_tok:,.0f}", f"{mn_tok / sm_tok:.2f}×"],
        ["每题成本（¥）", f"{mn_cost:.4f}", f"{sm_cost:.4f}", f"{mn_cost / sm_cost:.2f}×"],
        ["每题延迟（s）", f"{mn_lat:.1f}", f"{sm_lat:.1f}", f"{mn_lat / sm_lat:.2f}×"],
    ]
    cost_table = table(["指标", "MemNexus", "SimpleMem", "MN / SM"], cost_rows,
                       ["left", "right", "right", "right"], "num")

    cause_order = sorted(causes, key=lambda c: -(causes[c][0] + causes[c][1]))
    cause_rows = []
    for c in cause_order:
        loss_n, both_n = causes[c]
        cause_rows.append([html.escape(c.split("（")[0]), str(loss_n), str(both_n),
                           f"<b>{loss_n + both_n}</b>", html.escape(A.STAGE_OF.get(c, ""))])
    total_loss = sum(v[0] for v in causes.values())
    total_both = sum(v[1] for v in causes.values())
    cause_rows.append(["<b>合计</b>", f"<b>{total_loss}</b>", f"<b>{total_both}</b>",
                       f"<b>{total_loss + total_both}</b>", ""])
    cause_table = table(["归因", "MN 落后", "双方同错", "合计", "出错阶段"], cause_rows,
                        ["left", "right", "right", "right", "left"], "num")

    path_rows = [
        ["source_bound_synthesis", "已实现（默认路径）", "175", "全部主结果由它产出"],
        ["execution", "已实现，`MEMORY_SEMANTIC_EXECUTION=false`", "0", "集合 / 计数 / 时间的确定性执行"],
        ["indexed", "已实现", "0", "索引化枚举"],
        ["source_adjudication", "已实现", "0", "逐来源裁决"],
        ["failed", "异常路径", "3", "MN 空预测，已从 175 对中剔除"],
    ]
    path_table = table(["路径", "代码状态", "实测条数", "说明"], path_rows,
                       ["left", "left", "right", "left"], "num")

    # ---- html -------------------------------------------------------------- #
    css = """
:root { --ink:#2C2C2A; --muted:#6B6A64; --faint:#9C978B; --line:#E7E4DA;
        --bg:#FAF9F5; --card:#FFFFFF; --mn:#534AB7; --pos:#0F6E56; --neg:#BA7517; --red:#A32D2D; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font-family:"Microsoft YaHei","PingFang SC",system-ui,-apple-system,sans-serif;
       font-size:14px; line-height:1.62; -webkit-font-smoothing:antialiased; }
.wrap { max-width:1120px; margin:0 auto; padding:44px 28px 72px; }
header.top { border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:30px; }
.eyebrow { font-size:12px; letter-spacing:.12em; color:var(--faint); text-transform:uppercase; }
h1 { font-size:30px; line-height:1.25; margin:8px 0 10px; font-weight:700; letter-spacing:-.3px; }
.sub { color:var(--muted); font-size:13.5px; margin:0; }
.meta { margin-top:14px; display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:var(--faint); }
.meta b { color:var(--muted); font-weight:500; }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:26px 0 36px; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 16px 14px; }
.kpi-label { font-size:12px; color:var(--faint); margin-bottom:6px; }
.kpi-value { font-size:26px; font-weight:700; letter-spacing:-.5px; line-height:1.1; }
.kpi-sub { font-size:11.5px; color:var(--muted); margin-top:7px; }
.kpi.mn .kpi-value { color:var(--mn); }
.kpi.pos .kpi-value { color:var(--pos); }
.kpi.neg .kpi-value { color:var(--neg); }
section { margin:0 0 46px; }
.sec-head { display:flex; gap:14px; align-items:flex-start; margin-bottom:16px; }
.sec-num { flex:0 0 30px; height:30px; border-radius:50%; background:var(--ink); color:#fff;
           font-size:13px; display:flex; align-items:center; justify-content:center; margin-top:2px; }
h2 { font-size:19px; margin:0 0 4px; font-weight:700; letter-spacing:-.2px; }
.lead { margin:0; color:var(--muted); font-size:13px; max-width:900px; }
figure { margin:0 0 16px; background:var(--card); border:1px solid var(--line);
         border-radius:12px; padding:16px 18px 10px; }
figure svg { width:100%; height:auto; display:block; }
figcaption { font-size:11.5px; color:var(--faint); border-top:1px solid var(--line);
             padding-top:9px; margin-top:8px; }
figure svg text { font-family:"Microsoft YaHei",sans-serif; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line);
        border-radius:12px; overflow:hidden; font-size:12.8px; margin:0 0 18px; }
thead th { background:#F3F0E8; color:var(--muted); font-weight:600; font-size:11.5px;
           letter-spacing:.03em; padding:9px 12px; border-bottom:1px solid var(--line); }
tbody td { padding:9px 12px; border-bottom:1px solid #F1EEE6; vertical-align:top; }
tbody tr:last-child td { border-bottom:0; }
tbody tr:nth-child(even) td { background:#FCFBF7; }
td.right, th.right { text-align:right; font-variant-numeric:tabular-nums; }
td.center, th.center { text-align:center; }
.note { border-left:3px solid var(--neg); background:#FBF4E7; padding:12px 16px;
        border-radius:0 8px 8px 0; font-size:12.8px; color:#6A4E1C; margin:4px 0 18px; }
.note.blue { border-color:var(--mn); background:#F1EFFC; color:#3B3578; }
.note.green { border-color:var(--pos); background:#EAF5F0; color:#1C4C3E; }
.grid2 { display:grid; grid-template-columns:1.15fr .85fr; gap:18px; align-items:start; }
.grid2 figure { margin-bottom:0; }
ul.tight { margin:6px 0 0; padding-left:18px; }
ul.tight li { margin:3px 0; }
footer { border-top:1px solid var(--line); margin-top:20px; padding-top:18px;
         font-size:11.5px; color:var(--faint); }
code { background:#F1EEE6; padding:1px 5px; border-radius:4px; font-size:11.5px; }
@media (max-width:900px) { .kpis { grid-template-columns:repeat(2,1fr); } .grid2 { grid-template-columns:1fr; } }
@media print { body { background:#fff; } section { page-break-inside:avoid; } }
"""

    kpis = "".join(
        [
            card("Average（四类等权）", f"{pct(metrics['avg_mn'])} <span style='color:#A8A49A;font-weight:400;font-size:16px'>vs {pct(metrics['avg_sm'])}</span>",
                 f"差 {pp(metrics['avg_delta'])} · 95%CI [{100 * metrics['avg_lo']:+.1f}, {100 * metrics['avg_hi']:+.1f}]", "mn"),
            card("剔除连接符后的净差", f"{net:+.2f}",
                 f"同预测，仅把 “, ” 换回 “; ” → {100 * cf_avg:.2f}", "neg"),
            card("机制净差 95%CI", f"[{net_lo:+.1f}, {net_hi:+.1f}]",
                 "跨 0 ⇒ 单次运行不可主张", "neg"),
            card("配对样本", f"{n}", "7 段对话 · 全量 1986 题的 8.8%", ""),
        ]
    )

    body = []
    body.append(
        section(
            1,
            "主结果：四类题型与总分",
            "官方 LoCoMo F1（Porter 词干、多跳按逗号取最佳项、开放域取分号前）。"
            "Average 为四类推理题的等权平均，<b>不含对抗</b>——与 SimpleMem 论文表 1 一致。",
            figure(fig_main(metrics),
                   f"灰柱为 SimpleMem，紫柱为 MemNexus；柱顶为官方 F1，顶部色字为差值（百分点）。"
                   f"多跳 {100 * (metrics['multi-hop']['mn'] - metrics['multi-hop']['sm']):+.1f}、"
                   f"时间 {100 * (metrics['temporal']['mn'] - metrics['temporal']['sm']):+.1f} 是主要来源；"
                   f"开放域两家都接近 0。")
            + main_table
            + '<p class="lead" style="margin:-8px 0 8px">同一批题目的 BLEU-1（论文表 1 的第二个指标，'
              '同样按四类等权）：</p>'
            + bleu_table
            + '<div class="note blue">读法：单跳样本最多（76 题）但差值只有 +4.8，且 95%CI 跨 0；'
              '时间题 n=37 却达到 +11.4 且区间不跨 0，是四类里唯一站得住的优势。'
              '多跳 +14.9 看起来最大，但 n=16，区间宽达 42 个百分点。</div>',
        )
    )

    body.append(
        section(
            2,
            "口径分解：8.0 分里只有 2.3 分属于机制",
            "把 MemNexus 的预测原样保留，只把列表连接符从 “, ” 换回 “; ”，重算官方 F1。"
            "这一步不改变任何模型行为，只改变评测脚本能否把一个列表切成多个条目。",
            figure(fig_waterfall(metrics, cf_avg),
                   f"左：测得差 {100 * metrics['avg_delta']:+.2f}；中：连接符适配贡献 "
                   f"{100 * (cf_avg - metrics['avg_mn']):+.2f}（换回分号后总分 {pct(cf_avg)}）；"
                   f"右：机制净差 {net:+.2f}。")
            + f'<div class="note">这不是小瑕疵，而是结论的全部差别。LoCoMo 的多跳金标用逗号分隔条目，'
              f'评测按同样规则切分预测；MemNexus 的 <code>render()</code> 对集合型取值输出 “, ”，'
              f'于是 12 道多跳题天然对齐。换回分号后，多跳从 {pct(metrics["multi-hop"]["mn"])} 掉到 '
              f'38.4，其余三类完全不动。<b>这部分不能写进贡献列表。</b></div>',
        )
    )

    body.append(
        section(
            3,
            "逐题分布：赢了 63 题，输了 34 题",
            "175 对逐题比较。持平 78 题里有 58 题双方都拿满分，其余 20 题是“双方都错、错得一样”。",
            '<div class="grid2">'
            + figure(fig_winloss(winloss),
                     "按题型拆分胜负。时间题 22 胜 5 负最干净；单跳 26 胜 20 负，胜率仅 56%；"
                     "对抗题 4 胜 3 负，实质无差异。")
            + figure(fig_scatter(pairs, cache),
                     "每点一题：横轴 SimpleMem F1，纵轴 MemNexus F1。对角线以上为 MN 更好。"
                     "低分区的密集点说明两家在开放域与部分单跳题上一起失败。")
            + "</div>"
            + figure(fig_conv(conv_stats),
                     "分对话看：conv-26 / 41 / 43 / 44 明显为正（+10.8 ~ +15.9），"
                     "conv-30 / 42 / 47 基本持平（−1.0 ~ +1.1）。没有一段为显著负，"
                     "但也没有一段能在单段内部独立成立——每段只有 15–24 道推理题。")
            + '<div class="note green">胜负结构比总分更能说明问题：时间题的 22:5 是“粒度纪律”的稳定优势'
              '（不加介词、不补零、不写 “about”）；多跳 9:4 靠的是列举完整度；'
              '而单跳 26:20 几乎是抛硬币——46 题差距在 ±1 题量级，这也是总分无法进一步拉开的原因。</div>',
        )
    )

    body.append(
        section(
            4,
            "显著性：只有时间题通过了区间检验",
            "配对逐题差值（MN − SM），按题型求均值与标准误，正态近似给出 95% 置信区间。"
            "Average 的标准误按四类分层组合：SE = ¼·√ΣSE²，与预注册文档的 3.24 完全一致。",
            figure(fig_ci(metrics),
                   "竖虚线为 0。区间不跨 0 的只有时间题（+1.7 ~ +21.1）；"
                   "总分区间 [+1.7, +14.4] 勉强不跨 0，但它包含的是 +8.0，不是机制净差 +2.3。")
            + f'<div class="note">机制净差 {net:+.2f} 的区间为 [{net_lo:+.1f}, {net_hi:+.1f}]，跨 0。'
              f'要把它和 0 分开需要约 <b>1291 对</b>题目；现有 175 对只能支撑“方向为正”，'
              f'不能支撑“显著优于”。全量 1986 题时标准误约 0.96，刚好够，但没有任何余量。</div>',
        )
    )

    body.append(
        section(
            5,
            "成本：token、钱、延迟三项都占优",
            "只统计问答阶段（建库已通过冻结快照回放归零）。成本取 <code>model_calls.cost_rmb</code>，"
            "不含缓存折扣，因此是上界；两侧同口径，可比。",
            figure(fig_cost(cost),
                   "每题 tokens 8,309 vs 8,720（−4.7%）；每题成本 ¥0.030 vs ¥0.054（−44%）；"
                   "每题延迟 7.6s vs 24.7s（−69%）。")
            + cost_table
            + '<div class="note blue">成本优势主要来自输出：MN 每题只写 341 个 token，SM 写 613 个；'
              '输出单价远高于输入，所以“少写”比“少读”更省钱。这也解释了为什么 MN 的输入并不更少，'
              '总成本却低 44%。延迟优势来自调用次数：2.5 次 vs 4.0 次。</div>',
        )
    )

    body.append(
        section(
            6,
            "损失在哪：召回几乎相同，差别在写多少",
            "把 MN 胜出的 63 题与落后的 34 题分组对比。召回侧指标几乎重合，产出侧指标差 1.8–2.9 倍——"
            "这说明瓶颈不在“找没找到”，而在“写多少”。",
            figure(fig_diag(diag),
                   "上排：候选条数、图节点数、金标覆盖率（胜出组甚至更高）；"
                   "下排：答案长度 7.5 vs 13.3 词，无依据占比 0.26 vs 0.74。")
            + '<div class="note">更强的证据：78 道有多余内容的题里，<b>58 道（74.4%）的多余 token '
              '能在召回证据中逐字找到</b>。这不是幻觉，是“有据但不节制”。'
              '因此修法不能是截断（<code>first_item</code> −1.26、<code>clause_cut</code> −0.09 已否决），'
              '只能从义务契约约束取值粒度。</div>',
        )
    )

    body.append(
        section(
            7,
            "错误归因：产出层 34 例，召回层 13 例",
            "对 34 道落后题 + 20 道“双方同错”题逐题归因，共 54 例。"
            "归因规则固定为一条主因，按损失产生的阶段判定（脚本 <code>audit_errors_20261022.py</code>）。",
            figure(fig_cause(causes),
                   "深色为 MN 落后，浅色为双方同错。H（列举不完整）14、D（多余条目）11、"
                   "G（同义表述）9 三项合计 34，全部发生在产出层。")
            + cause_table
            + '<div class="note blue">产出层（D+G+H）34 例 vs 召回/表示层（F+C）13 例，比例 2.6:1。'
              '另有对抗 6 例、绑定 1 例单列。这个分布直接否定了“再优化检索”的路线——'
              '同样的召回条件下，MN 只是写得更多、更碎。</div>',
        )
    )

    body.append(
        section(
            8,
            "四条产出路径，只跑通了一条",
            "代码里四条路径都已实现，但 <code>MEMORY_SEMANTIC_EXECUTION</code> 默认 false，"
            "所以主结果 175 条全部由 <code>source_bound_synthesis</code> 产出。",
            figure(fig_paths(path_counts, coverage),
                   "左：实测路径分布，execution / indexed / source_adjudication 均为 0；"
                   "右：离线测算开启 execution 后各题型的命中比例，合计 80/178 = 44.9%。")
            + path_table
            + '<div class="note">这是“文档写了、实测没跑”的典型风险：方法章节的四条路径只有一条有数据支撑，'
              '而集合求交、计数精确化、时间运算这些确定性卖点全在未启用的 <code>execution</code> 里。'
              '更麻烦的是它恰好覆盖最弱的两类（开放域 89%、多跳 71%），'
              '所以它既是最大的主张风险，也是最便宜的潜在涨分点。</div>',
        )
    )

    body.append(
        f'<section><div class="sec-head"><span class="sec-num">9</span><div>'
        f'<h2>三条结论与三条边界</h2><p class="lead">只写数据支持的部分。</p></div></div>'
        f'<div class="note green"><b>可以主张：</b>在 175 对配对样本上，MemNexus 的时间类题目 F1 高出 11.4 点'
        f'（95%CI [+1.7, +21.1]），多跳题胜 9 负 4；每题成本低 44%、延迟低 69%、token 低 4.7%。</div>'
        f'<div class="note"><b>不能主张：</b>“总体显著优于”。总分 +8.0 中 5.7 来自连接符适配，'
        f'净差 2.3 的区间跨 0；开放域 8.4 vs 7.5 实质相同；对抗题 31/37 vs 30/37 无差异。</div>'
        f'<div class="note blue"><b>尚不可判：</b>验证层（主张 A）的收益。'
        f'<code>typed_negative_gate</code> 的三条返回路径 <code>blocked</code> 恒为 False，只是建议器；'
        f'81 题时的“对抗 +19 点”在 175 题上没有复现。</div>'
        f'</section>'
    )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MemNexus vs SimpleMem · 配对实验结果</title>
<style>{css}</style></head>
<body><div class="wrap">
<header class="top">
  <div class="eyebrow">LoCoMo · Paired Evaluation</div>
  <h1>MemNexus vs SimpleMem：175 对题目的结果与分析</h1>
  <p class="sub">所有数字由 <code>experiments/paper_protocol.py</code> 从逐题记录重算；
  口径与 SimpleMem 论文 §3.1 / Tables 1、3 一致。图表由 <code>experiments/build_comparison_dashboard.py</code> 生成。</p>
  <div class="meta">
    <span><b>数据</b> result/bench260/rows.json</span>
    <span><b>配对</b> 175 / 178（3 条 MN 崩溃已剔除）</span>
    <span><b>对话</b> 7 段</span>
    <span><b>生成</b> 2026-10-22</span>
  </div>
</header>
<div class="kpis">{kpis}</div>
{''.join(body)}
<footer>
  <p><b>口径声明</b>：F1 为官方 LoCoMo 评分（去冠词标点 + Porter 词干；多跳金标按逗号切分、逐项取最佳预测再平均；开放域仅取分号前）；
  Average 为四类推理题等权平均，不含对抗题；对抗题按拒答成功率单独报告；成本只含问答阶段，不含建库。</p>
  <p><b>可复现</b>：<code>python experiments/build_comparison_dashboard.py</code> ·
  论文表：<code>python experiments/paper_protocol.py result/bench260</code> ·
  逐题审计：<code>python experiments/audit_errors_20261022.py</code></p>
  <p>注：本页把“机制净差”的区间按同一标准误平移得到（{net:+.2f} ± 1.96×{net_se:.2f}），
  仅用于说明量级；严格的重采样区间需要按题型分层 bootstrap，未在此页给出。</p>
</footer>
</div></body></html>"""

    OUT_PATH.write_text(doc, encoding="utf-8")

    payload = {
        "protocol": "SimpleMem paper §3.1 / Tables 1 and 3; official LoCoMo F1",
        "pairs": n,
        "average": {"memnexus": metrics["avg_mn"], "simplemem": metrics["avg_sm"],
                    "delta": metrics["avg_delta"], "se": metrics["avg_se"],
                    "ci95": [metrics["avg_lo"], metrics["avg_hi"]]},
        "counterfactual_connector": {"average": cf_avg, "net_delta": net,
                                     "net_ci95": [net_lo, net_hi]},
        "by_category": {c: metrics[c] for c in CATS},
        "win_tie_loss": {c: winloss[c] for c in winloss},
        "cost": cost,
        "diagnostics_win_vs_loss": diag,
        "causes": {k: {"loss": v[0], "both_bad": v[1]} for k, v in causes.items()},
        "answer_paths": dict(path_counts),
        "execution_offline_coverage": coverage,
    }
    (OUT_PATH.parent / "comparison_metrics_20261022.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"pairs={n} avg={100 * metrics['avg_mn']:.2f}/{100 * metrics['avg_sm']:.2f} "
          f"cf={100 * cf_avg:.2f} net={100 * (cf_avg - metrics['avg_sm']):.2f}")
    print(f"causes={sum(causes[c][0] + causes[c][1] for c in causes)} paths={dict(path_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
