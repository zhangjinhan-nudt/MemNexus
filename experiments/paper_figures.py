"""Generate the paper figures for MemNexus.

All labels are English: these files go into the ACL submission, and keeping the
text ASCII avoids font substitution on the authors' machines. Every number or
mechanism named here exists in the code; nothing is illustrative-only.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

OUT = pathlib.Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

INK = "#2C2C2A"
MUTED = "#5F5E5A"
LINE = "#B4B2A9"
PURPLE = "#534AB7"
PURPLE_L = "#EEEDFE"
TEAL = "#0F6E56"
TEAL_L = "#E1F5EE"
AMBER = "#BA7517"
AMBER_L = "#FAEEDA"
GRAY_L = "#F1EFE8"
RED = "#A32D2D"
RED_L = "#FCEBEB"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0,
    "savefig.dpi": 300,
})


def canvas(w: float, h: float, xmax: float, ymax: float):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, title, sub="", fc=GRAY_L, ec=LINE, tc=INK, fs=8.0, sfs=6.8,
        bold=True, lw=0.8, radius=0.04):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold" if bold else "normal", zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=sfs, color=MUTED, zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold" if bold else "normal", zorder=3)


def arrow(ax, p0, p1, color=LINE, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1, shrinkB=1, zorder=1))


def caption(ax, x, y, text, fs=7.2, color=MUTED, style="normal", weight="normal"):
    ax.text(x, y, text, ha="left", va="center", fontsize=fs, color=color,
            style=style, weight=weight, zorder=3)


# --------------------------------------------------------------------------- #
# Fig. 1  Two pipelines: where the judgement sits
# --------------------------------------------------------------------------- #
def fig1() -> None:
    fig, ax = canvas(7.2, 3.5, 7.2, 3.5)
    caption(ax, 0.18, 3.28, "(a) Compress-retrieve-generate: one model does both recall and assertion",
            fs=7.4, color=INK, weight="bold")
    xs = [0.18, 1.55, 2.92, 4.29, 5.66]
    w, h = 1.25, 0.66
    ys = 2.28
    labels = [
        ("Transcript", "sessions 1..n", GRAY_L, LINE),
        ("Compress", "summaries, atoms", GRAY_L, LINE),
        ("Top-k recall", "one similarity score", AMBER_L, AMBER),
        ("Generate", "reads, decides, writes", AMBER_L, AMBER),
        ("Answer", "no per-value support", RED_L, RED),
    ]
    for x, (t, s, fc, ec) in zip(xs, labels):
        box(ax, x, ys, w, h, t, s, fc=fc, ec=ec)
    for i in range(4):
        arrow(ax, (xs[i] + w, ys + h / 2), (xs[i + 1], ys + h / 2))
    ax.add_patch(FancyBboxPatch((2.80, 1.88), 3.95, 0.28,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                linewidth=0.8, edgecolor=RED, facecolor="none",
                                linestyle=(0, (3, 2)), zorder=4))
    caption(ax, 2.86, 2.02, "relevance is read as truth confidence; no audit unit exists",
            fs=6.5, color=RED)

    caption(ax, 0.18, 1.58, "(b) MemNexus: recall stays in the model, assertion moves outside it",
            fs=7.4, color=INK, weight="bold")
    ys = 0.58
    labels = [
        ("Transcript", "sessions 1..n", GRAY_L, LINE),
        ("Typed assertion", "+ evidence span", PURPLE_L, PURPLE),
        ("Budgeted recall", "3 views, budget object", PURPLE_L, PURPLE),
        ("Bind", "1 call: value + quote", AMBER_L, AMBER),
        ("Validate", "0 calls: 5 checks", TEAL_L, TEAL),
    ]
    for x, (t, s, fc, ec) in zip(xs, labels):
        box(ax, x, ys, w, h, t, s, fc=fc, ec=ec)
    for i in range(4):
        arrow(ax, (xs[i] + w, ys + h / 2), (xs[i + 1], ys + h / 2), color=MUTED)
    ax.add_patch(FancyBboxPatch((5.62, 0.24), 1.33, 0.26,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                linewidth=0.8, edgecolor=TEAL, facecolor="none", zorder=4))
    caption(ax, 6.28, 0.37, "no model here", fs=6.3, color=TEAL, style="italic")
    fig.savefig(OUT / "fig1_pipelines.png", facecolor="white")
    fig.savefig(OUT / "fig1_pipelines.svg", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 2  Layers and their invariants
# --------------------------------------------------------------------------- #
def fig2() -> None:
    fig, ax = canvas(7.0, 4.3, 7.0, 4.3)
    layers = [
        ("L1  Contract", "data shapes; extra fields forbidden"),
        ("L2  State", "authoritative store, rebuildable index, snapshot"),
        ("L3  Ingestion", "window, extract, relation, lifecycle"),
        ("L4  Retrieval", "proposal, route override, 3 views, budget"),
        ("L5  Grounding", "resolve to utterances, block revoked, coverage"),
        ("L6  Decision", "typed gate, binding, 5 checks, operators"),
        ("L7  Realization", "path choice, render, citation mapping"),
    ]
    caption(ax, 0.25, 4.05, "One invariant per layer; a violation has no representation",
            fs=7.6, color=INK, weight="bold")
    top, h, gap = 3.72, 0.36, 0.10
    x, w = 0.25, 4.10
    for i, (name, mech) in enumerate(layers):
        y = top - i * (h + gap) - h
        highlight = name.startswith("L6")
        fc = TEAL_L if highlight else (PURPLE_L if i < 2 else GRAY_L)
        ec = TEAL if highlight else (PURPLE if i < 2 else LINE)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                    linewidth=0.9 if highlight else 0.7,
                                    edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + 0.12, y + h / 2, name, ha="left", va="center", fontsize=7.4,
                color=ec if highlight else INK, weight="bold", zorder=3)
        ax.text(x + 1.55, y + h / 2, mech, ha="left", va="center", fontsize=6.3,
                color=MUTED, zorder=3)
        if i < len(layers) - 1:
            arrow(ax, (x + w / 2, y), (x + w / 2, y - gap), color=LINE, lw=0.8)
    caption(ax, x, 0.16, "L6 changes an answer without calling a model; L7 only writes it down.",
            fs=6.6, color=TEAL)

    gx, gw = 4.72, 2.03
    ax.add_patch(FancyBboxPatch((gx, 0.42), gw, 3.30, boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=0.8, edgecolor=AMBER, facecolor=AMBER_L, zorder=2))
    ax.text(gx + gw / 2, 3.52, "L8  Governance", ha="center", fontsize=8.0,
            color=INK, weight="bold", zorder=3)
    items = [
        ("version guard", "revoked evidence voids\nthe answer"),
        ("time guard", "expired scope is\nnot returned"),
        ("citation guard", "out-of-range ids are\ndropped and logged"),
        ("ledger", "sources, plan, gaps,\ntokens per question"),
    ]
    for i, (head, body) in enumerate(items):
        y = 3.20 - i * 0.74
        ax.text(gx + 0.14, y, head, ha="left", va="top", fontsize=6.8, color=INK,
                weight="bold", zorder=3)
        ax.text(gx + 0.14, y - 0.14, body, ha="left", va="top", fontsize=6.2,
                color=MUTED, zorder=3, linespacing=1.45)
    fig.savefig(OUT / "fig2_architecture.png", facecolor="white")
    fig.savefig(OUT / "fig2_architecture.svg", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 3  13 steps, three stages, who acts
# --------------------------------------------------------------------------- #
def fig3() -> None:
    fig, ax = canvas(7.2, 4.0, 7.2, 4.0)
    stages = [
        ("A. Write", 3.15, [
            ("1", "ingest turn", "store"),
            ("2", "build window", "rule"),
            ("3", "extract", "model"),
            ("4", "store short", "rule"),
            ("5", "promote, lifecycle", "rule"),
        ]),
        ("B. Acquire", 1.95, [
            ("6", "read question", "model"),
            ("7", "override route", "rule"),
            ("8", "3-view recall", "rule"),
            ("9", "resolve to speech", "rule"),
            ("10", "time / set algebra", "rule"),
        ]),
        ("C. Decide", 0.75, [
            ("11", "bind value + quote", "model"),
            ("12", "5 checks", "rule"),
            ("13", "render + ledger", "rule"),
        ]),
    ]
    colors = {"rule": (GRAY_L, LINE), "model": (AMBER_L, AMBER), "store": (PURPLE_L, PURPLE)}
    w, h, gap = 1.20, 0.58, 0.24
    for label, y, steps in stages:
        ax.text(0.18, y + h + 0.12, label, fontsize=7.8, color=INK, weight="bold")
        for i, (num, text, who) in enumerate(steps):
            x = 0.18 + i * (w + gap)
            fc, ec = colors[who]
            box(ax, x, y, w, h, num, text, fc=fc, ec=ec, fs=8.2, sfs=6.4)

    ax.add_patch(FancyBboxPatch((4.95, 0.62), 2.05, 0.86,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                linewidth=0.8, edgecolor=TEAL, facecolor=TEAL_L,
                                linestyle=(0, (3, 2)), zorder=2))
    ax.text(5.97, 1.32, "no model call", ha="center", fontsize=7.2, color=TEAL, weight="bold")
    ax.text(5.97, 1.00, "steps 12-13 validate,\nrender and record", ha="center", va="center",
            fontsize=6.3, color=MUTED, linespacing=1.4)

    ly = 0.30
    for i, (label, fc, ec) in enumerate((("model", AMBER_L, AMBER),
                                        ("rule", GRAY_L, LINE),
                                        ("store", PURPLE_L, PURPLE))):
        x = 0.18 + i * 1.05
        ax.add_patch(FancyBboxPatch((x, ly - 0.10), 0.16, 0.16,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    linewidth=0.7, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(x + 0.22, ly - 0.02, label, fontsize=6.4, color=MUTED, va="center")
    ax.text(3.45, ly - 0.02,
            "3 model calls total (3, 6, 11); every other step is deterministic.",
            fontsize=6.6, color=INK, va="center")
    fig.savefig(OUT / "fig3_pipeline.png", facecolor="white")
    fig.savefig(OUT / "fig3_pipeline.svg", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 4  Determinism ladder
# --------------------------------------------------------------------------- #
def fig4() -> None:
    fig, ax = canvas(6.8, 3.5, 6.8, 3.5)
    rows = [
        ("L0", "typed enumeration / set algebra", "0", "per member + span", TEAL, TEAL_L),
        ("L1", "contract execution + 5 checks", "1", "per slot: value or gap", TEAL, TEAL_L),
        ("L2", "answer table + 5 checks", "1", "per cell quote", PURPLE, PURPLE_L),
        ("L3", "free reading", "1", "citation ids only", AMBER, AMBER_L),
        ("A", "typed abstention (gate, gap)", "0", "gap code + checked count", RED, RED_L),
    ]
    caption(ax, 0.20, 3.22, "Selection prefers the highest rung whose preconditions hold",
            fs=7.4, color=INK, weight="bold")
    caption(ax, 0.20, 3.02, "calls per question", fs=6.2, color=MUTED)
    caption(ax, 3.85, 3.02, "smallest auditable unit", fs=6.2, color=MUTED)
    top, h, gap = 2.82, 0.42, 0.09
    for i, (tag, name, calls, audit, ec, fc) in enumerate(rows):
        y = top - i * (h + gap) - h
        ax.add_patch(FancyBboxPatch((0.20, y), 3.45, h, boxstyle="round,pad=0,rounding_size=0.03",
                                    linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(0.34, y + h / 2, tag, fontsize=7.8, color=ec, weight="bold", va="center")
        ax.text(0.70, y + h / 2, name, fontsize=7.3, color=INK, va="center")
        ax.text(3.52, y + h / 2, calls, fontsize=7.8, color=INK, ha="center", va="center")
        ax.text(3.85, y + h / 2, audit, fontsize=6.6, color=MUTED, va="center")
    caption(ax, 0.20, 0.16,
            "Results are reported per rung, so a change in accuracy cannot hide behind a change in path mix.",
            fs=6.4, color=MUTED, style="italic")
    fig.savefig(OUT / "fig4_ladder.png", facecolor="white")
    fig.savefig(OUT / "fig4_ladder.svg", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 5  Answer table -> five checks -> value or typed gap
# --------------------------------------------------------------------------- #
def fig5() -> None:
    fig, ax = canvas(6.8, 3.4, 6.8, 3.4)
    caption(ax, 0.18, 3.22, "obligation", fs=6.4)
    caption(ax, 1.32, 3.22, "bound values, each with a verbatim quote", fs=6.4)
    caption(ax, 3.48, 3.22, "five deterministic checks", fs=6.8, color=TEAL, weight="bold")
    caption(ax, 5.72, 3.22, "outcome", fs=6.4)

    box(ax, 0.18, 2.42, 0.98, 0.50, "slot 1", "when?", fc=PURPLE_L, ec=PURPLE, fs=7.4, sfs=6.2)
    box(ax, 0.18, 1.72, 0.98, 0.50, "slot 2", "who else?", fc=PURPLE_L, ec=PURPLE, fs=7.4, sfs=6.2)
    box(ax, 1.32, 2.42, 1.98, 0.50, "29 January 2023", "\"launched it on Jan 29\"",
        fc="white", ec=LINE, fs=7.0, sfs=6.0)
    box(ax, 1.32, 1.72, 1.98, 0.50, "Gina, Tom", "\"Gina\" and \"Tom\"", fc="white", ec=LINE,
        fs=7.0, sfs=6.0)
    caption(ax, 1.32, 1.56, "an empty slot must declare a reason", fs=6.0, style="italic")

    gates = ["1  every target present, in order",
             "2  source id resolves to a hit",
             "3  quote occurs verbatim",
             "4  one factual utterance, not a\n     question or a past reply",
             "5  operator anchored: date from\n     source, endpoints exact"]
    for i, g in enumerate(gates):
        y = 2.86 - i * 0.40
        ax.add_patch(FancyBboxPatch((3.48, y - 0.26), 2.05, 0.34,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    linewidth=0.8, edgecolor=TEAL, facecolor=TEAL_L, zorder=2))
        ax.text(3.58, y - 0.09, g, fontsize=6.1, color=INK, va="center", zorder=3,
                linespacing=1.3)
    arrow(ax, (3.32, 2.67), (3.46, 2.67), color=MUTED)
    arrow(ax, (3.32, 1.97), (3.46, 1.97), color=MUTED)

    box(ax, 5.78, 2.36, 0.88, 0.62, "value", "render", fc="white", ec=TEAL, fs=7.4, sfs=6.2)
    box(ax, 5.78, 1.30, 0.88, 0.62, "gap", "retry or abstain", fc=RED_L, ec=RED, fs=7.4, sfs=6.0)
    arrow(ax, (5.58, 2.60), (5.76, 2.67), color=TEAL)
    arrow(ax, (5.58, 1.10), (5.76, 1.45), color=RED)

    caption(ax, 0.18, 0.88, "A violation is not repaired by the model. It becomes a typed gap, and the gap",
            fs=6.8, color=INK)
    caption(ax, 0.18, 0.68, "decides the next action (reread, retrieve, abstain) before any text is written.",
            fs=6.8, color=INK)
    caption(ax, 0.18, 0.38, "Hence the audit unit is a cell, not an answer: the checker can point at the failed quote.",
            fs=6.5, style="italic")
    fig.savefig(OUT / "fig5_validation.png", facecolor="white")
    fig.savefig(OUT / "fig5_validation.svg", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 6  Retrieval: four views, one bounded score, one budget object
# --------------------------------------------------------------------------- #
def fig6() -> None:
    fig, ax = canvas(7.2, 4.3, 7.2, 4.3)
    caption(ax, 0.15, 4.12, "Recall: four independent views, fused by max, then bounded by a budget object",
            fs=7.6, color=INK, weight="bold")

    box(ax, 0.15, 2.95, 1.30, 0.80, "question + plan", "route, depth,\nrequired_info",
        fc=GRAY_L, ec=LINE, fs=7.0, sfs=6.0)

    views = [
        (1.72, 3.30, "lexical", "BM25 over FTS5\ncandidates, s/(1+s)"),
        (2.86, 3.30, "symbolic", "subject & predicate\nexact, s = 1.0"),
        (1.72, 2.50, "semantic", "cosine >= 0.35,\n(cos+1)/2"),
        (2.86, 2.50, "hierarchy", "path overlap,\n0.4 x overlap"),
    ]
    for x, y, t, s in views:
        box(ax, x, y, 1.04, 0.78, t, s, fc=AMBER_L, ec=AMBER, fs=6.8, sfs=5.9, radius=0.03)
    arrow(ax, (1.45, 3.35), (1.70, 3.69), color=MUTED)
    arrow(ax, (1.45, 3.35), (1.70, 2.89), color=MUTED)

    box(ax, 4.16, 2.86, 1.30, 1.00, "fusion (max)",
        "s = max over views\nviews kept as a set\nnot a confidence",
        fc=PURPLE_L, ec=PURPLE, fs=7.0, sfs=6.0)
    arrow(ax, (3.92, 3.69), (4.14, 3.48), color=MUTED)
    arrow(ax, (3.92, 2.89), (4.14, 3.24), color=MUTED)

    box(ax, 5.72, 2.86, 1.36, 1.00, "bounded re-rank",
        "s x (0.95 + 0.05 x 2^(-t/H))\nH = 7 d short / 90 d long\nfacts never decay",
        fc=PURPLE_L, ec=PURPLE, fs=7.0, sfs=5.8)
    arrow(ax, (5.46, 3.36), (5.70, 3.36), color=MUTED)

    ax.add_patch(FancyBboxPatch((0.15, 1.42), 6.93, 1.04,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                linewidth=0.8, edgecolor=TEAL, facecolor=TEAL_L, zorder=2))
    caption(ax, 0.28, 2.28, "budget object (returned with every answer, not a hyper-parameter)",
            fs=6.6, color=TEAL, weight="bold")
    lines = [
        "candidate_limit = min(max_candidates, m x depth),   m = 10 recall-first / 6 focused",
        "safety_cap = min(top_k, max_top_k);   target_k = min(safety_cap, max(3, min(20, depth)))",
        "selection adds whole support bundles: a candidate whose evidence does not fit is dropped, never truncated",
    ]
    for i, line in enumerate(lines):
        caption(ax, 0.28, 2.04 - i * 0.21, line, fs=6.2, color=INK)
    arrow(ax, (6.40, 2.84), (6.40, 2.50), color=TEAL)

    caption(ax, 0.15, 1.14, "What this buys:", fs=6.9, color=INK, weight="bold")
    buys = [
        ("auditable", "each hit reports the views that\nfired and its rank formula", TEAL),
        ("bounded", "token cost is decided before\ngeneration, not measured after", PURPLE),
        ("honest", "score_semantics = relevance,\nnot truth confidence", AMBER),
    ]
    for i, (head, body, colour) in enumerate(buys):
        x = 0.15 + i * 2.36
        box(ax, x, 0.22, 2.24, 0.78, head, body, fc="white", ec=colour, fs=7.0, sfs=6.0, radius=0.03)
    fig.savefig(OUT / "fig6_recall.png", facecolor="white")
    fig.savefig(OUT / "fig6_recall.svg", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 7  Four answer paths: where the model calls are, and where they are not
# --------------------------------------------------------------------------- #
def fig7() -> None:
    fig, ax = canvas(7.2, 4.5, 7.2, 4.5)
    caption(ax, 0.15, 4.34, "Answer paths: the same evidence, four decision costs",
            fs=7.6, color=INK, weight="bold")
    caption(ax, 0.15, 4.14, "0 = deterministic, no model call. The route is chosen from the question form.",
            fs=6.3, style="italic")

    box(ax, 0.15, 2.62, 1.30, 0.90, "question form", "classify_question:\nset / count / relation /\nfact / temporal",
        fc=GRAY_L, ec=LINE, fs=7.0, sfs=5.8, radius=0.03)

    paths = [
        (1.80, 2.92, "indexed", "typed index already covers\nthe asked subjects", "0 calls", PURPLE, PURPLE_L),
        (3.62, 2.92, "source adjudication", "per-source candidate + quote,\nthen execute() 0 calls", "1 call", AMBER, AMBER_L),
        (1.80, 1.82, "execution (default)", "bind values to contract,\ncheck_and_execute 0 calls", "1 call", AMBER, AMBER_L),
        (3.62, 1.82, "source synthesis", "answer table, five checks,\nrender_table 0 calls", "1 call", AMBER, AMBER_L),
    ]
    for x, y, name, body, calls, ec, fc in paths:
        box(ax, x, y, 1.72, 0.90, name, body, fc=fc, ec=ec, fs=6.8, sfs=5.8, radius=0.03)
        ax.text(x + 0.86, y + 0.99, calls, ha="center", va="center", fontsize=6.2,
                color=ec, weight="bold", zorder=3)
    arrow(ax, (1.45, 3.07), (1.78, 3.37), color=MUTED)
    arrow(ax, (1.45, 3.07), (1.78, 2.27), color=MUTED)

    box(ax, 5.60, 2.42, 0.86, 0.80, "answer", "or typed gap", fc="white", ec=TEAL, fs=7.0, sfs=6.0)
    arrow(ax, (5.34, 3.37), (5.58, 2.95), color=MUTED)
    arrow(ax, (5.34, 2.27), (5.58, 2.70), color=MUTED)

    ax.add_patch(FancyBboxPatch((0.15, 0.92), 6.93, 0.72,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                linewidth=0.8, edgecolor=RED, facecolor=RED_L, zorder=2))
    caption(ax, 0.28, 1.46, "only when a gap is actionable   (a single extra read budget, then re-bind and re-validate)",
            fs=6.4, color=RED, weight="bold")
    caption(ax, 0.28, 1.24, "reread the cited source, or retrieve the missing premise;  call budget exhausted is reported, never hidden",
            fs=6.2, color=INK)
    caption(ax, 0.28, 1.05, "member_scope_unknown is deliberately not actionable: an open list cannot be completed by search",
            fs=6.1, style="italic", color=INK)
    arrow(ax, (4.48, 1.82), (4.48, 1.66), color=RED, ls=(0, (3, 2)))
    arrow(ax, (4.56, 1.66), (4.56, 1.82), color=RED, ls=(0, (3, 2)))

    caption(ax, 0.15, 0.72, "Cost consequence:", fs=6.8, color=INK, weight="bold")
    caption(ax, 0.15, 0.52, "the cheap part of the budget is the part that decides truth: validation, coverage and rendering are free;",
            fs=6.2)
    caption(ax, 0.15, 0.34, "the model is paid once to propose bindings, never to grade them. Measured: 8,309 vs 8,720 tokens per",
            fs=6.2)
    caption(ax, 0.15, 0.16, "question against SimpleMem (-4.7%), with 7.6 s vs 24.7 s latency at equal evidence width.",
            fs=6.2)
    fig.savefig(OUT / "fig7_paths.png", facecolor="white")
    fig.savefig(OUT / "fig7_paths.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        fn()
        print("ok", fn.__name__)
