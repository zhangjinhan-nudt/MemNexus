"""Paired A/B comparison of two runs that consumed the same frozen memory base.

Why this file exists
--------------------
Three 150-question runs of unchanged code moved MemNexus by 6 questions
(120 -> 114 -> 114) and flipped 14-18 individual verdicts. The optimisations we
want to measure move 2-4 questions. An unpaired accuracy delta is therefore
unreadable: it is smaller than the noise it sits in.

Freezing the build stage removes most of that noise, but the answering stage is
still a sampling model. So the only defensible comparison is:

    same frozen base + same cohort, one behaviour change, paired per question

and the statistic has to be paired too: McNemar's exact test on discordant
pairs, not a difference of two proportions. With 150 questions, a change that
wins 6 and loses 0 is the smallest one that clears p < 0.05; winning 4 and
losing 1 (p ~ 0.38) is indistinguishable from a coin flip, no matter how good
the aggregate line looks.

Usage
-----
    python experiments/compare_paired_runs.py \
        --run-a result/benchmark150_paper/<baseline> \
        --run-b result/benchmark150_paper/<candidate> \
        --label-a baseline --label-b candidate

Exit code 0 means the protocol accepts the candidate; 1 means it rejected or
could not judge it (unpaired, crashed, or not significant).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

CORRECT = "CORRECT"


def load_rows(run: Path, system: str) -> tuple[dict, dict]:
    """Return ``(rows keyed by (sample, question), stage metadata)``."""
    payload = json.loads((run / "results.json").read_text(encoding="utf-8"))
    rows = {}
    for row in payload["rows"]:
        if row.get("system") != system:
            continue
        rows[(row["sample_id"], row["question"])] = row
    return rows, payload


def replay_condition(run: Path) -> dict:
    """What base did this run actually consume?

    A run can only be paired with another if both replayed the same sha256 for
    every conversation. A run that rebuilt part of its base is a different
    experiment wearing the same name.
    """
    digests, rebuilt, stages = set(), [], {}
    for path in sorted(run.glob("attempts/*/usage.stage.json")):
        try:
            stage = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if stage.get("system") != "MemNexus":
            continue
        digest = stage.get("memory_snapshot_sha256")
        if digest:
            digests.add(digest)
        else:
            rebuilt.append(stage.get("sample_id"))
        stages[stage.get("sample_id")] = stage
    return {"digests": digests, "rebuilt": rebuilt, "stages": stages}


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact binomial test on discordant pairs (p0 = 0.5)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def minimum_detectable() -> str:
    """Smallest discordant pattern that clears p < 0.05 with fewest flips."""
    for total in range(2, 40):
        for wins in range(total + 1):
            losses = total - wins
            if wins > losses and mcnemar_exact(wins, losses) < 0.05:
                return f"{wins} 胜 {losses} 负（n={total}）"
    return "n/a"


def usage_totals(rows: dict) -> dict:
    totals = Counter()
    for row in rows.values():
        calls = row.get("model_calls") or {}
        totals["input_tokens"] += calls.get("input_tokens", 0) or 0
        totals["output_tokens"] += calls.get("output_tokens", 0) or 0
        totals["cost_rmb"] += calls.get("cost_rmb", 0.0) or 0.0
        totals["questions"] += 1
    return totals


def pct(new: float, old: float) -> str:
    if not old:
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True, help="baseline run directory")
    parser.add_argument("--run-b", type=Path, required=True, help="candidate run directory")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--system", default="MemNexus")
    parser.add_argument("--max-token-increase", type=float, default=2.0,
                        help="问答 token 增幅上限（百分比），超过则判为有条件通过")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    run_a, run_b = args.run_a.resolve(), args.run_b.resolve()
    rows_a, _ = load_rows(run_a, args.system)
    rows_b, _ = load_rows(run_b, args.system)
    cond_a, cond_b = replay_condition(run_a), replay_condition(run_b)

    print(f"A = {args.label_a}  {run_a.name}")
    print(f"B = {args.label_b}  {run_b.name}")
    print()

    # --- gate 1: did anything crash? Accuracy read off a crashed run is noise.
    crashes_a = [k for k, r in rows_a.items() if r.get("forced_abstain")]
    crashes_b = [k for k, r in rows_b.items() if r.get("forced_abstain")]
    print(f"崩溃/强制弃答：A={len(crashes_a)}  B={len(crashes_b)}")
    if crashes_a or crashes_b:
        print("  判定：INVALID —— 有题目在答题阶段崩过，准确率不可读。先修崩溃再比较。")
        return 1

    # --- gate 2: same frozen base?
    shared_digests = cond_a["digests"] & cond_b["digests"]
    paired = bool(shared_digests) and cond_a["digests"] == cond_b["digests"]
    print(f"记忆库快照：A={len(cond_a['digests'])} 份  B={len(cond_b['digests'])} 份  "
          f"共同={len(shared_digests)} 份")
    if cond_a["rebuilt"] or cond_b["rebuilt"]:
        print(f"  未回放的对话：A={len(cond_a['rebuilt'])}  B={len(cond_b['rebuilt'])}")
    if not paired:
        print("  判定：UNPAIRED —— 两侧不是同一个冻结基座，只能看方向，不能判显著。")

    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        print("没有可配对的题目。")
        return 1

    # --- accuracy, paired
    only_a = sum(1 for k in shared if rows_a[k].get("judge_result") == CORRECT)
    only_b = sum(1 for k in shared if rows_b[k].get("judge_result") == CORRECT)
    wins = [k for k in shared
            if rows_a[k].get("judge_result") != CORRECT and rows_b[k].get("judge_result") == CORRECT]
    losses = [k for k in shared
              if rows_a[k].get("judge_result") == CORRECT and rows_b[k].get("judge_result") != CORRECT]
    p_value = mcnemar_exact(len(wins), len(losses))
    print()
    print(f"配对题数 {len(shared)}")
    print(f"正确率   A={only_a}/{len(shared)} ({only_a/len(shared)*100:.1f}%)  "
          f"B={only_b}/{len(shared)} ({only_b/len(shared)*100:.1f}%)  "
          f"净 {only_b-only_a:+d}")
    print(f"配对翻转 A错B对={len(wins)}  A对B错={len(losses)}  "
          f"McNemar 精确 p={p_value:.4f}")
    print(f"最小可检测模式：{minimum_detectable()}")

    # --- per-category
    categories = sorted({rows_a[k]["category"] for k in shared})
    print()
    print(f"{'题型':<12}{'n':>4}{'A':>5}{'B':>5}{'净':>5}{'胜':>4}{'负':>4}")
    for category in categories:
        keys = [k for k in shared if rows_a[k]["category"] == category]
        a_ok = sum(1 for k in keys if rows_a[k].get("judge_result") == CORRECT)
        b_ok = sum(1 for k in keys if rows_b[k].get("judge_result") == CORRECT)
        w = sum(1 for k in keys if k in set(wins))
        l = sum(1 for k in keys if k in set(losses))
        print(f"{category:<12}{len(keys):>4}{a_ok:>5}{b_ok:>5}{b_ok-a_ok:>+5}{w:>4}{l:>4}")

    # --- cost
    ua, ub = usage_totals(rows_a), usage_totals(rows_b)
    qa = ua.get("input_tokens", 0) + ua.get("output_tokens", 0)
    qb = ub.get("input_tokens", 0) + ub.get("output_tokens", 0)
    per_q_a = qa / max(ua.get("questions", 1), 1)
    per_q_b = qb / max(ub.get("questions", 1), 1)
    print()
    print(f"问答 token  A={qa} ({per_q_a:.0f}/题)  B={qb} ({per_q_b:.0f}/题)  {pct(qb, qa)}")
    print(f"问答成本    A=¥{ua.get('cost_rmb', 0):.4f}  B=¥{ub.get('cost_rmb', 0):.4f}  "
          f"{pct(ub.get('cost_rmb', 0), ua.get('cost_rmb', 0))}")

    # --- verdict, per the pre-registered rule
    token_delta = (qb - qa) / qa * 100 if qa else 0.0
    accuracy_delta = only_b - only_a
    if not paired:
        verdict = "UNPAIRED：只能记方向，不得写进结论"
        accepted = False
    elif accuracy_delta < 0:
        verdict = "否决：净正确数下降"
        accepted = False
    elif p_value >= 0.05:
        verdict = "不可判定：与噪声无法区分（p>=0.05），不得当作收益"
        accepted = False
    elif token_delta > args.max_token_increase:
        verdict = f"有条件通过：准确率提升显著，但 token {token_delta:+.1f}% 超过上限，需按主张裁决取舍"
        accepted = True
    else:
        verdict = f"通过：准确率提升且 token {token_delta:+.1f}% 在允许范围内"
        accepted = True

    print()
    print(f"判定：{verdict}")
    print()
    print("A 输给 B 的题：")
    for key in [k for k in wins][:5]:
        print(f"  [{rows_a[key]['category']}] {key[1][:70]}")
        print(f"    A: {str(rows_a[key].get('prediction'))[:90]}")
        print(f"    B: {str(rows_b[key].get('prediction'))[:90]}")
    if losses:
        print("B 输给 A 的题：")
        for key in losses[:5]:
            print(f"  [{rows_a[key]['category']}] {key[1][:70]}")
            print(f"    A: {str(rows_a[key].get('prediction'))[:90]}")
            print(f"    B: {str(rows_b[key].get('prediction'))[:90]}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "run_a": str(run_a), "run_b": str(run_b),
            "paired": paired, "shared": len(shared),
            "accuracy_a": only_a, "accuracy_b": only_b,
            "wins": len(wins), "losses": len(losses),
            "mcnemar_p": p_value,
            "token_a": qa, "token_b": qb, "token_delta_pct": token_delta,
            "cost_a_rmb": ua.get("cost_rmb", 0.0), "cost_b_rmb": ub.get("cost_rmb", 0.0),
            "verdict": verdict, "accepted": accepted,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {args.json_out}")

    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
