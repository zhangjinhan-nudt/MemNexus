"""量化问答阶段的噪声地板：同一冻结基座上重复 R 次，看判定到底翻不翻。

门禁结果（result/gate_replay_20261020_150249）已经证明两件事：
  1. 快照机制可用：建库 ¥0、sha256 一致、73 条记忆可回放；
  2. 但问答不可复现：29 题两遍，文本 25/29 一致，证据集 21/29 一致。

现在要回答的是"这个不一致有多大"——不是文本差异率，而是**判定翻不翻**。
因为 A/B 比较的是正确率，文本换个说法但判定不变，不影响读数。

做法：在已有 2 遍的基础上补跑到 R 遍，逐条判分，按题统计"R 遍里对了几遍"。
输出三个数，它们决定后续实验能不能读出 ±2~4 题：
  - 稳定对 / 稳定错 / 翻转题 各多少道
  - 单次运行的准确率波动区间（min~max across passes）
  - 最小可判读效应量 = 翻转题占比

用法::

    python experiments/measure_qa_noise_floor.py --base-dir result/gate_replay_20261020_150249 --passes 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "MemNexus" / "memory-system" / "src"))
sys.path.insert(0, str(ROOT / "SimpleMem"))

from gate_replay_determinism import (  # noqa: E402
    SNAPSHOTS,
    configure_condition,
    load_questions,
    one_pass,
)


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # A resumed pass may append the same question twice; keep the last write.
    by_question: dict[str, dict] = {}
    for row in rows:
        by_question[row["question"]] = row
    return list(by_question.values())


def complete_passes(base: Path, total: int) -> dict[int, Path]:
    """Pass folders that actually finished. A crashed pass leaves a short file
    and must be re-run, never silently mixed into the noise estimate."""
    found: dict[int, Path] = {}
    for folder in base.glob("full*"):
        number = "".join(ch for ch in folder.name if ch.isdigit())
        ledger = folder / "usage.rows.jsonl"
        if not number or not ledger.is_file():
            continue
        if len(read_rows(ledger)) == total:
            found[int(number)] = folder
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--sample", default="conv-26")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--judge-budget", type=float, default=8.0)
    args = parser.parse_args()

    configure_condition()
    base = args.base_dir
    sample, questions, max_session = load_questions(args.sample, None)
    total = len(questions)
    index_of = {q["question"]: i for i, q in enumerate(questions)}

    finished = complete_passes(base, total)
    for number, folder in sorted(finished.items()):
        if number > args.passes:
            continue
        print(f"复用第 {number} 遍：{folder.name}", flush=True)

    for number in range(1, args.passes + 1):
        if number in finished:
            continue
        target = base / f"full{number}"
        if target.exists():
            for stale in target.glob("usage.rows.jsonl"):
                stale.unlink()
        print(f"跑第 {number} 遍…", flush=True)
        asyncio.run(one_pass(sample, questions, max_session, target, f"full-{number}"))

    passes = [complete_passes(base, total)[number] for number in range(1, args.passes + 1)]

    from run_locomo_pilot import judge_rows

    judged: list[dict] = []
    for folder in passes:
        rows = read_rows(folder / "usage.rows.jsonl")
        for row in rows:
            row.setdefault("qa_index", index_of.get(row["question"], -1))
        missing = [row for row in rows if row.get("judge_result") is None]
        if missing:
            print(f"判分 {folder.name}：{len(missing)} 条", flush=True)
            judge_rows(missing, args.judge_budget, usage_ledger=folder / "__judge.json")
        judged.append({row["qa_index"]: row for row in rows})

    common = set(judged[0])
    for table in judged[1:]:
        common &= set(table)
    common = sorted(common)

    per_question = []
    for qa_index in common:
        sample_row = judged[0][qa_index]
        verdicts = [table[qa_index].get("judge_result") == "CORRECT" for table in judged]
        predictions = [table[qa_index]["prediction"].strip() for table in judged]
        retrievals = [
            tuple(str(s.get("chunk_id")) + "|" + str(s.get("content"))[:60] for s in table[qa_index]["retrieved_sources"])
            for table in judged
        ]
        per_question.append(
            {
                "qa_index": qa_index,
                "category": sample_row["category"],
                "question": sample_row["question"],
                "gold": sample_row["gold_answer"],
                "correct_count": sum(verdicts),
                "passes": len(verdicts),
                "distinct_predictions": len(set(predictions)),
                "distinct_retrievals": len(set(retrievals)),
                "prediction": predictions[0],
                "f1": [round(float(table[qa_index].get("f1") or 0), 3) for table in judged],
            }
        )

    stable_correct = [r for r in per_question if r["correct_count"] == r["passes"]]
    stable_wrong = [r for r in per_question if r["correct_count"] == 0]
    flips = [r for r in per_question if 0 < r["correct_count"] < r["passes"]]
    per_pass_accuracy = [
        sum(table[index].get("judge_result") == "CORRECT" for index in common) / len(common)
        for table in judged
    ]

    print()
    print(f"题数 {len(common)}  遍数 {len(passes)}")
    print(f"稳定对 {len(stable_correct)}  稳定错 {len(stable_wrong)}  翻转 {len(flips)}")
    print(f"单次准确率：{', '.join(f'{value:.3f}' for value in per_pass_accuracy)}")
    print(f"  区间 {min(per_pass_accuracy):.3f}~{max(per_pass_accuracy):.3f} "
          f"跨度 {max(per_pass_accuracy) - min(per_pass_accuracy):.3f}")
    if flips:
        print(f"\n翻转题（{len(flips)} 道）：")
        for row in flips:
            print(f"  Q{row['qa_index']} [{row['category']}] {row['question'][:56]}")
            print(f"     对 {row['correct_count']}/{row['passes']}  "
                  f"文本变体 {row['distinct_predictions']}  证据集变体 {row['distinct_retrievals']}  "
                  f"F1 {row['f1']}")

    report = {
        "questions": len(common),
        "passes": len(passes),
        "stable_correct": len(stable_correct),
        "stable_wrong": len(stable_wrong),
        "flipping": len(flips),
        "per_pass_accuracy": [round(value, 4) for value in per_pass_accuracy],
        "accuracy_span": round(max(per_pass_accuracy) - min(per_pass_accuracy), 4),
        "details": per_question,
    }
    (base / "NOISE_FLOOR.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# {args.sample} 问答噪声地板（{len(passes)} 遍 × {len(common)} 题）",
        "",
        f"- 稳定答对：{len(stable_correct)}",
        f"- 稳定答错：{len(stable_wrong)}",
        f"- 判定翻转：{len(flips)}（{len(flips) / max(len(common), 1) * 100:.1f}%）",
        f"- 单次准确率序列：{', '.join(f'{value:.3f}' for value in per_pass_accuracy)}",
        f"- 跨度：{max(per_pass_accuracy) - min(per_pass_accuracy):.3f}",
        "",
        "## 翻转题",
        "",
    ]
    for row in flips:
        lines += [
            f"### Q{row['qa_index']} [{row['category']}] {row['question']}",
            f"- gold：{row['gold']}",
            f"- 答对次数：{row['correct_count']}/{row['passes']}",
            f"- 文本变体 {row['distinct_predictions']} 种，证据集变体 {row['distinct_retrievals']} 种",
            f"- F1 序列：{row['f1']}",
            f"- 首遍回答：{row['prediction'][:180]}",
            "",
        ]
    (base / "NOISE_FLOOR.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n写入 {base / 'NOISE_FLOOR.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
