"""读出可信度的门禁：同一份冻结记忆库，跑两遍，答案必须逐题相同。

为什么这件事必须先做：三次同配置 150 题运行，MemNexus 的逐题判定翻转了 14~18 题，
而任何待测优化的效应只有 ±2~4 题。也就是说在证明"读数可复现"之前，
所有 before/after 差值都不具备判读价值（见 experiments/OPTIMIZATION_PROTOCOL.md §7）。

两段式，自己给自己设门禁，不需要人守着：

  阶段 1（接线冒烟） 3 题 × 2 遍，约 ¥0.15、3 分钟。
      验证：快照命中、建库花费为 0、两次 sha256 相同、无异常。
      不过就中止，不烧钱。
  阶段 2（定量） 整段对话全部题 × 2 遍，约 ¥1.3、25 分钟。
      测量：逐题预测完全一致的比例。这是后续所有 A/B 的噪声地板。

用法::

    python experiments/gate_replay_determinism.py --sample conv-26
    python experiments/gate_replay_determinism.py --sample conv-26 --skip-smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json"
COHORT = ROOT / "result" / "locomo_150_cohort_20261018.json"
SNAPSHOTS = ROOT / "result" / "memnexus_snapshots" / "main"

sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "MemNexus" / "memory-system" / "src"))
sys.path.insert(0, str(ROOT / "SimpleMem"))


def configure_condition() -> None:
    """把实验条件钉成 150 题论文条件，避免"跑到一半发现少设一个环境变量"。

    条件与 run 20261020_000448_049755（MN 114 / SM 116）一致：
    关闭语义执行路径、价格 2.4/9.6、截断上限 8192。
    """
    from agent_memory.settings import Settings  # noqa: F401  导入即加载私有 .env

    os.environ["MEMORY_SEMANTIC_EXECUTION"] = "false"
    os.environ["MEMORY_INPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "2.4"
    os.environ["MEMORY_OUTPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "9.6"
    os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "8192"


def load_questions(sample_id: str, limit: int | None) -> tuple[dict, list[dict], int]:
    """按 150 题队列取某段对话的题，返回 (官方样本, 原始 qa 列表, max_session)。"""
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    items = [q for q in cohort["questions"] if q["sample_id"] == sample_id]
    if not items:
        raise ValueError(f"题组里没有 {sample_id}")
    items.sort(key=lambda q: q["qa_index"])
    if limit:
        items = items[:limit]
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    sample = next(row for row in dataset if str(row.get("sample_id")) == sample_id)
    questions = [sample["qa"][item["qa_index"]] for item in items]
    return sample, questions, cohort["max_session"]


async def one_pass(sample, questions, max_session, out_dir: Path, tag: str) -> tuple[dict, list[dict]]:
    from run_locomo_pilot import run_memnexus

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage, rows = await run_memnexus(
        sample,
        questions,
        8.0,
        usage_ledger=out_dir / "usage.json",
        max_session=max_session,
        snapshot_dir=SNAPSHOTS,
        require_snapshot=True,
    )
    elapsed = time.perf_counter() - started
    print(
        f"  [{tag}] 回放={stage['memory_base_replayed']} "
        f"记忆={stage['memory_count']} "
        f"建库=¥{stage['build_usage'].get('cost_rmb', 0):.4f} "
        f"答题=¥{stage['total_usage'].get('cost_rmb', 0) - stage['build_usage'].get('cost_rmb', 0):.4f} "
        f"sha={(stage['memory_snapshot_sha256'] or '')[:12]} "
        f"{elapsed / 60:.1f}min",
        flush=True,
    )
    return stage, rows


def compare(a_stage, a_rows, b_stage, b_rows) -> dict:
    problems: list[str] = []
    if not a_stage.get("memory_base_replayed") or not b_stage.get("memory_base_replayed"):
        problems.append("有遍次没有回放快照（等于重新建库，读数不可比）")
    if a_stage.get("memory_snapshot_sha256") != b_stage.get("memory_snapshot_sha256"):
        problems.append("两遍用的不是同一份快照")
    for tag, stage in (("第一遍", a_stage), ("第二遍", b_stage)):
        if stage["build_usage"].get("cost_rmb", 0) != 0:
            problems.append(f"{tag}回放仍产生建库花费 ¥{stage['build_usage']['cost_rmb']:.4f}")
        if stage.get("memory_count", 0) <= 0:
            problems.append(f"{tag}记忆条数为 0")
        if stage.get("failed_windows", 0):
            problems.append(f"{tag}建库有失败窗口")
    if len(a_rows) != len(b_rows):
        problems.append(f"两遍题数不同 {len(a_rows)} vs {len(b_rows)}")

    details = []
    identical = 0
    for index, (left, right) in enumerate(zip(a_rows, b_rows), 1):
        same = left["prediction"].strip() == right["prediction"].strip()
        identical += int(same)
        details.append(
            {
                "n": index,
                "category": left["category"],
                "question": left["question"],
                "gold": left["gold_answer"],
                "pass1": left["prediction"],
                "pass2": right["prediction"],
                "identical": same,
                "retrieval_1": left["retrieval_count"],
                "retrieval_2": right["retrieval_count"],
                "f1_1": round(left["f1"], 4),
                "f1_2": round(right["f1"], 4),
            }
        )
    return {
        "questions": len(details),
        "identical": identical,
        "identical_rate": round(identical / max(len(details), 1), 4),
        "problems": problems,
        "details": details,
        "cost_rmb": round(
            (a_stage["total_usage"].get("cost_rmb", 0) or 0)
            + (b_stage["total_usage"].get("cost_rmb", 0) or 0),
            4,
        ),
    }


def report(result: dict, path: Path, title: str) -> None:
    lines = [f"# {title}", "", f"- 题数：{result['questions']}",
             f"- 逐题答案完全一致：{result['identical']}/{result['questions']} "
             f"({result['identical_rate'] * 100:.1f}%)",
             f"- 两遍总花费：¥{result['cost_rmb']:.4f}", ""]
    if result["problems"]:
        lines += ["## 接线问题（必须修）", ""] + [f"- {p}" for p in result["problems"]] + [""]
    diff = [row for row in result["details"] if not row["identical"]]
    lines += ["## 不一致的题", ""]
    if not diff:
        lines += ["无。", ""]
    for row in diff:
        lines += [
            f"### Q{row['n']} [{row['category']}] {row['question']}",
            f"- gold: {row['gold']}",
            f"- 第一遍: {row['pass1']}",
            f"- 第二遍: {row['pass2']}",
            f"- 证据数: {row['retrieval_1']} vs {row['retrieval_2']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="conv-26")
    parser.add_argument("--smoke-questions", type=int, default=3)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    configure_condition()
    from agent_memory import snapshot as memory_snapshot

    if not memory_snapshot.available(SNAPSHOTS, args.sample):
        print(f"失败：{SNAPSHOTS} 里没有 {args.sample} 的快照。先建库或改路径。", flush=True)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or ROOT / "result" / f"gate_replay_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{out_dir}", flush=True)

    if not args.skip_smoke:
        print(f"\n== 阶段 1 接线冒烟：{args.sample} 前 {args.smoke_questions} 题 × 2 遍 ==", flush=True)
        sample, questions, max_session = load_questions(args.sample, args.smoke_questions)
        stage_a, rows_a = asyncio.run(one_pass(sample, questions, max_session, out_dir / "smoke1", "smoke-1"))
        stage_b, rows_b = asyncio.run(one_pass(sample, questions, max_session, out_dir / "smoke2", "smoke-2"))
        smoke = compare(stage_a, rows_a, stage_b, rows_b)
        report(smoke, out_dir / "SMOKE.md", "接线冒烟")
        if smoke["problems"]:
            print("阶段 1 失败：接线有问题，停止，不进入阶段 2。", flush=True)
            for p in smoke["problems"]:
                print("  -", p, flush=True)
            return 1
        print(f"阶段 1 通过：接线正常，{smoke['identical']}/{smoke['questions']} 一致", flush=True)

    print(f"\n== 阶段 2 定量：{args.sample} 全部题 × 2 遍 ==", flush=True)
    sample, questions, max_session = load_questions(args.sample, None)
    print(f"题数 {len(questions)}，预计 ¥{len(questions) * 0.045:.2f}", flush=True)
    stage_a, rows_a = asyncio.run(one_pass(sample, questions, max_session, out_dir / "full1", "full-1"))
    stage_b, rows_b = asyncio.run(one_pass(sample, questions, max_session, out_dir / "full2", "full-2"))
    full = compare(stage_a, rows_a, stage_b, rows_b)
    report(full, out_dir / "REPORT.md", f"{args.sample} 冻结基座问答确定性")
    (out_dir / "report.json").write_text(
        json.dumps({"full": full, "snapshot_sha256": stage_a.get("memory_snapshot_sha256")},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print("\n== 结果 ==", flush=True)
    print(f"逐题一致：{full['identical']}/{full['questions']} ({full['identical_rate'] * 100:.1f}%)")
    print(f"总花费：¥{full['cost_rmb']:.4f}")
    if full["problems"]:
        for p in full["problems"]:
            print("  ! ", p)
        return 1
    if full["identical"] == full["questions"]:
        print("判定：PASS —— 冻结基座下问答可复现，后续 A/B 差值可判读。", flush=True)
        return 0
    print("判定：PARTIAL —— 冻结只消掉了建库噪声，问答本身仍有随机性。"
          "后续必须 N 次重复取均值，不得读单次差值。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
