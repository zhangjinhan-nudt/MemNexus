"""构造"目标集分层子集" cohort，用于阶段 B 的廉价 A/B 迭代。

为什么需要它
------------
阶段 B 要在小 cohort 上反复迭代 E5/E3/P2/E6。随机子集对"针对特定错因的改动"
几乎无效——目标题可能一道都没抽到，等于用不含病灶的样本测药。因此子集必须：
  1. **强制包含全部目标题**（A1 复核判为 fixable 的题）；
  2. 再按类别分层补足**对照题**（非目标题），用于监测改动是否误伤原本答对的题；
  3. 所有题的 (sample_id, qa_index) 必须来自本轮 1000 题 cohort，这样才能复用
     已跑好的 SimpleMem 基线（配合 run_parallel_bench.py --only-system MemNexus），
     每轮省一半成本与时间，并消除网关漂移带来的混杂。

口径纪律
--------
- selection_sha256 与主 cohort 同算法（对 (sample_id, qa_index) 列表求 SHA256），
  保证 benchmark100/finalize 的哈希校验能过。
- verdict 默认取 CSV 的 human_verdict（人工复核结论）。未复核时 human_verdict 为空，
  脚本**拒绝生成**，除非显式传 --use-rule-suggestion，且此时 cohort 会被标记
  partial_review=true，protocol 里写明基于规则初判，禁止当正式迭代依据。
- 只读输入，只写新 cohort 文件；不改任何结果或算法代码。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

FIXABLE = "fixable"
VALID_VERDICTS = {"fixable", "not_fixable", "uncertain"}


def load_pool(cohort_path: Path) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    """读取 1000 题母 cohort，返回 (cohort 头信息, key→question)。"""
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    questions = cohort["questions"]
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for question in questions:
        key = (str(question["sample_id"]), int(question["qa_index"]))
        if key in index:
            raise ValueError(f"母 cohort 存在重复键：{key}")
        index[key] = question
    return cohort, index


def load_target_keys(csv_path: Path, verdict_column: str,
                     experiments: set[str] | None) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """从复核 CSV 读取目标题键，返回 (keys, 诊断信息)。

    verdict_column: "human_verdict"（正式）或 "suggested_verdict"（规则初判，需显式开启）。
    experiments: 若给定，只纳入这些 target_experiment 的题（如只跑 E5）。
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV 为空：{csv_path}")
    if verdict_column not in rows[0]:
        raise ValueError(f"CSV 缺少列 {verdict_column}；实际列：{list(rows[0])}")

    keys: list[tuple[str, int]] = []
    verdict_counter: Counter[str] = Counter()
    experiment_counter: Counter[str] = Counter()
    for row in rows:
        verdict = (row.get(verdict_column) or "").strip().lower()
        verdict_counter[verdict or "(空)"] += 1
        if verdict != FIXABLE:
            continue
        target_experiment = (row.get("target_experiment") or "").strip()
        if experiments and target_experiment not in experiments:
            continue
        keys.append((str(row["sample_id"]), int(row["qa_index"])))
        experiment_counter[target_experiment] += 1
    diagnostics = {
        "verdict_column": verdict_column,
        "verdict_distribution": dict(verdict_counter),
        "target_keys": len(keys),
        "by_experiment": dict(experiment_counter),
        "unique_keys": len(set(keys)),
    }
    if len(keys) != len(set(keys)):
        raise ValueError("目标键存在重复")
    return keys, diagnostics


def load_keys_file(keys_path: Path) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """读取 evidence_recall_audit 产出的目标集文件（如 e4_target_keys.json）。

    与 CSV 目标源的本质区别：keys 文件来自**机械归因**——金标 not/partial 召回
    且建库可达（三层判据，快照直接验证），是事实性判定而非价值判断，因此
    不要求人工复核、不触发 partial_review 标记。CSV 的 fixable/not_fixable
    才是价值判断（改了是否算过拟合），必须有人的裁决。

    文件格式（evidence_recall_audit.py 产出）：
        {"n": int, "keys": ["sample_id/qa_index", ...], "definition": str}
    n 与 keys 数不符或 key 格式非法都直接报错——目标集错了整轮 A/B 白跑。
    """
    payload = json.loads(keys_path.read_text(encoding="utf-8"))
    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError(f"keys 文件无有效 keys 列表：{keys_path}")
    keys: list[tuple[str, int]] = []
    for raw in raw_keys:
        text = str(raw)
        sample_id, sep, qa_text = text.rpartition("/")
        if not sep or not sample_id or not qa_text.isdigit():
            raise ValueError(f"key 格式非法（应为 sample_id/qa_index）：{text!r}")
        keys.append((sample_id, int(qa_text)))
    if len(keys) != len(set(keys)):
        raise ValueError("keys 文件存在重复键")
    declared = payload.get("n")
    if isinstance(declared, int) and declared != len(keys):
        raise ValueError(f"声明的 n={declared} 与实际 keys 数 {len(keys)} 不符")
    diagnostics = {
        "keys_file": str(keys_path),
        "definition": payload.get("definition"),
        "generated_at": payload.get("generated_at"),
        "target_keys": len(keys),
    }
    return keys, diagnostics


def pick_controls(pool: dict[tuple[str, int], dict[str, Any]],
                  target_keys: set[tuple[str, int]],
                  need: int, seed: int) -> list[tuple[str, int]]:
    """从非目标题里按类别分层、确定性抽取对照题，补到 need 个。

    分层按类别比例分配，类别内用固定 seed 打乱后轮转，保证可复现且不偏袒某对话。
    对照题的作用：监测改动是否误伤原本答对的题（回归护栏）。
    """
    if need <= 0:
        return []
    candidates = [key for key in pool if key not in target_keys]
    by_category: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in candidates:
        by_category[pool[key]["category"]].append(key)

    rng = random.Random(seed)
    for category in by_category:
        by_category[category].sort()          # 先排序，保证与字典顺序无关
        rng.shuffle(by_category[category])    # 再用固定 seed 打乱，可复现

    total_candidates = len(candidates)
    # 按类别在候选池中的占比分配名额（向下取整），余额按剩余量最大者优先补。
    quotas = {cat: int(need * len(keys) / total_candidates) for cat, keys in by_category.items()}
    for cat in sorted(by_category, key=lambda c: -len(by_category[c])):
        if sum(quotas.values()) >= need:
            break
        quotas[cat] += 1
    while sum(quotas.values()) > need:
        cat = max(quotas, key=lambda c: (quotas[c], -len(by_category[c])))
        quotas[cat] -= 1

    selected: list[tuple[str, int]] = []
    cursors = {cat: 0 for cat in by_category}
    for cat, quota in quotas.items():
        bucket = by_category[cat]
        take = min(quota, len(bucket))
        selected.extend(bucket[:take])
        cursors[cat] = take
    # 若某些类别题不够导致总数不足，从其余类别的剩余题轮转补足。
    if len(selected) < need:
        remaining_pool = [
            key for cat in sorted(by_category) for key in by_category[cat][cursors[cat]:]
        ]
        rng.shuffle(remaining_pool)
        for key in remaining_pool:
            if len(selected) >= need:
                break
            if key not in selected:
                selected.append(key)
    return selected[:need]


def adversarial_guard_keys(pool: dict[tuple[str, int], dict[str, Any]],
                           exclude: set[tuple[str, int]], need: int) -> list[tuple[str, int]]:
    """按对话轮转抽取对抗题，作为 AdvSR 守护护栏。

    为什么必须有：P2（放松拒答判定）的一票否决指标是 AdvSR，而对抗题正是靠拒答得分。
    若子集里对抗题太少（随机分层可能只剩个位数），AdvSR 回归根本测不出来，
    P2 的安全护栏形同虚设——一个打掉 AdvSR 的坏变体会在子集阶段被放过，
    直到阶段 C 全量验证才暴露，白白浪费 15h。

    确定性：按 (sample_id, qa_index) 排序后跨对话轮转，不依赖 RNG，可复现。
    """
    candidates = sorted(
        key for key, question in pool.items()
        if question["category"] == "adversarial" and key not in exclude
    )
    if need >= len(candidates):
        return candidates
    by_conversation: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in candidates:
        by_conversation[key[0]].append(key)
    order = sorted(by_conversation)
    selected: list[tuple[str, int]] = []
    cursor = 0
    while len(selected) < need:
        progressed = False
        for offset in range(len(order)):
            position = (cursor + offset) % len(order)
            queue = by_conversation[order[position]]
            if queue:
                selected.append(queue.pop(0))
                cursor = (position + 1) % len(order)
                progressed = True
                break
        if not progressed:
            break
    return selected[:need]


def build_subset(pool_cohort: dict[str, Any], pool: dict[tuple[str, int], dict[str, Any]],
                 target_keys: list[tuple[str, int]], total: int, seed: int,
                 verdict_column: str, experiments: set[str] | None,
                 diagnostics: dict[str, Any], adversarial_guard: int) -> dict[str, Any]:
    target_set = set(target_keys)
    missing_in_pool = [key for key in target_keys if key not in pool]
    if missing_in_pool:
        raise ValueError(
            f"目标题不在母 cohort 中（无法复用 SM 基线）：{missing_in_pool[:5]}"
        )

    # AdvSR 守护层：目标题里已有的对抗题计入配额，不足部分从母 cohort 补。
    # 没有这层，P2（放松拒答）的一票否决指标在小 cohort 上根本测不出来。
    adversarial_in_targets = {key for key in target_set if pool[key]["category"] == "adversarial"}
    guard_need = max(0, adversarial_guard - len(adversarial_in_targets))
    guard_keys = adversarial_guard_keys(pool, exclude=target_set, need=guard_need)
    guard_set = set(guard_keys)

    reserved = target_set | guard_set
    remaining_budget = total - len(reserved)
    if remaining_budget > 0:
        controls = pick_controls(pool, reserved, remaining_budget, seed)
        overflow = 0
    else:
        # 目标题+守护题已超出目标规模：不裁剪它们，据实生成并告警。
        controls = []
        overflow = -remaining_budget

    chosen_keys = sorted(reserved | set(controls),
                         key=lambda key: (pool[key]["sample_index"], key[1]))
    questions = [pool[key] for key in chosen_keys]

    signature = hashlib.sha256(
        json.dumps([(q["sample_id"], q["qa_index"]) for q in questions],
                   separators=(",", ":")).encode()
    ).hexdigest()

    adversarial_total = sum(1 for q in questions if q["category"] == "adversarial")
    # keys_file 是事实性机械归因（金标未进 prompt 且建库可达，快照直接验证），
    # 不是"未经人工复核的价值判断"，不标 partial；suggested_verdict 才标。
    partial = verdict_column not in ("human_verdict", "keys_file")
    keys_file_info = diagnostics.get("keys_file")
    source_note = (
        f"; keys-file source = {keys_file_info['keys_file']}"
        if keys_file_info else ""
    )
    subset = dict(pool_cohort)  # 继承 dataset / dataset_sha256 / max_session
    subset.update({
        "protocol": (
            f"targeted {len(questions)}-question subset of the 1000-question cohort; "
            f"force-include all '{FIXABLE}' target questions"
            + (f" for experiments {sorted(experiments)}" if experiments else "")
            + f"; adversarial guard quota {adversarial_guard} for AdvSR monitoring; "
            f"stratified control fill with seed {seed}; "
            f"verdict source = {verdict_column}"
            + source_note
            + ("（规则初判，未经人工复核，仅供流程验证，禁止当正式迭代依据）" if partial else "（人工复核或事实性机械归因）")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "partial_review": partial,
        "target_experiment_filter": sorted(experiments) if experiments else None,
        "question_count": len(questions),
        "conversation_count": len({q["sample_id"] for q in questions}),
        "categories": dict(Counter(q["category"] for q in questions)),
        "per_conversation": dict(Counter(q["sample_id"] for q in questions)),
        "target_questions": len(target_set),
        "adversarial_guard_requested": adversarial_guard,
        "adversarial_guard_added": len(guard_set),
        "adversarial_in_targets": len(adversarial_in_targets),
        "adversarial_total": adversarial_total,
        "control_questions": len(controls),
        "overflow_beyond_total": overflow,
        "target_diagnostics": diagnostics,
        "selection_sha256": signature,
        "questions": questions,
    })
    return subset


def self_check(subset: dict[str, Any], pool: dict[tuple[str, int], dict[str, Any]],
               target_keys: list[tuple[str, int]]) -> list[str]:
    """生成后自检：返回失败原因列表，空列表表示通过。"""
    problems: list[str] = []
    questions = subset["questions"]
    keys = [(q["sample_id"], q["qa_index"]) for q in questions]

    if len(keys) != len(set(keys)):
        problems.append("子集存在重复键")
    if subset["question_count"] != len(questions):
        problems.append("question_count 与实际题数不符")

    # 所有题必须来自母 cohort（否则无法复用 SM 基线）
    not_in_pool = [key for key in keys if key not in pool]
    if not_in_pool:
        problems.append(f"{len(not_in_pool)} 题不在母 cohort：{not_in_pool[:5]}")

    # 目标题必须全部纳入
    missing_targets = [key for key in target_keys if key not in set(keys)]
    if missing_targets:
        problems.append(f"{len(missing_targets)} 个目标题未纳入：{missing_targets[:5]}")

    # AdvSR 守护：对抗题数必须达到配额，否则 P2 的护栏失效。
    # 母 cohort 对抗题总量不足时按实际上限放行，但必须与声明一致。
    guard = subset["adversarial_guard_requested"]
    pool_adversarial = sum(1 for q in pool.values() if q["category"] == "adversarial")
    expected_guard = min(guard, pool_adversarial)
    if subset["adversarial_total"] < expected_guard:
        problems.append(
            f"对抗题守护不足：实际 {subset['adversarial_total']} < 要求 {expected_guard}"
            f"（AdvSR 回归将测不出来）"
        )

    # 哈希可复算
    recomputed = hashlib.sha256(
        json.dumps(keys, separators=(",", ":")).encode()
    ).hexdigest()
    if recomputed != subset["selection_sha256"]:
        problems.append("selection_sha256 不可复算，口径不一致")

    # 排序稳定（benchmark100 的校验依赖文件顺序重算哈希）
    if keys != sorted(keys, key=lambda k: (pool[k]["sample_index"], k[1])):
        problems.append("questions 未按 (sample_index, qa_index) 排序")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构造目标集分层子集 cohort（只读输入）")
    parser.add_argument("--pool-cohort", type=Path,
                        default=ROOT / "result" / "locomo_1000_cohort_20261023.json")
    parser.add_argument("--review-csv", type=Path,
                        default=ROOT / "result" / "OUTPUT_LAYER_LOSSES_20261109" / "output_layer_losses.csv")
    parser.add_argument("--keys-file", type=Path, default=None,
                        help="evidence_recall_audit 产出的目标集文件（如 e4_target_keys.json）。"
                             "来自事实性机械归因（三层判据），不需要人工复核；"
                             "与 --review-csv 可同时给出（并集），此时 CSV 仍是 verdict 口径。")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "result" / "locomo_targeted_subset_cohort.json")
    parser.add_argument("--total", type=int, default=150, help="子集目标规模（含目标题+守护题+对照题）")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--adversarial-guard", type=int, default=34,
                        help="对抗题最小额度，用于监测 AdvSR 回归（P2 的一票否决指标）。"
                             "母 cohort 对抗题占 22.5%%，150 题子集默认 34 题；"
                             "跑 P2 时建议调高到 50 以上以获得足够监测功效。")
    parser.add_argument("--experiments", default="",
                        help="逗号分隔的 target_experiment 过滤；留空=纳入全部 fixable 题")
    parser.add_argument("--use-rule-suggestion", action="store_true",
                        help="用规则初判 suggested_verdict 代替 human_verdict（仅供流程验证，产出标记 partial）")
    parser.add_argument("--force", action="store_true",
                        help="跳过'人工复核未完成'的拒绝（仅在 --use-rule-suggestion 下有意义）")
    args = parser.parse_args(argv)

    for path, label in ((args.pool_cohort, "pool-cohort"),):
        if not path.exists():
            print(f"[FAIL] {label} 不存在：{path}", file=sys.stderr)
            return 3
    if args.keys_file is not None and not args.keys_file.exists():
        print(f"[FAIL] keys-file 不存在：{args.keys_file}", file=sys.stderr)
        return 3

    experiments = {name.strip() for name in args.experiments.split(",") if name.strip()} or None
    pool_cohort, pool = load_pool(args.pool_cohort)

    # ---- 目标源合并：keys-file（事实归因）+ CSV（价值判断），并集去重保序 ----
    target_keys: list[tuple[str, int]] = []
    diagnostics: dict[str, Any] = {"sources": []}
    verdict_column: str | None = None  # CSV 的 verdict 口径；None = 未使用 CSV

    if args.keys_file is not None:
        keys_file_keys, keys_file_diag = load_keys_file(args.keys_file)
        target_keys.extend(keys_file_keys)
        diagnostics["keys_file"] = keys_file_diag
        diagnostics["sources"].append("keys_file")

    csv_available = args.review_csv.exists()
    if not target_keys and not csv_available:
        print(f"[FAIL] 目标源不可用：--keys-file 未提供且 {args.review_csv} 不存在",
              file=sys.stderr)
        return 3

    if csv_available:
        verdict_column = "suggested_verdict" if args.use_rule_suggestion else "human_verdict"
        csv_keys, csv_diag = load_target_keys(args.review_csv, verdict_column, experiments)
        # 正式模式（human_verdict）下复核未完成：CSV 是唯一来源时拒绝（避免空/半截
        # 结论误导）；keys-file 已提供目标题时降级为警告（CSV 零贡献不阻塞）。
        if verdict_column == "human_verdict":
            unfilled = csv_diag["verdict_distribution"].get("(空)", 0)
            if unfilled and not args.force:
                if not target_keys:
                    print(f"[FAIL] 人工复核未完成：{unfilled} 题 human_verdict 为空。", file=sys.stderr)
                    print("       完成复核后重跑；或加 --use-rule-suggestion 用规则初判做流程验证"
                          "（产出会标记 partial_review=true，禁止当正式依据）。", file=sys.stderr)
                    return 4
                print(f"[WARN] CSV 复核未完成（{unfilled} 题空值），本子集目标题全部来自 keys-file")
        if csv_keys:
            before = len(target_keys)
            target_keys = list(dict.fromkeys(target_keys + csv_keys))
            diagnostics["csv"] = csv_diag
            diagnostics["sources"].append("csv")
            print(f"[INFO] CSV 贡献目标题 {len(target_keys) - before} 个（verdict={verdict_column}）")
        elif target_keys:
            print(f"[WARN] CSV 贡献 0 个目标题（verdict={verdict_column}）；"
                  f"目标题全部来自 keys-file")

    if not target_keys:
        print("[FAIL] 没有 fixable 目标题；检查 verdict 列与 --experiments 过滤", file=sys.stderr)
        return 4

    # effective verdict 用于 build_subset 的 protocol/partial 判定：
    #   仅 keys-file（或 CSV 零贡献）→ "keys_file"（事实归因，不标 partial）
    #   CSV 有贡献 → CSV 的 verdict 口径（suggested_verdict 会如实标 partial）
    csv_contributed = "csv" in diagnostics["sources"]
    effective_verdict = (
        verdict_column if (verdict_column is not None and csv_contributed)
        else "keys_file"
    )

    subset = build_subset(pool_cohort, pool, target_keys, args.total, args.seed,
                          effective_verdict, experiments, diagnostics, args.adversarial_guard)
    problems = self_check(subset, pool, target_keys)
    if problems:
        print("[FAIL] 子集自检未通过：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 5

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(subset, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] 写出 {args.out}")
    print(f"     规模 {subset['question_count']}（目标题 {subset['target_questions']} + "
          f"守护对抗 {subset['adversarial_guard_added']} + 对照题 {subset['control_questions']}）；"
          f"对话 {subset['conversation_count']}")
    print(f"     类别 {subset['categories']}")
    print(f"     对抗题合计 {subset['adversarial_total']}（守护配额 {subset['adversarial_guard_requested']}）")
    print(f"     目标源 = {' + '.join(diagnostics['sources'])}；"
          f"verdict 口径 = {effective_verdict}；partial_review = {subset['partial_review']}")
    if subset["overflow_beyond_total"]:
        print(f"[WARN] 目标题已超过 --total，实际 {subset['question_count']} 题（未裁剪目标题）")
    if subset["control_questions"] == 0:
        print("[WARN] 对照题为 0：目标题+守护题已占满 --total。"
              "对照题是监测「改动误伤原本答对的题」的回归护栏，为 0 时无法发现回退。"
              "建议调高 --total，或用 --experiments 只纳入单个实验的目标题。")
    print(f"     selection_sha256 = {subset['selection_sha256'][:16]}…")
    if subset["partial_review"]:
        print("[WARN] 基于规则初判，仅供流程验证；正式迭代前必须完成人工复核并去掉 --use-rule-suggestion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
