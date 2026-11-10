"""证据级检索保真度审计（离线，零 API 调用）。

目的
----
把 MemNexus 的输题拆成两类病因，决定后续该改哪个模块：
  A. 金标证据根本没被召回  → 表示层/检索层问题（E1 双轨表示、E4 证据补全）
  B. 金标证据已召回但仍答错 → 生成/渲染/义务判定问题（E3 时间渲染、E5 槽位）

对应综述 arXiv:2606.24775 §4.2 / Finding 2：检索不是 top-1 排序问题，
而是"证据补全"问题。本脚本只读已有结果，不重跑任何模型调用。

为什么这样设计（都由实测决定，不是假设）
--------------------------------------
1. 金标证据 `gold_evidence` 形如 `D1:12`（对话号:轮次号），实测 1411/1411 可解析。
2. MN 的 `retrieved_sources[].metadata.source_events[].turn_sequence` 是**全局累计
   轮次序号**，实测与 `dia_id` 严格对齐（D1:12 → seq 12），34968/34968 事件全部
   可对回 dia_id，零失败。因此 MN 的召回判定是**精确 provenance**，不是词面猜测。
3. `retrieved_sources` 有 997/1000 行**不是**按 score 降序；它按视图分块排列
   （indexed_sweep / compressed_memory / raw_fallback / relation_index，1000/1000 行
   均为连续块）。且 score 跨视图**不可比**（compressed_memory 中位 0.8975 vs
   indexed_sweep 0.5235）。因此 Recall@K 一律按**存储顺序**（系统交给答案器的呈现
   顺序）计算，绝不做全局 score 重排。
4. SimpleMem 的 `retrieved_sources` 是改写后的合成记忆字符串，**没有 provenance**。
   用金标原文的词面 bigram 去匹配时，SM 的 exact-all 命中率为 **0.0%**，而 MN 为
   77.9%。这个差异来自"SM 改写、MN 保留原文"，**不是**召回能力差异。所以：
   SM 不产出可比 Recall；只输出一个**明确标注为有偏、不可用于跨系统比较**的词面
   包含度诊断。任何"SM 召回低"的说法都不被本脚本支持。
5. adversarial 的官方指标是 AdvSR（拒答成功率），召回保真度对它不适用，
   因此单列且排除在主召回指标之外。

口径纪律
--------
- 本脚本输出**诊断指标**，不进入论文主表。主表仍只用 paper_protocol 的四类等权
  Average F1 + AdvSR。
- Recall@K 的分母是该题全部金标 dia_id；多金标题按集合覆盖率计。
- 只读输入，不修改任何结果文件。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

SYSTEMS = ("MemNexus", "SimpleMem")
K_VALUES = (1, 3, 5, 10, 20)
GAP_BINS = ("1-5", "6-10", "11-15", "16-20", "21-25", "26+")
# 综述用 1-5..26-31 的 session gap 分桶；本项目 max gap 由数据决定，最后一桶开放。
DIA_RE = re.compile(r"^D(\d+):(\d+)$")
SESSION_RE = re.compile(r"^session_(\d+)$")
# 与 finalize_locomo1000 保持一致的数值容差，避免 win/tie/loss 判定漂移。
TOLERANCE = 1e-12
ADVERSARIAL = "adversarial"
# 配额扫描的候选值；cap=0 与"剔除该视图"等价，用于一致性自检。
QUOTA_CAPS = (0, 1, 2, 3, 4, 5, 6, 8, 11)
QUOTA_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# 数据集侧：dia_id ↔ 全局轮次序号 ↔ session
# ---------------------------------------------------------------------------

def build_dialog_maps(sample: dict) -> dict[str, Any]:
    """为一个对话建立 turn_sequence ↔ dia_id ↔ session 的双向映射。

    turn_sequence 是跨 session 的全局累计序号（从 1 开始），这一点已由实测
    确认；若数据集结构变化导致映射不唯一，必须让下面的断言失败而不是静默继续。
    """
    sample_id = str(sample["sample_id"])
    conversation = sample["conversation"]
    session_keys = sorted(
        (key for key in conversation if SESSION_RE.fullmatch(key)),
        key=lambda key: int(SESSION_RE.fullmatch(key).group(1)),
    )
    if not session_keys:
        raise ValueError(f"{sample_id}: 未找到 session_* 结构")

    seq_to_dia: dict[int, str] = {}
    dia_to_seq: dict[str, int] = {}
    dia_to_session: dict[str, int] = {}
    global_index = 0
    for position, session_key in enumerate(session_keys, start=1):
        for turn in conversation[session_key]:
            global_index += 1
            dia_id = turn.get("dia_id")
            if not dia_id:
                continue
            if global_index in seq_to_dia or dia_id in dia_to_seq:
                raise ValueError(f"{sample_id}: dia_id 映射冲突于 seq {global_index}")
            seq_to_dia[global_index] = dia_id
            dia_to_seq[dia_id] = global_index
            dia_to_session[dia_id] = position

    # 自检：dia_id 自带的 D{session}:{turn} 必须与遍历得到的 session 一致。
    mismatched = []
    for dia_id, position in dia_to_session.items():
        parsed = DIA_RE.fullmatch(dia_id)
        if parsed and int(parsed.group(1)) != position:
            mismatched.append((dia_id, position))
    if mismatched:
        raise ValueError(f"{sample_id}: dia_id 与 session 遍历不一致 {mismatched[:5]}")

    return {
        "sample_id": sample_id,
        "n_sessions": len(session_keys),
        "n_turns": global_index,
        "seq_to_dia": seq_to_dia,
        "dia_to_seq": dia_to_seq,
        "dia_to_session": dia_to_session,
    }


def load_dialog_maps(dataset_payload: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(dataset_payload, list):
        raise ValueError("数据集应为对话列表")
    return {str(item["sample_id"]): build_dialog_maps(item) for item in dataset_payload}


# ---------------------------------------------------------------------------
# 检索结果侧：把 retrieved_sources 展开成有序的 dia_id 覆盖序列
# ---------------------------------------------------------------------------

def item_dia_ids(item: Any, dialog: dict[str, Any]) -> tuple[set[str], int, str]:
    """返回单个检索项覆盖的 dia_id 集合、事件数、所属视图 origin。

    只信任 turn_sequence → dia_id 的精确映射；无法映射的事件计入 unresolved，
    由调用方汇总，绝不悄悄丢弃（丢弃会高估召回）。
    """
    origin = "unknown"
    events: list[dict[str, Any]] = []
    if isinstance(item, dict):
        metadata = item.get("metadata") or {}
        origin = str(metadata.get("origin") or "unknown")
        raw_events = metadata.get("source_events") or []
        if isinstance(raw_events, list):
            events = [event for event in raw_events if isinstance(event, dict)]
    seq_to_dia = dialog["seq_to_dia"]
    covered: set[str] = set()
    unresolved = 0
    for event in events:
        sequence = event.get("turn_sequence")
        dia_id = seq_to_dia.get(sequence) if isinstance(sequence, int) else None
        if dia_id:
            covered.add(dia_id)
        else:
            unresolved += 1
    return covered, unresolved, origin


def provenance_trace(row: dict, dialog: dict[str, Any]) -> dict[str, Any]:
    """按存储顺序把检索项展开为累计覆盖序列，并记录 K 截断点与首次命中位置。"""
    items = row.get("retrieved_sources")
    if not isinstance(items, list):
        items = []
    gold_ids = [str(value) for value in (row.get("gold_evidence") or [])]
    gold_known = [value for value in gold_ids if value in dialog["dia_to_seq"]]
    gold_set = set(gold_known)

    cumulative: set[str] = set()
    first_hit_rank: int | None = None
    per_position: list[dict[str, Any]] = []
    unresolved_events = 0
    origin_items: Counter[str] = Counter()
    origin_hit_items: Counter[str] = Counter()

    for position, item in enumerate(items, start=1):
        covered, unresolved, origin = item_dia_ids(item, dialog)
        unresolved_events += unresolved
        origin_items[origin] += 1
        newly = covered & gold_set
        if newly and first_hit_rank is None:
            first_hit_rank = position
        if newly:
            origin_hit_items[origin] += 1
        cumulative |= covered
        per_position.append({
            "position": position,
            "origin": origin,
            "covered_dia": sorted(covered),
            "cumulative_gold_covered": len(cumulative & gold_set),
        })

    covered_gold = cumulative & gold_set
    recall_at_k = {}
    hit_at_k = {}
    running: set[str] = set()
    for position, entry in enumerate(per_position, start=1):
        running |= set(entry["covered_dia"]) & gold_set
        if position in K_VALUES:
            recall_at_k[position] = (len(running) / len(gold_set)) if gold_set else None
            hit_at_k[position] = bool(running) if gold_set else None
    for budget in K_VALUES:
        if budget not in recall_at_k:
            recall_at_k[budget] = (len(covered_gold) / len(gold_set)) if gold_set else None
            hit_at_k[budget] = bool(covered_gold) if gold_set else None

    gold_sessions = [dialog["dia_to_session"][value] for value in gold_known]
    evidence_gap = (dialog["n_sessions"] - min(gold_sessions)) if gold_sessions else None

    return {
        "n_items": len(items),
        "gold_total": len(gold_ids),
        "gold_resolvable": len(gold_known),
        "gold_set_size": len(gold_set),
        "gold_covered": len(covered_gold),
        "recall_full_budget": (len(covered_gold) / len(gold_set)) if gold_set else None,
        "recall_at_k": recall_at_k,
        "hit_at_k": hit_at_k,
        "first_hit_rank": first_hit_rank,
        "mrr": (1.0 / first_hit_rank) if first_hit_rank else (0.0 if gold_set else None),
        "unresolved_events": unresolved_events,
        "origin_items": dict(origin_items),
        "origin_hit_items": dict(origin_hit_items),
        "evidence_gap": evidence_gap,
        "evidence_sessions": sorted(set(gold_sessions)),
        "multi_session_evidence": len(set(gold_sessions)) > 1,
    }


# ---------------------------------------------------------------------------
# SimpleMem 侧：明确有偏的词面包含度诊断
# ---------------------------------------------------------------------------

def bigrams(text: Any) -> set[tuple[str, str]]:
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {(words[index], words[index + 1]) for index in range(len(words) - 1)}


def lexical_containment(row: dict, sample: dict, dialog: dict[str, Any]) -> dict[str, Any]:
    """词面 bigram 包含度。**已知对改写型记忆系统性偏低，不可用于跨系统比较。**

    保留它只为透明：读者能看到 SM 侧为什么拿不到精确召回，以及偏差有多大。
    """
    conversation = sample["conversation"]
    gold_text: list[str] = []
    for dia_id in (row.get("gold_evidence") or []):
        session_match = DIA_RE.fullmatch(str(dia_id))
        if not session_match:
            continue
        session_key = f"session_{session_match.group(1)}"
        turn_index = int(session_match.group(2)) - 1
        turns = conversation.get(session_key) or []
        if 0 <= turn_index < len(turns):
            gold_text.append(turns[turn_index].get("text") or "")
    gold_grams = set().union(*[bigrams(text) for text in gold_text]) if gold_text else set()

    pool_grams: set[tuple[str, str]] = set()
    items = row.get("retrieved_sources")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, str):
                pool_grams |= bigrams(item)
            elif isinstance(item, dict):
                pool_grams |= bigrams(item.get("content"))
                metadata = item.get("metadata") or {}
                for event in (metadata.get("source_events") or []):
                    if isinstance(event, dict):
                        pool_grams |= bigrams(event.get("text"))

    if not gold_grams:
        return {"n_items": len(items) if isinstance(items, list) else 0,
                "gold_bigrams": 0, "contained_bigrams": 0, "containment": None,
                "exact_all": None, "half_overlap": None}
    contained = len(gold_grams & pool_grams)
    return {
        "n_items": len(items) if isinstance(items, list) else 0,
        "gold_bigrams": len(gold_grams),
        "contained_bigrams": contained,
        "containment": contained / len(gold_grams),
        "exact_all": int(contained == len(gold_grams)),
        "half_overlap": int(contained >= max(1, len(gold_grams) // 2)),
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def gap_bin(gap: int | None) -> str | None:
    if gap is None:
        return None
    if gap <= 5:
        return "1-5"
    if gap <= 10:
        return "6-10"
    if gap <= 15:
        return "11-15"
    if gap <= 20:
        return "16-20"
    if gap <= 25:
        return "21-25"
    return "26+"


def mean_or_none(values: list[float]) -> float | None:
    usable = [value for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def summarize(records: list[dict[str, Any]], category_filter: set[str] | None = None) -> dict[str, Any]:
    selected = [
        record for record in records
        if category_filter is None or record["category"] in category_filter
    ]
    grouped_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_gap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        grouped_by_category[record["category"]].append(record)
        bucket = gap_bin(record["evidence_gap"])
        if bucket:
            grouped_by_gap[bucket].append(record)

    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(group),
            "recall_at_k": {
                str(budget): mean_or_none([item["recall_at_k"].get(budget) for item in group])
                for budget in K_VALUES
            },
            "hit_at_k": {
                str(budget): mean_or_none(
                    [float(item["hit_at_k"][budget]) for item in group
                     if item["hit_at_k"].get(budget) is not None]
                )
                for budget in K_VALUES
            },
            "recall_full_budget": mean_or_none([item["recall_full_budget"] for item in group]),
            "mrr": mean_or_none([item["mrr"] for item in group]),
            "first_hit_rank_median": (
                statistics.median([item["first_hit_rank"] for item in group
                                   if item["first_hit_rank"] is not None])
                if any(item["first_hit_rank"] is not None for item in group) else None
            ),
            "mean_items_retrieved": mean_or_none([float(item["n_items"]) for item in group]),
        }

    return {
        "overall": block(selected),
        "by_category": {category: block(group) for category, group in sorted(grouped_by_category.items())},
        "by_gap_bin": {bucket: block(grouped_by_gap[bucket]) for bucket in GAP_BINS if bucket in grouped_by_gap},
    }


def loss_decomposition(records: list[dict[str, Any]]) -> dict[str, Any]:
    """把 MN 输题拆成"没召回"与"召回了仍答错"，用于决定该优先修检索层还是产出层。"""
    nonadversarial = [record for record in records if record["category"] != ADVERSARIAL]

    def state(record: dict[str, Any]) -> str:
        recall = record["recall_full_budget"]
        if recall is None:
            return "no_gold_reference"
        if recall == 0.0:
            return "gold_not_recalled"
        if recall < 1.0:
            return "gold_partially_recalled"
        return "gold_fully_recalled"

    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    # 两张 gap 表口径不同，必须分开：一张只统计输题（用于病因归因），
    # 一张统计该桶全部题（用于看召回随距离的整体退化）。混用会产生
    # "占输题比例 > 100%" 这种自相矛盾的读数。
    gap_loss_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    gap_all_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for record in nonadversarial:
        bucket = state(record)
        matrix[record["outcome"]][bucket] += 1
        gap_bucket = gap_bin(record["evidence_gap"])
        if gap_bucket:
            gap_all_matrix[gap_bucket][bucket] += 1
            if record["outcome"] == "loss":
                gap_loss_matrix[gap_bucket][bucket] += 1
        if record["outcome"] == "loss" and len(examples[bucket]) < 8:
            examples[bucket].append(record["key"])

    losses = matrix["loss"]
    total_losses = sum(losses.values())
    return {
        "outcome_x_recall_state": {
            outcome: dict(counter) for outcome, counter in sorted(matrix.items())
        },
        "gap_bin_x_recall_state_losses_only": {
            bucket: dict(gap_loss_matrix[bucket]) for bucket in GAP_BINS if bucket in gap_loss_matrix
        },
        "gap_bin_x_recall_state_all_items": {
            bucket: dict(gap_all_matrix[bucket]) for bucket in GAP_BINS if bucket in gap_all_matrix
        },
        "loss_by_gap_bin": {
            bucket: {
                "losses": sum(counter.values()),
                "retrieval_side": counter["gold_not_recalled"] + counter["gold_partially_recalled"],
                "generation_side": counter["gold_fully_recalled"],
            }
            for bucket, counter in sorted(gap_loss_matrix.items())
        },
        "loss_totals": {
            "losses": total_losses,
            "gold_not_recalled": losses["gold_not_recalled"],
            "gold_partially_recalled": losses["gold_partially_recalled"],
            "gold_fully_recalled": losses["gold_fully_recalled"],
            "no_gold_reference": losses["no_gold_reference"],
            "retrieval_side_share": (
                (losses["gold_not_recalled"] + losses["gold_partially_recalled"]) / total_losses
            ) if total_losses else None,
            "generation_side_share": (
                losses["gold_fully_recalled"] / total_losses
            ) if total_losses else None,
        },
        "loss_examples_by_state": {key: sorted(values) for key, values in sorted(examples.items())},
        "adversarial_excluded": sum(1 for record in records if record["category"] == ADVERSARIAL),
        "note": "gold_not_recalled / partial → 表示与检索层（E1/E4）；"
                "gold_fully_recalled 仍输 → 生成/渲染/义务判定层（E3/E5）。",
    }


def origin_effectiveness(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """各视图（origin）贡献了多少命中金标的检索项，用于判断预算该往哪个视图倾斜。"""
    items: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for record in records:
        for origin, count in record["origin_items"].items():
            items[origin] += count
        for origin, count in record["origin_hit_items"].items():
            hits[origin] += count
    return {
        origin: {
            "items": items[origin],
            "items_covering_gold": hits[origin],
            "precision": (hits[origin] / items[origin]) if items[origin] else None,
        }
        for origin in sorted(items)
    }


def load_judged(judged_path: Path | None) -> dict[tuple[Any, Any, str], str]:
    """读取辅助 Judge 结果；缺失时返回空字典（Judge 只是辅助，不阻塞主诊断）。"""
    if judged_path is None or not judged_path.exists():
        return {}
    payload = json.loads(judged_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("rows_judged.json 应为行列表")
    labels: dict[tuple[Any, Any, str], str] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        key = (row.get("sample_id"), row.get("qa_index"), row.get("system"))
        label = row.get("judge_result")
        if label in ("CORRECT", "WRONG"):
            labels[key] = label
    return labels


def view_counterfactual(records: list[dict[str, Any]], rows_by_key: dict[str, dict[str, Any]],
                        dialogs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """逐视图剔除 + 配额扫描：量化"减冗余 token"的可执行空间与召回代价。

    这是**离线重放**——用已记录的 retrieved_sources 重新截断，不等价于线上改配额后
    重跑（线上截断会改变 grounding 与答案器输入分布）。因此结果只能作为 E6 的
    **方向性依据与参数候选**，不能当成端到端收益。
    """
    nonadversarial = [record for record in records if record["category"] != ADVERSARIAL]

    def origin_of(item: Any) -> str:
        if isinstance(item, dict):
            return str((item.get("metadata") or {}).get("origin") or "unknown")
        return "unknown"

    def item_dia(item: Any, dialog: dict[str, Any]) -> set[str]:
        covered, _, _ = item_dia_ids(item, dialog)
        return covered

    def evaluate(select_fn) -> dict[str, float | None]:
        recalls_full: list[float] = []
        recalls_5: list[float] = []
        recalls_10: list[float] = []
        chars: list[float] = []
        counts: list[float] = []
        for record in nonadversarial:
            row = rows_by_key.get(record["key"])
            if row is None:
                continue
            dialog = dialogs.get(record["sample_id"])
            if dialog is None:
                continue
            gold = {value for value in (row.get("gold_evidence") or [])
                    if value in dialog["dia_to_seq"]}
            if not gold:
                continue
            items = select_fn(row.get("retrieved_sources") or [])
            cumulative: set[str] = set()
            at_k: dict[int, float] = {}
            for position, item in enumerate(items, start=1):
                cumulative |= item_dia(item, dialog) & gold
                if position in (5, 10):
                    at_k[position] = len(cumulative & gold) / len(gold)
            at_k.setdefault(5, len(cumulative & gold) / len(gold))
            at_k.setdefault(10, len(cumulative & gold) / len(gold))
            recalls_full.append(len(cumulative & gold) / len(gold))
            recalls_5.append(at_k[5])
            recalls_10.append(at_k[10])
            chars.append(sum(len(item.get("content") or "") for item in items
                             if isinstance(item, dict)))
            counts.append(len(items))
        if not recalls_full:
            return {"n": 0, "recall_full": None, "recall_5": None, "recall_10": None,
                    "chars_per_question": None, "items_per_question": None}
        return {
            "n": len(recalls_full),
            "recall_full": statistics.fmean(recalls_full),
            "recall_5": statistics.fmean(recalls_5),
            "recall_10": statistics.fmean(recalls_10),
            "chars_per_question": statistics.fmean(chars),
            "items_per_question": statistics.fmean(counts),
        }

    origins = sorted({origin_of(item) for record in nonadversarial
                      for item in ((rows_by_key.get(record["key"]) or {}).get("retrieved_sources") or [])
                      if isinstance(item, dict)})

    def keep_all(items: list[Any]) -> list[Any]:
        return list(items)

    def drop_origin(origin: str):
        return lambda items: [item for item in items if origin_of(item) != origin]

    def only_origin(origin: str):
        return lambda items: [item for item in items if origin_of(item) == origin]

    def cap_origin(origin: str, cap: int):
        """按存储顺序保留该视图前 cap 项，其余项原样保留、顺序不变。"""
        def select(items: list[Any]) -> list[Any]:
            selected: list[Any] = []
            used = 0
            for item in items:
                if origin_of(item) == origin:
                    if used >= cap:
                        continue
                    used += 1
                selected.append(item)
            return selected
        return select

    baseline = evaluate(keep_all)
    drop_view = {origin: evaluate(drop_origin(origin)) for origin in origins}
    only_view = {origin: evaluate(only_origin(origin)) for origin in origins}
    # 配额扫描：cap=0 等价于 drop_view，用作一致性自检。
    quota_sweep = {
        origin: {str(cap): evaluate(cap_origin(origin, cap)) for cap in QUOTA_CAPS}
        for origin in origins
    }

    # 一致性自检：cap=0 必须与"剔除该视图"完全一致，否则选择器实现有 bug。
    for origin in origins:
        zero = quota_sweep[origin]["0"]
        dropped = drop_view[origin]
        for field in ("recall_full", "recall_5", "recall_10", "items_per_question"):
            if zero[field] is None or dropped[field] is None:
                if zero[field] != dropped[field]:
                    raise AssertionError(f"{origin} cap=0 与 drop_view 不一致：{field}")
            elif abs(zero[field] - dropped[field]) > 1e-9:
                raise AssertionError(f"{origin} cap=0 与 drop_view 不一致：{field}")

    # 甜点选择：在"R@10 不低于基线"的前提下，取检索字符最省的 cap。
    # 不追求 R@full 无损——R@full 是上界口径，实际答案器只看到有限预算。
    sweet_spots: dict[str, dict[str, Any]] = {}
    for origin in origins:
        candidates = []
        for cap in QUOTA_CAPS:
            block = quota_sweep[origin][str(cap)]
            if block["recall_10"] is None or baseline["recall_10"] is None:
                continue
            if block["recall_10"] >= baseline["recall_10"] - QUOTA_TOLERANCE:
                candidates.append((cap, block))
        if candidates:
            cap, block = min(candidates, key=lambda pair: pair[1]["chars_per_question"])
            sweet_spots[origin] = {
                "cap": cap,
                "chars_per_question": block["chars_per_question"],
                "chars_saved_pct": (
                    100 * (1 - block["chars_per_question"] / baseline["chars_per_question"])
                    if baseline["chars_per_question"] else None
                ),
                "items_per_question": block["items_per_question"],
                "recall_10": block["recall_10"],
                "recall_full": block["recall_full"],
                "recall_full_delta_pp": (
                    (block["recall_full"] - baseline["recall_full"]) * 100
                    if block["recall_full"] is not None and baseline["recall_full"] is not None else None
                ),
                "selection_rule": "在 R@10 不低于基线（容差 1e-9）的前提下取检索字符最省的 cap",
            }

    return {
        "offline_replay_only": True,
        "not_end_to_end": True,
        "origins": origins,
        "quota_caps": list(QUOTA_CAPS),
        "baseline": baseline,
        "drop_view": drop_view,
        "only_view": only_view,
        "quota_sweep": quota_sweep,
        "sweet_spots": sweet_spots,
        "warning": (
            "离线重放：用已记录的检索结果重新截断，不等价于线上改配额后重跑。"
            "线上截断会改变 grounding 与答案器输入分布，cap 甜点只能作为 E6 的参数候选，"
            "端到端收益必须实测。检索字符节省也不等于 accepted token 等比下降"
            "（prompt 大头是静态契约）。"
        ),
    }


def judge_cross_tabulation(records: list[dict[str, Any]],
                           labels: dict[tuple[Any, Any, str], str]) -> dict[str, Any] | None:
    """召回状态 × Judge 交叉表：检验"F1 输但语义对"的规模，以及两指标是否同向。

    没有 Judge 数据时返回 None，调用方必须显式标注不可用，不得用 0 冒充。
    """
    usable = [record for record in records
              if record.get("mn_judge") in ("CORRECT", "WRONG")
              or record.get("sm_judge") in ("CORRECT", "WRONG")]
    if not usable:
        return None

    nonadversarial = [record for record in usable if record["category"] != ADVERSARIAL]

    def accuracy(group: list[dict[str, Any]], field: str) -> tuple[float | None, int]:
        values = [record[field] for record in group if record.get(field) in ("CORRECT", "WRONG")]
        if not values:
            return None, 0
        return sum(value == "CORRECT" for value in values) / len(values), len(values)

    def recall_state(record: dict[str, Any]) -> str:
        recall = record["recall_full_budget"]
        if recall is None:
            return "no_gold_reference"
        if recall == 0.0:
            return "gold_not_recalled"
        if recall < 1.0:
            return "gold_partially_recalled"
        return "gold_fully_recalled"

    by_state: dict[str, dict[str, Any]] = {}
    for state in ("gold_fully_recalled", "gold_partially_recalled", "gold_not_recalled"):
        group = [record for record in nonadversarial if recall_state(record) == state]
        if not group:
            continue
        mn_acc, mn_n = accuracy(group, "mn_judge")
        sm_acc, sm_n = accuracy(group, "sm_judge")
        by_state[state] = {
            "n": len(group), "mn_accuracy": mn_acc, "mn_n": mn_n,
            "sm_accuracy": sm_acc, "sm_n": sm_n,
            "delta_pp": ((mn_acc - sm_acc) * 100) if (mn_acc is not None and sm_acc is not None) else None,
        }

    # "金标全召回 + 官方F1判MN输 + 双方Judge都CORRECT" 是最需要人工复核的子集：
    # 它同时说明"F1 词面失配"与"SM 语义仍更强"，不能只取其中一个解释。
    focus = [record for record in nonadversarial
             if recall_state(record) == "gold_fully_recalled" and record.get("outcome") == "loss"]
    focus_both_correct = [record for record in focus
                          if record.get("mn_judge") == "CORRECT" and record.get("sm_judge") == "CORRECT"]
    mn_acc_focus, _ = accuracy(focus, "mn_judge")
    sm_acc_focus, _ = accuracy(focus, "sm_judge")

    mn_overall, mn_n = accuracy(nonadversarial, "mn_judge")
    sm_overall, sm_n = accuracy(nonadversarial, "sm_judge")
    mn_all, _ = accuracy(usable, "mn_judge")
    sm_all, _ = accuracy(usable, "sm_judge")

    return {
        "available": True,
        "auxiliary_only": True,
        "matched_labels": len(labels),
        "nonadversarial": {
            "n": len(nonadversarial),
            "mn_accuracy": mn_overall, "mn_n": mn_n,
            "sm_accuracy": sm_overall, "sm_n": sm_n,
            "delta_pp": ((mn_overall - sm_overall) * 100)
            if (mn_overall is not None and sm_overall is not None) else None,
        },
        "all_categories": {"mn_accuracy": mn_all, "sm_accuracy": sm_all},
        "by_recall_state": by_state,
        "full_recall_but_f1_loss": {
            "n": len(focus),
            "mn_accuracy": mn_acc_focus,
            "sm_accuracy": sm_acc_focus,
            "both_judge_correct": len(focus_both_correct),
            "both_judge_correct_share": (len(focus_both_correct) / len(focus)) if focus else None,
            "example_keys": sorted(record["key"] for record in focus_both_correct)[:20],
        },
        "interpretation_warning": (
            "官方 F1 与 LLM Judge 可能方向相反：MN 保留原文轮次措辞→词面 F1 占优；"
            "SM 改写为贴近问题的合成句→语义 Judge 占优。两个指标必须同时报告，"
            "不得只取有利者。Judge 仍只是辅助指标。"
        ),
    }


# ---------------------------------------------------------------------------
# 可达集归因：把「未召回」拆成 建库层 / 检索层 / 视界截断（2026-11-09 新增）
# ---------------------------------------------------------------------------

def load_memory_snapshots(attempts_dir: Path | None) -> dict[str, dict[str, Any]]:
    """读取建库快照（attempts/<sid>_memnexus/usage.memory.json）。

    为什么需要它：仅凭「其他题能否召回」推断建库状况是**间接证据**——一个轮次从未被
    任何题召回，分不开「建库里没有」与「建库有但没有 query 触发」。usage.memory.json
    含建库输入的全部 turns（sequence → turn_id）与全部 memories（evidence[].turn_id），
    可以**直接验证**金标轮次是否进了建库、是否被至少一条记忆引用。

    快照缺失时返回空字典：归因自动退化为只用可达集（间接口径），并在报告中标注。
    """
    if attempts_dir is None or not attempts_dir.exists():
        return {}
    snapshots: dict[str, dict[str, Any]] = {}
    for path in sorted(attempts_dir.glob("*_memnexus/usage.memory.json")):
        sample_id = path.parent.name.removesuffix("_memnexus")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        seq_to_turn = {int(t["sequence"]): t["turn_id"] for t in (payload.get("turns") or [])}
        mem_turn_ids: set[str] = set()
        for mem in payload.get("memories") or []:
            for ev in mem.get("evidence") or []:
                turn_id = ev.get("turn_id")
                if turn_id:
                    mem_turn_ids.add(turn_id)
        snapshots[sample_id] = {
            "seq_to_turn": seq_to_turn,
            "mem_turn_ids": mem_turn_ids,
            "n_memories": len(payload.get("memories") or []),
            "n_turns": len(seq_to_turn),
        }
    return snapshots


def reachability_attribution(mn_records: list[dict[str, Any]],
                             mn_rows_by_key: dict[str, dict[str, Any]],
                             dialogs: dict[str, dict[str, Any]],
                             snapshots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """三层归因：prompt 召回 → 可达集 → 建库快照直接验证。

    判据链（由弱到强）：
      1. prompt 召回状态：金标 dia 是否出现在本题 retrieved_sources（已有 provenance）。
      2. 可达集（间接）：金标 dia 是否被**同对话其他任何题**召回过。可达 ⇒ 建库和
         检索管道都能触达它，本题没召回是 query/排序/配额问题（E4 的靶子）。
         注意这是间接判据：不可达 ≠ 建库缺失，需要第 3 层验证。
      3. 建库快照（直接）：金标 dia → sequence → turn_id，检查该 turn 是否
         (a) 在建库输入 turns 里（不在 ⇒ 视界截断，物理上不可能召回）；
         (b) 被至少一条记忆的 evidence 引用（在 ⇒ 建库没问题，是检索问题）。

    口径纪律：可达集用**全部非对抗 MN 行**累积（与内联分析一致，adversarial 的召回
    同样能证明可达性，但为与已发布数字保持同口径，此处排除 adversarial）。
    """
    # 第一遍：每题召回过的全部 dia（不限金标），按对话累积为可达集。
    covered_by_conv: dict[str, set[str]] = defaultdict(set)
    for record in mn_records:
        if record["category"] == ADVERSARIAL:
            continue
        row = mn_rows_by_key.get(record["key"])
        dialog = dialogs.get(record["sample_id"])
        if row is None or dialog is None:
            continue
        for item in (row.get("retrieved_sources") or []):
            covered, _, _ = item_dia_ids(item, dialog)
            covered_by_conv[record["sample_id"]] |= covered

    # 第二遍 + 第三遍：对每道非对抗题做三层归因。
    matrix: Counter[str] = Counter()
    e4_keys: list[str] = []       # not/partial 且金标全部可达 → 检索/排序问题
    e1_keys: list[str] = []       # not 且快照证实金标不在记忆 → 建库遗漏
    horizon_keys: list[str] = []  # 金标轮次在建库输入之外 → 视界截断
    snapshot_detail: list[dict[str, Any]] = []

    for record in mn_records:
        if record["category"] == ADVERSARIAL:
            continue
        row = mn_rows_by_key.get(record["key"])
        dialog = dialogs.get(record["sample_id"])
        if row is None or dialog is None:
            continue
        # 金标 dia 必须从原始 row 取（mn_record 只带 trace 计数，不带金标列表）。
        gold_dias = {str(v) for v in (row.get("gold_evidence") or [])
                     if str(v) in dialog["dia_to_seq"]}
        gold_set_size = record.get("gold_set_size") or 0
        if not gold_set_size or not gold_dias:
            continue
        gold_covered = record.get("gold_covered") or 0
        state = "full" if gold_covered == gold_set_size else (
            "partial" if gold_covered else "not")
        reachable = covered_by_conv.get(record["sample_id"], set())
        # 注意：可达集含本题自己的召回，但 state=not/partial 的题自己没召回到这部分金标，
        # 所以 gold∩reachable 的增量必然来自其他题——这正是可达集的定义。
        gold_reach = gold_dias & reachable
        reach_state = "reachable" if len(gold_reach) == gold_set_size else (
            "part_reach" if gold_reach else "unreachable")
        matrix[f"{state}/{reach_state}"] += 1

        if state == "full":
            continue  # 产出层，不属于本归因
        if reach_state == "reachable":
            e4_keys.append(record["key"])
            continue
        # 不可达或部分可达：用快照直接验证金标在建库里的状态。
        snapshot = snapshots.get(record["sample_id"])
        if snapshot is None:
            # 无快照 ⇒ 无法直接验证，保守归入 E4（间接证据不足以断言建库缺失）。
            e4_keys.append(record["key"])
            continue
        turn_ids, cited = 0, 0
        in_horizon = 0
        for dia in gold_dias:
            seq = dialog["dia_to_seq"].get(dia)
            turn_id = snapshot["seq_to_turn"].get(seq) if seq is not None else None
            if turn_id is None:
                continue  # 该轮次不在建库输入（视界截断）
            in_horizon += 1
            turn_ids += 1
            if turn_id in snapshot["mem_turn_ids"]:
                cited += 1
        entry = {
            "key": record["key"], "category": record["category"],
            "state": state, "reach_state": reach_state,
            "gold_turns_in_horizon": in_horizon,
            "gold_turns_cited_by_memory": cited,
        }
        snapshot_detail.append(entry)
        if in_horizon == 0:
            # 金标轮次全部在建库输入之外：物理上不可能召回，E4/E1 都救不了。
            horizon_keys.append(record["key"])
        elif cited == 0:
            e1_keys.append(record["key"])  # 进了建库输入但没被任何记忆引用
        else:
            e4_keys.append(record["key"])  # 记忆里有（至少部分），是检索没捞到

    per_conversation = {
        sid: {
            "total_turns": dialogs[sid]["n_turns"],
            "reachable_dias": len(covered_by_conv.get(sid, set())),
            "reachable_pct": round(100 * len(covered_by_conv.get(sid, set()))
                                   / max(1, dialogs[sid]["n_turns"]), 1),
        }
        for sid in sorted(dialogs)
    }

    return {
        "definition": (
            "可达集 = 同对话所有非对抗题召回过的 dia 全集（含 raw_fallback 直读原文轮次，"
            "因此可达率 80–95% 高于建库记忆引用率 54–61%）；不可达 ≠ 建库缺失，"
            "以 usage.memory.json 的 turns/memories 直接验证为准。"
        ),
        "snapshot_available": bool(snapshots),
        "per_conversation": per_conversation,
        "attribution_matrix": dict(matrix),
        "e4_pool_keys": sorted(e4_keys),
        "e1_residual_keys": sorted(e1_keys),
        "horizon_excluded_keys": sorted(horizon_keys),
        "snapshot_verification": snapshot_detail,
        "counts": {
            "e4_pool": len(e4_keys),
            "e1_residual": len(e1_keys),
            "horizon_excluded": len(horizon_keys),
        },
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_rows(rows_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(rows_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("rows.json 应为行列表")
    return [row for row in payload if isinstance(row, dict)]


def official_scores(row: dict[str, Any]) -> float:
    """复用主报告的唯一评分实现，避免诊断与主表口径漂移。"""
    import paper_protocol as protocol

    return protocol.f1_of(row)


def audit(rows_path: Path, dataset_path: Path, systems: tuple[str, ...] = SYSTEMS,
          judged_path: Path | None = None,
          attempts_dir: Path | None = None) -> dict[str, Any]:
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    dialogs = load_dialog_maps(dataset_payload)
    rows = load_rows(rows_path)
    judge_labels = load_judged(judged_path)
    snapshots = load_memory_snapshots(attempts_dir)

    # 官方 F1 需要与主表一致；按 (sample_id, qa_index, system) 建索引以配对。
    index: dict[tuple[Any, Any, str], dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.get("system") not in systems:
            continue
        key = (row.get("sample_id"), row.get("qa_index"), row.get("system"))
        if key in index:
            duplicates.append("|".join(str(part) for part in key))
        index[key] = row
    if duplicates:
        raise ValueError(f"存在重复键，拒绝继续：{duplicates[:5]}")

    samples = {str(item["sample_id"]): item for item in dataset_payload}
    validation = {
        "conversations_in_dataset": len(dialogs),
        "turns_total": sum(dialog["n_turns"] for dialog in dialogs.values()),
        "sessions_per_conversation": {key: dialog["n_sessions"] for key, dialog in sorted(dialogs.items())},
        "unresolved_events_total": 0,
        "gold_refs_total": 0,
        "gold_refs_resolvable": 0,
        "mn_rows": 0,
        "mn_rows_with_items": 0,
        "mn_rows_missing_items": [],
        "duplicate_keys": duplicates,
    }

    mn_records: list[dict[str, Any]] = []
    sm_records: list[dict[str, Any]] = []
    per_item: list[dict[str, Any]] = []

    for (sample_id, qa_index, system), row in sorted(
        index.items(), key=lambda pair: (str(pair[0][0]), pair[0][1] or 0, pair[0][2])
    ):
        dialog = dialogs.get(str(sample_id))
        if dialog is None:
            raise ValueError(f"行 {sample_id}/{qa_index}/{system} 的对话不在数据集中")
        gold_ids = [str(value) for value in (row.get("gold_evidence") or [])]
        validation["gold_refs_total"] += len(gold_ids)
        validation["gold_refs_resolvable"] += sum(1 for value in gold_ids if value in dialog["dia_to_seq"])

        sm_row = index.get((sample_id, qa_index, "SimpleMem"))
        mn_row = index.get((sample_id, qa_index, "MemNexus"))
        if system == "MemNexus":
            validation["mn_rows"] += 1
        outcome = None
        if sm_row is not None and mn_row is not None:
            mn_f1, sm_f1 = official_scores(mn_row), official_scores(sm_row)
            delta = mn_f1 - sm_f1
            outcome = "win" if delta > TOLERANCE else "loss" if delta < -TOLERANCE else "tie"

        category = str(row.get("category") or "unknown")
        if system == "MemNexus":
            trace = provenance_trace(row, dialog)
            validation["unresolved_events_total"] += trace["unresolved_events"]
            if trace["n_items"]:
                validation["mn_rows_with_items"] += 1
            else:
                validation["mn_rows_missing_items"].append(f"{sample_id}/{qa_index}")
            record = {
                "key": f"{sample_id}/{qa_index}",
                "sample_id": str(sample_id),
                "qa_index": qa_index,
                "category": category,
                "outcome": outcome,
                "f1": official_scores(row),
                # Judge 标签随行携带，供 §4.5 交叉表使用；缺失时为 None。
                "mn_judge": judge_labels.get((sample_id, qa_index, "MemNexus")),
                "sm_judge": judge_labels.get((sample_id, qa_index, "SimpleMem")),
                **trace,
            }
            mn_records.append(record)
            per_item.append({
                "key": record["key"], "system": "MemNexus", "category": category,
                "outcome": outcome, "evidence_gap": trace["evidence_gap"],
                "gap_bin": gap_bin(trace["evidence_gap"]),
                "gold_set_size": trace["gold_set_size"], "gold_covered": trace["gold_covered"],
                "recall_full_budget": trace["recall_full_budget"],
                "first_hit_rank": trace["first_hit_rank"], "mrr": trace["mrr"],
                "recall_at_k": {str(k): trace["recall_at_k"].get(k) for k in K_VALUES},
                "hit_at_k": {str(k): trace["hit_at_k"].get(k) for k in K_VALUES},
                "n_items": trace["n_items"],
                "multi_session_evidence": trace["multi_session_evidence"],
                "origin_items": trace["origin_items"],
                "unresolved_events": trace["unresolved_events"],
            })
        else:
            containment = lexical_containment(row, samples[str(sample_id)], dialog)
            record = {
                "key": f"{sample_id}/{qa_index}", "sample_id": str(sample_id),
                "qa_index": qa_index, "category": category, "outcome": outcome,
                **containment,
            }
            sm_records.append(record)
            per_item.append({
                "key": record["key"], "system": "SimpleMem", "category": category,
                "outcome": outcome, "lexical": containment,
            })

    nonadversarial = {category for category in {r["category"] for r in mn_records}
                      if category != ADVERSARIAL}
    validation["gold_all_resolvable"] = (
        validation["gold_refs_total"] == validation["gold_refs_resolvable"]
    )
    validation["mapping_exact"] = validation["unresolved_events_total"] == 0

    sm_containments = [record["containment"] for record in sm_records if record["containment"] is not None]
    sm_exact = [record["exact_all"] for record in sm_records if record["exact_all"] is not None]
    mn_exact_proxy_bias_note = (
        "SM 的词面 exact-all 命中率实测为 0.0%（MN 同口径 77.9%）。该差异来自 SM 记忆被改写、"
        "MN 保留原文轮次，**不代表召回能力差异**。因此 SM 不产出 Recall@K，本诊断不可跨系统比较。"
    )

    # 视图反事实与配额扫描只针对 MN（SM 无 provenance，不可对称比较）。
    mn_rows_by_key = {
        record["key"]: index[(record["sample_id"], record["qa_index"], "MemNexus")]
        for record in mn_records
        if (record["sample_id"], record["qa_index"], "MemNexus") in index
    }
    counterfactual = view_counterfactual(mn_records, mn_rows_by_key, dialogs)
    reachability = reachability_attribution(
        mn_records, mn_rows_by_key, dialogs, snapshots)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"rows": str(rows_path), "dataset": str(dataset_path)},
        "method": {
            "provenance_join": "retrieved_sources[].metadata.source_events[].turn_sequence "
                               "→ 全局轮次序号 → dia_id（与 gold_evidence 同命名空间）",
            "ordering": "按 retrieved_sources 存储顺序计算 Recall@K；不做跨视图 score 重排"
                        "（score 跨 origin 不可比，997/1000 行非 score 降序）",
            "k_values": list(K_VALUES),
            "gap_definition": "gap = n_sessions(conversation) − min(金标证据所在 session)",
            "gap_bins": list(GAP_BINS),
            "scoring": "官方 F1 复用 paper_protocol.f1_of，与主表一致；容差 1e-12",
            "adversarial": "单列并排除在主召回指标外（其官方指标为 AdvSR）",
            "simplemem": mn_exact_proxy_bias_note,
            "not_main_table": True,
        },
        "validation": validation,
        "memnexus": {
            "all": summarize(mn_records),
            "nonadversarial": summarize(mn_records, nonadversarial),
            "origin_effectiveness": origin_effectiveness(mn_records),
            "loss_decomposition": loss_decomposition(mn_records),
            "multi_session_evidence": {
                "n": sum(1 for record in mn_records if record["multi_session_evidence"]),
                "recall_full_budget": mean_or_none(
                    [record["recall_full_budget"] for record in mn_records
                     if record["multi_session_evidence"]]
                ),
            },
            "view_counterfactual": counterfactual,
            "reachability": reachability,
        },
        "simplemem_lexical_diagnostic": {
            "biased_and_not_comparable": True,
            "note": mn_exact_proxy_bias_note,
            "n": len(sm_records),
            "mean_containment": mean_or_none(sm_containments),
            "exact_all_rate": mean_or_none([float(value) for value in sm_exact]),
            "by_category": {
                category: {
                    "n": len(group),
                    "mean_containment": mean_or_none([r["containment"] for r in group]),
                    "exact_all_rate": mean_or_none(
                        [float(r["exact_all"]) for r in group if r["exact_all"] is not None]
                    ),
                }
                for category, group in sorted(
                    {
                        category: [r for r in sm_records if r["category"] == category]
                        for category in {r["category"] for r in sm_records}
                    }.items()
                )
            },
        },
        "judge_cross": judge_cross_tabulation(mn_records, judge_labels),
        "per_item": per_item,
    }


def render_markdown(report: dict[str, Any]) -> str:
    validation = report["validation"]
    memnexus = report["memnexus"]
    loss = memnexus["loss_decomposition"]
    # 旧格式 report（2026-11-09 之前）没有 reachability 章节：缺失时优雅跳过，
    # 不要求所有历史产物都补齐新键。
    reachability = memnexus.get("reachability")
    lines: list[str] = [
        "# 证据级检索保真度审计（Recall@K + evidence-gap 分桶）",
        "",
        f"生成：{report['generated_at']}",
        "",
        "**这是诊断指标，不进入论文主表。** 主表仍为 `paper_protocol` 的四类等权 Average F1 + AdvSR。",
        "",
        "## 1. 映射自检（不通过则全表作废）",
        "",
        f"- 对话数：**{validation['conversations_in_dataset']}**；轮次总数：**{validation['turns_total']}**",
        f"- 金标证据引用：{validation['gold_refs_resolvable']}/{validation['gold_refs_total']} 可解析 "
        f"→ **{'全部可解析' if validation['gold_all_resolvable'] else '存在不可解析'}**",
        f"- MN 检索事件未解析数：**{validation['unresolved_events_total']}** "
        f"→ 映射{'精确' if validation['mapping_exact'] else '存在缺口'}",
        f"- MN 行数：{validation['mn_rows']}；有检索项：{validation['mn_rows_with_items']}；"
        f"缺检索项：{len(validation['mn_rows_missing_items'])}",
        f"- 重复键：{len(validation['duplicate_keys'])}",
        "",
        "映射方式：`source_events[].turn_sequence` → 全局轮次序号 → `dia_id`，"
        "与 `gold_evidence` 同命名空间；Recall@K 按存储顺序，不做跨视图 score 重排"
        "（score 跨 origin 不可比，997/1000 行非降序）。",
        "",
        "## 2. MN 召回保真度（非对抗题）",
        "",
    ]

    nonadv = memnexus["nonadversarial"]
    header = "| 分组 | n | " + " | ".join(f"R@{k}" for k in K_VALUES) + " | R@全部 | MRR | 首命中位(中位) |"
    sep = "|---|---:|" + "---:|" * len(K_VALUES) + "---:|---:|---:|"
    lines += [header, sep]

    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "—"

    def fmt_pct(value: float | None) -> str:
        return f"{100 * value:.1f}%" if value is not None else "—"

    def emit(label: str, block: dict[str, Any]) -> None:
        cells = [fmt(block["recall_at_k"][str(k)]) for k in K_VALUES]
        lines.append(
            f"| {label} | {block['n']} | " + " | ".join(cells)
            + f" | {fmt(block['recall_full_budget'])} | {fmt(block['mrr'])} "
            + f" | {block['first_hit_rank_median'] if block['first_hit_rank_median'] is not None else '—'} |"
        )

    emit("**全部非对抗**", nonadv["overall"])
    for category, block in nonadv["by_category"].items():
        emit(f"题型:{category}", block)
    for bucket, block in nonadv["by_gap_bin"].items():
        emit(f"证据距离 gap {bucket}", block)
    lines.append("")

    lines += [
        "## 3. MN 输题病因分解（决定优先修哪一层）",
        "",
        f"非对抗输题 **{loss['loss_totals']['losses']}** 题；adversarial 已排除 "
        f"（{loss['adversarial_excluded']} 题，其官方指标为 AdvSR）。",
        "",
        "| 病因 | 题数 | 占输题 | 指向 |",
        "|---|---:|---:|---|",
    ]
    totals = loss["loss_totals"]
    rows_for_table = [
        ("金标完全未召回", totals["gold_not_recalled"], "表示/检索层：E1 双轨表示、E4 证据补全"),
        ("金标部分召回", totals["gold_partially_recalled"], "证据补全：E4 多跳覆盖感知选择"),
        ("金标已全召回仍输", totals["gold_fully_recalled"], "生成/渲染/义务判定：E3 时间渲染、E5 槽位"),
        ("无金标参照", totals["no_gold_reference"], "不参与归因"),
    ]
    denominator = totals["losses"] or 1
    for label, count, target in rows_for_table:
        lines.append(f"| {label} | {count} | {100 * count / denominator:.1f}% | {target} |")
    lines += [
        "",
        f"- 检索侧合计占比：**{fmt(loss['loss_totals']['retrieval_side_share'])}**",
        f"- 生成侧合计占比：**{fmt(loss['loss_totals']['generation_side_share'])}**",
        "",
        "### outcome × 召回状态",
        "",
        "| outcome | 完全未召回 | 部分召回 | 全召回 | 无参照 |",
        "|---|---:|---:|---:|---:|",
    ]
    for outcome, counter in loss["outcome_x_recall_state"].items():
        lines.append(
            f"| {outcome} | {counter.get('gold_not_recalled', 0)} "
            f"| {counter.get('gold_partially_recalled', 0)} "
            f"| {counter.get('gold_fully_recalled', 0)} "
            f"| {counter.get('no_gold_reference', 0)} |"
        )
    lines += [
        "",
        "### gap 分桶 × 召回状态（**仅输题**）",
        "",
        "| gap | 输题 | 完全未召回 | 部分召回 | 全召回仍输 | 检索侧占该桶输题 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bucket, counter in loss["gap_bin_x_recall_state_losses_only"].items():
        total = sum(counter.values())
        retrieval_side = counter.get("gold_not_recalled", 0) + counter.get("gold_partially_recalled", 0)
        share = f"{100 * retrieval_side / total:.1f}%" if total else "—"
        lines.append(
            f"| {bucket} | {total} | {counter.get('gold_not_recalled', 0)} "
            f"| {counter.get('gold_partially_recalled', 0)} "
            f"| {counter.get('gold_fully_recalled', 0)} | {share} |"
        )
    lines += [
        "",
        "### gap 分桶 × 召回状态（该桶全部非对抗题，看整体退化）",
        "",
        "| gap | n | 完全未召回 | 部分召回 | 全召回 | 检索侧占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bucket, counter in loss["gap_bin_x_recall_state_all_items"].items():
        total = sum(counter.values())
        retrieval_side = counter.get("gold_not_recalled", 0) + counter.get("gold_partially_recalled", 0)
        lines.append(
            f"| {bucket} | {total} | {counter.get('gold_not_recalled', 0)} "
            f"| {counter.get('gold_partially_recalled', 0)} "
            f"| {counter.get('gold_fully_recalled', 0)} "
            f"| {100 * retrieval_side / total:.1f}% |" if total else ""
        )
    lines += ["", "### 各视图（origin）命中金标的效率", "",
              "| origin | 检索项 | 覆盖金标的项 | 命中率 |", "|---|---:|---:|---:|"]
    for origin, block in memnexus["origin_effectiveness"].items():
        lines.append(
            f"| {origin} | {block['items']} | {block['items_covering_gold']} | {fmt(block['precision'])} |"
        )
    lines += [
        "",
        f"多会话证据题：{memnexus['multi_session_evidence']['n']} 题，"
        f"全预算召回 {fmt(memnexus['multi_session_evidence']['recall_full_budget'])}。",
    ]

    counterfactual = memnexus.get("view_counterfactual") or {}
    if counterfactual:
        baseline = counterfactual["baseline"]
        lines += [
            "",
            "## 4. 视图反事实与配额扫描（离线重放，非端到端）",
            "",
            f"基线（全部视图，n={baseline['n']}）：R@5 {fmt(baseline['recall_5'])}、"
            f"R@10 {fmt(baseline['recall_10'])}、R@全部 {fmt(baseline['recall_full'])}；"
            f"检索字符 {baseline['chars_per_question']:.0f}/题、检索项 "
            f"{baseline['items_per_question']:.1f}/题。",
            "",
            "### 4.1 剔除 / 只留单一视图后的召回",
            "",
            "| 场景 | R@5 | R@10 | R@全部 | 字符/题 |",
            "|---|---:|---:|---:|---:|",
        ]
        for origin in counterfactual["origins"]:
            dropped = counterfactual["drop_view"][origin]
            kept = counterfactual["only_view"][origin]
            lines.append(
                f"| 去 {origin} | {fmt(dropped['recall_5'])} | {fmt(dropped['recall_10'])} "
                f"| {fmt(dropped['recall_full'])} | {dropped['chars_per_question']:.0f} |"
            )
            lines.append(
                f"| 只留 {origin} | {fmt(kept['recall_5'])} | {fmt(kept['recall_10'])} "
                f"| {fmt(kept['recall_full'])} | {kept['chars_per_question']:.0f} |"
            )
        lines += [
            "",
            "### 4.2 配额甜点（约束：R@10 不低于基线）",
            "",
            "| 视图 | cap | 字符/题 | 字符节省 | 项/题 | R@10 | R@全部 | ΔR@全部(pp) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for origin in counterfactual["origins"]:
            spot = counterfactual["sweet_spots"].get(origin)
            if not spot:
                lines.append(f"| {origin} | 无满足约束的 cap | — | — | — | — | — | — |")
                continue
            saved = spot["chars_saved_pct"]
            delta = spot["recall_full_delta_pp"]
            saved_text = f"{saved:.1f}%" if saved is not None else "—"
            delta_text = f"{delta:+.2f}" if delta is not None else "—"
            lines.append(
                f"| {origin} | {spot['cap']} | {spot['chars_per_question']:.0f} | {saved_text} "
                f"| {spot['items_per_question']:.1f} | {fmt(spot['recall_10'])} "
                f"| {fmt(spot['recall_full'])} | {delta_text} |"
            )
        lines += ["", f"> {counterfactual['warning']}", ""]

    lines += [
        "",
        "## 5. SimpleMem 侧：为什么没有可比 Recall",
        "",
        report["simplemem_lexical_diagnostic"]["note"],
        "",
        f"- SM 行数 {report['simplemem_lexical_diagnostic']['n']}；"
        f"平均词面包含度 {fmt(report['simplemem_lexical_diagnostic']['mean_containment'])}；"
        f"exact-all 率 {fmt(report['simplemem_lexical_diagnostic']['exact_all_rate'])}",
        "",
        "| 题型 | n | 平均包含度 | exact-all 率 |",
        "|---|---:|---:|---:|",
    ]
    for category, block in report["simplemem_lexical_diagnostic"]["by_category"].items():
        lines.append(
            f"| {category} | {block['n']} | {fmt(block['mean_containment'])} "
            f"| {fmt(block['exact_all_rate'])} |"
        )
    lines += [
        "",
        "## 6. 与辅助 Judge 交叉（若可用）",
        "",
    ]
    judge_cross = report.get("judge_cross")
    if not judge_cross:
        lines += [
            "**Judge 数据不可用**：未找到 `rows_judged.json` 或无有效判定。",
            "本节保持空缺，不用 0 冒充，也不影响上文 provenance 诊断。",
        ]
    else:
        nonadv_judge = judge_cross["nonadversarial"]
        focus = judge_cross["full_recall_but_f1_loss"]
        lines += [
            f"匹配判定 **{judge_cross['matched_labels']}** 条；仅辅助指标，主表仍为官方 F1。",
            "",
            f"- 非对抗题（n={nonadv_judge['n']}）：MN Judge **{fmt_pct(nonadv_judge['mn_accuracy'])}**、"
            f"SM Judge **{fmt_pct(nonadv_judge['sm_accuracy'])}**、差 "
            f"**{nonadv_judge['delta_pp']:+.1f} pp**",
            f"- 全部题（含对抗）：MN {fmt_pct(judge_cross['all_categories']['mn_accuracy'])}、"
            f"SM {fmt_pct(judge_cross['all_categories']['sm_accuracy'])}",
            "",
            "### 按召回状态分层",
            "",
            "| 召回状态 | n | MN Judge | SM Judge | 差(pp) |",
            "|---|---:|---:|---:|---:|",
        ]
        for state, block in judge_cross["by_recall_state"].items():
            delta = block["delta_pp"]
            lines.append(
                f"| {state} | {block['n']} | {fmt_pct(block['mn_accuracy'])} "
                f"| {fmt_pct(block['sm_accuracy'])} "
                f"| {delta:+.1f} |" if delta is not None else
                f"| {state} | {block['n']} | {fmt_pct(block['mn_accuracy'])} "
                f"| {fmt_pct(block['sm_accuracy'])} | — |"
            )
        lines += [
            "",
            "### 「金标全召回但官方 F1 判 MN 输」子集",
            "",
            f"- n={focus['n']}；MN Judge {fmt_pct(focus['mn_accuracy'])}、"
            f"SM Judge {fmt_pct(focus['sm_accuracy'])}",
            f"- 双方 Judge 都 CORRECT：**{focus['both_judge_correct']} 题**"
            f"（{fmt_pct(focus['both_judge_correct_share'])}）",
            "",
            "**不得只取一个解释**：这批题既说明官方 F1 存在词面失配，也说明 SM 的语义正确率"
            "仍高于 MN（两者 Judge 差见上表）。人工复核样本："
            + ", ".join(focus["example_keys"][:10]),
            "",
            f"> {judge_cross['interpretation_warning']}",
        ]
    if reachability is not None:
        lines += [
            "",
            "## 7. 可达集归因：未召回 ≠ 建库缺失（三层判据）",
            "",
            reachability["definition"],
            "",
        ]
        if reachability["snapshot_available"]:
            cited = [e for e in reachability["snapshot_verification"]
                     if e["gold_turns_cited_by_memory"] > 0]
            lines += [
                f"- 建库快照（usage.memory.json）可用：对不可达题做了**直接验证**，"
                f"其中金标被记忆引用的 {len(cited)} 题实为检索问题（归 E4）。",
            ]
        else:
            lines += [
                "- **建库快照不可用**：归因仅基于可达集（间接判据），"
                "「不可达」不能断言为建库缺失，E1 残余数不可信。",
            ]
        rc = reachability["counts"]
        lines += [
            f"- **E4 检索侧目标集：{rc['e4_pool']} 题**（金标 not/partial 且可达）"
            f"——已落盘 `e4_target_keys.json` 供靶向子集构造。",
            f"- E1 建库残余：{rc['e1_residual']} 题（快照证实金标进了建库输入但没被任何记忆引用）。",
            f"- 视界截断：{rc['horizon_excluded']} 题（金标轮次在建库输入之外，E4/E1 都救不了）。",
            "",
            "| 召回状态 × 可达性 | 题数 |",
            "|---|---:|",
        ]
        for cell, count in sorted(reachability["attribution_matrix"].items()):
            lines.append(f"| {cell} | {count} |")
        lines += [
            "",
            "各对话可达率（可达 dia / 全部轮次）："
            + "、".join(
                f"{sid} {info['reachable_pct']}%"
                for sid, info in reachability["per_conversation"].items()
            ) + "。",
            "",
        ]
    lines += [
        "",
        "## 8. 使用边界",
        "",
        "- 本表回答的是「MN 输题是没召回还是答错」，**不是**「MN 比 SM 召回强/弱」。",
        "- Recall@K 依赖 MN 自己记录的 provenance；SM 未记录，故无法对称比较。",
        "- 召回覆盖是集合级 provenance 判定，不等于语义充分性：金标轮次被召回，"
        "不代表答案所需的语义关系已被正确绑定。",
        "- gap 定义为 session 级距离，不是轮次级；桶边界沿用综述 §4.2 的分桶习惯。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="证据级检索保真度审计（离线只读）")
    parser.add_argument("--rows", type=Path,
                        default=ROOT / "result" / "bench1000_locomo_20261023" / "rows.json")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "result" / "EVIDENCE_RECALL_20261108")
    parser.add_argument("--judged", type=Path, default=None,
                        help="rows_judged.json；缺省自动探测，缺失时 Judge 交叉表标为不可用")
    parser.add_argument("--attempts", type=Path, default=None,
                        help="attempts 目录（含 <sid>_memnexus/usage.memory.json）；"
                             "缺省自动探测 rows 同级目录，用于建库快照直接验证")
    parser.add_argument("--json", type=Path, default=None, help="额外写出 JSON 的路径")
    args = parser.parse_args(argv)

    for path, label in ((args.rows, "rows"), (args.dataset, "dataset")):
        if not path.exists():
            print(f"[FAIL] {label} 不存在：{path}", file=sys.stderr)
            return 3

    # Judge 是辅助指标：探测到就用，探测不到就显式标为不可用，绝不静默当成全 0。
    judged_path = args.judged
    if judged_path is None:
        candidate = args.rows.parent / "rows_judged.json"
        judged_path = candidate if candidate.exists() else None

    # 建库快照同理：自动探测，缺失时归因退化为只用可达集（间接口径）并显式标注。
    attempts_dir = args.attempts
    if attempts_dir is None:
        candidate = args.rows.parent / "attempts"
        attempts_dir = candidate if candidate.is_dir() else None

    report = audit(args.rows, args.dataset, judged_path=judged_path,
                   attempts_dir=attempts_dir)
    validation = report["validation"]

    # 自检门禁：映射不精确或有重复键就拒绝出报告，避免产出误导性诊断。
    blockers = []
    if not validation["gold_all_resolvable"]:
        blockers.append("存在不可解析的 gold_evidence 引用")
    if not validation["mapping_exact"]:
        blockers.append(f"存在 {validation['unresolved_events_total']} 个未解析检索事件")
    if validation["duplicate_keys"]:
        blockers.append(f"存在 {len(validation['duplicate_keys'])} 个重复键")
    if validation["mn_rows_missing_items"]:
        blockers.append(f"{len(validation['mn_rows_missing_items'])} 行 MN 无检索项")
    if blockers:
        print("[FAIL] 自检未通过，拒绝生成报告：", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 4

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.json or (args.out_dir / "evidence_recall.json")
    markdown_path = args.out_dir / "evidence_recall.md"
    items_path = args.out_dir / "evidence_recall_items.jsonl"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # 逐题明细单独落 JSONL：主 JSON 保持可读，明细仍可 grep 审计。
    per_item = report.pop("per_item", [])
    with items_path.open("w", encoding="utf-8") as handle:
        for entry in per_item:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    # E4/E1 目标集单独落盘：供 build_targeted_subset 构造靶向子集直接引用。
    reachability = report["memnexus"]["reachability"]
    e4_path = args.out_dir / "e4_target_keys.json"
    e4_path.write_text(json.dumps({
        "generated_at": report["generated_at"],
        "definition": ("金标 not/partial 召回且建库可达（其他题能召回或快照证实记忆中存在）"
                       "⇒ 检索/排序/配额问题，E4 的目标集"),
        "n": len(reachability["e4_pool_keys"]),
        "keys": reachability["e4_pool_keys"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    loss = report["memnexus"]["loss_decomposition"]["loss_totals"]
    nonadv = report["memnexus"]["nonadversarial"]["overall"]
    print(f"[OK] 自检通过：金标 {validation['gold_refs_resolvable']}/{validation['gold_refs_total']} "
          f"可解析；未解析事件 {validation['unresolved_events_total']}")
    print(f"[OK] 非对抗 R@5={nonadv['recall_at_k']['5']:.4f} "
          f"R@10={nonadv['recall_at_k']['10']:.4f} "
          f"R@全部={nonadv['recall_full_budget']:.4f} MRR={nonadv['mrr']:.4f}")
    print(f"[OK] MN 输题 {loss['losses']}：未召回 {loss['gold_not_recalled']}、"
          f"部分 {loss['gold_partially_recalled']}、全召回仍输 {loss['gold_fully_recalled']}")
    print(f"[OK] 检索侧占比 {loss['retrieval_side_share']:.4f}、"
          f"生成侧占比 {loss['generation_side_share']:.4f}")
    rc = reachability["counts"]
    snap_note = "建库快照直接验证" if reachability["snapshot_available"] else "仅可达集（间接）"
    print(f"[OK] 可达集归因（{snap_note}）：E4 检索侧目标 {rc['e4_pool']} 题、"
          f"E1 建库残余 {rc['e1_residual']} 题、视界截断 {rc['horizon_excluded']} 题")
    print(f"[OK] 输出 {json_path}")
    print(f"[OK] 输出 {markdown_path}")
    print(f"[OK] 输出 {e4_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
