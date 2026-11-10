"""导出"金标全召回但官方 F1 判 MN 输"题目的可复核清单（E5/E3/P2 的目标集分流）。

为什么需要这个脚本
------------------
`EVIDENCE_RECALL_FINDINGS_20261108.md §4.5` 显示：263 道非对抗输题里有 143 道
"金标证据已 100% 召回却仍输给 SM"。这 143 题的失败发生在**产出层**而非召回层，
但它们并非同一种病：有的该改答案渲染（E5），有的该改时间表达（E3），有的其实是
**证据在手却拒答**（P2 判定层）。混在一起做 A/B 会既改不准、也无法归因。

本脚本把该子集连同复核所需的全部上下文落盘为 Markdown + CSV + JSON：
  - 精确 provenance 召回状态（复用 evidence_recall_audit 的映射，不重造）
  - 官方 F1 / BLEU-1（复用 paper_protocol，与主表同源）
  - 双方 Judge 判定与理由（语义层面是否真的错）
  - 规则化初判类别 + 应归属实验（target_experiment）
  - **留空的 human_verdict 列**（人工填 fixable / not_fixable / uncertain）

纪律
----
- 只读输入，只写新文件；不修改任何结果或算法代码。
- 拒答判定复用 `paper_protocol.adversarial_success`，与主表 AdvSR 同一口径，
  不自造第三套标记（项目里已存在两套不同的 ABSTAIN_MARKERS，再增一套必然分裂）。
- 规则类别明确标注为"初判"，人工复核结论才是目标集依据。
- human_verdict 默认留空，脚本不替人做判断。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import evidence_recall_audit as era  # noqa: E402
import paper_protocol as pp  # noqa: E402

ADVERSARIAL = "adversarial"
TOLERANCE = 1e-12

# 规则初判：只用于给人工复核提供索引，不是结论。
RELATIVE_TIME = re.compile(r"\b(last|this|next|yesterday|today|tonight|ago|recent|past)\b|\(as of ", re.IGNORECASE)
POLAR_LEAD = re.compile(
    r"^(did|do|does|was|were|is|are|has|have|had|can|could|will|would|should|shall)\b", re.IGNORECASE
)
DURATION_ASK = re.compile(r"\bhow (long|many years|much time)\b", re.IGNORECASE)


def is_abstention(prediction: Any) -> bool:
    """判定是否为拒答，直接复用主表 AdvSR 的标记，避免口径分裂。

    `paper_protocol._ABSTAIN_MARKERS` 是官方主表判定对抗题拒答的唯一来源；
    本清单若另立一套标记，会出现"AdvSR 算拒答、错因分析不算"的矛盾。
    """
    return pp.adversarial_success({"prediction": prediction})


def word_count(text: Any) -> int:
    return len(re.findall(r"[a-z0-9]+", str(text or "").lower()))


ITEM_SPLIT = re.compile(r",|;|\band\b|\n")
LISTY_GOLD = re.compile(r",|;|\band\b", re.IGNORECASE)


def split_items(text: Any) -> list[str]:
    """把答案切成条目。金标常是 `A, B and C` 形式，必须按条目而非整句比对。"""
    return [part.strip() for part in ITEM_SPLIT.split(str(text or "")) if part.strip()]


def gold_coverage(gold: Any, prediction: Any) -> float:
    """金标条目中，被预测覆盖的比例（0.0–1.0）。

    覆盖判定用「子串 + 词集包含」双通道：
      - 子串：金标条目整体出现在预测里（大小写无关）；
      - 词集：金标条目的实词全部出现在预测词集里（容忍语序/冠词差异）。

    为什么需要这个量：「多余条目」「过度概括」都属于**颗粒度**问题，其成立前提是
    MN 已经答到了正确内容、只是多给或少给。若 MN 根本没覆盖金标，那是答错内容
    （例：金标 `a painting of Aragorn`，MN 答 `a man smoking a cigarette`），
    E5 的答案归一化改不动它，误归入 E5 会让目标集虚高、A/B 必然测不出效果。
    """
    gold_items = split_items(gold)
    if not gold_items:
        return 1.0
    pred_text = str(prediction or "").casefold()
    pred_tokens = set(re.findall(r"[a-z0-9]+", pred_text))
    covered = 0
    for item in gold_items:
        item_lower = item.casefold()
        item_tokens = set(re.findall(r"[a-z0-9]+", item_lower))
        if item_lower in pred_text:
            covered += 1
        elif item_tokens and item_tokens <= pred_tokens:
            covered += 1
    return covered / len(gold_items)


def gold_is_enumeration(gold: Any) -> bool:
    """金标本身是否为列举型（要求多值）。

    这类题**不能**压成单值：金标 `politics and the government` 若被规则压成
    `politics`，recall 直接掉。是否该保留多值属人工判断，脚本只标记不裁决。
    """
    return bool(LISTY_GOLD.search(str(gold or ""))) or len(split_items(gold)) > 1


STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be been being by with his her its "
    "their my your our as that this these those it they he she".split()
)


def content_tokens(text: Any) -> set[str]:
    """去虚词后的实词集合。虚词差异是官方 F1 的已知伪影（`in 2022` vs `2022`）。"""
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if w not in STOPWORDS}


def word_set_relation(gold: Any, prediction: Any) -> tuple[str, str]:
    """返回 (关系代号, 说明)：金标与预测在去虚词后的实词集合关系。

    这是把「语义正确但词面不匹配」进一步定性的机械判据。五种关系的处置**完全相反**，
    因此必须分开，否则会把"不该改"的题当成"该改"的：

      1_等价     : 去虚词后完全相同（`by biking` vs `Biking`）→ 纯评分器伪影，改了是倒退。
      2_真子集   : MN 词集 ⊂ 金标（`October 2022` vs `2022`）→ MN 丢了信息，真实缺陷。
      3_超集     : MN 词集 ⊃ 金标（`gripping with great actors` vs 更长表述）→ 冗余，可修。
      4_部分重叠 : 有交集但互不包含 → 方向不明，需人工判是"丢信息"还是"多给"。
      5_不相交   : 无共同实词 → 可能是 MN 更精确（`In France` vs `Paris`）、
                   同义改写（`get back on the road` vs `cruising around`）、
                   或 MN 更概括（`Super Mario and Zelda` vs `Nintendo`）——三者处置相反，必须人工。
    """
    gold_tokens = content_tokens(gold)
    pred_tokens = content_tokens(prediction)
    if not gold_tokens or not pred_tokens:
        return ("5_不相交", "一侧无实词，无法机械比对")
    if gold_tokens == pred_tokens:
        return ("1_等价", "去虚词后实词集合完全相同；差异纯属虚词/大小写，是 F1 伪影")
    if pred_tokens < gold_tokens:
        missing = sorted(gold_tokens - pred_tokens)[:6]
        return ("2_真子集", f"MN 缺金标实词 {missing}；属信息丢失")
    if gold_tokens < pred_tokens:
        extra = sorted(pred_tokens - gold_tokens)[:6]
        return ("3_超集", f"MN 含全部金标实词并多出 {extra}；属冗余")
    if gold_tokens & pred_tokens:
        return ("4_部分重叠",
                f"共有 {sorted(gold_tokens & pred_tokens)[:4]}，"
                f"金标独有 {sorted(gold_tokens - pred_tokens)[:4]}，"
                f"MN 独有 {sorted(pred_tokens - gold_tokens)[:4]}")
    return ("5_不相交",
            f"无共同实词（金标 {sorted(gold_tokens)[:4]} vs MN {sorted(pred_tokens)[:4]}）")


def initial_category(row: dict[str, Any], mn_judge: str | None = None) -> tuple[str, str]:
    """返回 (初判类别, 判定依据)。规则顺序即优先级，先匹配先生效。

    `mn_judge` 是 LLM Judge 的语义判定（CORRECT/WRONG/None），用作**语义真值**。
    词面判据（词数、覆盖度）只能说明「形式不像金标」，不能说明「答错了」；
    实测证明单靠词面会把 83% 的语义正确题误判成答错，因此必须两者联用。

    拒答检测必须放在最前：本清单已用 provenance 证明金标证据 100% 召回，
    因此"证据在手却拒答"是判定层过度拒答（对应 P2/E3），与答案形态无关。
    若先按词数/相对时间分类，会把这类题误判成"同义失配（不可修）"或
    "多余条目"，直接导致 E5 目标集错误。
    """
    prediction = str(row.get("prediction") or "")
    question = str(row.get("question") or "")
    gold = str(row.get("gold_answer") or "")
    pred_words, gold_words = word_count(prediction), word_count(gold)

    # 0. 拒答：用与主表 AdvSR 完全相同的判定标记，保证口径一致。
    if is_abstention(prediction):
        return ("证据已召回却拒答（判定过度）",
                "金标证据 100% 召回但 MN 输出拒答措辞；属判定层问题，非答案形态问题")

    # 1. 问时长却答绝对年份/日期：问题槽类型错配，确定性可修。
    if DURATION_ASK.search(question) and re.search(r"\b(19|20)\d{2}\b", prediction) \
            and not DURATION_ASK.search(gold):
        return ("duration-答成绝对时间", "问题问 how long，答案给了年份/日期")

    # 2. 相对时间未展开：答案里已有锚点日期，却没合成绝对日期。
    if RELATIVE_TIME.search(prediction) and not RELATIVE_TIME.search(gold):
        return ("相对时间未展开", "答案含相对时间表达，金标是绝对日期")

    # 3. 是非题答成实体：金标极短（Yes/No）而答案展开成句子。
    if POLAR_LEAD.match(question.strip()) and gold_words <= 2 and pred_words > gold_words + 3:
        return ("是非题未给Yes/No", "金标为 Yes/No 类极短答案，MN 输出了实体描述")

    # 4/5 前置门槛：颗粒度类别（过度概括、多余条目）要求 MN 已覆盖金标内容。
    # 未覆盖 => 词面对不上。但「词面对不上」有两种完全相反的情形，必须用 Judge 分开：
    #   - Judge=CORRECT：语义其实对了，只是措辞/粒度与金标不同（Paris vs In France、
    #     Nintendo vs Super Mario and Zelda、cruising around vs get back on the road）。
    #   - Judge=WRONG：真的答错内容（gold=coffee and pastries，MN 答成日期）。
    # 实测：63 题未覆盖里 Judge 判 CORRECT 占 52（83%），因此**词面覆盖度不能单独
    # 作为「答错」的判据**，必须以 Judge 为语义真值，覆盖度只作辅助说明。
    coverage = gold_coverage(gold, prediction)
    if coverage < 1.0:
        if mn_judge == "CORRECT":
            return ("语义正确但词面/粒度不匹配",
                    f"Judge 判语义正确，但金标条目词面覆盖率仅 {coverage:.2f}；"
                    f"需人工判断 MN 是更精确、更概括还是同义改写")
        if mn_judge == "WRONG":
            return ("答错内容（未覆盖金标）",
                    f"金标条目覆盖率 {coverage:.2f} < 1.00 且 Judge 判语义错误；"
                    f"属内容错误，非颗粒度问题")
        # Judge 缺失时**不得**断言内容错误：同义改写（her headspace vs her mental health）
        # 词面覆盖度同样为 0，但没有语义真值就无法区分「改写」与「答错」。
        return ("同义/词面失配",
                f"词面覆盖率 {coverage:.2f} < 1.00 且 Judge 不可用；无语义真值，保守归为词面失配")

    # 4. 过度概括：答案比金标更短且金标含具体值。
    if pred_words < gold_words and gold_words >= 2:
        return ("过度概括", f"答案 {pred_words} 词 vs 金标 {gold_words} 词，且已覆盖金标全部条目")

    # 5. 多余条目：答案显著长于金标。
    if pred_words >= gold_words + 3:
        return ("多余条目", f"答案 {pred_words} 词 vs 金标 {gold_words} 词，且已覆盖金标全部条目")

    return ("同义/词面失配", "长度相近但词面不匹配；改渲染无法解决")


# 初判类别 → (建议 verdict, 应归属的实验)。
# 归属实验比 verdict 更重要：本清单是为 E5 导出的，但其中一部分题的病因不在
# 答案渲染层，把它们放进 E5 目标集会导致"改了没效果"或"改错地方"。
CATEGORY_ROUTING = {
    "证据已召回却拒答（判定过度）": ("fixable", "P2/E3 义务与拒答判定"),
    "duration-答成绝对时间": ("fixable", "E3 时间渲染"),
    "相对时间未展开": ("fixable", "E3 时间渲染"),
    "是非题未给Yes/No": ("fixable", "E5 答案归一化"),
    "过度概括": ("fixable", "E5 答案归一化"),
    "多余条目": ("fixable", "E5 答案归一化"),
    # 金标未覆盖 = 答错内容，归一化改不动；应属检索/绑定或表示层（E4/E1）。
    "答错内容（未覆盖金标）": ("not_fixable", "E4/E1 检索与表示层（非产出层，E5 改不动）"),
    # 语义对但词面对不上金标：混合体，**不能给统一 verdict**。三种相反情形需人工分开：
    #   (a) MN 比金标更精确（gold=In France, MN=Paris）→ 不该改，改了是倒退；
    #   (b) MN 比金标更概括/丢信息（gold=Super Mario and Zelda, MN=Nintendo）→ 真实缺陷，可修；
    #   (c) 纯同义改写（gold=get back on the road, MN=cruising around）→ 评分器伪影，不该改。
    "语义正确但词面/粒度不匹配": ("uncertain", "需人工裁决（更精确/更概括/同义改写三者相反）"),
    "同义/词面失配": ("not_fixable", "不优化（改渲染无法解决，强行优化=过拟合评分器）"),
}
FIXABLE_SUGGESTION = {key: value[0] for key, value in CATEGORY_ROUTING.items()}
TARGET_EXPERIMENT = {key: value[1] for key, value in CATEGORY_ROUTING.items()}

# 拒答类改动的高风险提示：放松拒答会直接威胁 AdvSR（对抗题靠拒答得分）。
ABSTENTION_RISK = (
    "放松拒答判定会直接影响 AdvSR（对抗题正是靠拒答得分，当前 MN 86.67% vs SM 84.00%）。"
    "任何针对本类的改动必须把 AdvSR 列为守护指标，且只在非对抗路由上生效"
    "（参考 temporal_rescue 的做法：negative_gate 路由上永不触发）。"
)


def load_judge_reasons(judged_path: Path | None) -> dict[tuple[Any, Any, str], str]:
    """一次性读取 Judge 理由并建索引，避免在主循环里对每条记录重读整个文件。

    字段名 `judge_reason` 已在 rows_judged.json 实测确认；缺失时返回空字典，
    调用方按 None 处理，不得用空串冒充"无理由"。
    """
    if judged_path is None or not judged_path.exists():
        return {}
    payload = json.loads(judged_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("rows_judged.json 应为行列表")
    reasons: dict[tuple[Any, Any, str], str] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        reason = row.get("judge_reason")
        if not reason:
            continue
        reasons[(row.get("sample_id"), row.get("qa_index"), row.get("system"))] = str(reason)
    return reasons


def build_records(rows_path: Path, dataset_path: Path, judged_path: Path | None) -> list[dict[str, Any]]:
    """筛出"金标全召回 + F1 判 MN 输"的题，并附复核所需全部上下文。"""
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    dialogs = era.load_dialog_maps(dataset_payload)
    rows = era.load_rows(rows_path)
    judge_labels = era.load_judged(judged_path)
    # Judge 理由单独建一次索引：避免在主循环里对每条记录重读整个 judged 文件。
    judge_reasons = load_judge_reasons(judged_path)
    samples = {str(item["sample_id"]): item for item in dataset_payload}

    index: dict[tuple[Any, Any, str], dict[str, Any]] = {}
    for row in rows:
        index[(row.get("sample_id"), row.get("qa_index"), row.get("system"))] = row

    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("system") != "MemNexus" or row.get("category") == ADVERSARIAL:
            continue
        sample_id = str(row.get("sample_id"))
        qa_index = row.get("qa_index")
        dialog = dialogs.get(sample_id)
        if dialog is None:
            raise ValueError(f"{sample_id}/{qa_index} 的对话不在数据集中")
        sm_row = index.get((sample_id, qa_index, "SimpleMem"))
        if sm_row is None:
            raise ValueError(f"{sample_id}/{qa_index} 缺少 SimpleMem 配对行")

        # 只保留金标证据被完整召回的题：这些题的失败发生在产出层，不是召回层。
        trace = era.provenance_trace(row, dialog)
        if trace["recall_full_budget"] is None or trace["recall_full_budget"] < 1.0:
            continue

        mn_f1, sm_f1 = pp.f1_of(row), pp.f1_of(sm_row)
        delta = mn_f1 - sm_f1
        if delta >= -TOLERANCE:
            continue  # 只保留 MN 输的题

        mn_judge = judge_labels.get((sample_id, qa_index, "MemNexus"))
        sm_judge = judge_labels.get((sample_id, qa_index, "SimpleMem"))
        mn_judge_reason = judge_reasons.get((sample_id, qa_index, "MemNexus"))
        # Judge 必须在归类之前取得：词面覆盖度分不开「更精确/更概括/同义改写」，
        # 必须以 Judge 作为语义真值，否则会误把 83% 的语义正确题判成「答错内容」。
        category, basis = initial_category(row, mn_judge)

        records.append({
            "key": f"{sample_id}/{qa_index}",
            "sample_id": sample_id,
            "qa_index": qa_index,
            "category": row.get("category"),
            "question": row.get("question"),
            "gold_answer": row.get("gold_answer"),
            "mn_prediction": row.get("prediction"),
            "sm_prediction": sm_row.get("prediction"),
            "mn_f1": round(mn_f1, 4),
            "sm_f1": round(sm_f1, 4),
            "f1_delta": round(delta, 4),
            "mn_judge": mn_judge,
            "sm_judge": sm_judge,
            "mn_judge_reason": mn_judge_reason,
            "evidence_gap": trace["evidence_gap"],
            "gold_set_size": trace["gold_set_size"],
            "n_retrieved": trace["n_items"],
            "mn_pred_words": word_count(row.get("prediction")),
            "gold_words": word_count(row.get("gold_answer")),
            "sm_pred_words": word_count(sm_row.get("prediction")),
            # 三个判据字段：复核时可直接看到分类依据，而不只看到结论。
            "gold_coverage": round(gold_coverage(row.get("gold_answer"), row.get("prediction")), 4),
            "gold_is_enumeration": gold_is_enumeration(row.get("gold_answer")),
            "word_set_relation": word_set_relation(row.get("gold_answer"), row.get("prediction"))[0],
            "word_set_note": word_set_relation(row.get("gold_answer"), row.get("prediction"))[1],
            "initial_category": category,
            "initial_basis": basis,
            "suggested_verdict": FIXABLE_SUGGESTION[category],
            "target_experiment": TARGET_EXPERIMENT[category],
            "human_verdict": "",  # 人工填：fixable / not_fixable / uncertain
            "human_note": "",
            "gold_evidence": row.get("gold_evidence"),
            # 金标证据原文，供人工判断"答案是否真的能从证据推出"
            "gold_evidence_text": gold_text_for(samples[sample_id], row.get("gold_evidence") or []),
        })

    # 复核档位必须在记录字段齐全后计算（依赖 mn_judge / gold_is_enumeration）。
    for record in records:
        tier, reason = review_tier(record)
        record["review_tier"] = tier
        record["review_tier_reason"] = reason

    # 排序：先按复核档位（T1 最该看），档位内按 ΔF1 升序（损失最大的在前）。
    # 这样打开 CSV 从上往下读，就是「优先级最高的待判断题」的顺序。
    records.sort(key=lambda record: (record["review_tier"], record["f1_delta"], record["key"]))
    return records


def gold_text_for(sample: dict[str, Any], gold_evidence: list[Any]) -> str:
    """按 dia_id 取出金标证据原文，便于人工判断答案是否可由证据推出。"""
    conversation = sample.get("conversation") or {}
    parts: list[str] = []
    for dia_id in gold_evidence:
        parsed = era.DIA_RE.fullmatch(str(dia_id))
        if not parsed:
            continue
        turns = conversation.get(f"session_{parsed.group(1)}") or []
        index = int(parsed.group(2)) - 1
        if 0 <= index < len(turns):
            speaker = turns[index].get("speaker") or "?"
            parts.append(f"[{dia_id}] {speaker}: {turns[index].get('text') or ''}")
    return "\n".join(parts)


def review_tier(record: dict[str, Any]) -> tuple[str, str]:
    """返回 (复核档位, 理由)。目的：把人工复核压到真正有歧义的题上。

    三档的划分依据是「结论能否由机械判据独立证明」，而不是题量：

    T3_可批量确认：语义与形态判据一致，机械可证。
      例：相对时间未展开 / duration 答成绝对时间 —— MN Judge 判 CORRECT（语义对），
      而金标是绝对日期、MN 输出相对表达（形态错），两条独立证据同向，无需逐题读。
      复核动作：抽查 2–3 题确认规则没跑偏，然后整批接受。

    T2_抽样确认：规则给出方向但存在已知反例风险。
      例：金标本身是列举型（`A and B`）却被归为「多余条目」——压成单值会掉 recall，
      但是否该保留多值取决于问题措辞，需要人读题。这类只需抽样定策略，不必逐题。

    T1_必须逐题：机械判据互相矛盾，或涉及会威胁守护指标的改动。
      例：证据已在 prompt 里却拒答（MN Judge 判 WRONG、AdvSR 是一票否决项）。
      这类错判的代价最高（P2 实验整个消失或 AdvSR 回退），必须逐题确认。
    """
    category = record["initial_category"]
    judge = record.get("mn_judge")
    enumeration = bool(record.get("gold_is_enumeration"))
    relation = str(record.get("word_set_relation") or "")

    if category in ("证据已召回却拒答（判定过度）",):
        return ("T1_必须逐题",
                "证据已进 prompt 却拒答，且放松拒答直接威胁 AdvSR（一票否决项）")
    if category == "答错内容（未覆盖金标）":
        return ("T1_必须逐题",
                "Judge 判语义错且词面未覆盖金标；属内容错误，需确认是否真该归 E4/E1")

    # 「语义正确但词面/粒度不匹配」与「同义/词面失配」都靠词集关系定性。
    # 关系 1/2/3 的方向可机械证明（等价=伪影、子集=丢信息、超集=冗余），
    # 关系 4/5 的方向不明（部分重叠、完全不相交），才真正需要人读题。
    if category in ("语义正确但词面/粒度不匹配", "同义/词面失配"):
        if relation.startswith("1_"):
            return ("T3_可批量确认",
                    "去虚词后与金标实词完全等价 ⇒ 纯 F1 伪影，改了是倒退；抽 2 题确认即可整批接受")
        if relation.startswith("2_"):
            return ("T2_抽样确认",
                    "MN 词集是金标真子集 ⇒ 方向确定为「丢信息」；抽样确认补全策略即可")
        if relation.startswith("3_"):
            return ("T2_抽样确认",
                    "MN 词集是金标超集 ⇒ 方向确定为「冗余」；抽样确认压缩策略即可")
        # 关系 4/5：方向机械不可判。但若**双方 Judge 都判 CORRECT**，则该题在语义层
        # 并不存在输赢，F1 判 MN 输只是表示风格差异 ⇒ 属评分器口径，不该为它改渲染。
        # 此时需要人做的不是逐题裁决，而是**抽样确认 Judge 没有放水**（一次性判断）。
        if judge == "CORRECT" and record.get("sm_judge") == "CORRECT":
            return ("T2_抽样确认",
                    "双方 Judge 均判 CORRECT 而 F1 判 MN 输 ⇒ 属 F1 评分器口径差异，"
                    "不应为提分改渲染；抽样确认 Judge 未放水即可整批判 not_fixable")
        if relation.startswith("4_"):
            return ("T1_必须逐题",
                    "与金标部分重叠、互不包含 ⇒ 方向不明（丢信息 or 多给），必须人工判")
        return ("T1_必须逐题",
                "与金标无共同实词 ⇒ 可能是 MN 更精确 / 同义改写 / 更概括，三者处置相反，必须人工判")

    if category in ("相对时间未展开", "duration-答成绝对时间", "是非题未给Yes/No"):
        if judge == "CORRECT":
            return ("T3_可批量确认",
                    "Judge 判语义正确 + 形态判据明确，两条独立证据同向")
        return ("T2_抽样确认", f"形态判据成立但 Judge={judge or '缺失'}，语义需抽查")
    # 颗粒度类：多余条目 / 过度概括
    if enumeration:
        return ("T2_抽样确认",
                "金标本身是列举型；压成单值会掉 recall，是否保留多值需读题定策略")
    if judge == "CORRECT":
        return ("T2_抽样确认",
                "已覆盖金标且语义判对，属真颗粒度冗余；抽样确认压缩策略即可")
    return ("T1_必须逐题",
            f"判为颗粒度问题但 Judge={judge or '缺失'}，语义与形态判据不一致")


CSV_COLUMNS = (
    "key", "category", "initial_category", "target_experiment", "suggested_verdict",
    "review_tier", "review_tier_reason",
    "human_verdict", "human_note",
    "f1_delta", "mn_f1", "sm_f1", "mn_judge", "sm_judge", "question", "gold_answer",
    "mn_prediction", "sm_prediction", "mn_pred_words", "gold_words", "sm_pred_words",
    "gold_coverage", "gold_is_enumeration", "word_set_relation", "word_set_note",
    "initial_basis", "evidence_gap", "gold_set_size", "n_retrieved", "gold_evidence_text",
    "mn_judge_reason", "sample_id", "qa_index", "gold_evidence",
)


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["gold_evidence"] = ";".join(str(value) for value in (record.get("gold_evidence") or []))
            writer.writerow(row)


def render_markdown(records: list[dict[str, Any]], judged_available: bool) -> str:
    by_category = Counter(record["initial_category"] for record in records)
    by_type = Counter(record["category"] for record in records)
    both_correct = sum(
        1 for record in records
        if record["mn_judge"] == "CORRECT" and record["sm_judge"] == "CORRECT"
    )
    mn_correct = sum(1 for record in records if record["mn_judge"] == "CORRECT")

    lines: list[str] = [
        "# 产出层输题复核清单：金标全召回但 F1 判 MN 输",
        "",
        f"生成：{datetime.now(timezone.utc).isoformat()}",
        "",
        f"共 **{len(records)}** 题。筛选条件（三条同时成立，均由 `evidence_recall_audit` 的精确 "
        "provenance 判定，非词面启发式）：",
        "",
        "1. 非 adversarial（其官方指标是 AdvSR，不进本清单）；",
        "2. 金标证据 **100% 被召回**（`recall_full_budget == 1.0`）⇒ 失败发生在产出层，不是召回层；",
        "3. 官方 F1 判定 **MN 输给 SM**（ΔF1 < 0，容差 1e-12）。",
        "",
        "> **本清单不只服务 E5。** 其中一部分题的病因不在答案渲染层（例如"
        "「证据已召回却拒答」属判定层），已用 `target_experiment` 列分流。"
        "把判定层问题塞进 E5 会导致「改了没效果」。",
        "",
        "## 0. 怎么用这份清单",
        "",
        "- 逐题填 CSV 的 `human_verdict` 列：`fixable`（确定性可修）/ `not_fixable`（评分器伪影，改了等于过拟合）"
        "/ `uncertain`。`human_note` 写一句理由。",
        "- **先看 `review_tier` 列，它决定你要花多少力气**：",
        "  - `T1_必须逐题`：机械判据互相矛盾（如形态判为冗余、Judge 却判语义错），错判代价最高，必须逐题读；",
        "  - `T2_抽样确认`：规则方向可信，每类抽 3–5 题定策略后整批填同一 verdict；",
        "  - `T3_可批量确认`：语义（Judge）与形态（规则）两条独立证据同向，抽 2 题确认后可整批接受。",
        "- 判据字段（供你自行核验分类是否成立，不必信任规则）：",
        "  - `gold_coverage`：金标条目被 MN 预测覆盖的比例。**< 1.0 说明 MN 没答到正确内容**，"
        "属内容错误而非颗粒度问题，E5 的答案归一化改不动它；",
        "  - `gold_is_enumeration`：金标本身是否为列举型（如 `A and B`）。为 True 时**不能压成单值**，"
        "否则 recall 直接掉——这类题是否保留多值需读题判断。",
        "- `initial_category`、`suggested_verdict`、`target_experiment` 都只是**规则初判**，用于排序与分流；"
        "**最终目标集 = human_verdict 为 fixable 的题**，不是初判类别。",
        "- `initial_category` 为「同义/词面失配」的题默认建议 `not_fixable`：这类题答案与金标语义相近、"
        "只是措辞不同，改渲染无法解决，强行优化等于过拟合 F1 评分器。",
        "- `initial_category` 为「答错内容（未覆盖金标）」的题**不属产出层**：证据虽在 prompt 里，"
        "但模型没从中得出正确内容，应归 E4/E1（检索与表示层），不要塞进 E5。",
        "- 复核完成后运行本脚本的 `--stats-from-csv` 模式，用人工结论替换规则初判的统计。",
        "",
        "> **不想逐题看？** 同目录 `DECISIONS_REQUIRED.md` 已把 143 题归并成 **6 个决策**，"
        "每项附实测题量、回收上限（pp pooled）、建议与风险；其中 4 项只需确认一条策略，"
        "真正需要逐题读的只有 30 题（29 题分属 2 条策略）。",
        "",
        f"> **拒答类改动的高风险提示**：{ABSTENTION_RISK}",
        "",
        "## 1. 规则初判分布（**未经人工复核，不可直接当目标题量**）",
        "",
        "| 初判类别 | 题数 | 建议 verdict | 应归属实验 | 平均ΔF1 |",
        "|---|---:|---|---|---:|",
    ]
    for category, count in by_category.most_common():
        group = [record for record in records if record["initial_category"] == category]
        mean_delta = sum(record["f1_delta"] for record in group) / len(group)
        lines.append(
            f"| {category} | {count} | {FIXABLE_SUGGESTION[category]} "
            f"| {TARGET_EXPERIMENT[category]} | {mean_delta:+.3f} |"
        )

    by_experiment = Counter(record["target_experiment"] for record in records)
    lines += [
        "",
        "### 按应归属实验分流",
        "",
        "| 实验 | 题数 | 平均ΔF1 |",
        "|---|---:|---:|",
    ]
    for experiment, count in by_experiment.most_common():
        group = [record for record in records if record["target_experiment"] == experiment]
        mean_delta = sum(record["f1_delta"] for record in group) / len(group)
        lines.append(f"| {experiment} | {count} | {mean_delta:+.3f} |")

    # 复核工作量表：把「逐题判断」压到真正有歧义的题上。
    # 每档的动作不同——T3 不需要逐题读，T1 必须逐题读。
    by_tier = Counter(record["review_tier"] for record in records)
    tier_action = {
        "T1_必须逐题": "逐题填 `human_verdict`（机械判据互相矛盾，错判代价最高）",
        "T2_抽样确认": "每类抽 3–5 题定策略，其余整批填同一 verdict（规则方向可信）",
        "T3_可批量确认": "抽 2 题确认规则没跑偏，整批接受（语义+形态两条独立证据同向）",
    }
    lines += [
        "",
        "### 复核工作量（**不需要逐题看 143 题**）",
        "",
        "档位划分依据是「结论能否由机械判据独立证明」，不是题量。CSV 已按档位排序，",
        "打开后从上往下读即是优先级顺序。",
        "",
        "| 档位 | 题数 | 你要做的动作 |",
        "|---|---:|---|",
    ]
    for tier in ("T1_必须逐题", "T2_抽样确认", "T3_可批量确认"):
        if not by_tier.get(tier):
            continue
        lines.append(f"| {tier} | {by_tier[tier]} | {tier_action[tier]} |")
    must_read = by_tier.get("T1_必须逐题", 0)
    lines += [
        "",
        f"**逐题必读只有 {must_read} 题**（占 {100 * must_read / max(1, len(records)):.0f}%）；"
        f"其余 {len(records) - must_read} 题可按档位批量或抽样确认。",
        "",
    ]

    lines += [
        "",
        "### 按题型",
        "",
        "| 题型 | 题数 |",
        "|---|---:|",
    ]
    for category, count in sorted(by_type.items()):
        lines.append(f"| {category} | {count} |")

    lines += [
        "",
        "### Judge 交叉（辅助，不替代人工复核）",
        "",
    ]
    if judged_available:
        lines += [
            f"- MN Judge CORRECT：**{mn_correct}/{len(records)}**（{100 * mn_correct / len(records):.1f}%）",
            f"- 双方 Judge 都 CORRECT：**{both_correct}/{len(records)}**"
            f"（{100 * both_correct / len(records):.1f}%）",
            "",
            "**这两个数字必须一起读**：`both_correct` 高说明官方 F1 存在词面失配；"
            "但 MN Judge 低于 SM（本子集 SM 约 95.8%）说明 MN 语义上也确实略逊。"
            "**不能只取前者当作「纯属评分器问题」的证据。**",
        ]
    else:
        lines += [
            "**Judge 数据不可用**（未找到 `rows_judged.json`）。本清单仍可复核，"
            "但缺少语义层面的交叉参照，`mn_judge`/`sm_judge` 列为空。",
        ]

    lines += [
        "",
        "## 2. 逐题清单（按应归属实验分组，组内按 ΔF1 升序）",
        "",
        "完整字段见同目录 CSV（UTF-8 BOM，Excel 可直接打开）。按实验分组是为了让复核"
        "能按改动批次进行：同一实验的题应一起判断，避免把判定层问题混进渲染层改动。",
        "",
    ]

    for experiment in sorted(by_experiment, key=lambda name: (-by_experiment[name], name)):
        experiment_records = [record for record in records if record["target_experiment"] == experiment]
        mean_delta = sum(record["f1_delta"] for record in experiment_records) / len(experiment_records)
        lines += [
            f"## 2.{list(sorted(by_experiment, key=lambda n: (-by_experiment[n], n))).index(experiment) + 1} "
            f"{experiment}（{len(experiment_records)} 题，平均ΔF1 {mean_delta:+.3f}）",
            "",
        ]
        if "拒答" in experiment:
            lines += [f"> {ABSTENTION_RISK}", ""]
        sub_counts = Counter(record["initial_category"] for record in experiment_records)
        lines += [
            "初判构成：" + "、".join(f"{name} {count}" for name, count in sub_counts.most_common()),
            "",
        ]
        for record in experiment_records:
            judge_text = (
                f"MN Judge={record['mn_judge'] or '—'} / SM Judge={record['sm_judge'] or '—'}"
                if judged_available else "Judge 不可用"
            )
            lines += [
                f"### `{record['key']}` · {record['category']} · ΔF1 {record['f1_delta']:+.3f} "
                f"(MN {record['mn_f1']:.3f} vs SM {record['sm_f1']:.3f}) · {judge_text}",
                "",
                f"- **Q**：{record['question']}",
                f"- **GOLD**：{record['gold_answer']}",
                f"- **MN**：{record['mn_prediction']}",
                f"- **SM**：{record['sm_prediction']}",
                f"- 词数 MN/GOLD/SM：{record['mn_pred_words']}/{record['gold_words']}/{record['sm_pred_words']}"
                f"；evidence gap={record['evidence_gap']}；金标条数={record['gold_set_size']}；"
                f"检索项={record['n_retrieved']}",
                f"- 初判：{record['initial_category']}（{record['initial_basis']}）；"
                f"建议 {record['suggested_verdict']}",
                f"- 金标证据原文：",
            ]
            for evidence_line in str(record["gold_evidence_text"] or "（无）").splitlines():
                lines.append(f"  > {evidence_line}")
            if record["mn_judge_reason"]:
                lines.append(f"- MN Judge 理由：{record['mn_judge_reason']}")
            lines += [
                f"- **human_verdict**：`________`（fixable / not_fixable / uncertain）",
                f"- **human_note**：`________`",
                "",
            ]

    lines += [
        "## 3. 复核后的处理规则",
        "",
        "1. `fixable` 的题按 `target_experiment` 分流后，分别构成 **E5（答案归一化）**、"
        "**E3（时间渲染）**、**P2（义务与拒答判定）** 的目标集；各实验的题量决定其收益上限，"
        "须用人工复核后的数字重新估算（此前 ≤+2.7 pp pooled 的估算基于规则初判，且把判定层"
        "与渲染层混在一起，人工复核后必须分开计）。",
        "2. `not_fixable` 的题**不得**为提升 F1 而改动渲染逻辑；若它们占比很高，"
        "说明官方 F1 对本系统的表示风格存在系统性低估，应在论文中作为 limitation 说明，"
        "并同时报 Judge。",
        "3. `uncertain` 的题不进目标集，单独记录；若数量超过 20%，说明初判规则不可用，"
        "应改进规则后重新导出，而不是强行二分。",
        "4. **拒答类（P2）单独成实验，不得与 E5 合并**：它改的是「是否作答」的判定，"
        "E5 改的是「作答后怎么渲染」。两者混在一个 A/B 里无法归因，且拒答改动直接威胁 AdvSR。",
        "5. 复核结论落盘后，各实验的预注册文档必须引用**本清单的人工统计**，不得引用规则初判。",
        "",
    ]
    return "\n".join(lines)


def stats_from_csv(csv_path: Path) -> dict[str, Any]:
    """读取人工复核后的 CSV，用 human_verdict 产出正式目标集统计。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV 为空：{csv_path}")
    verdicts = Counter((row.get("human_verdict") or "").strip().lower() for row in rows)
    unfilled = verdicts.get("", 0)
    fixable = [row for row in rows if (row.get("human_verdict") or "").strip().lower() == "fixable"]
    by_initial = Counter(row["initial_category"] for row in fixable)
    by_type = Counter(row["category"] for row in fixable)
    # 人工确认 fixable 的题按实验分流：这才是各实验真正的目标题量。
    by_experiment = Counter(
        (row.get("target_experiment") or "未分流") for row in fixable
    )
    # 初判为 fixable 但人工判 not_fixable 的题：规则过宽的证据
    overridden = Counter(
        row["initial_category"] for row in rows
        if row.get("suggested_verdict") == "fixable"
        and (row.get("human_verdict") or "").strip().lower() == "not_fixable"
    )
    return {
        "total": len(rows),
        "unfilled": unfilled,
        "verdicts": dict(verdicts),
        "fixable_count": len(fixable),
        "fixable_by_experiment": dict(by_experiment),
        "fixable_by_initial_category": dict(by_initial),
        "fixable_by_type": dict(by_type),
        "suggested_fixable_overridden_to_not_fixable": dict(overridden),
        "review_complete": unfilled == 0,
        "note": "只有 review_complete 为 true 时，fixable_count 与 fixable_by_experiment "
                "才能作为各实验（E5/E3/P2）的目标题量；否则只是规则初判。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="导出产出层输题复核清单并按 E5/E3/P2 分流（只读）")
    parser.add_argument("--rows", type=Path,
                        default=ROOT / "result" / "bench1000_locomo_20261023" / "rows.json")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json")
    parser.add_argument("--judged", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "result" / "OUTPUT_LAYER_LOSSES_20261109")
    parser.add_argument("--stats-from-csv", type=Path, default=None,
                        help="读取已人工复核的 CSV，输出正式目标集统计（JSON）")
    args = parser.parse_args(argv)

    if args.stats_from_csv is not None:
        if not args.stats_from_csv.exists():
            print(f"[FAIL] CSV 不存在：{args.stats_from_csv}", file=sys.stderr)
            return 3
        stats = stats_from_csv(args.stats_from_csv)
        out_path = args.stats_from_csv.with_suffix(".stats.json")
        out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"[OK] 统计写入 {out_path}")
        if not stats["review_complete"]:
            print(f"[WARN] 尚有 {stats['unfilled']} 题未填 human_verdict，"
                  f"fixable_count 不能作为最终目标题量", file=sys.stderr)
        return 0

    for path, label in ((args.rows, "rows"), (args.dataset, "dataset")):
        if not path.exists():
            print(f"[FAIL] {label} 不存在：{path}", file=sys.stderr)
            return 3

    judged_path = args.judged
    if judged_path is None:
        candidate = args.rows.parent / "rows_judged.json"
        judged_path = candidate if candidate.exists() else None
    judged_available = judged_path is not None and judged_path.exists()

    records = build_records(args.rows, args.dataset, judged_path)
    if not records:
        print("[FAIL] 没有符合条件的题目；请检查筛选条件或输入路径", file=sys.stderr)
        return 4

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "output_layer_losses.csv"
    markdown_path = args.out_dir / "OUTPUT_LAYER_LOSSES_REVIEW.md"
    json_path = args.out_dir / "output_layer_losses.json"
    write_csv(records, csv_path)
    markdown_path.write_text(render_markdown(records, judged_available), encoding="utf-8")
    json_path.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "judge_available": judged_available,
         "filter": "non-adversarial AND gold fully recalled AND official F1 judges MN loss",
         "initial_category_counts": dict(Counter(r["initial_category"] for r in records)),
         "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(record["initial_category"] for record in records)
    print(f"[OK] 导出 {len(records)} 题（judge_available={judged_available}）")
    for category, count in counts.most_common():
        print(f"     {category}: {count} → 建议 {FIXABLE_SUGGESTION[category]}")
    print(f"[OK] CSV     {csv_path}")
    print(f"[OK] Markdown {markdown_path}")
    print(f"[OK] JSON    {json_path}")
    print("[NEXT] 人工填写 CSV 的 human_verdict 列后，运行：")
    print(f"       python experiments/export_output_layer_losses.py --stats-from-csv {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
