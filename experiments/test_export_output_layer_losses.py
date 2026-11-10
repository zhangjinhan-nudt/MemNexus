"""export_output_layer_losses 的回归测试。

重点锁住本轮修复的核心 bug：拒答检测必须优先于所有词数/形态规则。
此前 18 道"证据已召回却拒答"的题被词数规则误判成「同义失配（不可修）」或
「多余条目」，导致 P2（判定层）问题被错误塞进 E5（渲染层）目标集。
若不锁住，后续任何人调整规则顺序都可能让这个 bug 复活。

运行：
  python -m pytest experiments/test_export_output_layer_losses.py -q --basetemp=.pytest_tmp/output_layer
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import export_output_layer_losses as m  # noqa: E402
import paper_protocol as pp  # noqa: E402

ABSTENTION_CATEGORY = "证据已召回却拒答（判定过度）"


class AbstentionPriorityTest(unittest.TestCase):
    """拒答必须最先归类，且复用主表 AdvSR 口径。"""

    def test_is_abstention_reuses_paper_protocol_markers(self):
        # 与 paper_protocol.adversarial_success 完全一致，不得自造第三套标记。
        for text in ("Not mentioned in the conversation",
                     "No information available about this.",
                     "I do not have enough information.",
                     "I don't have enough information."):
            self.assertTrue(m.is_abstention(text), f"应判为拒答：{text}")
            self.assertEqual(m.is_abstention(text),
                             pp.adversarial_success({"prediction": text}))
        for text in ("Miso the cat", "2013", "a shiny orange car"):
            self.assertFalse(m.is_abstention(text))

    def test_abstention_beats_word_count_rules(self):
        """关键回归：拒答答案即使词数很多（像"多余条目"）也必须归为拒答。"""
        row = {
            # 长答案 + 拒答措辞：旧逻辑会因 pred_words 多而误判"多余条目"
            "prediction": "Not mentioned in the conversation, no further details were given at all",
            "question": "What did Melanie paint?",
            "gold_answer": "a sunrise",
        }
        category, basis = m.initial_category(row)
        self.assertEqual(category, ABSTENTION_CATEGORY)
        self.assertIn("拒答", basis)

    def test_abstention_beats_overgeneralization(self):
        """拒答答案词数少于金标时，旧逻辑会误判"过度概括"。"""
        row = {
            "prediction": "Not mentioned",
            "question": "What does the bowl remind Melanie of?",
            "gold_answer": "art and self-expression",
        }
        category, _ = m.initial_category(row)
        self.assertEqual(category, ABSTENTION_CATEGORY)

    def test_abstention_routes_to_judgement_layer_not_e5(self):
        """拒答必须分流到判定层（P2），不能进 E5 渲染层。"""
        self.assertEqual(m.TARGET_EXPERIMENT[ABSTENTION_CATEGORY], "P2/E3 义务与拒答判定")
        self.assertNotIn("E5", m.TARGET_EXPERIMENT[ABSTENTION_CATEGORY])
        self.assertEqual(m.FIXABLE_SUGGESTION[ABSTENTION_CATEGORY], "fixable")


class CategoryRoutingTest(unittest.TestCase):
    """每个 initial_category 的返回值都必须有完整路由，避免 KeyError。"""

    def test_every_category_has_routing(self):
        produced = {
            m.initial_category(row)[0]
            for row in [
                {"prediction": "Not mentioned", "question": "Q", "gold_answer": "a b c"},
                {"prediction": "2013", "question": "How long ago was it?", "gold_answer": "10 years ago"},
                {"prediction": "this past weekend (as of 2023-10-20)",
                 "question": "When did they go?", "gold_answer": "14 October 2023"},
                {"prediction": "Yes both had childhood dogs and they loved them a lot",
                 "question": "Did they have dogs?", "gold_answer": "Yes"},
                {"prediction": "a", "question": "What sparked it?",
                 "gold_answer": "Super Mario and Zelda games"},
                {"prediction": "self-doubt, tech stuff, workplace stuff, more stuff",
                 "question": "What challenge?", "gold_answer": "self-doubt"},
                {"prediction": "her mental health", "question": "Great for what?",
                 "gold_answer": "her headspace"},
            ]
        }
        for category in produced:
            self.assertIn(category, m.CATEGORY_ROUTING, f"类别缺路由：{category}")
            verdict, target = m.CATEGORY_ROUTING[category]
            self.assertIn(verdict, ("fixable", "not_fixable"))
            self.assertTrue(target)

    def test_synonym_mismatch_is_not_fixable(self):
        """同义失配默认不可修：改渲染无法解决，强行优化=过拟合评分器。"""
        row = {"prediction": "her mental health", "question": "Great for what?",
               "gold_answer": "her headspace"}
        category, _ = m.initial_category(row)
        self.assertEqual(category, "同义/词面失配")
        self.assertEqual(m.FIXABLE_SUGGESTION[category], "not_fixable")

    def test_duration_question_answered_as_year(self):
        row = {"prediction": "2013", "question": "How long ago was Caroline's 18th birthday?",
               "gold_answer": "10 years ago"}
        category, _ = m.initial_category(row)
        self.assertEqual(category, "duration-答成绝对时间")
        self.assertEqual(m.TARGET_EXPERIMENT[category], "E3 时间渲染")


class JudgeAsSemanticTruthTest(unittest.TestCase):
    """词面判据必须与 Judge 联用：单靠词面会把语义正确的题误判成「答错内容」。

    实测证据：143 题中 63 题词面未覆盖金标，其中 52 题（83%）Judge 判 CORRECT。
    若只按覆盖度归类，这 52 题会被错误踢出目标集、并被误报成「内容错误」。
    """

    def test_uncovered_but_judge_correct_is_not_content_error(self):
        # 真实样本 conv-48/18：gold='In France', MN='Paris' —— MN 更精确，不是答错。
        row = {"prediction": "Paris", "question": "In what country did Jolene buy Seraphim?",
               "gold_answer": "In France"}
        self.assertLess(m.gold_coverage(row["gold_answer"], row["prediction"]), 1.0)
        category, _ = m.initial_category(row, mn_judge="CORRECT")
        self.assertEqual(category, "语义正确但词面/粒度不匹配")
        self.assertNotEqual(category, "答错内容（未覆盖金标）")

    def test_uncovered_and_judge_wrong_is_content_error(self):
        # 真实样本 conv-48/74：gold='coffee and fresh pastries', MN 答成日期 —— 真答错。
        row = {"prediction": "11 September 2023", "question": "What did they buy?",
               "gold_answer": "coffee and fresh pastries"}
        category, _ = m.initial_category(row, mn_judge="WRONG")
        self.assertEqual(category, "答错内容（未覆盖金标）")

    def test_missing_judge_never_asserts_content_error(self):
        """无语义真值时保守归为词面失配，不得断言「答错内容」。

        同义改写（her headspace vs her mental health）的覆盖度同样是 0，
        没有 Judge 就分不开「改写」与「答错」，此时断言答错会误导实验分流。
        """
        row = {"prediction": "her mental health", "question": "Great for what?",
               "gold_answer": "her headspace"}
        category, _ = m.initial_category(row, mn_judge=None)
        self.assertEqual(category, "同义/词面失配")
        self.assertNotEqual(category, "答错内容（未覆盖金标）")
        self.assertEqual(m.FIXABLE_SUGGESTION[category], "not_fixable")


class WordSetRelationTest(unittest.TestCase):
    """词集关系判据：五种关系的处置完全相反，必须能正确区分。"""

    def test_stopword_difference_is_equivalence(self):
        # 真实样本：gold='in 2022' vs MN='2022' —— 差异纯属虚词，是 F1 伪影。
        relation, _ = m.word_set_relation("in 2022", "2022")
        self.assertEqual(relation, "1_等价")
        relation, _ = m.word_set_relation("The summer of 2022", "summer 2022")
        self.assertEqual(relation, "1_等价")

    def test_subset_means_lost_information(self):
        # 真实样本：gold='October 2022' vs MN='2022' —— MN 丢了月份。
        relation, note = m.word_set_relation("October 2022", "2022")
        self.assertEqual(relation, "2_真子集")
        self.assertIn("october", note.lower())

    def test_superset_means_redundancy(self):
        relation, _ = m.word_set_relation("gripping with great actors",
                                          "the story was gripping and the actors were great")
        self.assertEqual(relation, "3_超集")

    def test_disjoint_when_no_common_content_word(self):
        # 真实样本：gold='In France' vs MN='Paris' —— 无共同实词，方向机械不可判。
        relation, _ = m.word_set_relation("In France", "Paris")
        self.assertEqual(relation, "5_不相交")

    def test_partial_overlap(self):
        relation, _ = m.word_set_relation("A shiny orange car", "a red sports car")
        self.assertEqual(relation, "4_部分重叠")


class ReviewTierTest(unittest.TestCase):
    """复核档位：把人工逐题判断压到真正有歧义的题上。"""

    def _record(self, **overrides):
        base = {
            "initial_category": "语义正确但词面/粒度不匹配",
            "mn_judge": "CORRECT",
            "sm_judge": "CORRECT",
            "gold_is_enumeration": False,
            "word_set_relation": "4_部分重叠",
        }
        base.update(overrides)
        return base

    def test_both_judges_correct_demotes_to_sampling(self):
        """双方 Judge 都判对而 F1 判 MN 输 ⇒ 属评分器口径，不需逐题裁决。

        这是工作量从 143 降到 30 的关键：这类题「该不该修」已有答案（不该改渲染），
        人要做的只是抽样确认 Judge 没放水。
        """
        tier, reason = m.review_tier(self._record())
        self.assertEqual(tier, "T2_抽样确认")
        self.assertIn("评分器口径", reason)

    def test_judge_wrong_keeps_per_item_review(self):
        tier, _ = m.review_tier(self._record(mn_judge="WRONG"))
        self.assertEqual(tier, "T1_必须逐题")

    def test_equivalence_relation_is_batch_confirmable(self):
        tier, _ = m.review_tier(self._record(word_set_relation="1_等价"))
        self.assertEqual(tier, "T3_可批量确认")

    def test_subset_and_superset_have_determined_direction(self):
        for relation in ("2_真子集", "3_超集"):
            tier, _ = m.review_tier(self._record(word_set_relation=relation))
            self.assertEqual(tier, "T2_抽样确认", f"{relation} 方向已确定，不必逐题")

    def test_disjoint_without_judge_agreement_needs_human(self):
        tier, _ = m.review_tier(self._record(word_set_relation="5_不相交", sm_judge="WRONG"))
        self.assertEqual(tier, "T1_必须逐题")

    def test_abstention_always_per_item(self):
        """拒答类必须逐题：放松拒答直接威胁 AdvSR（一票否决项）。"""
        tier, reason = m.review_tier(self._record(
            initial_category="证据已召回却拒答（判定过度）", mn_judge="WRONG"))
        self.assertEqual(tier, "T1_必须逐题")
        self.assertIn("AdvSR", reason)

    def test_deterministic_form_errors_are_batch_confirmable(self):
        """形态判据 + Judge 同向 ⇒ 可批量。"""
        tier, _ = m.review_tier(self._record(
            initial_category="相对时间未展开", mn_judge="CORRECT",
            word_set_relation="5_不相交"))
        self.assertEqual(tier, "T3_可批量确认")


class GoldTextTest(unittest.TestCase):
    def test_gold_text_resolves_dia_id(self):
        sample = {
            "conversation": {
                "session_1": [
                    {"dia_id": "D1:1", "speaker": "Ann", "text": "hello"},
                    {"dia_id": "D1:2", "speaker": "Bob", "text": "her name is Miso"},
                ],
            }
        }
        text = m.gold_text_for(sample, ["D1:2"])
        self.assertIn("Miso", text)
        self.assertIn("Bob", text)
        self.assertIn("D1:2", text)

    def test_gold_text_handles_missing_and_malformed(self):
        sample = {"conversation": {"session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "x"}]}}
        self.assertEqual(m.gold_text_for(sample, ["D9:9"]), "")  # 越界不抛错
        self.assertEqual(m.gold_text_for(sample, ["not-a-dia"]), "")  # 非法格式跳过


class StatsFromCsvTest(unittest.TestCase):
    def _write_csv(self, rows: list[dict]) -> Path:
        folder = Path(tempfile.mkdtemp(prefix="stats_"))
        path = folder / "review.csv"
        fields = ["key", "category", "initial_category", "target_experiment",
                  "suggested_verdict", "human_verdict", "human_note"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        return path

    def test_review_incomplete_when_verdict_blank(self):
        path = self._write_csv([
            {"key": "a/1", "category": "single-hop", "initial_category": "多余条目",
             "target_experiment": "E5 答案归一化", "suggested_verdict": "fixable",
             "human_verdict": ""},
        ])
        stats = m.stats_from_csv(path)
        self.assertFalse(stats["review_complete"])
        self.assertEqual(stats["unfilled"], 1)

    def test_fixable_count_and_experiment_split(self):
        path = self._write_csv([
            {"key": "a/1", "category": "single-hop", "initial_category": "多余条目",
             "target_experiment": "E5 答案归一化", "suggested_verdict": "fixable",
             "human_verdict": "fixable"},
            {"key": "a/2", "category": "temporal", "initial_category": "相对时间未展开",
             "target_experiment": "E3 时间渲染", "suggested_verdict": "fixable",
             "human_verdict": "fixable"},
            {"key": "a/3", "category": "single-hop", "initial_category": "同义/词面失配",
             "target_experiment": "不优化", "suggested_verdict": "not_fixable",
             "human_verdict": "not_fixable"},
        ])
        stats = m.stats_from_csv(path)
        self.assertTrue(stats["review_complete"])
        self.assertEqual(stats["fixable_count"], 2)
        self.assertEqual(stats["fixable_by_experiment"]["E5 答案归一化"], 1)
        self.assertEqual(stats["fixable_by_experiment"]["E3 时间渲染"], 1)
        self.assertNotIn("不优化", stats["fixable_by_experiment"])

    def test_overridden_suggestion_is_tracked(self):
        """规则建议 fixable 但人工判 not_fixable → 规则过宽的证据，必须单列。"""
        path = self._write_csv([
            {"key": "a/1", "category": "single-hop", "initial_category": "多余条目",
             "target_experiment": "E5 答案归一化", "suggested_verdict": "fixable",
             "human_verdict": "not_fixable"},
        ])
        stats = m.stats_from_csv(path)
        self.assertEqual(stats["fixable_count"], 0)
        self.assertEqual(stats["suggested_fixable_overridden_to_not_fixable"]["多余条目"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
