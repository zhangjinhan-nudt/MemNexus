"""evidence_recall_audit 的回归测试。

重点锁住四条容易被后续改动破坏的语义：
  1. provenance 映射：turn_sequence 是全局累计序号，映射错必须抛错而不是静默继续；
  2. 排序：Recall@K 必须按 retrieved_sources 存储顺序，**不得**按 score 全局重排
     （score 跨 origin 不可比，实测 997/1000 行非降序）；
  3. 病因分解：adversarial 必须排除，输题数必须与主报告一致；
  4. SimpleMem 侧只能产出"有偏、不可跨系统比较"的词面诊断，不得伪装成 Recall。

运行：
  python -m pytest experiments/test_evidence_recall_audit.py -q --basetemp=.pytest_tmp/evidence_recall
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import evidence_recall_audit as era  # noqa: E402


def make_dataset() -> list[dict]:
    """两个 session、各两个 turn；全局序号 D1:1=1, D1:2=2, D2:1=3, D2:2=4。"""
    return [{
        "sample_id": "conv-test",
        "conversation": {
            "speaker_a": "Ann", "speaker_b": "Bob",
            "session_1_date_time": "1:00 pm on 1 May, 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Ann", "text": "I adopted a cat last spring."},
                {"dia_id": "D1:2", "speaker": "Bob", "text": "What did you name the cat?"},
            ],
            "session_2_date_time": "2:00 pm on 9 June, 2023",
            "session_2": [
                {"dia_id": "D2:1", "speaker": "Ann", "text": "Her name is Miso."},
                {"dia_id": "D2:2", "speaker": "Bob", "text": "Miso is a lovely name."},
            ],
        },
        "qa": [],
    }]


def mn_source(dia_sequences: list[int], score: float, origin: str = "indexed_sweep",
              content: str = "retrieved text") -> dict:
    return {
        "content": content,
        "score": score,
        "engine": "simple_memory",
        "metadata": {
            "origin": origin,
            "source_events": [
                {"turn_sequence": seq, "text": f"turn {seq}",
                 "occurred_at": "2023-05-01T13:00:00+00:00"}
                for seq in dia_sequences
            ],
        },
    }


def mn_row(qa_index: int, gold: list[str], sources: list[dict], category: str = "single-hop",
           category_id: int = 4, prediction: str = "Miso", gold_answer: str = "Miso") -> dict:
    return {
        "system": "MemNexus", "sample_id": "conv-test", "qa_index": qa_index,
        "category": category, "category_id": category_id,
        "question": "What is the cat name?", "gold_answer": gold_answer,
        "evaluation_query": "What is the cat name?", "prediction": prediction,
        "gold_evidence": gold, "retrieved_sources": sources,
        "model_calls": {"input_tokens": 100, "output_tokens": 10},
    }


def sm_row(qa_index: int, gold: list[str], sources: list[str], category: str = "single-hop",
           category_id: int = 4, prediction: str = "Miso", gold_answer: str = "Miso") -> dict:
    return {
        "system": "SimpleMem", "sample_id": "conv-test", "qa_index": qa_index,
        "category": category, "category_id": category_id,
        "question": "What is the cat name?", "gold_answer": gold_answer,
        "evaluation_query": "What is the cat name?", "prediction": prediction,
        "gold_evidence": gold, "retrieved_sources": sources,
        "model_calls": {"input_tokens": 100, "output_tokens": 10},
    }


class MappingTest(unittest.TestCase):
    def test_global_sequence_matches_dia_id(self):
        dialog = era.build_dialog_maps(make_dataset()[0])
        self.assertEqual(dialog["n_turns"], 4)
        self.assertEqual(dialog["n_sessions"], 2)
        self.assertEqual(dialog["seq_to_dia"][1], "D1:1")
        self.assertEqual(dialog["seq_to_dia"][3], "D2:1")
        self.assertEqual(dialog["seq_to_dia"][4], "D2:2")
        self.assertEqual(dialog["dia_to_session"]["D2:2"], 2)

    def test_dia_id_session_mismatch_raises(self):
        """dia_id 自带的 session 号与遍历顺序冲突时，必须失败而不是静默出错误指标。"""
        sample = make_dataset()[0]
        sample["conversation"]["session_2"][0]["dia_id"] = "D9:1"
        with self.assertRaises(ValueError):
            era.build_dialog_maps(sample)

    def test_missing_session_structure_raises(self):
        sample = make_dataset()[0]
        sample["conversation"] = {"speaker_a": "Ann"}
        with self.assertRaises(ValueError):
            era.build_dialog_maps(sample)


class OrderingTest(unittest.TestCase):
    """Recall@K 必须按存储顺序；按 score 重排会得到相反结论。"""

    def test_recall_uses_storage_order_not_score(self):
        dialog = era.build_dialog_maps(make_dataset()[0])
        # 存储顺序：position 1 = 非金标但 score 高；position 2 = 金标但 score 低。
        # 若错误地按 score 降序重排，金标会被提到 position 1，R@1 变成 1.0。
        sources = [mn_source([1], score=0.99), mn_source([4], score=0.10)]
        trace = era.provenance_trace({"gold_evidence": ["D2:2"], "retrieved_sources": sources}, dialog)
        self.assertEqual(trace["recall_at_k"][1], 0.0)
        self.assertEqual(trace["recall_at_k"][3], 1.0)
        self.assertEqual(trace["first_hit_rank"], 2)
        self.assertAlmostEqual(trace["mrr"], 0.5)

    def test_recall_at_k_monotonic_nondecreasing(self):
        dialog = era.build_dialog_maps(make_dataset()[0])
        sources = [mn_source([1], 0.9), mn_source([2], 0.8), mn_source([3], 0.7), mn_source([4], 0.6)]
        trace = era.provenance_trace(
            {"gold_evidence": ["D1:1", "D2:1", "D2:2"], "retrieved_sources": sources}, dialog)
        values = [trace["recall_at_k"][k] for k in era.K_VALUES]
        for earlier, later in zip(values, values[1:], strict=False):
            self.assertLessEqual(earlier, later + 1e-12)

    def test_multi_gold_uses_set_coverage(self):
        """多金标题按集合覆盖率计，不是"命中任一即满分"。"""
        dialog = era.build_dialog_maps(make_dataset()[0])
        trace = era.provenance_trace(
            {"gold_evidence": ["D1:1", "D2:2"], "retrieved_sources": [mn_source([1], 0.9)]}, dialog)
        self.assertEqual(trace["gold_set_size"], 2)
        self.assertEqual(trace["gold_covered"], 1)
        self.assertAlmostEqual(trace["recall_full_budget"], 0.5)

    def test_unresolvable_turn_sequence_is_counted_not_dropped(self):
        dialog = era.build_dialog_maps(make_dataset()[0])
        sources = [mn_source([999], 0.9), mn_source([4], 0.8)]
        trace = era.provenance_trace({"gold_evidence": ["D2:2"], "retrieved_sources": sources}, dialog)
        self.assertEqual(trace["unresolved_events"], 1)
        self.assertEqual(trace["gold_covered"], 1)


class DecompositionTest(unittest.TestCase):
    def _records(self) -> list[dict]:
        return [
            {"key": "a/1", "sample_id": "a", "qa_index": 1, "category": "single-hop",
             "outcome": "loss", "recall_full_budget": 0.0, "evidence_gap": 3},
            {"key": "a/2", "sample_id": "a", "qa_index": 2, "category": "temporal",
             "outcome": "loss", "recall_full_budget": 0.5, "evidence_gap": 8},
            {"key": "a/3", "sample_id": "a", "qa_index": 3, "category": "multi-hop",
             "outcome": "loss", "recall_full_budget": 1.0, "evidence_gap": 30},
            {"key": "a/4", "sample_id": "a", "qa_index": 4, "category": "single-hop",
             "outcome": "win", "recall_full_budget": 1.0, "evidence_gap": 2},
            {"key": "a/5", "sample_id": "a", "qa_index": 5, "category": "adversarial",
             "outcome": "loss", "recall_full_budget": 0.0, "evidence_gap": 1},
        ]

    def test_adversarial_excluded_from_loss_decomposition(self):
        loss = era.loss_decomposition(self._records())
        totals = loss["loss_totals"]
        self.assertEqual(totals["losses"], 3)
        self.assertEqual(totals["gold_not_recalled"], 1)
        self.assertEqual(totals["gold_partially_recalled"], 1)
        self.assertEqual(totals["gold_fully_recalled"], 1)
        self.assertEqual(loss["adversarial_excluded"], 1)
        self.assertAlmostEqual(totals["retrieval_side_share"], 2 / 3)
        self.assertAlmostEqual(totals["generation_side_share"], 1 / 3)

    def test_gap_bin_boundaries(self):
        self.assertEqual(era.gap_bin(5), "1-5")
        self.assertEqual(era.gap_bin(6), "6-10")
        self.assertEqual(era.gap_bin(25), "21-25")
        self.assertEqual(era.gap_bin(26), "26+")
        self.assertEqual(era.gap_bin(99), "26+")
        self.assertIsNone(era.gap_bin(None))

    def test_gap_loss_table_counts_only_losses(self):
        """回归：gap × 召回状态表曾错误统计全量题，导致"检索侧占输题 > 100%"。

        现在 losses_only 表每桶合计必须等于该桶输题数，且各桶输题之和等于总输题。
        """
        loss = era.loss_decomposition(self._records())
        losses_only = loss["gap_bin_x_recall_state_losses_only"]
        all_items = loss["gap_bin_x_recall_state_all_items"]
        # 3 条非对抗输题分别落在 gap 3 / 8 / 30 → 桶 1-5 / 6-10 / 26+
        self.assertEqual(sum(sum(c.values()) for c in losses_only.values()), 3)
        # 全量表还包含 win 与 adversarial-excluded 之外的非对抗题 → 4 条非对抗题
        self.assertEqual(sum(sum(c.values()) for c in all_items.values()), 4)
        self.assertEqual(loss["loss_by_gap_bin"]["1-5"]["losses"], 1)
        self.assertEqual(loss["loss_by_gap_bin"]["26+"]["generation_side"], 1)
        for bucket, counter in losses_only.items():
            total = sum(counter.values())
            retrieval_side = counter.get("gold_not_recalled", 0) + counter.get("gold_partially_recalled", 0)
            self.assertLessEqual(retrieval_side, total,
                                 f"{bucket} 检索侧计数超过该桶输题数，标签又错了")

    def test_origin_effectiveness_precision(self):
        records = [{
            "origin_items": {"indexed_sweep": 10, "compressed_memory": 4},
            "origin_hit_items": {"indexed_sweep": 2},
        }]
        result = era.origin_effectiveness(records)
        self.assertAlmostEqual(result["indexed_sweep"]["precision"], 0.2)
        self.assertEqual(result["compressed_memory"]["items_covering_gold"], 0)
        self.assertEqual(result["compressed_memory"]["precision"], 0.0)


class SimpleMemDiagnosticTest(unittest.TestCase):
    def test_rewritten_memory_scores_zero_and_is_flagged(self):
        """SM 的合成记忆不含原文措辞时，词面命中为 0 —— 必须被标记为不可比。"""
        sample = make_dataset()[0]
        dialog = era.build_dialog_maps(sample)
        row = sm_row(1, ["D2:1"], ["Ann said her pet is called Miso in June."])
        result = era.lexical_containment(row, sample, dialog)
        self.assertEqual(result["exact_all"], 0)
        self.assertIsNotNone(result["containment"])

    def test_verbatim_memory_scores_one(self):
        sample = make_dataset()[0]
        dialog = era.build_dialog_maps(sample)
        row = sm_row(1, ["D2:1"], ["Her name is Miso."])
        result = era.lexical_containment(row, sample, dialog)
        self.assertEqual(result["exact_all"], 1)
        self.assertAlmostEqual(result["containment"], 1.0)

    def test_report_flags_sm_as_not_comparable(self):
        with TemporaryRun() as paths:
            report = era.audit(paths["rows"], paths["dataset"])
        diagnostic = report["simplemem_lexical_diagnostic"]
        self.assertTrue(diagnostic["biased_and_not_comparable"])
        self.assertIn("不可跨系统比较", diagnostic["note"])


class EndToEndGateTest(unittest.TestCase):
    def _write(self, folder: Path, rows: list[dict]) -> dict[str, Path]:
        dataset = folder / "locomo.json"
        rows_path = folder / "rows.json"
        dataset.write_text(json.dumps(make_dataset(), ensure_ascii=False), encoding="utf-8")
        rows_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return {"dataset": dataset, "rows": rows_path}

    def _base_rows(self) -> list[dict]:
        return [
            mn_row(1, ["D2:1"], [mn_source([1], 0.9), mn_source([3], 0.8)]),
            sm_row(1, ["D2:1"], ["Ann said her pet is called Miso."]),
            mn_row(2, ["D1:1"], [mn_source([1], 0.9)], category="adversarial", category_id=5,
                   prediction="I do not have enough information in my memory.",
                   gold_answer="Not mentioned"),
            sm_row(2, ["D1:1"], ["nothing relevant"], category="adversarial", category_id=5,
                   prediction="I do not have enough information in my memory.",
                   gold_answer="Not mentioned"),
        ]

    def test_gate_passes_and_metrics_are_reproducible(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            paths = self._write(folder, self._base_rows())
            report = era.audit(paths["rows"], paths["dataset"])
            validation = report["validation"]
            self.assertTrue(validation["gold_all_resolvable"])
            self.assertTrue(validation["mapping_exact"])
            self.assertEqual(validation["unresolved_events_total"], 0)
            self.assertEqual(validation["mn_rows"], 2)
            self.assertEqual(validation["mn_rows_missing_items"], [])
            nonadv = report["memnexus"]["nonadversarial"]["overall"]
            self.assertEqual(nonadv["n"], 1)
            # 金标 D2:1 = 全局序号 3，位于 position 2 → R@1=0，R@3=1
            self.assertEqual(nonadv["recall_at_k"]["1"], 0.0)
            self.assertEqual(nonadv["recall_at_k"]["3"], 1.0)
            self.assertEqual(report["memnexus"]["loss_decomposition"]["adversarial_excluded"], 1)

    def test_duplicate_keys_are_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            rows = self._base_rows() + [mn_row(1, ["D2:1"], [mn_source([3], 0.5)])]
            paths = self._write(folder, rows)
            with self.assertRaises(ValueError) as context:
                era.audit(paths["rows"], paths["dataset"])
            self.assertIn("重复键", str(context.exception))

    def test_unmappable_sequence_fails_gate(self):
        """未解析事件会让 mapping_exact=False，主流程必须拒绝出报告。"""
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            rows = self._base_rows()
            rows[0]["retrieved_sources"].append(mn_source([4242], 0.7))
            paths = self._write(folder, rows)
            report = era.audit(paths["rows"], paths["dataset"])
            self.assertFalse(report["validation"]["mapping_exact"])
            self.assertEqual(report["validation"]["unresolved_events_total"], 1)
            self.assertNotEqual(era.main([
                "--rows", str(paths["rows"]), "--dataset", str(paths["dataset"]),
                "--out-dir", str(folder / "out")]), 0)

    def test_missing_retrieved_items_blocks_report(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            rows = self._base_rows()
            rows[0]["retrieved_sources"] = []
            paths = self._write(folder, rows)
            report = era.audit(paths["rows"], paths["dataset"])
            self.assertEqual(report["validation"]["mn_rows_missing_items"], ["conv-test/1"])
            self.assertNotEqual(era.main([
                "--rows", str(paths["rows"]), "--dataset", str(paths["dataset"]),
                "--out-dir", str(folder / "out")]), 0)


class TemporaryRun:
    """在真实数据上跑一次端到端审计，供跨系统语义测试复用。"""

    def __enter__(self) -> dict[str, Path]:
        self.paths = {
            "rows": ROOT / "result" / "bench1000_locomo_20261023" / "rows.json",
            "dataset": ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json",
        }
        if not all(path.exists() for path in self.paths.values()):
            raise unittest.SkipTest("真实 1000 题结果不可用，跳过端到端语义测试")
        return self.paths

    def __exit__(self, *exc_info) -> bool:
        return False


class JudgeCrossTest(unittest.TestCase):
    """Judge 交叉表：缺失时必须显式不可用，绝不用 0 冒充。"""

    def _records(self) -> list[dict]:
        return [
            # 全召回 + F1 输 + 双方 Judge 都对 → 最需人工复核的子集
            {"key": "a/1", "category": "single-hop", "outcome": "loss",
             "recall_full_budget": 1.0, "mn_judge": "CORRECT", "sm_judge": "CORRECT"},
            # 全召回 + F1 输 + MN 判错
            {"key": "a/2", "category": "single-hop", "outcome": "loss",
             "recall_full_budget": 1.0, "mn_judge": "WRONG", "sm_judge": "CORRECT"},
            # 未召回 + F1 输
            {"key": "a/3", "category": "temporal", "outcome": "loss",
             "recall_full_budget": 0.0, "mn_judge": "WRONG", "sm_judge": "CORRECT"},
            # 对抗题必须排除
            {"key": "a/4", "category": "adversarial", "outcome": "loss",
             "recall_full_budget": 0.0, "mn_judge": "WRONG", "sm_judge": "CORRECT"},
        ]

    def test_cross_tab_excludes_adversarial_and_computes_focus_subset(self):
        result = era.judge_cross_tabulation(self._records(), {"dummy": "CORRECT"})
        self.assertTrue(result["available"])
        nonadv = result["nonadversarial"]
        self.assertEqual(nonadv["n"], 3)
        self.assertAlmostEqual(nonadv["mn_accuracy"], 1 / 3)
        self.assertAlmostEqual(nonadv["sm_accuracy"], 1.0)
        self.assertAlmostEqual(nonadv["delta_pp"], (1 / 3 - 1.0) * 100)
        focus = result["full_recall_but_f1_loss"]
        self.assertEqual(focus["n"], 2)
        self.assertEqual(focus["both_judge_correct"], 1)
        self.assertAlmostEqual(focus["both_judge_correct_share"], 0.5)
        self.assertEqual(focus["example_keys"], ["a/1"])
        self.assertIn("gold_fully_recalled", result["by_recall_state"])
        self.assertNotIn("adversarial", str(result["by_recall_state"]))

    def test_missing_judge_returns_none_not_zero(self):
        records = [{"key": "a/1", "category": "single-hop", "outcome": "loss",
                    "recall_full_budget": 1.0, "mn_judge": None, "sm_judge": None}]
        self.assertIsNone(era.judge_cross_tabulation(records, {}))

    def test_load_judged_ignores_invalid_labels(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "rows_judged.json"
            path.write_text(json.dumps([
                {"sample_id": "a", "qa_index": 1, "system": "MemNexus", "judge_result": "CORRECT"},
                {"sample_id": "a", "qa_index": 2, "system": "MemNexus", "judge_result": "ERROR"},
                {"sample_id": "a", "qa_index": 3, "system": "MemNexus", "judge_result": None},
                "not-a-dict",
            ]), encoding="utf-8")
            labels = era.load_judged(path)
            self.assertEqual(labels, {("a", 1, "MemNexus"): "CORRECT"})

    def test_load_judged_missing_file_returns_empty(self):
        self.assertEqual(era.load_judged(None), {})
        self.assertEqual(era.load_judged(Path("does/not/exist.json")), {})

    def test_markdown_marks_judge_unavailable(self):
        report = {
            "generated_at": "now",
            "validation": {"conversations_in_dataset": 1, "turns_total": 4,
                           "gold_refs_total": 2, "gold_refs_resolvable": 2,
                           "gold_all_resolvable": True, "unresolved_events_total": 0,
                           "mapping_exact": True, "mn_rows": 1, "mn_rows_with_items": 1,
                           "mn_rows_missing_items": [], "duplicate_keys": []},
            "memnexus": {
                "nonadversarial": {"overall": {"n": 1, "recall_at_k": {str(k): 1.0 for k in era.K_VALUES},
                                               "recall_full_budget": 1.0, "mrr": 1.0,
                                               "first_hit_rank_median": 1},
                                   "by_category": {}, "by_gap_bin": {}},
                "loss_decomposition": {
                    "loss_totals": {"losses": 0, "gold_not_recalled": 0, "gold_partially_recalled": 0,
                                    "gold_fully_recalled": 0, "no_gold_reference": 0,
                                    "retrieval_side_share": None, "generation_side_share": None},
                    "adversarial_excluded": 0, "outcome_x_recall_state": {},
                    "gap_bin_x_recall_state_losses_only": {},
                    "gap_bin_x_recall_state_all_items": {}, "loss_by_gap_bin": {}},
                "origin_effectiveness": {}, "multi_session_evidence": {"n": 0, "recall_full_budget": None},
            },
            "simplemem_lexical_diagnostic": {"biased_and_not_comparable": True, "note": "x",
                                             "n": 1, "mean_containment": 0.5, "exact_all_rate": 0.0,
                                             "by_category": {}},
            "judge_cross": None,
        }
        text = era.render_markdown(report)
        self.assertIn("Judge 数据不可用", text)


class ViewCounterfactualTest(unittest.TestCase):
    """视图反事实与配额扫描：cap=0 必须等价于剔除该视图（内置自检也守这条）。"""

    def _setup(self):
        dialogs = {"conv-test": era.build_dialog_maps(make_dataset()[0])}
        row_a = mn_row(1, ["D2:1"], [
            mn_source([3], 0.9, origin="compressed_memory", content="a" * 100),
            mn_source([1], 0.5, origin="indexed_sweep", content="b" * 200),
            mn_source([2], 0.4, origin="indexed_sweep", content="c" * 200),
            mn_source([4], 0.3, origin="indexed_sweep", content="d" * 200),
        ])
        rows_by_key = {"conv-test/1": row_a}
        records = [{
            "key": "conv-test/1", "sample_id": "conv-test", "qa_index": 1,
            "category": "single-hop", "outcome": "loss", "recall_full_budget": 1.0,
        }]
        return records, rows_by_key, dialogs

    def test_drop_view_removes_gold_only_from_that_view(self):
        records, rows_by_key, dialogs = self._setup()
        result = era.view_counterfactual(records, rows_by_key, dialogs)
        # 金标 D2:1 = 全局序号 3，只由 compressed_memory 提供
        self.assertAlmostEqual(result["drop_view"]["compressed_memory"]["recall_full"], 0.0)
        self.assertAlmostEqual(result["only_view"]["compressed_memory"]["recall_full"], 1.0)
        self.assertAlmostEqual(result["drop_view"]["indexed_sweep"]["recall_full"], 1.0)

    def test_cap_zero_equals_drop_view(self):
        records, rows_by_key, dialogs = self._setup()
        result = era.view_counterfactual(records, rows_by_key, dialogs)
        for origin in result["origins"]:
            zero = result["quota_sweep"][origin]["0"]
            dropped = result["drop_view"][origin]
            for field in ("recall_full", "recall_5", "recall_10", "items_per_question"):
                self.assertEqual(zero[field], dropped[field], f"{origin}/{field} 不一致")

    def test_cap_limits_item_count(self):
        records, rows_by_key, dialogs = self._setup()
        result = era.view_counterfactual(records, rows_by_key, dialogs)
        # indexed_sweep 有 3 项；cap=1 后每题只剩 1 项该视图 + 1 项其他视图 = 2 项
        self.assertAlmostEqual(result["quota_sweep"]["indexed_sweep"]["1"]["items_per_question"], 2.0)
        self.assertAlmostEqual(result["quota_sweep"]["indexed_sweep"]["3"]["items_per_question"], 4.0)

    def test_sweet_spot_respects_r10_constraint(self):
        records, rows_by_key, dialogs = self._setup()
        result = era.view_counterfactual(records, rows_by_key, dialogs)
        baseline_r10 = result["baseline"]["recall_10"]
        for origin, spot in result["sweet_spots"].items():
            self.assertGreaterEqual(spot["recall_10"] + era.QUOTA_TOLERANCE, baseline_r10,
                                    f"{origin} 甜点违反 R@10 约束")

    def test_flags_offline_replay_only(self):
        records, rows_by_key, dialogs = self._setup()
        result = era.view_counterfactual(records, rows_by_key, dialogs)
        self.assertTrue(result["offline_replay_only"])
        self.assertTrue(result["not_end_to_end"])
        self.assertIn("不等价于线上", result["warning"])


class ReachabilityAttributionTest(unittest.TestCase):
    """三层归因：不可达 ≠ 建库缺失，必须用快照直接验证后才允许归 E1。

    本测试是最容易被改错的语义：若把「可达集不可达」直接当建库问题（E1），
    会把 23 题里 19 题金标其实在记忆里的题判错方向，E1 的目标集虚大 5 倍。
    """

    def _dataset(self) -> list[dict]:
        # 6 轮对话：D1:1..D1:3 (seq1-3) + D2:1..D2:3 (seq4-6)
        return [{
            "sample_id": "conv-reach",
            "conversation": {
                "speaker_a": "Ann", "speaker_b": "Bob",
                "session_1_date_time": "1:00 pm on 1 May, 2023",
                "session_1": [{"dia_id": f"D1:{i}", "speaker": "Ann", "text": f"s1 turn {i}"}
                              for i in (1, 2, 3)],
                "session_2_date_time": "2:00 pm on 9 June, 2023",
                "session_2": [{"dia_id": f"D2:{i}", "speaker": "Bob", "text": f"s2 turn {i}"}
                              for i in (1, 2, 3)],
            },
            "qa": [],
        }]

    def _snapshot(self) -> dict:
        # 建库输入只有 seq1-5（seq6 被视界截断）；记忆只引用 t1-t4（t5 未被引用）。
        return {
            "conv-reach": {
                "seq_to_turn": {i: f"t{i}" for i in range(1, 6)},
                "mem_turn_ids": {f"t{i}" for i in range(1, 5)},
                "n_memories": 4, "n_turns": 5,
            }
        }

    def _inputs(self):
        dialogs = era.load_dialog_maps(self._dataset())
        rows = {
            # A：金标 D1:1，自己召回 → full（产出层，不进任何池）
            "conv-reach/1": {"gold_evidence": ["D1:1"],
                             "retrieved_sources": [mn_source([1], 0.9)]},
            # B：金标 D2:1，未召回且无人召回过 → 不可达；但快照里 t4 被记忆引用 ⇒ E4
            "conv-reach/2": {"gold_evidence": ["D2:1"],
                             "retrieved_sources": [mn_source([1], 0.9)]},
            # C：金标 D2:2，不可达；快照里 t5 未被任何记忆引用 ⇒ E1（真建库遗漏）
            "conv-reach/3": {"gold_evidence": ["D2:2"],
                             "retrieved_sources": [mn_source([1], 0.9)]},
            # D：金标 D2:3，不可达；seq6 不在建库输入 ⇒ 视界截断，E4/E1 都救不了
            "conv-reach/4": {"gold_evidence": ["D2:3"],
                             "retrieved_sources": [mn_source([1], 0.9)]},
            # E：金标 [D1:2, D2:1]，部分召回（D1:2）→ partial；part_reach；快照引用>0 ⇒ E4
            "conv-reach/5": {"gold_evidence": ["D1:2", "D2:1"],
                             "retrieved_sources": [mn_source([2], 0.9)]},
        }
        records = [
            {"key": "conv-reach/1", "sample_id": "conv-reach", "qa_index": 1,
             "category": "single-hop", "gold_set_size": 1, "gold_covered": 1},
            {"key": "conv-reach/2", "sample_id": "conv-reach", "qa_index": 2,
             "category": "single-hop", "gold_set_size": 1, "gold_covered": 0},
            {"key": "conv-reach/3", "sample_id": "conv-reach", "qa_index": 3,
             "category": "single-hop", "gold_set_size": 1, "gold_covered": 0},
            {"key": "conv-reach/4", "sample_id": "conv-reach", "qa_index": 4,
             "category": "single-hop", "gold_set_size": 1, "gold_covered": 0},
            {"key": "conv-reach/5", "sample_id": "conv-reach", "qa_index": 5,
             "category": "multi-hop", "gold_set_size": 2, "gold_covered": 1},
        ]
        return records, rows, dialogs

    def test_three_layer_attribution(self):
        records, rows, dialogs = self._inputs()
        result = era.reachability_attribution(records, rows, dialogs, self._snapshot())
        # A 是 full → 产出层，不进任何池
        self.assertNotIn("conv-reach/1", result["e4_pool_keys"])
        self.assertNotIn("conv-reach/1", result["e1_residual_keys"])
        # B 不可达但记忆里有 → E4（不是 E1！这是核心断言）
        self.assertIn("conv-reach/2", result["e4_pool_keys"])
        self.assertNotIn("conv-reach/2", result["e1_residual_keys"])
        # C 不可达且记忆里没有 → E1
        self.assertIn("conv-reach/3", result["e1_residual_keys"])
        self.assertNotIn("conv-reach/3", result["e4_pool_keys"])
        # D 金标轮次在建库输入外 → 视界截断，两个池都不进
        self.assertIn("conv-reach/4", result["horizon_excluded_keys"])
        self.assertNotIn("conv-reach/4", result["e4_pool_keys"])
        self.assertNotIn("conv-reach/4", result["e1_residual_keys"])
        # E partial + part_reach + 快照引用>0 → E4
        self.assertIn("conv-reach/5", result["e4_pool_keys"])
        self.assertEqual(result["counts"], {"e4_pool": 2, "e1_residual": 1,
                                            "horizon_excluded": 1})

    def test_reachable_gold_routes_to_e4_without_snapshot(self):
        """可达（其他题召回过）⇒ E4，且不需要快照。"""
        records, rows, dialogs = self._inputs()
        # 给题 2 增加一个邻居题，它召回了 seq4（D2:1），使 D2:1 可达。
        rows["conv-reach/6"] = {"gold_evidence": ["D1:3"],
                                "retrieved_sources": [mn_source([4], 0.9)]}
        records.append({"key": "conv-reach/6", "sample_id": "conv-reach", "qa_index": 6,
                        "category": "single-hop", "gold_set_size": 1, "gold_covered": 0})
        result = era.reachability_attribution(records, rows, dialogs, {})
        self.assertIn("conv-reach/2", result["e4_pool_keys"])
        self.assertTrue(result["snapshot_available"] is False)

    def test_missing_snapshot_defaults_unreachable_to_e4(self):
        """快照缺失时不可达题保守归 E4：间接证据不足以断言建库缺失。"""
        records, rows, dialogs = self._inputs()
        result = era.reachability_attribution(records, rows, dialogs, {})
        # B/C 都是 not+unreachable，无快照 → 都保守进 E4，不误报 E1。
        self.assertIn("conv-reach/2", result["e4_pool_keys"])
        self.assertIn("conv-reach/3", result["e4_pool_keys"])
        self.assertEqual(result["counts"]["e1_residual"], 0)

    def test_adversarial_excluded_from_pools(self):
        records, rows, dialogs = self._inputs()
        records.append({"key": "conv-reach/9", "sample_id": "conv-reach", "qa_index": 9,
                        "category": "adversarial", "gold_set_size": 1, "gold_covered": 0})
        rows["conv-reach/9"] = {"gold_evidence": ["D2:2"],
                                "retrieved_sources": [mn_source([1], 0.9)]}
        result = era.reachability_attribution(records, rows, dialogs, self._snapshot())
        self.assertNotIn("conv-reach/9", result["e4_pool_keys"])
        self.assertNotIn("conv-reach/9", result["e1_residual_keys"])

    def test_load_memory_snapshots(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            unit = tmp_path / "conv-x_memnexus"
            unit.mkdir()
            (unit / "usage.memory.json").write_text(json.dumps({
                "turns": [{"sequence": 1, "turn_id": "t1"}],
                "memories": [{"evidence": [{"turn_id": "t1"}]},
                             {"evidence": []}],
            }), encoding="utf-8")
            (tmp_path / "conv-y_memnexus").mkdir()  # 空目录不产生快照
            snapshots = era.load_memory_snapshots(tmp_path)
            self.assertIn("conv-x", snapshots)
            self.assertEqual(snapshots["conv-x"]["seq_to_turn"], {1: "t1"})
            self.assertEqual(snapshots["conv-x"]["mem_turn_ids"], {"t1"})
            self.assertEqual(snapshots["conv-x"]["n_memories"], 2)
            self.assertNotIn("conv-y", snapshots)

    def test_load_memory_snapshots_none_dir(self):
        self.assertEqual(era.load_memory_snapshots(None), {})
        self.assertEqual(era.load_memory_snapshots(Path("Z:/nonexistent")), {})


class RealDataInvariantTest(unittest.TestCase):
    """在真实 1000 题上验证不变量；这些是后续改动最可能破坏的性质。"""

    @classmethod
    def setUpClass(cls) -> None:
        paths = {
            "rows": ROOT / "result" / "bench1000_locomo_20261023" / "rows.json",
            "dataset": ROOT / "benchmark-data" / "official" / "locomo" / "locomo10.json",
        }
        if not all(path.exists() for path in paths.values()):
            raise unittest.SkipTest("真实结果不可用")
        # 与 main() 一致的 Judge 与建库快照自动探测，保证测试与产物口径相同。
        judged = paths["rows"].parent / "rows_judged.json"
        attempts = paths["rows"].parent / "attempts"
        cls.report = era.audit(paths["rows"], paths["dataset"],
                               judged_path=judged if judged.exists() else None,
                               attempts_dir=attempts if attempts.is_dir() else None)

    def test_mapping_is_exact_on_full_run(self):
        validation = self.report["validation"]
        self.assertTrue(validation["mapping_exact"])
        self.assertTrue(validation["gold_all_resolvable"])
        self.assertEqual(validation["unresolved_events_total"], 0)
        self.assertEqual(validation["mn_rows"], 1000)
        self.assertEqual(validation["mn_rows_missing_items"], [])

    def test_loss_count_matches_main_report(self):
        """主报告 276 输题 = 263 非对抗 + 13 对抗；分解口径必须与主表对齐。"""
        loss = self.report["memnexus"]["loss_decomposition"]
        self.assertEqual(loss["adversarial_excluded"], 225)
        totals = loss["loss_totals"]
        self.assertEqual(totals["losses"], 263)
        self.assertEqual(
            totals["gold_not_recalled"] + totals["gold_partially_recalled"]
            + totals["gold_fully_recalled"] + totals["no_gold_reference"],
            totals["losses"],
        )

    def test_recall_is_monotonic_and_bounded(self):
        for block in [self.report["memnexus"]["nonadversarial"]["overall"],
                      *self.report["memnexus"]["nonadversarial"]["by_category"].values()]:
            values = [block["recall_at_k"][str(k)] for k in era.K_VALUES]
            for earlier, later in zip(values, values[1:], strict=False):
                if earlier is not None and later is not None:
                    self.assertLessEqual(earlier, later + 1e-12)
            for value in values:
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_adversarial_absent_from_nonadversarial_view(self):
        categories = self.report["memnexus"]["nonadversarial"]["by_category"]
        self.assertNotIn("adversarial", categories)
        self.assertIn("adversarial", self.report["memnexus"]["all"]["by_category"])

    def test_reachability_counts_on_full_run(self):
        """E4/E1/视界 的三层归因计数（2026-11-09 口径，快照直接验证）。

        这些数字是「砍掉 E1」决策与 E4 目标集规模的全部依据；
        若有人改动了归因逻辑导致数字漂移，此测试会立刻暴露。
        """
        reachability = self.report["memnexus"]["reachability"]
        if not reachability["snapshot_available"]:
            self.skipTest("建库快照不可用，无法验证直接口径")
        counts = reachability["counts"]
        self.assertEqual(counts["e4_pool"], 220)
        self.assertEqual(counts["e1_residual"], 4)
        self.assertEqual(counts["horizon_excluded"], 0)
        # E4 与产出层池互斥：E4 池里的题都不可能是 full 召回。
        matrix = reachability["attribution_matrix"]
        self.assertNotIn("full/unreachable", matrix)
        self.assertNotIn("full/part_reach", matrix)
        # 池的规模可从矩阵复核：E4 ⊇ not/reachable + partial/reachable。
        self.assertGreaterEqual(
            counts["e4_pool"],
            matrix.get("not/reachable", 0) + matrix.get("partial/reachable", 0))

    def test_e4_pool_excludes_full_recall_rows(self):
        """E4 目标集的每道题都必须是 not/partial：full 的题归产出层。"""
        reachability = self.report["memnexus"]["reachability"]
        per_item = {e["key"]: e for e in self.report.get("per_item", [])
                    if e.get("system") == "MemNexus"}
        for key in reachability["e4_pool_keys"]:
            entry = per_item.get(key)
            if entry is None:
                continue
            self.assertLess(entry["recall_full_budget"], 1.0 + 1e-12,
                            f"{key} 金标已全召回，不应出现在 E4 目标集")

    def test_simplemem_diagnostic_never_reports_recall(self):
        diagnostic = self.report["simplemem_lexical_diagnostic"]
        self.assertTrue(diagnostic["biased_and_not_comparable"])
        for block in diagnostic["by_category"].values():
            self.assertNotIn("recall_at_k", block)
            self.assertNotIn("mrr", block)

    def test_counterfactual_on_real_data(self):
        """真实数据上：compressed_memory 必须是召回主力，indexed_sweep 必须有过配特征。"""
        cf = self.report["memnexus"]["view_counterfactual"]
        baseline = cf["baseline"]["recall_full"]
        # compressed_memory 去掉后召回大幅崩塌（实测 0.778 → 0.229）
        self.assertLess(cf["drop_view"]["compressed_memory"]["recall_full"], baseline * 0.5)
        # indexed_sweep 去掉后召回只小幅下降，且字符节省显著（过配特征）
        self.assertGreater(cf["drop_view"]["indexed_sweep"]["recall_full"], baseline * 0.85)
        spot = cf["sweet_spots"]["indexed_sweep"]
        self.assertGreater(spot["chars_saved_pct"], 25.0)
        self.assertGreaterEqual(spot["recall_10"] + era.QUOTA_TOLERANCE, cf["baseline"]["recall_10"])
        # cap=0 与剔除一致（脚本内也有断言，这里再锁一次真实数据）
        for origin in cf["origins"]:
            self.assertEqual(cf["quota_sweep"][origin]["0"]["recall_full"],
                             cf["drop_view"][origin]["recall_full"])

    def test_judge_cross_present_when_judged_available(self):
        judged = ROOT / "result" / "bench1000_locomo_20261023" / "rows_judged.json"
        if not judged.exists():
            self.skipTest("rows_judged.json 不可用")
        cross = self.report["judge_cross"]
        self.assertIsNotNone(cross)
        self.assertEqual(cross["matched_labels"], 2000)
        nonadv = cross["nonadversarial"]
        self.assertEqual(nonadv["n"], 775)
        # 两指标方向相反：F1 侧 MN 略优（pooled +0.08），Judge 侧 MN 落后约 5.2pp
        self.assertLess(nonadv["delta_pp"], 0)
        focus = cross["full_recall_but_f1_loss"]
        self.assertEqual(focus["n"], 143)
        self.assertEqual(focus["both_judge_correct"], 113)


if __name__ == "__main__":
    unittest.main(verbosity=2)
