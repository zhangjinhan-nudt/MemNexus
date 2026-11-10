"""compare_subset_ab 的回归测试。

最重要的是"口径等价"测试：比较器在变体=母轮时必须精确复现主报告的
Average Δ 与 cluster CI（同 seed/bootstrap）。若哪天口径漂移，这条会红。

运行：
  python -m pytest experiments/test_compare_subset_ab.py -q --basetemp=.pytest_tmp/compare_ab
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import compare_subset_ab as cab  # noqa: E402

PARENT_ROWS = ROOT / "result" / "bench1000_locomo_20261023" / "rows.json"


def make_row(system, sample_id, qa_index, category_id, prediction, gold,
             input_tokens=100, output_tokens=10, latency=1000.0):
    return {
        "system": system, "sample_id": sample_id, "qa_index": qa_index,
        "category": {1: "multi-hop", 2: "temporal", 3: "open-domain",
                     4: "single-hop", 5: "adversarial"}[category_id],
        "category_id": category_id, "question": "Q", "evaluation_query": "Q",
        "gold_answer": gold, "prediction": prediction,
        "gold_evidence": ["D1:1"],
        "model_calls": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "latency_ms": latency,
    }


class SyntheticPairingTest(unittest.TestCase):
    def test_paired_items_and_outcome(self):
        # qa0: MN 精确命中(F1=1.0)，SM 带冗余词(F1=0.4) → MN win。
        # qa1: MN 答错(F1=0)，SM 精确命中(F1=1.0) → MN loss。
        # 注意：冗余词会降 precision，所以"答案更长"通常不是 win——
        # 这正是官方 F1 的特性，也是 E5 要治的病。
        mn = [make_row("MemNexus", "c1", 0, 4, "Miso", "Miso"),
              make_row("MemNexus", "c1", 1, 4, "wrong", "right")]
        sm = [make_row("SimpleMem", "c1", 0, 4, "cat named Miso maybe", "Miso"),
              make_row("SimpleMem", "c1", 1, 4, "right", "right")]
        items, diag = cab.build_paired_items(mn, sm)
        self.assertEqual(diag["paired"], 2)
        self.assertEqual(diag["unpaired_variant_only"], 0)
        outcomes = {item["qa_index"]: item["outcome"] for item in items}
        self.assertEqual(outcomes[0], "win")
        self.assertEqual(outcomes[1], "loss")
        by_index = {item["qa_index"]: item for item in items}
        self.assertAlmostEqual(by_index[0]["mn_f1"], 1.0)
        self.assertLess(by_index[0]["sm_f1"], 1.0)
        self.assertAlmostEqual(by_index[1]["mn_f1"], 0.0)

    def test_unpaired_keys_are_reported_not_dropped(self):
        mn = [make_row("MemNexus", "c1", 0, 4, "a", "a"),
              make_row("MemNexus", "c1", 9, 4, "b", "b")]
        sm = [make_row("SimpleMem", "c1", 0, 4, "a", "a")]
        items, diag = cab.build_paired_items(mn, sm)
        self.assertEqual(diag["paired"], 1)
        self.assertEqual(diag["unpaired_variant_only"], 1)
        self.assertEqual(diag["unpaired_examples"]["variant_only"], ["c1/9"])

    def test_category_mismatch_raises(self):
        """同键但类别不一致 → 配对无效，必须报错而非静默错配。"""
        mn = [make_row("MemNexus", "c1", 0, 4, "a", "a")]
        sm = [make_row("SimpleMem", "c1", 0, 2, "a", "a")]
        with self.assertRaises(ValueError):
            cab.build_paired_items(mn, sm)

    def test_duplicate_valid_rows_raise(self):
        mn = [make_row("MemNexus", "c1", 0, 4, "a", "a"),
              make_row("MemNexus", "c1", 0, 4, "b", "a")]
        sm = [make_row("SimpleMem", "c1", 0, 4, "a", "a")]
        with self.assertRaises(ValueError):
            cab.build_paired_items(mn, sm)

    def test_empty_predictions_excluded(self):
        mn = [make_row("MemNexus", "c1", 0, 4, "", "a"),
              make_row("MemNexus", "c1", 1, 4, "a", "a")]
        sm = [make_row("SimpleMem", "c1", 0, 4, "a", "a"),
              make_row("SimpleMem", "c1", 1, 4, "a", "a")]
        items, diag = cab.build_paired_items(mn, sm)
        self.assertEqual(diag["variant_mn_valid"], 1)  # 空预测被排除
        self.assertEqual(diag["paired"], 1)


class AverageAndAdvSRTest(unittest.TestCase):
    def test_average_is_four_category_equal_weight_excluding_adversarial(self):
        items = [
            {"category_id": 1, "mn_f1": 0.4, "sm_f1": 0.2},
            {"category_id": 2, "mn_f1": 0.6, "sm_f1": 0.4},
            {"category_id": 3, "mn_f1": 0.2, "sm_f1": 0.0},
            {"category_id": 4, "mn_f1": 0.8, "sm_f1": 0.6},
            {"category_id": 5, "mn_f1": 1.0, "sm_f1": 1.0},  # adversarial 不进 Average
        ]
        mn_means = cab.category_means([i for i in items if i["category_id"] != 5], "mn_f1")
        self.assertAlmostEqual(mn_means["Average"], (0.4 + 0.6 + 0.2 + 0.8) / 4)
        self.assertNotIn("adversarial", mn_means)

    def test_adversarial_sr_uses_official_markers(self):
        # 通过 build_paired_items 生成配对项，验证 AdvSR 走的是同题配对口径。
        variant = [make_row("MemNexus", "c1", 0, 5, "Not mentioned in the conversation", "x"),
                   make_row("MemNexus", "c1", 1, 5, "some wrong answer", "x")]
        baseline = [make_row("SimpleMem", "c1", 0, 5, "No information available", "x"),
                    make_row("SimpleMem", "c1", 1, 5, "another answer", "x")]
        items, _ = cab.build_paired_items(variant, baseline)
        result = cab.adversarial_sr(items)
        self.assertTrue(result["paired"])
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["variant_mn"]["adv_sr"], 0.5)
        self.assertAlmostEqual(result["baseline_sm"]["adv_sr"], 0.5)
        self.assertAlmostEqual(result["delta_pp"], 0.0)

    def test_adversarial_sr_empty_when_no_adversarial(self):
        items = [{"category_id": 4, "mn_adv_success": None, "sm_adv_success": None}]
        result = cab.adversarial_sr(items)
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["variant_mn"]["adv_sr"])
        self.assertIsNone(result["delta_pp"])


class DriftDetectionTest(unittest.TestCase):
    def test_no_drift_when_identical(self):
        rows = [make_row("MemNexus", "c1", 0, 4, "Miso", "Miso")]
        result = cab.detect_drift(rows, rows)
        self.assertFalse(result["drift_detected"])
        self.assertEqual(result["shared_keys"], 1)

    def test_drift_detected_on_f1_change(self):
        """baseline 臂的 MN F1 变了 → 说明快照/代码/网关漂了，必须报警。"""
        parent = [make_row("MemNexus", "c1", 0, 4, "Miso", "Miso")]
        variant = [make_row("MemNexus", "c1", 0, 4, "completely different wrong", "Miso")]
        result = cab.detect_drift(variant, parent)
        self.assertTrue(result["drift_detected"])
        self.assertEqual(result["drifted"], 1)


class GuardTest(unittest.TestCase):
    def test_adv_floor_guard_fails_when_below(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            variant = Path(raw) / "v.json"
            # MN 对抗题拒答率 0.5，低于 floor 0.84
            rows = [make_row("MemNexus", "c1", 0, 5, "Not mentioned", "x"),
                    make_row("MemNexus", "c1", 1, 5, "wrong answer", "x"),
                    make_row("MemNexus", "c1", 2, 4, "a", "a")]
            sm = [make_row("SimpleMem", "c1", 0, 5, "Not mentioned", "x"),
                  make_row("SimpleMem", "c1", 1, 5, "wrong", "x"),
                  make_row("SimpleMem", "c1", 2, 4, "a", "a")]
            variant.write_text(__import__("json").dumps(rows + sm), encoding="utf-8")
            report = cab.analyze(variant, variant, "variant", 100, 1, 0.84, None, None)
            self.assertFalse(report["guards"]["adv_sr_floor"]["passed"])


@unittest.skipUnless(PARENT_ROWS.exists(), "母轮 rows.json 不可用")
class RealDataEquivalenceTest(unittest.TestCase):
    """口径等价：变体=母轮时，必须精确复现主报告 LOCOMO1000_FINAL_20261025。"""

    def test_reproduces_main_report_average_and_ci(self):
        report = cab.analyze(
            PARENT_ROWS, PARENT_ROWS, sm_source="variant",
            bootstrap=5000, seed=20261025, adv_floor=None,
            token_ceiling_pct=None, drift_path=None)
        self.assertEqual(report["pairing"]["paired"], 1000)
        self.assertAlmostEqual(report["primary"]["mn_average_f1"], 0.47414784470552596, places=6)
        self.assertAlmostEqual(report["primary"]["sm_average_f1"], 0.46374925118278043, places=6)
        self.assertAlmostEqual(report["primary"]["delta_pp"], 1.039859352274556, places=6)
        ci = report["cluster_bootstrap"]["ci95"]["Average"]
        self.assertAlmostEqual(ci[0], -0.028371999976804313, places=6)
        self.assertAlmostEqual(ci[1], 0.058991057058042466, places=6)
        self.assertAlmostEqual(report["adversarial"]["variant_mn"]["adv_sr"],
                               0.8666666666666667, places=6)
        self.assertAlmostEqual(report["adversarial"]["baseline_sm"]["adv_sr"], 0.84, places=6)

    def test_drift_check_clean_on_self(self):
        report = cab.analyze(
            PARENT_ROWS, PARENT_ROWS, sm_source="variant",
            bootstrap=100, seed=1, adv_floor=None,
            token_ceiling_pct=None, drift_path=PARENT_ROWS)
        self.assertFalse(report["drift_check"]["drift_detected"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
