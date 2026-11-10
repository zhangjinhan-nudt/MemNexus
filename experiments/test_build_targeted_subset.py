"""build_targeted_subset 的回归测试。

锁住四个决定实验有效性的不变量：
  1. 目标题必须 100% 纳入（否则迭代会漏掉病灶）；
  2. 对抗题守护必须补足到配额（否则 P2 的 AdvSR 护栏失效）；
  3. selection_sha256 与母 cohort 同算法可复算（否则 GATE 校验会拒）；
  4. 目标题不在母 cohort 时必须报错（否则无法复用 SM 基线，配对失效）。

运行：
  python -m pytest experiments/test_build_targeted_subset.py -q --basetemp=.pytest_tmp/targeted_subset
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import build_targeted_subset as bts  # noqa: E402

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def make_pool() -> tuple[dict, dict[tuple[str, int], dict]]:
    """构造一个 10 题的合成母 cohort：含对抗题、跨两个对话。"""
    questions = []
    # conv-a: qa 0-4（其中 qa4 是对抗）；conv-b: qa 0-4（其中 qa4 是对抗）
    specs = [
        ("conv-a", 0, 0, 4), ("conv-a", 0, 1, 4), ("conv-a", 0, 2, 2),
        ("conv-a", 0, 3, 1), ("conv-a", 0, 4, 5),
        ("conv-b", 1, 0, 4), ("conv-b", 1, 1, 4), ("conv-b", 1, 2, 2),
        ("conv-b", 1, 3, 3), ("conv-b", 1, 4, 5),
    ]
    for sample_id, sample_index, qa_index, category_id in specs:
        questions.append({
            "sample_index": sample_index, "sample_id": sample_id, "qa_index": qa_index,
            "category_id": category_id, "category": CATEGORIES[category_id],
            "question": f"q-{sample_id}-{qa_index}", "evidence": ["D1:1"],
        })
    cohort = {
        "dataset": "benchmark-data/official/locomo/locomo10.json",
        "dataset_sha256": "deadbeef", "max_session": 30,
        "question_count": len(questions),
        "questions": questions,
    }
    pool = {(q["sample_id"], q["qa_index"]): q for q in questions}
    return cohort, pool


def make_review_csv(folder: Path, rows: list[dict], verdict_col: str = "human_verdict") -> Path:
    path = folder / "review.csv"
    fields = ["sample_id", "qa_index", "category", "initial_category",
              "target_experiment", "suggested_verdict", "human_verdict"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def make_keys_file(folder: Path, keys: list[str], n: int | None = None) -> Path:
    path = folder / "target_keys.json"
    path.write_text(json.dumps({
        "n": len(keys) if n is None else n,
        "keys": keys,
        "definition": "test target set",
    }, ensure_ascii=False), encoding="utf-8")
    return path


class KeysFileTest(unittest.TestCase):
    """keys-file 目标源：加载校验 + 与 CSV 的合并语义。

    keys-file 来自机械归因（事实判定），与 CSV 的价值判断（fixable/not_fixable）
    性质不同——partial_review 只对后者成立，这个区分若被破坏，正式子集会被误标
    （或反之：规则初判的子集被当成正式依据）。
    """

    def test_load_keys_file_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_keys_file(Path(tmp), ["conv-a/1", "conv-b/0"])
            keys, diag = bts.load_keys_file(path)
            self.assertEqual(keys, [("conv-a", 1), ("conv-b", 0)])
            self.assertEqual(diag["target_keys"], 2)
            self.assertEqual(diag["definition"], "test target set")

    def test_load_keys_file_rejects_bad_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (["no-slash"], ["conv-a/abc"], ["/1"], [""]):
                with self.subTest(bad=bad):
                    path = make_keys_file(Path(tmp), bad)
                    with self.assertRaises(ValueError):
                        bts.load_keys_file(path)

    def test_load_keys_file_rejects_n_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_keys_file(Path(tmp), ["conv-a/1"], n=5)
            with self.assertRaises(ValueError) as ctx:
                bts.load_keys_file(path)
            self.assertIn("n=5", str(ctx.exception))

    def test_load_keys_file_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_keys_file(Path(tmp), ["conv-a/1", "conv-a/1"])
            with self.assertRaises(ValueError):
                bts.load_keys_file(path)

    def test_keys_file_verdict_is_not_partial(self):
        """keys_file 是事实性归因：partial_review 必须为 False，protocol 必须记录来源。"""
        cohort, pool = make_pool()
        subset = bts.build_subset(
            cohort, pool, [("conv-a", 1), ("conv-b", 0)], total=8, seed=1,
            verdict_column="keys_file", experiments=None,
            diagnostics={"keys_file": {"keys_file": "x/e4_target_keys.json",
                                       "target_keys": 2}},
            adversarial_guard=2)
        self.assertFalse(subset["partial_review"])
        self.assertIn("keys-file source = x/e4_target_keys.json", subset["protocol"])

    def test_suggested_verdict_still_partial_when_merged(self):
        """合并场景下 CSV 用规则初判 → partial 必须为 True（不能被 keys-file 洗白）。"""
        cohort, pool = make_pool()
        subset = bts.build_subset(
            cohort, pool, [("conv-a", 1)], total=8, seed=1,
            verdict_column="suggested_verdict", experiments=None,
            diagnostics={"keys_file": {"keys_file": "x.json", "target_keys": 1}},
            adversarial_guard=2)
        self.assertTrue(subset["partial_review"])

    def test_main_keys_file_only_without_csv(self):
        """keys-file 单源：CSV 不存在也不报错（目标题全部来自 keys-file）。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cohort, _ = make_pool()
            pool_path = tmp_path / "pool.json"
            pool_path.write_text(json.dumps(cohort, ensure_ascii=False), encoding="utf-8")
            keys_path = make_keys_file(tmp_path, ["conv-a/1", "conv-b/2"])
            out = tmp_path / "subset.json"
            rc = bts.main([
                "--pool-cohort", str(pool_path),
                "--keys-file", str(keys_path),
                "--review-csv", str(tmp_path / "nonexistent.csv"),
                "--total", "6", "--adversarial-guard", "2",
                "--out", str(out),
            ])
            self.assertEqual(rc, 0)
            subset = json.loads(out.read_text(encoding="utf-8"))
            keys = {(q["sample_id"], q["qa_index"]) for q in subset["questions"]}
            self.assertIn(("conv-a", 1), keys)
            self.assertIn(("conv-b", 2), keys)
            self.assertFalse(subset["partial_review"])
            self.assertIn("keys_file", subset["target_diagnostics"]["sources"])

    def test_main_merged_sources_union_dedup(self):
        """keys-file 与 CSV 同时给出：并集、去重、CSV 贡献计入 diagnostics。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cohort, _ = make_pool()
            pool_path = tmp_path / "pool.json"
            pool_path.write_text(json.dumps(cohort, ensure_ascii=False), encoding="utf-8")
            # keys-file 有 conv-a/1、conv-b/2；CSV（规则初判）有 conv-a/1（重叠）、conv-a/3
            keys_path = make_keys_file(tmp_path, ["conv-a/1", "conv-b/2"])
            csv_path = make_review_csv(tmp_path, [
                {"sample_id": "conv-a", "qa_index": 1, "target_experiment": "E5 答案归一化",
                 "suggested_verdict": "fixable"},
                {"sample_id": "conv-a", "qa_index": 3, "target_experiment": "E5 答案归一化",
                 "suggested_verdict": "fixable"},
            ])
            out = tmp_path / "subset.json"
            rc = bts.main([
                "--pool-cohort", str(pool_path),
                "--keys-file", str(keys_path),
                "--review-csv", str(csv_path),
                "--use-rule-suggestion",
                "--total", "8", "--adversarial-guard", "2",
                "--out", str(out),
            ])
            self.assertEqual(rc, 0)
            subset = json.loads(out.read_text(encoding="utf-8"))
            # 并集 3 题（重叠的 conv-a/1 只算一次）
            self.assertEqual(subset["target_questions"], 3)
            self.assertEqual(subset["target_diagnostics"]["sources"], ["keys_file", "csv"])
            # CSV 用规则初判 ⇒ partial=True（即使 keys-file 也是来源之一）
            self.assertTrue(subset["partial_review"])


class RealDataKeysFileTest(unittest.TestCase):
    """在真实 E4 目标集（220 题）上验证端到端构造。"""

    def test_e4_subset_end_to_end(self):
        keys_path = ROOT / "result" / "EVIDENCE_RECALL_20261108" / "e4_target_keys.json"
        pool_path = ROOT / "result" / "locomo_1000_cohort_20261023.json"
        if not keys_path.exists() or not pool_path.exists():
            raise unittest.SkipTest("E4 目标集或母 cohort 不可用")
        keys, _ = bts.load_keys_file(keys_path)
        cohort, pool = bts.load_pool(pool_path)
        subset = bts.build_subset(
            cohort, pool, keys, total=300, seed=20261109,
            verdict_column="keys_file", experiments=None,
            diagnostics={"keys_file": {"keys_file": str(keys_path), "target_keys": len(keys)}},
            adversarial_guard=50)
        problems = bts.self_check(subset, pool, keys)
        self.assertEqual(problems, [])
        self.assertEqual(subset["target_questions"], 220)
        self.assertGreaterEqual(subset["adversarial_total"], 50)
        # E4 目标集定义 = 非对抗且金标未全召回；子集里的目标题必须保持这个性质。
        adversarial_targets = [k for k in keys if pool[k]["category"] == "adversarial"]
        self.assertEqual(adversarial_targets, [])


class TargetInclusionTest(unittest.TestCase):
    def test_all_fixable_targets_included(self):
        cohort, pool = make_pool()
        targets = [("conv-a", 1), ("conv-b", 0)]
        subset = bts.build_subset(
            cohort, pool, targets, total=8, seed=1, verdict_column="human_verdict",
            experiments=None, diagnostics={}, adversarial_guard=2)
        keys = {(q["sample_id"], q["qa_index"]) for q in subset["questions"]}
        for target in targets:
            self.assertIn(target, keys)
        self.assertEqual(subset["target_questions"], 2)

    def test_target_not_in_pool_raises(self):
        """目标题不在母 cohort → 无法复用 SM 基线，必须报错而非静默生成。"""
        cohort, pool = make_pool()
        with self.assertRaises(ValueError) as ctx:
            bts.build_subset(
                cohort, pool, [("conv-z", 9)], total=5, seed=1,
                verdict_column="human_verdict", experiments=None,
                diagnostics={}, adversarial_guard=0)
        self.assertIn("母 cohort", str(ctx.exception))


class AdversarialGuardTest(unittest.TestCase):
    def test_guard_fills_to_quota(self):
        """目标题无对抗题时，守护必须补足到配额。"""
        cohort, pool = make_pool()
        # 目标题都是非对抗
        targets = [("conv-a", 1), ("conv-b", 0)]
        subset = bts.build_subset(
            cohort, pool, targets, total=8, seed=1, verdict_column="human_verdict",
            experiments=None, diagnostics={}, adversarial_guard=2)
        self.assertEqual(subset["adversarial_total"], 2)
        self.assertEqual(subset["adversarial_guard_added"], 2)
        self.assertEqual(subset["adversarial_in_targets"], 0)

    def test_guard_counts_adversarial_already_in_targets(self):
        """目标题里已有对抗题时计入配额，不重复补。"""
        cohort, pool = make_pool()
        targets = [("conv-a", 4)]  # 这是对抗题
        subset = bts.build_subset(
            cohort, pool, targets, total=8, seed=1, verdict_column="human_verdict",
            experiments=None, diagnostics={}, adversarial_guard=2)
        self.assertEqual(subset["adversarial_in_targets"], 1)
        self.assertEqual(subset["adversarial_guard_added"], 1)
        self.assertEqual(subset["adversarial_total"], 2)

    def test_guard_capped_by_pool_availability(self):
        """母 cohort 对抗题不足时，按实际上限放行，self_check 不应误报。"""
        cohort, pool = make_pool()  # 只有 2 道对抗题
        targets = [("conv-a", 1)]
        subset = bts.build_subset(
            cohort, pool, targets, total=8, seed=1, verdict_column="human_verdict",
            experiments=None, diagnostics={}, adversarial_guard=99)
        self.assertEqual(subset["adversarial_total"], 2)  # 池里只有 2 道
        problems = bts.self_check(subset, pool, targets)
        self.assertEqual(problems, [])

    def test_guard_is_deterministic(self):
        """同输入必须产出同守护集，保证可复现。"""
        cohort, pool = make_pool()
        targets = [("conv-a", 1)]
        a = bts.build_subset(cohort, pool, targets, total=8, seed=1, verdict_column="human_verdict",
                             experiments=None, diagnostics={}, adversarial_guard=2)
        b = bts.build_subset(cohort, pool, targets, total=8, seed=1, verdict_column="human_verdict",
                             experiments=None, diagnostics={}, adversarial_guard=2)
        self.assertEqual(a["selection_sha256"], b["selection_sha256"])


class HashAndOrderTest(unittest.TestCase):
    def test_selection_hash_recomputable(self):
        cohort, pool = make_pool()
        subset = bts.build_subset(
            cohort, pool, [("conv-a", 1)], total=6, seed=1, verdict_column="human_verdict",
            experiments=None, diagnostics={}, adversarial_guard=2)
        self.assertEqual(bts.self_check(subset, pool, [("conv-a", 1)]), [])

    def test_questions_sorted_by_sample_index_then_qa(self):
        cohort, pool = make_pool()
        subset = bts.build_subset(
            cohort, pool, [("conv-b", 0), ("conv-a", 3)], total=8, seed=1,
            verdict_column="human_verdict", experiments=None, diagnostics={},
            adversarial_guard=2)
        keys = [(q["sample_index"], q["qa_index"]) for q in subset["questions"]]
        self.assertEqual(keys, sorted(keys))


class OverflowAndControlTest(unittest.TestCase):
    def test_no_controls_when_targets_fill_budget(self):
        """目标题+守护占满 total 时，对照题为 0（应触发告警，见 main）。"""
        cohort, pool = make_pool()
        targets = [("conv-a", 0), ("conv-a", 1), ("conv-a", 2), ("conv-a", 3)]
        subset = bts.build_subset(
            cohort, pool, targets, total=4, seed=1, verdict_column="human_verdict",
            experiments=None, diagnostics={}, adversarial_guard=0)
        self.assertEqual(subset["control_questions"], 0)
        self.assertGreaterEqual(subset["overflow_beyond_total"], 0)

    def test_controls_added_when_budget_remains(self):
        cohort, pool = make_pool()
        subset = bts.build_subset(
            cohort, pool, [("conv-a", 1)], total=8, seed=1, verdict_column="human_verdict",
            experiments=None, diagnostics={}, adversarial_guard=0)
        self.assertGreater(subset["control_questions"], 0)


class LoadTargetKeysTest(unittest.TestCase):
    def test_human_verdict_filter_and_experiment_scope(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            csv_path = make_review_csv(folder, [
                {"sample_id": "conv-a", "qa_index": 1, "category": "single-hop",
                 "initial_category": "多余条目", "target_experiment": "E5 答案归一化",
                 "suggested_verdict": "fixable", "human_verdict": "fixable"},
                {"sample_id": "conv-a", "qa_index": 2, "category": "temporal",
                 "initial_category": "相对时间未展开", "target_experiment": "E3 时间渲染",
                 "suggested_verdict": "fixable", "human_verdict": "not_fixable"},
                {"sample_id": "conv-b", "qa_index": 0, "category": "single-hop",
                 "initial_category": "同义/词面失配", "target_experiment": "不优化",
                 "suggested_verdict": "not_fixable", "human_verdict": ""},
            ])
            keys, diag = bts.load_target_keys(csv_path, "human_verdict", None)
            self.assertEqual(keys, [("conv-a", 1)])  # 只有 human=fixable 的入选
            self.assertEqual(diag["verdict_distribution"]["(空)"], 1)

            # 按实验过滤
            keys_e3, _ = bts.load_target_keys(csv_path, "human_verdict", {"E3 时间渲染"})
            self.assertEqual(keys_e3, [])  # E3 那题人工判 not_fixable，被排除

    def test_missing_verdict_column_raises(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            csv_path = make_review_csv(folder, [
                {"sample_id": "conv-a", "qa_index": 1, "human_verdict": "fixable"},
            ])
            with self.assertRaises(ValueError):
                bts.load_target_keys(csv_path, "nonexistent_column", None)

    def test_empty_csv_raises(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                bts.load_target_keys(path, "human_verdict", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
