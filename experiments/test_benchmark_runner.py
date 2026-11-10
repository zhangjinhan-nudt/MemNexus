import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("benchmark_runner.py")
SPEC = importlib.util.spec_from_file_location("benchmark_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class BudgetLedgerTests(unittest.TestCase):
    def test_reserve_and_commit_under_combined_limit(self):
        ledger = runner.BudgetLedger(8.0, 1.0, 2.0)
        self.assertEqual(ledger.reserve(1_000_000, 1_000_000), 3.0)
        self.assertEqual(ledger.commit(1_000_000, 500_000), 2.0)
        self.assertEqual(ledger.spent_rmb, 2.0)

    def test_zero_prices_fail_closed(self):
        with self.assertRaises(runner.PreflightError):
            runner.BudgetLedger(8.0, 0.0, 0.0)

    def test_reservation_cannot_cross_limit(self):
        ledger = runner.BudgetLedger(8.0, 4.0, 8.0)
        ledger.commit(1_000_000, 0)
        with self.assertRaises(runner.PreflightError):
            ledger.reserve(0, 1_000_000)


class DatasetAuditTests(unittest.TestCase):
    def test_manifest_and_official_evidence_mappings(self):
        manifest = runner.build_manifest()
        self.assertEqual(manifest["datasets"]["locomo"]["conversations"], 10)
        self.assertEqual(manifest["datasets"]["longmemeval_s"]["questions"], 500)
        # The downloaded official file contains two dangling references and two
        # malformed evidence entries. Preserve and report them; never repair gold
        # mappings silently inside one system's adapter.
        self.assertEqual(manifest["datasets"]["locomo"]["unresolved_evidence_references"], 2)
        self.assertEqual(manifest["datasets"]["locomo"]["unparseable_evidence_entries"], 2)
        self.assertEqual(
            manifest["datasets"]["longmemeval_s"]["unresolved_evidence_session_references"],
            0,
        )


def load_local_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Night1000ResumeTests(unittest.TestCase):
    def test_only_valid_rows_are_resumed_and_latest_valid_row_wins(self):
        pilot = load_local_module("run_locomo_pilot")
        base = {"sample_id": "conv-x", "qa_index": 0, "question": "Q", "evaluation_query": "Q", "gold_answer": "A"}
        temp_root = Path(__file__).parents[1] / ".pytest_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            ledger = Path(folder) / "usage.json"
            ledger.with_suffix(".rows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in (
                    {**base, "prediction": "old"},
                    {**base, "prediction": "", "error": "TimeoutError"},
                    {**base, "prediction": "new"},
                    {**base, "question": "Q2", "evaluation_query": "Q2", "prediction": ""},
                    {**base, "sample_id": "other", "prediction": "unrelated"},
                )) + "\n",
                encoding="utf-8",
            )
            rows = pilot.load_resumable_rows(ledger, "conv-x")
        self.assertEqual(len(rows), 1)
        self.assertEqual(next(iter(rows.values()))["prediction"], "new")

    def test_watchdog_counts_units_not_questions(self):
        watchdog = load_local_module("night_watchdog")
        root = Path(__file__).parents[1]
        self.assertEqual(
            watchdog.expected_unit_count(root / "result/locomo_1000_cohort_20261023.json"), 20)
        self.assertEqual(
            watchdog.expected_unit_count(root / "result/longmemeval_1000_cohort_20261023.json"), 2000)

    def test_watchdog_counts_only_unique_valid_checkpoint_answers(self):
        watchdog = load_local_module("night_watchdog")
        base = {"sample_id": "conv-x", "qa_index": 0, "question": "Q", "evaluation_query": "Q", "gold_answer": "A"}
        temp_root = Path(__file__).parents[1] / ".pytest_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            unit = Path(folder) / "attempts" / "conv-x_memnexus"
            unit.mkdir(parents=True)
            (unit / "usage.rows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in (
                    {**base, "prediction": "old"},
                    {**base, "prediction": "new"},
                    {**base, "question": "Q2", "evaluation_query": "Q2", "prediction": ""},
                    {**base, "question": "Q3", "evaluation_query": "Q3",
                     "prediction": "failed", "error": "TimeoutError"},
                )) + "\n", encoding="utf-8")
            (unit / "usage.json").write_text("{}", encoding="utf-8")
            valid, latest_activity = watchdog.checkpoint_progress(Path(folder))
        self.assertEqual(valid, 1)
        self.assertGreater(latest_activity, 0)

    def test_parallel_runner_injects_cohort_qa_index(self):
        parallel = load_local_module("run_parallel_bench")
        sample = {"qa": [{"question": "q0"}, {"question": "q1"}]}
        items = [{"qa_index": 1}, {"qa_index": 0}]
        questions = parallel.cohort_questions(sample, items)
        self.assertEqual([q["qa_index"] for q in questions], [1, 0])
        self.assertEqual([q["question"] for q in questions], ["q1", "q0"])
        self.assertNotIn("qa_index", sample["qa"][0])

    def test_resolve_systems_default_runs_both(self):
        """空串必须等于改动前的行为：两个系统都跑。向后兼容的硬约束。"""
        parallel = load_local_module("run_parallel_bench")
        self.assertEqual(parallel.resolve_systems(""), ("MemNexus", "SimpleMem"))

    def test_resolve_systems_single(self):
        parallel = load_local_module("run_parallel_bench")
        self.assertEqual(parallel.resolve_systems("MemNexus"), ("MemNexus",))
        self.assertEqual(parallel.resolve_systems("SimpleMem"), ("SimpleMem",))

    def test_resolve_systems_rejects_unknown(self):
        """非法系统名必须报错，不能静默退化成跑全部或跑空。"""
        parallel = load_local_module("run_parallel_bench")
        for bad in ("Foo", "memnexus", "MEMNEXUS", "SimpleMem,MemNexus"):
            with self.assertRaises(ValueError):
                parallel.resolve_systems(bad)

    def _write_unit(self, root, sample_id, system, rows):
        unit = root / "attempts" / f"{sample_id}_{system.lower()}"
        unit.mkdir(parents=True, exist_ok=True)
        (unit / "stage_result.json").write_text(
            json.dumps({"stage": {}, "rows": rows}), encoding="utf-8")

    def test_collect_reuses_untouched_system_when_keys_match(self):
        """只跑 MN 时，SM 既有结果被复用；键集与 cohort 一致则正常合并。"""
        parallel = load_local_module("run_parallel_bench")
        temp_root = Path(__file__).parents[1] / ".pytest_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            out = Path(folder)
            # cohort 期望 conv-x 有 qa_index 0 和 1 两题
            self._write_unit(out, "conv-x", "MemNexus", [
                {"sample_id": "conv-x", "qa_index": 0, "category": "single-hop", "system": "MemNexus"},
                {"sample_id": "conv-x", "qa_index": 1, "category": "single-hop", "system": "MemNexus"},
            ])
            self._write_unit(out, "conv-x", "SimpleMem", [
                {"sample_id": "conv-x", "qa_index": 0, "category": "single-hop", "system": "SimpleMem"},
                {"sample_id": "conv-x", "qa_index": 1, "category": "single-hop", "system": "SimpleMem"},
            ])
            dataset = [{"sample_id": "conv-x"}]
            groups = {0: [{"qa_index": 0}, {"qa_index": 1}]}
            rows, reused = parallel.collect_run_rows(
                out=out, dataset=dataset, indices=[0], groups=groups,
                active_systems=("MemNexus",))
            self.assertEqual(len(rows), 4)
            self.assertEqual(dict(reused), {"SimpleMem": 1})

    def test_collect_rejects_reused_unit_with_missing_keys(self):
        """复用的 SM 单元缺题（来自旧 cohort）必须报错，不能静默混入。"""
        parallel = load_local_module("run_parallel_bench")
        temp_root = Path(__file__).parents[1] / ".pytest_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            out = Path(folder)
            self._write_unit(out, "conv-x", "MemNexus", [
                {"sample_id": "conv-x", "qa_index": 0, "category": "single-hop", "system": "MemNexus"},
                {"sample_id": "conv-x", "qa_index": 1, "category": "single-hop", "system": "MemNexus"},
            ])
            # SM 只有 qa_index 0，缺 1 —— 模拟来自不同 cohort 的旧结果
            self._write_unit(out, "conv-x", "SimpleMem", [
                {"sample_id": "conv-x", "qa_index": 0, "category": "single-hop", "system": "SimpleMem"},
            ])
            dataset = [{"sample_id": "conv-x"}]
            groups = {0: [{"qa_index": 0}, {"qa_index": 1}]}
            with self.assertRaises(ValueError) as ctx:
                parallel.collect_run_rows(
                    out=out, dataset=dataset, indices=[0], groups=groups,
                    active_systems=("MemNexus",))
            self.assertIn("不匹配", str(ctx.exception))

    def test_collect_rejects_reused_unit_with_extra_keys(self):
        """复用的 SM 单元多出当前 cohort 没有的题，同样必须报错。"""
        parallel = load_local_module("run_parallel_bench")
        temp_root = Path(__file__).parents[1] / ".pytest_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            out = Path(folder)
            self._write_unit(out, "conv-x", "MemNexus", [
                {"sample_id": "conv-x", "qa_index": 0, "category": "single-hop", "system": "MemNexus"},
            ])
            # SM 有 qa_index 0 和 99，其中 99 不在当前 cohort
            self._write_unit(out, "conv-x", "SimpleMem", [
                {"sample_id": "conv-x", "qa_index": 0, "category": "single-hop", "system": "SimpleMem"},
                {"sample_id": "conv-x", "qa_index": 99, "category": "single-hop", "system": "SimpleMem"},
            ])
            dataset = [{"sample_id": "conv-x"}]
            groups = {0: [{"qa_index": 0}]}
            with self.assertRaises(ValueError):
                parallel.collect_run_rows(
                    out=out, dataset=dataset, indices=[0], groups=groups,
                    active_systems=("MemNexus",))

    def test_collect_does_not_validate_executed_system(self):
        """本次执行的系统（active）不做键集校验：它正在被写入，键集由本轮 cohort 决定。"""
        parallel = load_local_module("run_parallel_bench")
        temp_root = Path(__file__).parents[1] / ".pytest_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            out = Path(folder)
            self._write_unit(out, "conv-x", "MemNexus", [
                {"sample_id": "conv-x", "qa_index": 0, "category": "single-hop", "system": "MemNexus"},
            ])
            dataset = [{"sample_id": "conv-x"}]
            groups = {0: [{"qa_index": 0}]}
            # MemNexus 在 active_systems 内 → 即便只有它、无 SM，也不报错
            rows, reused = parallel.collect_run_rows(
                out=out, dataset=dataset, indices=[0], groups=groups,
                active_systems=("MemNexus",))
            self.assertEqual(len(rows), 1)
            self.assertEqual(dict(reused), {})

    def test_duplicate_text_questions_resume_independently(self):
        pilot = load_local_module("run_locomo_pilot")
        temp_root = Path(__file__).parents[1] / ".pytest_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            ledger = Path(folder) / "usage.json"
            rows = [
                {"sample_id": "conv-x", "qa_index": qa, "question": "same",
                 "evaluation_query": "same", "gold_answer": "A", "prediction": f"answer-{qa}"}
                for qa in (67, 74)
            ]
            ledger.with_suffix(".rows.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            completed = pilot.load_resumable_rows(ledger, "conv-x")
        self.assertEqual(len(completed), 2)
        self.assertEqual({row["qa_index"] for row in completed.values()}, {67, 74})

    def test_legacy_migration_keeps_unique_rows_and_quarantines_ambiguity(self):
        repair = load_local_module("repair_resume_identity")
        items = [
            {"sample_id": "conv-x", "qa_index": 1, "question": "same"},
            {"sample_id": "conv-x", "qa_index": 2, "question": "same"},
            {"sample_id": "conv-x", "qa_index": 3, "question": "unique"},
        ]
        records = [
            {"sample_id": "conv-x", "question": "same", "prediction": "old-1"},
            {"sample_id": "conv-x", "question": "same", "prediction": "old-2"},
            {"sample_id": "conv-x", "question": "unique", "prediction": "keep"},
        ]
        repaired, dropped = repair.repair_records(records, items)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0]["qa_index"], 3)


if __name__ == "__main__":
    unittest.main()
