import importlib.util
from pathlib import Path
import sys
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


if __name__ == "__main__":
    unittest.main()
