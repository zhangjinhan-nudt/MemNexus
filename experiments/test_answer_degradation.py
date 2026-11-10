"""单题校验失败降级路径的回归测试（2026-11-10 修复）。

背景：E6 cap=2 全量验证时 conv-44（79 题）、conv-48（115 题）连续 6 轮单元重试
全败——synthesis.py 的确定性校验（日期端点/证据引用）在 temp=0 下对同一道题
必然复现同一拒绝，而 run_memnexus 的 `if answer is None and resume: raise`
把单题失败升级为整单元失败，丢弃了该对话已答对的全部题（conv-44 已答对 51 题）。

修复语义（本文件锁住）：
  1. is_transient_error / is_deterministic_output_error 的分类判定；
  2. 降级只作用于确定性校验错误；网络瞬时错误与预算错误仍走 raise；
  3. 降级行的 prediction 必须非空（跑批器把空预测判为单元失败——
     那正是本次修复要消除的传染路径），且带 error 审计字段。

运行：
  python -m pytest experiments/test_answer_degradation.py -q --basetemp=.pytest_tmp/degradation
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_locomo_pilot as pilot  # noqa: E402


class _WithStatus(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class TransientClassificationTest(unittest.TestCase):
    def test_rate_limit_and_server_errors_are_transient(self):
        self.assertTrue(pilot.is_transient_error(_WithStatus(429)))
        self.assertTrue(pilot.is_transient_error(_WithStatus(500)))
        self.assertTrue(pilot.is_transient_error(_WithStatus(503)))

    def test_network_errors_are_transient(self):
        self.assertTrue(pilot.is_transient_error(TimeoutError("t")))
        self.assertTrue(pilot.is_transient_error(ConnectionError("c")))
        # httpx 的异常类名判定
        class ReadTimeout(Exception):
            pass
        self.assertTrue(pilot.is_transient_error(ReadTimeout("x")))

    def test_client_errors_not_transient(self):
        self.assertFalse(pilot.is_transient_error(_WithStatus(400)))
        self.assertFalse(pilot.is_transient_error(_WithStatus(404)))

    def test_validation_value_errors_not_transient(self):
        self.assertFalse(pilot.is_transient_error(
            ValueError("Calculation endpoints need exact source-bound dates")))
        # ModelOutputError 在 providers.py 中是 ValueError 子类
        class ModelOutputError(ValueError):
            pass
        self.assertFalse(pilot.is_transient_error(
            ModelOutputError("Selected time_expression is not supported")))


class DeterministicOutputErrorTest(unittest.TestCase):
    """修复的核心判定：ValueError 族且非瞬时 ⇒ 降级，不杀单元。"""

    def test_validation_errors_are_deterministic(self):
        class ModelOutputError(ValueError):
            pass
        for exc in (
            ValueError("cite at least one original factual utterance"),
            ModelOutputError("Calculation endpoints need exact source-bound dates"),
        ):
            self.assertTrue(pilot.is_deterministic_output_error(exc), repr(exc))

    def test_network_errors_are_not_deterministic(self):
        # 网络错误重试有意义：即使三次题内重试后仍失败，也必须 raise
        # 交给单元级重试（网络恢复后能成功），而不是降级成 0 分。
        self.assertFalse(pilot.is_deterministic_output_error(TimeoutError("t")))
        self.assertFalse(pilot.is_deterministic_output_error(_WithStatus(429)))

    def test_budget_and_unknown_errors_are_not_deterministic(self):
        # BudgetExceededError 继承 BaseException：预算耗尽必须停下，不能降级。
        class BudgetExceededError(BaseException):
            pass
        self.assertFalse(pilot.is_deterministic_output_error(BudgetExceededError()))
        self.assertFalse(pilot.is_deterministic_output_error(RuntimeError("unknown")))
        self.assertFalse(pilot.is_deterministic_output_error(None))


class DegradedRowContractTest(unittest.TestCase):
    """降级行的契约：非空 prediction + 审计字段。

    这里静态检查 run_memnexus 内的降级行构造（该函数依赖完整 runtime，
    不做单元级 mock），锁住两处最容易被改坏的性质：
      1. prediction 非空（跑批器契约：空预测 = 单元失败 = 传染路径复活）；
      2. error 字段存在（供事后按降级原因审计，区分真实拒答与降级拒答）。
    """

    def test_degraded_prediction_is_nonempty_abstention(self):
        # 降级措辞与主表 AdvSR 判定一致（paper_protocol 的 ABSTAIN_MARKERS）
        import paper_protocol as pp
        text = "Not mentioned in the conversation"
        self.assertTrue(pp.adversarial_success({"prediction": text}))
        self.assertTrue(str(text).strip())  # 非空：跑批器契约

    def test_run_memnexus_source_contains_degradation_contract(self):
        """源码级回归锁：降级行必须非空且带 error 字段，raise 必须被判定守卫。"""
        source = Path(pilot.__file__).read_text(encoding="utf-8")
        # 判定守卫存在：只有非确定性错误才 raise
        self.assertIn("if not is_deterministic_output_error(error):", source)
        # 降级行的非空措辞存在（替换旧的 "prediction": ""）
        self.assertIn('"prediction": "Not mentioned in the conversation"', source)
        self.assertNotIn('"prediction": "",\n                    # The answer request raised', source)
        # 审计字段存在
        self.assertIn('"error": f"{type(error).__name__}: {error}"', source)


if __name__ == "__main__":
    unittest.main()
