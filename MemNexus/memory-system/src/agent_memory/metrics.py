"""Per-runtime model call accounting for reproducible quality/cost experiments."""

from dataclasses import dataclass


@dataclass
class CallMetrics:
    extract_calls: int = 0
    plan_calls: int = 0
    answer_calls: int = 0
    relation_calls: int = 0
    summary_calls: int = 0
    errors: int = 0

    def record(self, operation: str) -> None:
        field = f"{operation}_calls"
        if not hasattr(self, field):
            raise ValueError(f"Unknown model operation: {operation}")
        setattr(self, field, getattr(self, field) + 1)

    def record_error(self) -> None:
        self.errors += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "extract_calls": self.extract_calls,
            "plan_calls": self.plan_calls,
            "answer_calls": self.answer_calls,
            "relation_calls": self.relation_calls,
            "summary_calls": self.summary_calls,
            "errors": self.errors,
        }

    def delta(self, before: dict[str, int]) -> dict[str, int]:
        current = self.snapshot()
        return {key: current[key] - before.get(key, 0) for key in current}
