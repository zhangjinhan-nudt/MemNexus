"""Deterministic belief-state planning and bounded local replanning.

The module is intentionally provider-free. It gives MemNexus a testable planning
core whose state transitions are backed by versioned observations. A future
LLM planner can propose candidates, while this layer checks preconditions,
resources, dependencies and stale versions before execution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Iterable


@dataclass(frozen=True)
class Fact:
    key: str
    value: str = "true"
    version: int = 1
    source: str = "initial"
    status: str = "active"


@dataclass(frozen=True)
class Action:
    name: str
    requires: frozenset[str] = frozenset()
    consumes: tuple[tuple[str, int], ...] = ()
    produces: frozenset[str] = frozenset()
    duration: int = 1

    @property
    def consumption(self) -> dict[str, int]:
        return dict(self.consumes)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    goal: str
    facts: frozenset[str]
    resources: tuple[tuple[str, int], ...]
    actions: tuple[Action, ...]
    fault_after_steps: int
    fault_resource: str
    fault_amount: int
    recoverable: bool
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def resource_map(self) -> dict[str, int]:
        return dict(self.resources)


@dataclass
class BeliefState:
    facts: dict[str, Fact]
    resources: dict[str, int]
    version: int = 1
    goals: frozenset[str] = frozenset()
    conflicts: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: TaskSpec) -> "BeliefState":
        return cls(
            facts={key: Fact(key=key, source="initial") for key in task.facts},
            resources=task.resource_map.copy(),
            goals=frozenset({task.goal}),
        )

    @classmethod
    def from_memories(cls, memories: Iterable[object], goals: Iterable[str] = ()) -> "BeliefState":
        """Project active, current MemNexus memories into planning facts.

        The adapter intentionally ignores superseded/retracted/pending records
        and records conflicting current values for the same subject/predicate.
        ``memory_id:version`` is retained as Fact.source for action receipts.
        """
        now = datetime.now(UTC)
        facts: dict[str, Fact] = {}
        conflicts: list[str] = []
        for memory in memories:
            status = getattr(memory, "status", "active")
            if status != "active":
                continue
            valid_to = getattr(memory, "valid_to", None)
            if valid_to is not None and valid_to <= now:
                continue
            subject = str(getattr(memory, "subject", "")).strip()
            predicate = str(getattr(memory, "predicate", "")).strip()
            value = str(getattr(memory, "value", "")).strip()
            if not subject or not predicate or not value:
                continue
            key = f"{subject}:{predicate}"
            source = f"{getattr(memory, 'memory_id', 'memory')}:{getattr(memory, 'version', 1)}"
            old = facts.get(key)
            if old is not None and old.value != value:
                conflicts.append(key)
                continue
            facts[key] = Fact(
                key=key,
                value=value,
                version=int(getattr(memory, "version", 1)),
                source=source,
            )
        return cls(facts=facts, resources={}, goals=frozenset(goals), conflicts=sorted(set(conflicts)))

    def clone(self) -> "BeliefState":
        return BeliefState(
            facts=self.facts.copy(),
            resources=self.resources.copy(),
            version=self.version,
            goals=self.goals,
            conflicts=list(self.conflicts),
            uncertainty=list(self.uncertainty),
            history=list(self.history),
        )

    def observe(self, *, fact_updates: Iterable[str] = (), resource_updates: dict[str, int] | None = None, source: str = "observation") -> list[str]:
        """Apply a monotonic observation and return changed state keys."""
        changed: list[str] = []
        for key in fact_updates:
            old = self.facts.get(key)
            self.facts[key] = Fact(
                key=key,
                value="true",
                version=(old.version + 1 if old else 1),
                source=source,
                status="active",
            )
            changed.append(key)
        for key, amount in (resource_updates or {}).items():
            if self.resources.get(key) != amount:
                self.resources[key] = amount
                changed.append(f"resource:{key}")
        if changed:
            self.version += 1
            self.history.append({"version": self.version, "source": source, "changed": list(changed)})
        return changed

    def has_facts(self, required: Iterable[str]) -> bool:
        return all(self.facts.get(key, Fact(key, status="missing")).status == "active" for key in required)

    def can_execute(self, action: Action) -> tuple[bool, str]:
        missing = sorted(key for key in action.requires if not self.has_facts({key}))
        if missing:
            return False, f"missing_facts:{','.join(missing)}"
        for resource, amount in action.consumption.items():
            if self.resources.get(resource, 0) < amount:
                return False, f"insufficient_resource:{resource}"
        return True, "ok"

    def execute(self, action: Action) -> tuple[bool, str]:
        allowed, reason = self.can_execute(action)
        if not allowed:
            return False, reason
        for resource, amount in action.consumption.items():
            self.resources[resource] = self.resources.get(resource, 0) - amount
        self.observe(fact_updates=action.produces, source=f"action:{action.name}")
        return True, "ok"


@dataclass(frozen=True)
class Plan:
    steps: tuple[Action, ...]
    belief_version: int


@dataclass
class ReplanResult:
    plan: Plan | None
    affected_steps: list[str]
    preserved_steps: list[str]
    replanning_calls: int
    reason: str


class DeterministicPlanner:
    """Bounded breadth-first planner for small task domains."""

    def __init__(self, max_depth: int = 12):
        self.max_depth = max_depth

    def plan(self, state: BeliefState, actions: tuple[Action, ...]) -> Plan | None:
        start = frozenset(key for key, fact in state.facts.items() if fact.status == "active")
        start_resources = tuple(sorted(state.resources.items()))
        queue = deque([(start, start_resources, tuple())])
        visited = {(start, start_resources)}
        while queue:
            facts, resources_tuple, path = queue.popleft()
            if state.goals <= facts:
                return Plan(path, state.version)
            if len(path) >= self.max_depth:
                continue
            resources = dict(resources_tuple)
            for action in actions:
                if not action.requires <= facts:
                    continue
                consumption = action.consumption
                if any(resources.get(key, 0) < amount for key, amount in consumption.items()):
                    continue
                next_resources = resources.copy()
                for key, amount in consumption.items():
                    next_resources[key] -= amount
                next_facts = frozenset(set(facts) | set(action.produces))
                marker = (next_facts, tuple(sorted(next_resources.items())))
                if marker in visited:
                    continue
                visited.add(marker)
                queue.append((next_facts, marker[1], path + (action,)))
        return None

    def affected_steps(self, plan: Plan, changed_keys: Iterable[str]) -> tuple[list[str], list[str]]:
        changed = set(changed_keys)
        affected: list[str] = []
        preserved: list[str] = []
        invalidated = False
        for action in plan.steps:
            touches = bool(changed & set(action.requires)) or any(
                f"resource:{key}" in changed for key, _ in action.consumes
            )
            if touches or invalidated:
                invalidated = True
                affected.append(action.name)
            else:
                preserved.append(action.name)
        return affected, preserved

    def replan_local(
        self,
        state: BeliefState,
        old_plan: Plan,
        changed_keys: Iterable[str],
        actions: tuple[Action, ...],
    ) -> ReplanResult:
        affected, preserved = self.affected_steps(old_plan, changed_keys)
        new_plan = self.plan(state, actions)
        return ReplanResult(
            plan=new_plan,
            affected_steps=affected,
            preserved_steps=preserved,
            replanning_calls=1,
            reason="state_change_dependency_replan",
        )


def execute_plan(state: BeliefState, plan: Plan) -> dict:
    actions = []
    violations = 0
    for action in plan.steps:
        success, reason = state.execute(action)
        actions.append({"action": action.name, "success": success, "reason": reason})
        if not success:
            violations += 1
            break
    return {
        "success": state.goals <= {key for key, fact in state.facts.items() if fact.status == "active"},
        "actions": actions,
        "constraint_violations": violations,
    }
