from datetime import timedelta

from agent_memory.models import Evidence, Memory, utcnow
from agent_memory.planning import Action, BeliefState, DeterministicPlanner, TaskSpec


def task(recoverable=True):
    return TaskSpec(
        task_id="t",
        goal="done",
        facts=frozenset({"ready"}),
        resources=(("primary", 1), ("backup", 1)),
        actions=(
            Action("primary", frozenset({"ready"}), (("primary", 1),), frozenset({"done"})),
            Action("backup", frozenset({"ready"}), (("backup", 1),), frozenset({"done"})),
        ),
        fault_after_steps=0,
        fault_resource="primary",
        fault_amount=0,
        recoverable=recoverable,
    )


def test_belief_state_versioned_observation_and_local_replan():
    planner = DeterministicPlanner()
    spec = task()
    state = BeliefState.from_task(spec)
    initial = planner.plan(state, spec.actions)
    assert initial is not None
    changed = state.observe(resource_updates={"primary": 0}, source="fault")
    result = planner.replan_local(state, initial, changed, spec.actions)
    assert result.plan is not None
    assert result.plan.steps[0].name == "backup"
    assert result.replanning_calls == 1
    assert state.version == 2


def test_planner_returns_none_when_all_routes_are_unavailable():
    planner = DeterministicPlanner()
    spec = task(recoverable=False)
    state = BeliefState.from_task(spec)
    initial = planner.plan(state, spec.actions)
    assert initial is not None
    state.observe(resource_updates={"primary": 0, "backup": 0}, source="fault")
    assert planner.plan(state, spec.actions) is None


def test_belief_projection_filters_stale_records_and_keeps_conflicts():
    now = utcnow()
    base = dict(
        memory_id="m",
        session_id="s",
        scope_id="s",
        subject="project",
        predicate="database",
        value="PostgreSQL",
        content="project database PostgreSQL",
        evidence=[Evidence(turn_id="t", event_index=0, quote="PostgreSQL")],
    )
    current = Memory(**base, version=2)
    conflicting = Memory(**{**base, "memory_id": "m2", "value": "SQLite", "content": "project database SQLite"})
    expired = Memory(**{**base, "memory_id": "m3", "valid_to": now - timedelta(seconds=1)})
    state = BeliefState.from_memories([current, conflicting, expired], goals=["project:database"])
    assert state.facts["project:database"].value == "PostgreSQL"
    assert state.facts["project:database"].source == "m:2"
    assert state.conflicts == ["project:database"]
