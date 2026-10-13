"""Deterministic fault-injection benchmark for the ICAPS planning loop.

The benchmark is a small synthetic planning environment created for mechanism
validation. It is not a public benchmark score. Every task is generated from a
fixed seed and every initial plan is checked before a fault is injected.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from agent_memory.planning import (
    Action,
    BeliefState,
    DeterministicPlanner,
    Plan,
    TaskSpec,
    execute_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def task_templates() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(
            task_id="logistics",
            goal="delivered",
            facts=frozenset({"order_ready"}),
            resources=(("box", 1), ("standard_slot", 1), ("express_slot", 1)),
            actions=(
                Action("pack_order", frozenset({"order_ready"}), (("box", 1),), frozenset({"packed"})),
                Action("standard_ship", frozenset({"packed"}), (("standard_slot", 1),), frozenset({"delivered"})),
                Action("express_ship", frozenset({"packed"}), (("express_slot", 1),), frozenset({"delivered"})),
            ),
            fault_after_steps=1,
            fault_resource="standard_slot",
            fault_amount=0,
            recoverable=True,
            metadata={"domain": "resource_routing"},
        ),
        TaskSpec(
            task_id="deployment",
            goal="verified",
            facts=frozenset({"artifact_ready"}),
            resources=(("primary_host", 1), ("backup_host", 1)),
            actions=(
                Action("prepare_release", frozenset({"artifact_ready"}), (), frozenset({"release_prepared"})),
                Action("deploy_primary", frozenset({"release_prepared"}), (("primary_host", 1),), frozenset({"deployed"})),
                Action("deploy_backup", frozenset({"release_prepared"}), (("backup_host", 1),), frozenset({"deployed"})),
                Action("verify_service", frozenset({"deployed"}), (), frozenset({"verified"})),
            ),
            fault_after_steps=1,
            fault_resource="primary_host",
            fault_amount=0,
            recoverable=True,
            metadata={"domain": "infrastructure"},
        ),
        TaskSpec(
            task_id="migration",
            goal="cutover_done",
            facts=frozenset({"backup_ready"}),
            resources=(("primary_db", 1), ("emergency_db", 1)),
            actions=(
                Action("prepare_migration", frozenset({"backup_ready"}), (), frozenset({"migration_ready"})),
                Action("migrate_primary", frozenset({"migration_ready"}), (("primary_db", 1),), frozenset({"migrated"})),
                Action("migrate_emergency", frozenset({"migration_ready"}), (("emergency_db", 1),), frozenset({"migrated"})),
                Action("validate_migration", frozenset({"migrated"}), (), frozenset({"validated"})),
                Action("cutover", frozenset({"validated"}), (), frozenset({"cutover_done"})),
            ),
            fault_after_steps=1,
            fault_resource="primary_db",
            fault_amount=0,
            recoverable=True,
            metadata={"domain": "data_migration"},
        ),
        TaskSpec(
            task_id="event",
            goal="event_ready",
            facts=frozenset({"agenda_ready"}),
            resources=(("venue", 1), ("remote_channel", 1)),
            actions=(
                Action("prepare_event", frozenset({"agenda_ready"}), (), frozenset({"event_prepared"})),
                Action("schedule_venue", frozenset({"event_prepared"}), (("venue", 1),), frozenset({"scheduled"})),
                Action("schedule_remote", frozenset({"event_prepared"}), (("remote_channel", 1),), frozenset({"scheduled"})),
                Action("publish_event", frozenset({"scheduled"}), (), frozenset({"event_ready"})),
            ),
            fault_after_steps=1,
            fault_resource="venue",
            fault_amount=0,
            recoverable=True,
            metadata={"domain": "temporal_resource"},
        ),
    )


def make_tasks(count: int, seed: int) -> list[TaskSpec]:
    rng = random.Random(seed)
    templates = task_templates()
    tasks: list[TaskSpec] = []
    for index in range(count):
        template = templates[index % len(templates)]
        recoverable = template.recoverable
        # Every fifth episode removes both primary and alternative resources.
        # This tests safe failure and prevents a benchmark where recovery is
        # guaranteed by construction.
        impossible = index % 5 == 4
        resources = dict(template.resources)
        if impossible:
            recoverable = False
        task = TaskSpec(
            task_id=f"{template.task_id}-{index:02d}",
            goal=template.goal,
            facts=template.facts,
            resources=tuple(sorted(resources.items())),
            actions=template.actions,
            fault_after_steps=template.fault_after_steps,
            fault_resource=template.fault_resource,
            fault_amount=template.fault_amount,
            recoverable=recoverable,
            metadata={**template.metadata, "seed": str(seed), "impossible": str(impossible).lower()},
        )
        tasks.append(task)
        rng.random()  # consume deterministic entropy for future template variants
    return tasks


def action_names(plan: Plan | None) -> list[str]:
    return [action.name for action in plan.steps] if plan else []


def run_episode(task: TaskSpec, planner: DeterministicPlanner) -> dict:
    initial = BeliefState.from_task(task)
    initial_plan = planner.plan(initial, task.actions)
    if initial_plan is None:
        raise AssertionError(f"No initial plan for {task.task_id}")

    # Execute the prefix before the fault. The fault is an external state update.
    prefix_state = initial.clone()
    prefix_actions = list(initial_plan.steps[: task.fault_after_steps])
    prefix_result = execute_plan(prefix_state, Plan(tuple(prefix_actions), prefix_state.version))
    if any(not action["success"] for action in prefix_result["actions"]):
        raise AssertionError(f"Initial prefix failed for {task.task_id}")
    fault_updates = {task.fault_resource: task.fault_amount}
    if not task.recoverable:
        fault_updates.update({key: 0 for key in prefix_state.resources if key != "box"})
    changed = prefix_state.observe(
        resource_updates=fault_updates,
        source=f"fault:{task.fault_resource}",
    )

    # Context-only baseline: it continues the old plan and cannot invalidate
    # stale resource assumptions after the injected fault.
    baseline_state = prefix_state.clone()
    baseline_tail = Plan(tuple(initial_plan.steps[task.fault_after_steps:]), baseline_state.version)
    baseline_result = execute_plan(baseline_state, baseline_tail)

    # Stronger baseline: a conventional global replanner observes the fault
    # and rebuilds the complete plan, but has no dependency analysis or
    # preservation guarantee.
    global_state = prefix_state.clone()
    global_plan = planner.plan(global_state, task.actions)
    global_result = execute_plan(global_state, global_plan) if global_plan else {
        "success": False,
        "actions": [],
        "constraint_violations": 0,
    }

    # MemNexus algorithm: versioned observation, dependency impact analysis and
    # one bounded local replan from the post-fault state.
    local = planner.replan_local(prefix_state, initial_plan, changed, task.actions)
    nexus_state = prefix_state.clone()
    nexus_result = execute_plan(nexus_state, local.plan) if local.plan else {
        "success": False,
        "actions": [],
        "constraint_violations": 0,
    }
    return {
        "task_id": task.task_id,
        "domain": task.metadata["domain"],
        "recoverable": task.recoverable,
        "initial_plan": action_names(initial_plan),
        "fault": {"resource": task.fault_resource, "amount": task.fault_amount, "changed": changed},
        "baseline": {
            "plan_success": baseline_result["success"],
            "replanning_success": False,
            "constraint_violations": baseline_result["constraint_violations"],
            "actions": baseline_result["actions"],
        },
        "global_replanner": {
            "plan_success": global_result["success"],
            "replanning_success": bool(task.recoverable and global_result["success"]),
            "constraint_violations": global_result["constraint_violations"],
            "actions": global_result["actions"],
            "replanned_plan": action_names(global_plan),
        },
        "memnexus": {
            "plan_success": nexus_result["success"],
            "replanning_success": bool(task.recoverable and nexus_result["success"]),
            "constraint_violations": nexus_result["constraint_violations"],
            "actions": nexus_result["actions"],
            "replanning_calls": local.replanning_calls,
            "affected_steps": local.affected_steps,
            "preserved_steps": local.preserved_steps,
            "replanned_plan": action_names(local.plan),
            "reason": local.reason,
        },
    }


def summarize(rows: list[dict]) -> dict:
    recoverable = [row for row in rows if row["recoverable"]]
    return {
        "episodes": len(rows),
        "recoverable_episodes": len(recoverable),
        "baseline_plan_success": sum(row["baseline"]["plan_success"] for row in rows) / len(rows),
        "memnexus_plan_success": sum(row["memnexus"]["plan_success"] for row in rows) / len(rows),
        "baseline_replanning_success": 0.0,
        "global_replanner_plan_success": sum(row["global_replanner"]["plan_success"] for row in rows) / len(rows),
        "global_replanner_replanning_success": (
            sum(row["global_replanner"]["replanning_success"] for row in recoverable) / len(recoverable)
            if recoverable else 0.0
        ),
        "memnexus_replanning_success": (
            sum(row["memnexus"]["replanning_success"] for row in recoverable) / len(recoverable)
            if recoverable else 0.0
        ),
        "baseline_constraint_violations": sum(row["baseline"]["constraint_violations"] for row in rows),
        "memnexus_constraint_violations": sum(row["memnexus"]["constraint_violations"] for row in rows),
        "mean_preserved_steps": sum(len(row["memnexus"]["preserved_steps"]) for row in rows) / len(rows),
    }


def main(args) -> int:
    planner = DeterministicPlanner(max_depth=12)
    tasks = make_tasks(args.episodes, args.seed)
    rows = [run_episode(task, planner) for task in tasks]
    report = {
        "schema_version": 1,
        "status": "completed",
        "benchmark": "synthetic_fault_injection_planning",
        "seed": args.seed,
        "episodes": args.episodes,
        "environment": {
            "action_model": "deterministic precondition/resource/effect transitions",
            "fault_model": "one external resource state change after initial plan prefix",
            "baseline": "stale-plan context-only execution without belief-state update",
            "memnexus": "versioned observation plus one dependency-aware local replan",
        },
        "summary": summarize(rows),
        "episodes_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output", type=Path, default=ROOT / "result" / "icaps_planning_fault_injection.json")
    options = parser.parse_args()
    if not 5 <= options.episodes <= 20:
        parser.error("--episodes must be between 5 and 20")
    raise SystemExit(main(options))
