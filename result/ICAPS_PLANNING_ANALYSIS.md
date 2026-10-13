# ICAPS planning loop fault-injection analysis

## Scope

This experiment validates the first planning prototype in a deterministic
synthetic environment. It contains 20 fixed-seed episodes across four domains:
resource routing, infrastructure deployment, data migration and temporal
resource scheduling. Each initial plan is checked, then one external resource
state change is injected. Sixteen episodes remain recoverable; four deliberately
remove all alternative resources to test safe failure.

The comparison is:

- **stale context baseline**: continues the original plan after the fault;
- **global replanner**: observes the fault and regenerates a complete plan;
- **MemNexus local replanner**: applies a versioned observation, computes affected
  steps and replans once from the current belief state.

This is a mechanism experiment, not a public benchmark score. It uses no model
calls and costs no API budget.

## Results

| Metric | Stale baseline | Global replanner | MemNexus local replanner |
|---|---:|---:|---:|
| Plan success | 0.00 | 0.80 | 0.80 |
| Replanning success on recoverable episodes | 0.00 | 1.00 | 1.00 |
| Constraint violations | 20 | 0 | 0 |
| Mean preserved prefix steps | not applicable | 0 | 1.00 |

The four unrecoverable episodes fail safely for both replanning methods because
no valid route remains. On the 16 recoverable episodes, both global and local
replanning recover all tasks. MemNexus preserves the already executed prefix
and repairs the affected suffix with one bounded replan call.

Raw per-episode traces are in
`result/icaps_planning_fault_injection.json`. The implementation is in
`MemNexus/memory-system/src/agent_memory/planning.py` and the runner is in
`experiments/run_icaps_planning.py`.

`MemoryRuntime.build_belief_state(...)` now projects the authorized active
short/long memory snapshot into this typed planning state, so scope, status,
validity and version filtering remain at the memory boundary.

## What this demonstrates

The prototype has the four required ICAPS mechanisms:

1. typed belief state with facts, resources, goals, conflicts, history and a
   monotonic state version;
2. actions with preconditions, resource consumption and effects;
3. dependency-based impact analysis after an external state change;
4. one bounded local replanning step with preserved unaffected actions.

The current result supports the mechanism claim that versioned local
replanning can avoid stale-resource actions while preserving an unaffected plan
prefix. It does not yet establish superiority on realistic planning domains.

## Next validation needed

The synthetic environment should be extended with temporal deadlines, negative
preconditions, conflicting observations, action durations and partial action
failure. The next paper-quality experiment should use standard planning-domain
tasks or a separately released task generator, compare against a planner that
replans globally, and report paired confidence intervals over multiple seeds.
