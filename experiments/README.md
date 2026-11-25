# MemNexus × SimpleMem comparison

Current MemNexus direct QA uses one `source_bound_synthesis` answer path. Collection
queries contribute checked original sources to it; the former collection IR,
semantic binding/execution, reader fallback, profile injection, subject sweep,
LLM bridge and temporal-execution configuration branches have been retired.
Canonical memory records, member operations and retrieval relations remain.

For new LongMemEval-S runs, the native MemNexus adapter explicitly uses
`source_read_scope="events"` for the question's complete supplied history;
`source_session_ids` contains only that history's other internal sessions. This
uses the same input history available to SimpleMem, not gold evidence labels.
LoCoMo keeps its current-session/default `quotes` access. Stage and per-question
records include the scope and grants; the paired protocol also pins this policy.
This is a QA protocol change: old `quotes` or unlabelled LME answers cannot be
resumed or combined into a new `events` run. Reuse the unchanged build snapshot
in a new results directory. Current runners validate both partial row checkpoints
and completed units (including reused judged rows); historical read-only
collection remains available without relabelling old answers. Source visibility
checks do not constitute newly measured Accuracy or F1.

Older experiment scripts are historical reproductions, not current entry points.
For example, `rehearse_deterministic_producer.py` and
`verify_collection_architecture.py` import retired collection APIs; semantic,
reader, subject-sweep and variant scripts may set removed flags that current
code no longer reads. Reproduce those runs with the recorded source revision or
the matching frozen snapshot (the pre-consolidation package and tests are in
`result/consolidation_20261114/source_before`). Do not interpret an ignored old
flag as a valid ablation, and do not combine historical results with a current
run without recording the source, model, data and accounting differences.
These scripts are retained in place; they have not been migrated to the new chain.

`benchmark_runner.py` is the budget-safe P0 entry point. It currently provides:

- complete LoCoMo and LongMemEval-S/oracle schema and evidence-map auditing;
- SHA-256 dataset identities and a locked experiment manifest;
- fairness checks for model, judge, temperature, Qwen3 encoding, window/stride,
  retrieval depth and the combined 8 RMB ceiling;
- a fail-closed paid preflight: blank keys or zero/unknown prices prevent calls.

Run the free audit from the workspace root:

```powershell
python experiments/benchmark_runner.py --mode audit
```

After filling a private API key and current provider prices, check readiness
without making a request:

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:INPUT_PRICE_RMB_PER_MILLION_TOKENS = "..."
$env:OUTPUT_PRICE_RMB_PER_MILLION_TOKENS = "..."
python experiments/benchmark_runner.py --mode paid-preflight
```

The 8 RMB limit is combined across MemNexus, SimpleMem and judging. A future
paid executor must reserve the worst-case completion cost through `BudgetLedger`
before every provider call and commit the provider-reported usage afterward.
