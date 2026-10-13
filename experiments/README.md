# MemNexus × SimpleMem comparison

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
