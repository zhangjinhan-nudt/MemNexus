# ICAPS 2027 experiment plan

## Main empirical claim

MemNexus should be evaluated as a memory-conditioned planning system. The
central claim should combine answer quality with planning state quality,
execution robustness and resource use. LoCoMo and LongMemEval-S measure memory
recall and answer quality; they do not by themselves establish planning
capability.

## Required benchmark tables

1. Full LoCoMo, stratified by single-hop, multi-hop, temporal, open-domain and
   adversarial questions.
2. Full LongMemEval-S, stratified by the official question types.
3. Paired per-question comparison against SimpleMem with bootstrap confidence
   intervals and McNemar tests.
4. Cost and latency table with actual provider tokens, attempts, retries,
   validation failures, P50/P95, and cost per correct answer.
5. Retrieval table with evidence Recall@K, MRR, NDCG and citation validity.

## ICAPS-specific planning evaluation

Create a deterministic task suite in which each episode contains a belief state,
goals, resources, temporal constraints, a plan, action outcomes and injected
state changes. Compare:

- SimpleMem context retrieval;
- MemNexus without replanning;
- MemNexus with evidence/version policy;
- MemNexus with local replanning;
- MemNexus full belief-state loop.

Report plan success rate, replanning success rate, steps to recovery, stale-memory
action rate, constraint violation rate, unsupported-action rate, resource
overuse, planning calls, total tokens and cost per successful episode.

The key ICAPS experiment should inject changes after the first action: a
resource becomes unavailable, a deadline changes, a task is completed by another
agent, or a prior fact is retracted. MemNexus must update the belief state,
identify affected plan steps and repair only the affected subplan when possible.

## Completion requirements before claiming superiority

- Full official datasets completed with no hidden failures.
- Every failed provider or validation attempt retained in the ledger.
- Three fixed seeds or paired bootstrap intervals.
- Main quality result uses English mode for both systems.
- Cache disabled for the quality table; cache only in an efficiency ablation.
- All ablations preserve the same raw inputs, questions, model, embedding and
  cost accounting.
- The planning episode suite reports execution outcomes, not only textual QA.
