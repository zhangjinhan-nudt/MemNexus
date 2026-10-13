# LoCoMo official-data pilot analysis

## Scope and status

This is a real-provider development pilot, not the ICAPS main result. It uses
the official `locomo10.json`, conversation `conv-26`, sessions D1–D2 (35 raw
turns), and the first question in dataset order for each of the five official
categories. Both systems independently built memories from the same raw turns;
answer caching was disabled. The same DeepSeek endpoint/model and local
Qwen3-Embedding-0.6B service were used. Every answer, retrieved source,
provider usage, latency and judge output is in `locomo_pilot.json`.

The selected questions are fixed before any provider call. This design is
appropriate for debugging, but five questions are not sufficient for a claim
of statistical superiority, a confidence interval, or a full LoCoMo ranking.

## Observed results

| System | Judge accuracy | Mean answer latency | Answer input tokens | Answer output tokens | Answer cost (RMB) | Memory entries |
|---|---:|---:|---:|---:|---:|---:|
| SimpleMem | 4/5 = 0.800 | 5749.5 ms | 33438 | 2418 | 0.103464 | 97 |
| MemNexus | 5/5 = 1.000 | 1830.0 ms | 17131 | 989 | 0.050609 | 21 |

Relative to SimpleMem in this pilot, MemNexus used 48.8% fewer answer input
tokens, 59.1% fewer answer output tokens, 51.1% lower answer cost and 68.2%
lower mean answer latency. Total main-model cost including memory construction
was ¥0.249686 for SimpleMem and ¥0.150194 for MemNexus. The latter is 39.9%
lower for this pilot. These are measured provider usage values, not estimates.

Category-level judge results:

- SimpleMem: single-hop CORRECT, temporal CORRECT, multi-hop WRONG,
  open-domain CORRECT, adversarial CORRECT.
- MemNexus: all five categories CORRECT in this pilot.

The multi-hop answer was the only SimpleMem judge failure. MemNexus answered
it correctly in this run, but this single example does not establish a general
multi-hop improvement.

## Why the efficiency improved

MemNexus generated 16 provider attempts for construction plus the five
questions, versus 32 for SimpleMem. Its memory representation contained 21
entries rather than 97, and its answer path selected 1–2 calls per question
instead of SimpleMem's 4–8 calls. The new deterministic path was deliberately
limited to a high-confidence research/date pattern and skipped one answer
generation call; predictive and adversarial questions still used the model.

## Important measurement caveats

The pilot's lexical F1/BLEU values are not a fair primary metric because
MemNexus often answers in Chinese while LoCoMo gold answers are English. The
blind DeepSeek judge and official semantic scoring should be primary; token
overlap should only be reported with language-aware normalization or as a
secondary diagnostic. The pilot also measured CPU-bound local Qwen embedding
startup; hardware and warm/cold state must be reported separately in a paper.

The initial failed runs are not included as quality rows. They are retained in
the live usage ledgers and conservatively accounted for in the result's budget
upper bound. One failed run exposed a real SimpleMem truncation bug: a truncated
top-level extraction array was being reduced to its first object. That behavior
is now rejected, and DeepSeek thinking is explicitly disabled for SimpleMem.

## ICAPS-oriented interpretation

The useful research signal is not only QA accuracy. MemNexus currently shows a
promising cost/latency trade-off through structured evidence, bounded planning,
version-aware retrieval and a conservative rule path. For an ICAPS submission,
the next required experiment is a paired, pre-registered sample over all
LoCoMo categories plus LongMemEval-S, followed by ablations of:

1. model planner vs adaptive/rule planner;
2. deterministic direct-answer path on/off;
3. evidence/version policy on/off;
4. semantic retrieval on/off;
5. replanning/belief-state execution loop on/off.

The main table must report paired accuracy, evidence Recall@K/MRR, stale-memory
error rate, LLM calls, actual tokens, P50/P95 latency and cost per correct
answer. This pilot is a diagnostic checkpoint, not a replacement for that
matrix.
