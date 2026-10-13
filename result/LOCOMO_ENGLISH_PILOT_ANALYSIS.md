# English output LoCoMo pilot

This report uses the official LoCoMo file, conversation `conv-26`, raw
sessions D1–D2 (35 turns), and one question from each official category. Both
systems independently built memory from the same raw turns. MemNexus ran with
`MEMORY_OUTPUT_LANGUAGE=en`; the original evidence quotes were preserved.
Answer caching was disabled, and the same DeepSeek model, judge and Qwen3
embedding service were used.

This is a development pilot with five questions. It is not a statistically
powered ICAPS main result.

| System | Judge accuracy | Mean answer latency | Answer input tokens | Answer output tokens | Answer cost (RMB) | Main-model cost including build |
|---|---:|---:|---:|---:|---:|---:|
| SimpleMem | 4/5 = 0.800 | 5703.2 ms | 35421 | 2502 | 0.109030 | 0.254856 |
| MemNexus | 5/5 = 1.000 | 2020.4 ms | 15657 | 1281 | 0.049874 | 0.156420 |

In this English pilot, MemNexus used 55.8% fewer answer input tokens, 48.8%
fewer answer output tokens, 54.2% lower answer cost and 64.6% lower mean
answer latency. Including memory construction, measured main-model cost was
38.6% lower. The only SimpleMem judge failure was the multi-hop question; the
MemNexus answer correctly stated that the data did not confirm a specific major
or degree while identifying the supported counseling/mental-health direction.

The lexical F1/BLEU values are now more comparable because both answers and
gold answers are English, but the blinded judge remains the primary quality
measure for this pilot. The five-question sample still cannot support a
significance claim.

## Why English is the formal mode

English is required for the main LoCoMo and LongMemEval-S comparison because
the official questions and gold answers are English and the target venue is an
international planning conference. It removes a language mismatch from the
quality metric and makes the raw answers readable to reviewers. Chinese user
interaction can remain a separate language ablation; it should not be mixed
with the English main table.

## Validation failure observed during the larger attempt

An attempted two-questions-per-category run produced no valid MemNexus quality
rows because the provider proposed a memory candidate supported only by a
question. The evidence validator rejected it, as required by the evidence
policy. The run is recorded in `FAILED_ENGLISH_PILOT.json`. The runner now
closes the SQLite retrieval projection before propagating such failures.

For the ICAPS evaluation, report this as an extraction validation failure rate
and provider repair rate. Do not silently discard these failures or turn them
into negative QA answers.
