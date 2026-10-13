# Multi-hop evidence retrieval and answer composition

This change addresses the observed gap between a fact being indexed and the fact reaching an answer. In the LoCoMo D1–D5 development cohort, 74/75 non-adversarial questions had all gold turn IDs referenced by stored MemNexus memories, while only 48/75 had all of them in the final answer context. Turn-ID coverage does not prove semantic support, but it localizes a substantial failure to retrieval and evidence selection.

## Query route

`classify_question()` now treats an explicit multi-person question as a relation query, even without the word “both.” Plural answer requests, “besides,” and count questions use a recall-first route. Ordinary singular facts keep the focused route. The model planner still supplies semantic queries; the route controls only evidence budget and candidate selection. It does not establish truth or change authorization.

## Authorized memory sweep

For recall-first questions, `MemoryRuntime.answer()` reads the owner's memories, takes each memory's latest version, and applies the existing session/project scope, status, validity and history checks. `indexed_memory_hits()` ranks up to 32 additional facts by question terms, named subject and rare two-word links to first-pass evidence. It keeps a per-subject candidate quota and excludes facts whose primary subject does not match a named subject. Source quotes are resolved and checked by the existing grounding layer before entering the prompt. The sweep adds no LLM calls; in this local implementation it scans the owner's eligible memories linearly.

`ground_evidence()` reserves part of the context for the most relevant first-pass evidence, then greedily selects additional facts using uncovered question terms, named subject coverage, requested field coverage, distinct predicates, source overlap and character cost. Raw turns remain available as a bounded fallback. This is a deterministic heuristic under a budget; there is no approximation guarantee or claim that a lexical match proves entailment.

Recall-first answers may use a larger conservative character budget (`complex_prompt_token_limit=22000`), while focused queries keep the existing limit. The effective evidence budget remains bounded by that limit and `retrieval_token_limit`. The answer model receives an internal subject-by-fact composition instruction for set, relation and count questions. The event graph no longer emits a numerical “event count” merely from the number of memory/source nodes: duplicate mentions and distinct events require a separate identity resolution step.

## Validation and limits

All MemNexus engineering tests pass, including an end-to-end owner/retraction check for the new indexed sweep. On saved memories for the same 13 LoCoMo category-1 development questions, offline full gold-turn coverage rose from 5/13 to 8/13. That is source coverage, not answer accuracy.

The final live rerun rebuilt eight conversation memories and scored all 13 questions with DeepSeek. Correct answers rose from 6/13 in the prior run to 8/13; the saved SimpleMem baseline is 9/13. Three formerly wrong answers became correct (shared shelter volunteering, the two favorite games, and James's pet count); one formerly correct answer became wrong. The median answer latency and input tokens increased. Rebuilding and judging with the same model can vary, so the difference is a development result, not a controlled single-component or held-out estimate.

Remaining errors include evidence-present list omissions (“hope”), a dietary restriction that is not explicitly labeled an allergy, a sport category bridge (surfing), and an omitted shop-opening dream. Source coverage alone cannot resolve these. Future tests should separate question-to-evidence retrieval, event identity, semantic entailment, and calibrated answer generation; the next held-out evaluation must use conversations not involved in this development analysis.
