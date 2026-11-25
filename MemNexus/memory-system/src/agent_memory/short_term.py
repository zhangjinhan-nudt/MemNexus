"""Report p9: five pending turns, fifteen context turns, token pressure first."""

import asyncio
import json

from .metrics import CallMetrics
from .models import ExtractionWindow, ReferenceFact, Scope, StatePatch
from .ports import ModelNotConfigured
from .retrieval import _terms, estimate_tokens


class ShortTermMemory:
    def __init__(self, store, settings, model=None, metrics: CallMetrics | None = None):
        self.store, self.settings, self.model, self.metrics = store, settings, model, metrics

    async def window(self, scope: Scope, session_id: str) -> ExtractionWindow:
        session, turns = await asyncio.to_thread(self.store.session_snapshot, scope, session_id)
        pending = [t for t in turns if t.sequence > session.processed_sequence]
        new_turns = []
        # Prompt/schema overhead is intentionally reserved; count is a conservative estimate.
        budget = self.settings.window_token_limit - estimate_tokens(session.model_dump_json()) - 1000
        if budget <= 0:
            raise ValueError("Session state exceeds extraction budget; shorten state before extracting")
        for turn in pending[: self.settings.pending_turns]:
            cost = estimate_tokens(json.dumps(turn.prompt_dump(), ensure_ascii=False))
            if cost > budget:
                if not new_turns:
                    if cost > self.settings.max_turn_chars:
                        raise ValueError(
                            "Single turn exceeds extraction budget; split large tool output into turns"
                        )
                    # A legitimate long utterance may occupy a batch on its own:
                    # the window budget governs how much *context* a batch
                    # carries, not whether such a turn can be ingested at all.
                    new_turns.append(turn)
                    budget = 0
                break
            new_turns.append(turn)
            budget -= cost
        references = []
        if new_turns:
            long = await asyncio.to_thread(self.store.memories, scope, None, "long")
            query_terms = _terms(" ".join(event.content for turn in new_turns for event in turn.events))
            ranked = []
            for memory in long:
                if (
                    memory.status != "active"
                    and not (memory.status == "superseded" and memory.valid_to is not None)
                ):
                    continue
                if memory.scope_type == "project" and memory.scope_id != session.project_id:
                    continue
                if memory.scope_type not in {"project", "user"}:
                    continue
                overlap = 2 * len(query_terms & _terms(memory.subject + " " + memory.predicate))
                value_terms = _terms(" ".join(filter(None, (
                    memory.value, memory.semantics.event_key, memory.semantics.event_identity,
                ))))
                # Shared dates/numbers alone do not identify a related field or event.
                overlap += len(query_terms & {term for term in value_terms if not any(c.isdigit() for c in term)})
                if overlap:
                    ranked.append((overlap, memory))
            reference_budget = min(budget, 1200)
            seen = set()
            for _, memory in sorted(ranked, key=lambda item: (-item[0], item[1].memory_id)):
                identity = (memory.scope_type, memory.subject, memory.predicate,
                            memory.assertion, memory.valid_from, memory.valid_to,
                            memory.semantics.event_key, memory.semantics.event_identity,
                            memory.value,
                            memory.memory_id if not memory.semantics.event_identity else None)
                if identity in seen:
                    continue
                ref = ReferenceFact(
                    subject=memory.subject,
                    predicate=memory.predicate,
                    value=memory.value,
                    scope_type=memory.scope_type,
                    cardinality=memory.semantics.cardinality,
                    event_key=memory.semantics.event_key,
                    event_identity=memory.semantics.event_identity,
                    assertion=memory.assertion,
                    status=memory.status,
                    valid_from=memory.valid_from,
                    valid_to=memory.valid_to,
                )
                cost = estimate_tokens(ref.model_dump_json()) + 2
                if cost > reference_budget:
                    continue
                references.append(ref)
                seen.add(identity)
                reference_budget -= cost
                budget -= cost
                if len(references) == 8:
                    break
        context = []
        previous = [t for t in turns if t.sequence <= session.processed_sequence]
        for turn in reversed(previous[-self.settings.context_turns :] if self.settings.context_turns else []):
            cost = estimate_tokens(json.dumps(turn.prompt_dump(), ensure_ascii=False))
            if cost > budget:
                break
            context.insert(0, turn)
            budget -= cost
        return ExtractionWindow(
            session=session, new_turns=new_turns, context_turns=context, reference_fields=references
        )

    async def extract(self, scope: Scope, session_id: str):
        window = await self.window(scope, session_id)
        if not window.new_turns:
            return {"status": "idle", "processed_sequence": window.session.processed_sequence, "memories": []}
        if self.model is None:
            raise ModelNotConfigured("Configure a model before extracting; pending turns are preserved")
        async with asyncio.timeout(self.settings.model_timeout):
            if self.metrics:
                self.metrics.record("extract")
            try:
                result = await self.model.extract(window)
            except Exception:
                if self.metrics:
                    self.metrics.record_error()
                raise
        next_session = window.session.model_copy(
            update={
                "summary": result.summary,
                **result.state_patch.model_dump(exclude_none=True),
                # Size commit metadata too. Each stored turn increments revision,
                # so it bounds the watermark even when commit skips trailing
                # receipt-bound assistant turns. These values are sizing only;
                # commit still receives the original session revision below.
                "processed_sequence": max(window.session.revision, window.new_turns[-1].sequence),
                "revision": window.session.revision + 1,
                "updated_at": window.session.updated_at.replace(microsecond=999999),
            }
        )
        # Reject model-generated state before persisting a state that would make the
        # next extraction permanently impossible.  A long-history session (e.g. a
        # benchmark haystack of ~500 turns) can legitimately produce a summary
        # larger than a short chat window allows, so clamp instead of failing the
        # whole question: truncate the summary first, then drop low-value lists.
        limit = self.settings.window_token_limit // 3
        if estimate_tokens(next_session.model_dump_json()) > limit:
            if not self.settings.state_clamp_enabled:
                raise ValueError(
                    "Extracted session state exceeds budget; model must return a shorter state"
                )
            overflow = estimate_tokens(next_session.model_dump_json()) - limit
            summary = next_session.summary or ""
            if summary:
                keep = max(0, len(summary) - overflow - 64)
                next_session = next_session.model_copy(update={"summary": summary[:keep]})
                overflow = estimate_tokens(next_session.model_dump_json()) - limit
            if overflow > 0:
                next_session = next_session.model_copy(update={"plan": [], "pending_tasks": []})
            if estimate_tokens(next_session.model_dump_json()) > limit:
                next_session = next_session.model_copy(
                    update={"summary": "", "goal": ""}
                )
            if estimate_tokens(next_session.model_dump_json()) > limit:
                # Topic/project identity is caller-owned, not generated state.
                raise ValueError("Session metadata exceeds extraction state budget")
            # The store commits result, not next_session. Persist the bounded
            # state we just checked without mutating the raw model response.
            result = result.model_copy(update={
                "summary": next_session.summary,
                "state_patch": StatePatch(
                    goal=next_session.goal,
                    plan=next_session.plan,
                    pending_tasks=next_session.pending_tasks,
                ),
            })
        memories = await asyncio.to_thread(
            self.store.commit_extraction, scope, window.session, window.new_turns, result,
            allow_direct_long_term=self.settings.promotion_enabled,
        )
        return {
            "status": "processed",
            "processed_sequence": window.new_turns[-1].sequence,
            "summary": result.summary,
            "candidate_count": len(result.candidates),
            "memories": memories,
        }
