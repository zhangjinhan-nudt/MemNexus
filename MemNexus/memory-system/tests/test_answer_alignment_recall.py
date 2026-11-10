"""Surface alignment, subject-scoped recall, and the abstention gate.

Each mechanism answers a measured failure from the 100-question LoCoMo run:

* an ISO date or a lower-bound phrase scores zero against the common written
  form even though the fact is correct;
* a hypernym query ("martial arts") never ranks the hyponym fact ("kickboxing")
  that shares the question's subject;
* an open-domain question declines to answer although the subject is present.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from agent_memory.grounding import ground_evidence
from agent_memory.models import Event, Evidence, Hit, Memory, QueryPlan, Turn
from agent_memory.reasoning import subject_recall_hits
from agent_memory.runtime import _is_abstention, _subject_present
from agent_memory.synthesis import AnswerTable, normalize_answer_text, render_table
from agent_memory.temporal import derive_times, render_phrase


# --------------------------------------------------------------------------- #
# Surface canonicalisation
# --------------------------------------------------------------------------- #

def test_iso_date_is_written_in_both_forms():
    assert normalize_answer_text("2023-01-29", "When?") == "29 January 2023"


def test_date_normalisation_is_idempotent():
    once = normalize_answer_text("2023-01-29", "When did she write?")
    assert normalize_answer_text(once, "When did she write?") == once


def test_lower_bound_restates_the_counted_noun():
    assert (
        normalize_answer_text("at least 3", "How many dogs does James have?")
        == "three dogs (at least 3)"
    )


def test_lower_bound_without_a_noun_keeps_the_bound():
    assert normalize_answer_text("at least 3", "How many?") == "three (at least 3)"


def test_lower_bound_with_noun_already_present_is_not_duplicated():
    assert (
        normalize_answer_text("at least 3 dogs", "How many dogs does James have?")
        == "three dogs (at least 3)"
    )


@pytest.mark.parametrize(
    "value",
    ["Max and Daisy", "Connecticut", "counseling or mental health", "taekwondo"],
)
def test_plain_answer_is_untouched(value):
    assert normalize_answer_text(value, "What are the names?") == value


def test_render_table_stays_verbatim_for_switch_off_runs():
    table = AnswerTable.model_validate({"slots": [{"target": "q", "values": [
        {"value": "at least 3", "supports": [{"source_id": 1, "quote": "x"}]}]}]})
    assert render_table(table) == "at least 3"
    assert normalize_answer_text("at least 3", "How many dogs does James have?") == "three dogs (at least 3)"


def test_normalisation_collapses_duplicate_forms_before_joining():
    once = normalize_answer_text("2023-01-29", "When?")
    twice = normalize_answer_text(once, "When?")
    assert list(dict.fromkeys([once, twice])) == ["29 January 2023"]


# --------------------------------------------------------------------------- #
# Subject-scoped recall
# --------------------------------------------------------------------------- #

def memory(memory_id, subject, predicate, value, quote="said so"):
    return Memory(
        memory_id=memory_id,
        session_id="s",
        scope_id="s",
        subject=subject,
        predicate=predicate,
        value=value,
        content=f"{subject} {predicate} {value}",
        evidence=[Evidence(turn_id="t1", event_index=0, quote=quote)],
    )


class KeywordEmbedder:
    """Stand-in embedder: only the measured hypernym/hyponym pair is a hit."""

    async def encode_query(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def encode_documents(self, texts):
        return [[1.0, 0.0] if "kickboxing" in text else [0.0, 1.0] for text in texts]


def test_subject_recall_ranks_the_hyponym_fact_first_and_excludes_others():
    memories = [
        memory("m1", "John", "workout activity", "kickboxing"),
        memory("m2", "John", "activity", "taekwondo"),
        memory("m3", "Maria", "activity", "yoga"),
        memory("m4", "John", "workout activity", "kickboxing", quote="duplicate"),
    ]
    hits = asyncio.run(
        subject_recall_hits(
            "What martial arts has John done?", ["John"], memories,
            exclude={"m4"}, embedder=KeywordEmbedder(), limit=8,
        )
    )
    assert [hit.metadata["memory_id"] for hit in hits] == ["m1", "m2"]
    assert hits[0].score > hits[1].score
    assert hits[0].metadata["origin"] == "subject_sweep"


def test_subject_recall_without_entities_returns_nothing():
    assert asyncio.run(
        subject_recall_hits("What happened?", [], [memory("m1", "John", "p", "v")], embedder=None)
    ) == []


def test_subject_recall_falls_back_to_lexical_overlap():
    memories = [
        memory("m1", "John", "workout activity", "kickboxing"),
        memory("m2", "John", "activity", "taekwondo"),
    ]
    hits = asyncio.run(
        subject_recall_hits("John taekwondo activity", ["John"], memories, embedder=None)
    )
    assert hits[0].metadata["memory_id"] == "m2"


# --------------------------------------------------------------------------- #
# Forced inclusion in grounding
# --------------------------------------------------------------------------- #

def _plan():
    return QueryPlan(
        query_type="set",
        recall_policy="recall_first",
        query_entities=["John"],
        required_info=["John martial arts practiced"],
    )


def _turn(text="I'm doing kickboxing and it's giving me so much energy."):
    return Turn(
        turn_id="t1", request_id="r1", session_id="s", sequence=1,
        events=[Event(role="user", speaker="John", content=text,
                      occurred_at=datetime(2023, 1, 1, tzinfo=UTC))],
    )


def _hit(memory_id="m1", subject="John", predicate="workout activity", value="kickboxing"):
    return Hit(
        content=f"{subject} {predicate} {value}", score=0.2, source_file=f"[memory:{memory_id}]",
        chunk_id=f"{memory_id}:1",
        metadata={
            "memory_id": memory_id, "subject": subject, "predicate": predicate,
            "value": value, "origin": "subject_sweep",
            "evidence": [{"turn_id": "t1", "event_index": 0, "quote": "I'm doing kickboxing"}],
        },
    )


def test_must_include_keeps_the_unranked_subject_fact():
    turn = _turn()
    result = ground_evidence(
        "What martial arts has John done?", _plan(), [_hit()],
        session_id="s", current_turns=[turn], resolved_turns={"t1": turn},
        blocked_events=set(), max_sources=4, budget=4000, must_include={"m1"},
    )
    kept = [hit.metadata.get("memory_id") for hit in result.sources]
    assert "m1" in kept
    assert result.trace.subject_recall_added == 1


def test_without_must_include_no_forced_slot_is_counted():
    turn = _turn()
    ranked = _hit(memory_id="m2", predicate="martial arts", value="taekwondo   ")
    result = ground_evidence(
        "What martial arts has John done?", _plan(), [_hit(), ranked],
        session_id="s", current_turns=[turn], resolved_turns={"t1": turn},
        blocked_events=set(), max_sources=1, budget=4000,
    )
    assert result.trace.subject_recall_added == 0
    assert result.trace.subject_recall_candidates == 0


def test_must_include_keeps_the_fact_even_at_one_slot():
    turn = _turn()
    ranked = _hit(memory_id="m2", predicate="martial arts", value="taekwondo   ")
    result = ground_evidence(
        "What martial arts has John done?", _plan(), [_hit(), ranked],
        session_id="s", current_turns=[turn], resolved_turns={"t1": turn},
        blocked_events=set(), max_sources=1, budget=4000, must_include={"m1"},
    )
    assert "m1" in [hit.metadata.get("memory_id") for hit in result.sources]
    assert result.trace.subject_recall_added == 1
    assert result.trace.subject_recall_candidates == 1


# --------------------------------------------------------------------------- #
# Abstention gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Not mentioned in the conversation", True),
        ("Insufficient evidence to determine the answer.", True),
        ("I cannot determine that from the conversation.", True),
        ("Connecticut.", False),
        ("taekwondo", False),
    ],
)
def test_abstention_detection(text, expected):
    assert _is_abstention(text) is expected


def test_subject_presence_reads_claims_and_source_text():
    hit = _hit()
    assert _subject_present(["John"], [hit]) is True
    assert _subject_present(["James"], [hit]) is False
    assert _subject_present([], [hit]) is False


def test_bounded_inference_contract_is_refusal_by_default():
    from agent_memory.models import BoundedInference

    step = BoundedInference(abstain=True, missing="no state named in the evidence")
    assert step.abstain is True
    assert step.confidence == "low"
    assert step.answer == ""


# --------------------------------------------------------------------------- #
# Relative time resolution
# --------------------------------------------------------------------------- #

def _time_source(occurred_at: str, text: str) -> Hit:
    event = Event(role="user", speaker="Jolene", content=text,
                  occurred_at=datetime.fromisoformat(occurred_at))
    turn = Turn(turn_id="t1", request_id="r1", session_id="s", sequence=1, events=[event])
    return Hit(content=text, score=0.9, source_file="source", chunk_id="t1",
               metadata={"source_events": [{"turn_id": "t1", "event_index": 0,
                                            "text": text, "occurred_at": occurred_at,
                                            "context_only": False}]})


def _value_table(value: str, quote: str) -> AnswerTable:
    return AnswerTable.model_validate({"slots": [{"target": "q", "values": [
        {"value": value, "supports": [{"source_id": 1, "quote": quote}]}]}]})


def test_last_summer_resolves_against_the_utterance_not_the_clock():
    sources = [_time_source("2023-02-04T10:00:00+00:00", "I was in Bogota last summer.")]
    assert render_phrase(derive_times("last summer", datetime.fromisoformat("2023-02-04T10:00:00+00:00"))[0]) == "summer 2022"
    assert render_table(_value_table("last summer", "I was in Bogota last summer."), sources) == "summer 2022"


def test_yesterday_resolves_to_its_calendar_day():
    sources = [_time_source("2023-01-26T10:00:00+00:00", "I got the letter yesterday.")]
    assert render_table(_value_table("yesterday", "I got the letter yesterday."), sources) == "25 January 2023"


def test_relative_value_is_kept_when_sources_disagree():
    first = _time_source("2023-02-04T10:00:00+00:00", "I was in Bogota last summer.")
    second = _time_source("2024-02-04T10:00:00+00:00", "Bogota last summer was hot.")
    table = AnswerTable.model_validate({"slots": [{"target": "q", "values": [
        {"value": "last summer", "supports": [{"source_id": 1, "quote": "I was in Bogota last summer."},
                                              {"source_id": 2, "quote": "Bogota last summer was hot."}]}]}]})
    assert render_table(table, [first, second]) == "last summer"


def test_absolute_value_is_not_rewritten():
    sources = [_time_source("2023-02-04T10:00:00+00:00", "I was in Bogota in June 2022.")]
    assert render_table(_value_table("June 2022", "I was in Bogota in June 2022."), sources) == "June 2022"


def test_multi_value_join_uses_commas_and_drops_duplicates():
    table = AnswerTable.model_validate({"slots": [{"target": "q", "values": [
        {"value": "taekwondo", "supports": [{"source_id": 1, "quote": "x"}]},
        {"value": "kickboxing", "supports": [{"source_id": 1, "quote": "x"}]},
        {"value": "taekwondo", "supports": [{"source_id": 1, "quote": "x"}]}]}]})
    assert render_table(table) == "taekwondo, kickboxing"
    assert "; " not in render_table(table)
