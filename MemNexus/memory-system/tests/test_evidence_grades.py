"""Evidence provenance: what a source event can support, and what it cannot.

These tests pin the distinction the older single ``context_only`` flag erased:
whether a turn asserts anything at all (a question does not), and whether the
assertion is first- or second-hand (an assistant turn is second-hand). They are
mechanism checks, not benchmark accuracy numbers.
"""

from datetime import UTC, datetime

import pytest

from agent_memory.grounding import _source_event, assistant_focus, collect_raw_records
from agent_memory.models import Event, Hit, QueryPlan, Turn
from agent_memory.reasoning import (
    _anchored_dates,
    _dedup_dates,
    _difference_allowed,
)
from agent_memory.synthesis import _assertion_grade, validate_table


def turn(identifier, content, role="user"):
    return Turn(
        turn_id=identifier,
        session_id="s",
        sequence=1,
        request_id=identifier,
        events=[
            Event(
                content=content,
                role=role,
                speaker="Nora" if role == "user" else "Assistant",
                occurred_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
        ],
    )


def hit(t):
    return Hit(
        content=t.events[0].content,
        score=0.9,
        source_file="source",
        chunk_id=t.turn_id,
        metadata={"source_events": [_source_event(t, 0, t.events[0].content, 0)]},
    )


def table(value="red", quote="red", target="colors"):
    from agent_memory.synthesis import AnswerTable

    return AnswerTable.model_validate(
        {"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": target, "values": [
            {"value": value, "supports": [{"source_id": 1, "quote": quote}]}]}]}
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What did the assistant recommend for the trip?", True),
        ("What did you say about my guitar?", True),
        ("Do you remember my sister's name?", False),
        ("When did I buy the amp?", False),
        ("你之前建议我买什么？", True),
    ],
)
def test_assistant_focus_detects_questions_about_the_assistant(query, expected):
    assert assistant_focus(query) is expected


def test_source_event_separates_assertion_from_speaker():
    user_statement = _source_event(turn("t1", "I bought a red car."), 0, "I bought a red car.", 0)
    assistant_statement = _source_event(
        turn("t2", "You bought a red car.", role="assistant"), 0, "You bought a red car.", 0
    )
    question = _source_event(turn("t3", "Did you buy a red car?"), 0, "Did you buy a red car?", 0)

    assert (user_statement["is_question"], user_statement["evidence_grade"]) == (False, "primary")
    assert (assistant_statement["is_question"], assistant_statement["evidence_grade"]) == (
        False,
        "secondary",
    )
    assert (question["is_question"], question["evidence_grade"]) == (True, "none")
    # Legacy meaning is preserved: an assistant turn is not a user fact.
    assert assistant_statement["context_only"] is True


def test_assistant_is_primary_when_the_question_asks_about_it():
    event = _source_event(
        turn("t1", "I suggested the coast road.", role="assistant"),
        0,
        "I suggested the coast road.",
        0,
        assistant_is_primary=True,
    )
    assert event["evidence_grade"] == "primary"


def test_secondary_assertion_supports_an_answer_but_a_question_does_not():
    # An assistant turn states something the conversation contains: it may
    # support an answer rather than voiding it.
    validate_table(
        table("red", "You prefer red."),
        [hit(turn("t1", "You prefer red.", role="assistant"))],
        ["colors"],
        "text",
    )
    # A question points at a topic without asserting anything.
    with pytest.raises(ValueError, match="alone"):
        validate_table(table("red", "Is it red?"), [hit(turn("t2", "Is it red?"))], ["colors"], "text")


def test_legacy_records_map_from_context_only():
    assert _assertion_grade({"context_only": True, "role": "assistant"}) == "secondary"
    assert _assertion_grade({"context_only": True, "role": "user"}) == "none"
    assert _assertion_grade({"context_only": False, "role": "user"}) == "primary"


def test_derived_tasks_keep_assistant_recaps_as_secondary_evidence(monkeypatch):
    monkeypatch.setenv("MEMORY_ALLOW_ASSISTANT_EVIDENCE", "true")
    turns = [
        turn("t1", "I watched Iron Man and Thor.", role="user"),
        turn("t2", "You watched four Marvel movies.", role="assistant"),
    ]
    count_plan = QueryPlan(query_type="count")
    kept = collect_raw_records(
        "How many Marvel movies did I watch?", count_plan, turns, "s", set()
    )
    assert {role for role, _ in _roles(kept)} == {"user", "assistant"}
    assert next(r for r, _ in kept if r["role"] == "assistant")["evidence_grade"] == "secondary"

    assistant_plan = QueryPlan(query_type="count")
    kept_focus = collect_raw_records(
        "How many movies did you say I watched?", assistant_plan, turns, "s", set()
    )
    assert "assistant" in {role for role, _ in _roles(kept_focus)}
    assert next(r for r, _ in kept_focus if r["role"] == "assistant")["evidence_grade"] == "primary"


def _roles(records):
    return [(record["role"], text) for record, text in records]


def test_interval_start_is_not_an_event_date():
    dates = _anchored_dates("I have had the amp for two weeks.", "2023-05-27T10:00:00+00:00")
    kinds = {kind for _stamp, _expr, _precision, kind in dates}
    assert "interval_start@2023-05-27" in kinds
    assert "point" not in kinds


def test_week_precision_is_orderable_but_not_subtractable():
    # "two weeks ago" snaps to a week bucket; its edge is not the stated
    # boundary, so it must never enter difference arithmetic.
    dates = _anchored_dates("I bought the amp two weeks ago.", "2023-05-27T10:00:00+00:00")
    week_kinds = [kind for _stamp, _expr, _precision, kind in dates if kind.startswith("interval")]
    assert week_kinds == ["interval_week@2023-05-27"]
    assert not any(_difference_allowed(kind, "interval_start@2023-05-27") for kind in week_kinds)


def test_explicit_dates_are_points():
    dates = _anchored_dates("It happened on 2023-05-13.", "2023-05-27T10:00:00+00:00")
    assert any(kind == "point" for _stamp, _expr, _precision, kind in dates)


def test_only_commensurable_dates_may_be_subtracted():
    assert _difference_allowed("point", "point")
    assert _difference_allowed("interval_start@2023-05-27", "interval_start@2023-05-27")
    assert not _difference_allowed("point", "interval_start@2023-05-27")
    assert not _difference_allowed(
        "interval_start@2023-05-27", "interval_start@2023-04-10"
    )
    assert not _difference_allowed("utterance", "utterance")


def test_two_durations_with_one_anchor_yield_the_elapsed_difference():
    dates = []
    for text in (
        "I have been taking guitar lessons for six weeks.",
        "I have had the amp for two weeks.",
    ):
        for stamp, expression, precision, kind in _anchored_dates(
            text, "2023-05-27T10:00:00+00:00"
        ):
            if precision == "day" and kind != "utterance":
                dates.append((stamp, expression, precision, kind))

    pairs = [
        (left, right)
        for i, left in enumerate(dates)
        for right in dates[i + 1 :]
        if left[0] != right[0] and _difference_allowed(left[3], right[3])
    ]
    assert pairs, "the two sibling spans share an anchor and must be comparable"
    (start, _e1, _p1, _k1), (end, _e2, _p2, _k2) = pairs[0]
    assert (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days == 28


def test_duplicate_stamps_with_different_kinds_are_kept():
    rows = [
        ("2023-05-27", "utterance date", "day", "utterance"),
        ("2023-05-27", "explicit date", "day", "point"),
    ]
    assert len(_dedup_dates(rows)) == 2


def _two_spans_same_anchor():
    first = turn("t1", "I have been taking lessons for six weeks.")
    second = turn("t2", "I have had the amp for two weeks.")
    source = hit(first)
    source.metadata["source_events"].append(
        _source_event(second, 0, second.events[0].content, 0)
    )
    return source


def test_unique_unbound_span_does_not_replace_a_point_offset():
    question = "How long had I been taking lessons?"
    result = table("two weeks ago", "for six weeks", target=question)
    original = result.model_dump()
    validate_table(result, [_two_spans_same_anchor()], [question], "duration")
    assert result.model_dump() == original


def test_how_long_ago_accepts_ago():
    # The question itself asks for a point offset, so "10 years ago" is the
    # right form and must not be re-typed.
    source = hit(turn("t1", "That was 10 years ago."))
    question = "How long ago was the birthday?"
    result = table("10 years ago", "10 years ago", target=question)
    validate_table(result, [source], [question], "duration")
    assert result.slots[0].values[0].value == "10 years ago"


def test_length_with_before_modifier_is_not_a_point_offset():
    source = hit(turn("t1", "I worked 30 hours before the move."))
    question = "How many hours did I work before the move?"
    result = table("30 hours before", "30 hours before", target=question)
    validate_table(result, [source], [question], "duration")
    assert result.slots[0].values[0].value == "30 hours before"


def test_unbound_supplied_derivations_cannot_repair_an_answer(monkeypatch):
    """Unbound pairwise arithmetic cannot become answer evidence."""
    from agent_memory import reasoning as reasoning_module

    def forbid(*_args, **_kwargs):
        raise AssertionError("validate_table must not rebuild the event graph")

    monkeypatch.setattr(reasoning_module.EventGraph, "from_hits", forbid)
    question = "How long had I been taking lessons?"
    result = table("two weeks ago", "for six weeks", target=question)
    supplied = [
        {
            "kind": "date_differences",
            "differences": [
                "2024-11-20 (for six weeks) -> 2024-12-18 (for two weeks): 28 days = 4 weeks"
            ],
        }
    ]
    original = result.model_dump()
    validate_table(result, [_two_spans_same_anchor()], [question], "duration", derivations=supplied)
    assert result.model_dump() == original
