"""Repair errors expose relevant source locations without changing acceptance."""
import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, render_table, validate_table


def source(text, role="assistant", context=False):
    return Hit(content=text, score=1, source_file="test", chunk_id="one",
               metadata={"source_events": [{"text": text, "role": role, "context_only": context}]})


def table(value, quote, basis="memory"):
    return AnswerTable.model_validate({"slots": [{"target": "q", "derivation": "Use supplied evidence.",
        "values": [{"value": value, "basis": basis, "supports": [{"source_id": 1, "quote": quote}]}]}]})


def error_for(answer, sources):
    before = answer.model_dump()
    try:
        validate_table(answer, sources, ["q"], "text")
    except ValueError as error:
        return str(error)
    finally:
        assert answer.model_dump() == before
    return None


def test_feedback_lists_original_cited_spellings_without_selecting_replacement():
    text = r"Options @north\_star and @south\_star. Unrelated @elsewhere."
    answer = table("@north_str", r"Options @north\_star and @south\_star.")
    message = error_for(answer, [source(text)])
    assert "source 1: @north_star" in message and "source 1: @south_star" in message
    assert "@elsewhere" not in message
    assert "not automatic replacements" in message


def test_uncited_event_text_cannot_supply_identifier_feedback():
    answer = table("@north_str", "A studio works here.")
    message = error_for(answer, [source("A studio works here. @north_star")])
    assert "none; provide a quote" in message and "source 1: @north_star" not in message


def test_recommendation_feedback_only_points_to_user_assertions():
    answer = table("Use the pass.", "Use a transit pass.", "recommendation")
    sources = [source("Use a transit pass."), source("I own a pass.", "user"),
               source("Do I own a pass?", "user", True), source("You own a pass.")]
    message = error_for(answer, sources)
    assert "eligible user assertions: 2." in message
    assert "does not prove relevance" in message


def test_no_user_assertion_feedback_cannot_create_a_fact():
    answer = table("Use the pass.", "Use a transit pass.", "recommendation")
    message = error_for(answer, [source("Use a transit pass.")])
    assert "No eligible user assertions" in message


@pytest.mark.parametrize("basis", ["memory", "inference"])
def test_accepted_identifier_needs_no_repair(basis):
    text = r"The account is @north\_star."
    assert error_for(table("@north_star", text, basis), [source(text)]) is None


def test_accepted_recommendation_needs_no_repair():
    answer = table("Use your pass.", "I own a pass.", "recommendation")
    assert error_for(answer, [source("I own a pass.", "user")]) is None


@pytest.mark.parametrize("operation,identities,amounts,quotes,provided,computed", [
    ("count", ["start endpoint", "end endpoint"], [None, None],
     ["I started in 2010.", "I finished in 2020."], "10 years", "2 years"),
    ("sum", ["first stage", "second stage"], ["4", "2"],
     ["The first stage took 4 years.", "The second stage took 2 years."], "8 years", "6 years"),
])
def test_arithmetic_feedback_exposes_conflict_without_erasing_the_proposed_answer(
    operation, identities, amounts, quotes, provided, computed,
):
    supports = [{"source_id": index, "quote": quote} for index, quote in enumerate(quotes, 1)]
    answer = AnswerTable.model_validate({"slots": [{"target": "q", "derivation": "Bind the requested facts.",
        "values": [{"operation": operation, "unit": "years", "value": provided, "supports": supports,
                    "inputs": [{"identity": identity, "statement": quote, "amount": amount,
                                "supports": [support]}
                               for identity, quote, amount, support in zip(
                                   identities, quotes, amounts, supports, strict=True)]}]}]})
    message = error_for(answer, [source(quote, "user") for quote in quotes])
    assert f"operation={operation!r} computes {computed!r}" in message
    assert f"provided value is {provided!r}" in message
    assert "Reconsider the operation and qualifying inputs" in message
    assert "do not merely erase" in message and "operation=none" in message
    assert answer.slots[0].values[0].value == provided


def test_arithmetic_feedback_does_not_forbid_legitimate_calendar_day_enumeration():
    quote = "I attended on May 1 and May 3, 2024."
    supports = [{"source_id": 1, "quote": quote}]
    answer = AnswerTable.model_validate({"slots": [{"target": "q", "derivation": "Count the two attended days.",
        "values": [{"operation": "count", "unit": "days", "inputs": [
            {"identity": day, "statement": f"Attended on {day}.", "supports": supports}
            for day in ["2024-05-01", "2024-05-03"]]}]}]})
    validate_table(answer, [source(quote, "user")], ["q"], "duration")
    assert render_table(answer) == "2 days"
