"""Coverage is an explicit supported-answer state, not a semantic proof."""

import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, render_table, validate_table


def slot(target, *, coverage="partial", value="Supported part", missing="Other part is unknown."):
    return {"target": target, "derivation": "The quoted fact supports this part only.",
            "coverage": coverage, "missing": missing,
            "values": ([{"value": value, "supports": [{"source_id": 1, "quote": "A fact"}]}]
                       if value else [])}


def source():
    return Hit(content="A fact", score=1, source_file="source", chunk_id="1",
               metadata={"source_events": [{"text": "A fact", "role": "user"}]})


@pytest.mark.parametrize("coverage", ["complete", "partial"])
def test_empty_answer_with_explicit_gap_normalizes_without_repair(coverage):
    result = AnswerTable(slots=[slot("q", coverage=coverage, value=None)])
    validate_table(result, [], ["q"], "text")
    assert result.slots[0].coverage == "unavailable"


def test_valid_abstention_keeps_reason_in_receipt_and_existing_public_text():
    result = AnswerTable(slots=[slot("q", coverage="unavailable", value=None)])
    validate_table(result, [], ["q"], "text")
    assert render_table(result) == "Not mentioned in the conversation"
    assert result.slots[0].missing == "Other part is unknown."


def test_every_partial_target_keeps_its_unmet_condition_at_publication():
    result = AnswerTable(slots=[
        slot("q1", value="One", missing="Condition A is unknown."),
        slot("q2", value="Two", missing="Condition B is unknown."),
    ])
    validate_table(result, [source()], ["q1", "q2"], "text")
    before = result.model_dump()
    rendered = render_table(result)
    assert rendered.startswith("One, Two.")
    assert "Condition A is unknown" in rendered and "Condition B is unknown" in rendered
    assert result.model_dump() == before


def test_repeated_gap_is_published_once_and_context_gap_stays_internal():
    context = slot("context", value="Context", missing="Internal gap.")
    context["role"] = "context"
    result = AnswerTable(slots=[slot("q1"), slot("q2"), context])
    validate_table(result, [source()], ["q1", "q2", "context"], "text")
    rendered = render_table(result)
    assert rendered.count("Other part is unknown") == 1
    assert "Internal gap" not in rendered and "Context" not in rendered


def test_partial_state_does_not_prove_a_numeric_conclusion():
    result = AnswerTable(slots=[slot("q", value="999", missing="Some members are unknown.")])
    # Shape/provenance checks must not be reported as entailment or exhaustive
    # set coverage. The synthesis audit still owns member qualification.
    validate_table(result, [source()], ["q"], "count")
    assert "999" in render_table(result)
