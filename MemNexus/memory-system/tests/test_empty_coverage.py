"""Derive empty publication coverage without re-deciding semantic sufficiency."""

import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, render_table, validate_table


def table(*, values=None, missing="The second event is unknown.", coverage="partial", role="answer"):
    return AnswerTable.model_validate({"slots": [{
        "target": "q", "derivation": "The model selected no supported comparison.",
        "values": values or [], "missing": missing, "coverage": coverage, "role": role,
        "missing_kind": "requested_part",
    }]})


def source():
    return Hit(content="A fact", score=1, source_file="source", chunk_id="1",
               metadata={"source_events": [{"text": "A fact", "role": "user"}]})


@pytest.mark.parametrize("coverage", ["complete", "partial", "unavailable"])
def test_empty_answer_explicit_gap_is_idempotent_and_preserves_reason(coverage):
    result = table(coverage=coverage)
    validate_table(result, [], ["q"], "text")
    assert result.slots[0].coverage == "unavailable"
    assert result.slots[0].missing == "The second event is unknown."
    assert result.slots[0].missing_kind == "requested_part"
    first = result.model_dump()
    validate_table(result, [], ["q"], "text")
    assert result.model_dump() == first
    assert render_table(result) == "Not mentioned in the conversation"


@pytest.mark.parametrize("missing", ["", " ", "\n\t"])
@pytest.mark.parametrize("coverage", ["complete", "partial", "unavailable"])
def test_missing_reason_is_still_required(missing, coverage):
    with pytest.raises(ValueError, match="explicit missing-evidence reason"):
        validate_table(table(missing=missing, coverage=coverage), [], ["q"], "text")


def test_blank_value_record_is_not_mistaken_for_no_values():
    result = table(values=[{"value": "", "supports": [{"source_id": 1, "quote": "A fact"}]}])
    with pytest.raises(ValueError, match="direct answer requires a value"):
        validate_table(result, [source()], ["q"], "text")
    assert result.slots[0].coverage == "partial"


def test_existing_partial_positive_answer_is_preserved():
    result = table(values=[{"value": "One", "supports": [{"source_id": 1, "quote": "A fact"}]}])
    validate_table(result, [source()], ["q"], "text")
    assert result.slots[0].coverage == "partial"
    assert result.slots[0].values[0].value == "One"
    assert render_table(result) == "One. The second event is unknown."


def test_unavailable_positive_value_still_fails():
    result = table(coverage="unavailable", values=[{"value": "One", "supports": [
        {"source_id": 1, "quote": "A fact"}]}])
    with pytest.raises(ValueError, match="Coverage representation conflict"):
        validate_table(result, [source()], ["q"], "text")


def test_context_slot_is_not_reclassified():
    result = table(role="context")
    validate_table(result, [], ["q"], "text")
    assert result.slots[0].coverage == "partial"


def test_independent_positive_slot_survives_other_empty_slot():
    result = table()
    positive = table(values=[{"value": "One", "supports": [
        {"source_id": 1, "quote": "A fact"}]}], coverage="complete", missing="").slots[0]
    positive.target = "q2"
    result.slots.append(positive)
    validate_table(result, [source()], ["q", "q2"], "text")
    assert result.slots[0].coverage == "unavailable"
    assert result.slots[1].coverage == "complete"
    assert result.slots[1].values[0].value == "One"
