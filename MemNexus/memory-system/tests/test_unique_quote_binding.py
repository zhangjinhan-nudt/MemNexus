from copy import deepcopy

import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, BoundInput, Support, validate_table


def hit(text, speaker="Alice", clock="2026-01-01T00:00:00+00:00", grade="primary"):
    return Hit(content=text, score=1, source_file="test", chunk_id=text,
               metadata={"source_events": [{"text": text, "speaker": speaker, "role": "user",
                                             "occurred_at": clock, "context_only": grade == "none",
                                             "evidence_grade": grade}]})


def table(source_id, quote):
    return AnswerTable.model_validate({"slots": [{"target": "Q", "derivation": "Use the stated fact.",
                                                  "values": [{"value": "blue", "supports": [
                                                      {"source_id": source_id, "quote": quote}]}]}]})


def test_unique_literal_wrong_id_is_rebound_without_changing_words_or_sources():
    sources = [hit("An unrelated statement."), hit("My favorite color is blue.")]
    before = deepcopy(sources)
    value = table(1, "My favorite color is blue.")
    validate_table(value, sources, ["Q"], "text")
    assert value.slots[0].values[0].supports[0].source_id == 2
    assert value.slots[0].values[0].supports[0].quote == "My favorite color is blue."
    assert sources == before


def test_unknown_id_remains_rejected_even_when_quote_is_unique():
    value = table(99, "My favorite color is blue.")
    with pytest.raises(ValueError, match="Unknown source"):
        validate_table(value, [hit("My favorite color is blue.")], ["Q"], "text")
    assert value.slots[0].values[0].supports[0].source_id == 99


def test_correct_id_kept_even_when_another_source_is_literal_and_current_is_formatted():
    value = table(1, "My favorite color is blue.")
    sources = [hit("MY favorite color is **blue**."), hit("My favorite color is blue.", speaker="Bob")]
    validate_table(value, sources, ["Q"], "text")
    assert value.slots[0].values[0].supports[0].source_id == 1


@pytest.mark.parametrize("other", [
    hit("My favorite color is blue.", speaker="Bob"),
    hit("My favorite color is blue.", clock="2026-02-01T00:00:00+00:00"),
    hit("My favorite color is blue."),
])
def test_multiple_sources_never_rebound_even_if_duplicate_event(other):
    value = table(1, "My favorite color is blue.")
    sources = [hit("An unrelated statement."), hit("My favorite color is blue."), other]
    with pytest.raises(ValueError, match="Source 1 does not contain exact quote"):
        validate_table(value, sources, ["Q"], "text")
    assert value.slots[0].values[0].supports[0].source_id == 1


@pytest.mark.parametrize("quote", ["My favorite color was blue.", "My favorite color is blue",
                                   "My favorite color is green."])
def test_paraphrase_or_nonliteral_only_alternative_not_rebound(quote):
    # Existing quote normalization can accept case/emphasis for an already valid
    # source, but cannot be used to search a replacement source.
    value = table(1, quote)
    sources = [hit("An unrelated statement."), hit("MY favorite color is **blue**.")]
    with pytest.raises(ValueError, match="Source 1 does not contain exact quote"):
        validate_table(value, sources, ["Q"], "text")


@pytest.mark.parametrize("shared_object", [False, True])
def test_bound_input_and_value_aliases_both_rebound_and_idempotent(shared_object):
    value = table(1, "I own one blue bike.")
    item = value.slots[0].values[0]
    shared = Support(source_id=1, quote="I own one blue bike.")
    item.supports = [shared]
    item.operation = "count"
    item.unit = "bikes"
    item.value = ""
    bound_support = shared if shared_object else shared.model_copy(deep=True)
    item.inputs = [BoundInput(identity="blue bike", statement="One owned bike", supports=[bound_support])]
    sources = [hit("An unrelated statement."), hit("I own one blue bike.")]
    validate_table(value, sources, ["Q"], "count")
    assert item.supports[0].source_id == item.inputs[0].supports[0].source_id == 2
    assert item.value == "1 bikes"
    once = value.model_dump()
    validate_table(value, sources, ["Q"], "count")
    assert value.model_dump() == once


def test_correcting_address_does_not_turn_question_into_fact():
    value = table(1, "Is the bike blue?")
    sources = [hit("An unrelated statement."), hit("Is the bike blue?", grade="none")]
    with pytest.raises(ValueError, match="question alone cannot establish a fact"):
        validate_table(value, sources, ["Q"], "text")
    assert value.slots[0].values[0].supports[0].source_id == 2


def test_input_only_reference_rebound_before_validator_collects_supports():
    value = table(1, "I own one blue bike.")
    item = value.slots[0].values[0]
    item.supports = []
    item.operation = "count"
    item.value = ""
    item.inputs = [BoundInput(identity="blue bike", statement="One owned bike", supports=[
        Support(source_id=1, quote="I own one blue bike.")])]
    sources = [hit("An unrelated statement."), hit("I own one blue bike.")]
    validate_table(value, sources, ["Q"], "count")
    assert item.inputs[0].supports[0].source_id == item.supports[0].source_id == 2
    assert item.value == "1"


def test_missing_premise_contract_still_rejects_after_unique_rebind():
    value = table(1, "My favorite color is blue.")
    slot = value.slots[0]
    slot.missing = "The required person's identity is unknown."
    slot.missing_kind = "necessary_premise"
    sources = [hit("An unrelated statement."), hit("My favorite color is blue.")]
    with pytest.raises(ValueError, match="necessary missing premise"):
        validate_table(value, sources, ["Q"], "text")
