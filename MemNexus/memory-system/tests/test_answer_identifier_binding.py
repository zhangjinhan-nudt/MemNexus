"""Exact identifiers bind to cited source spelling; no fuzzy publisher repair."""

import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, render_table, validate_table


def make_source(text):
    return Hit(content=text, score=1, source_file="fixture", chunk_id="one",
               metadata={"source_events": [{"text": text, "role": "assistant",
                                              "context_only": True}]})


def make_table(value, quote, basis="memory"):
    return AnswerTable.model_validate({"slots": [{
        "target": "Which account?", "derivation": "Select the account from the cited statement.",
        "values": [{"value": value, "basis": basis,
                    "supports": [{"source_id": 1, "quote": quote}]}],
    }]})


@pytest.mark.parametrize("basis", ["memory", "inference"])
def test_typo_rejected_without_rewriting_answer(basis):
    text = r"The official account is @river\_stone\_studio."
    table = make_table("@river_ston_studio", text, basis)
    saved = table.model_dump()
    with pytest.raises(ValueError, match="identifiers"):
        validate_table(table, [make_source(text)], ["Which account?"], "text")
    assert table.model_dump() == saved


@pytest.mark.parametrize("source, value", [
    (r"The account is @river\_stone\_studio.", "@river_stone_studio"),
    ("The account is `river_stone_studio`.", "@river_stone_studio"),
    ("The account is **@river.stone**.", "@river.stone"),
])
def test_exact_handle_and_markdown_escaping_accepted(source, value):
    table = make_table(value, source)
    validate_table(table, [make_source(source)], ["Which account?"], "text")
    assert render_table(table) == value


@pytest.mark.parametrize("value", ["@river", "@river_stone", "@river_stone_studi", "@River_stone_studio"])
def test_identifier_prefix_and_case_changes_are_not_fuzzy_matches(value):
    text = "The account is @river_stone_studio."
    with pytest.raises(ValueError, match="identifiers"):
        validate_table(make_table(value, text), [make_source(text)], ["Which account?"], "text")


@pytest.mark.parametrize("source, quote", [
    ("The account is @riverstone.", "The account is @river_stone."),
    ("The account is @river_stone.", "The account is @riverstone."),
])
def test_quote_format_normalization_cannot_remove_identifier_characters(source, quote):
    with pytest.raises(ValueError, match="identifiers"):
        validate_table(make_table("@river_stone", quote), [make_source(source)], ["Which account?"], "text")


def test_uncited_handle_elsewhere_in_event_cannot_license_value():
    source = "The account is @river_stone. The designer uses silver."
    with pytest.raises(ValueError, match="identifiers"):
        validate_table(make_table("@river_stone", "The designer uses silver."),
                       [make_source(source)], ["Which account?"], "text")


def test_inference_without_identifier_remains_semantically_unmodified():
    text = "We started in 2010 and completed in 2020."
    table = make_table("ten years", text, "inference")
    saved = table.model_dump()
    validate_table(table, [make_source(text)], ["Which account?"], "duration")
    assert table.model_dump() == saved
