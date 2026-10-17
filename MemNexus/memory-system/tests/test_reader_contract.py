"""Semantic regression fixtures, not benchmark observations or gold rules."""

import pytest

from agent_memory.models import Hit
from agent_memory.models import AnswerContract, AnswerOutput
from agent_memory.synthesis import (AnswerTable, BoundAnswer, BoundSlot, BoundValue, Support,
                                    check_and_execute, reader_profile, render_table, validate_table)
from agent_memory.runtime import _reasoning_query
from agent_memory.reasoning import classify_question


@pytest.mark.parametrize("question, expected", [
    ("Which carrier did I use most often in winter?", "entity_selection"),
    ("Who had the fewest delays?", "entity_selection"),
    ("How many flights did I take?", "single_quantity"),
    ("How long did the repair take?", "single_quantity"),
    ("How many hours did I work before and after the move?", "recall"),
    ("Which books did I read?", "recall"),
    ("Why does Inez sometimes read?", "recall"),
    ("Can you suggest ways to improve my garden?", "recommendation"),
    ("Do you have any ideas on how I can learn music?", "recommendation"),
    ("What did you recommend last month?", "recall"),
    ("Which ideas did I suggest?", "recall"),
])
def test_profile_uses_requested_operation_not_topic(question, expected):
    assert reader_profile(question) == expected


def test_longmemeval_timestamp_is_metadata_not_question_content():
    query = "Which airline did I fly with the most in March and April?\nQuestion date: 2023/04/27 (Thu) 05:28"
    assert _reasoning_query(query) == "Which airline did I fly with the most in March and April?"


def test_advice_question_uses_broad_personal_context_recall():
    route = classify_question("Do you have any ideas on how I can find new inspiration for painting?")
    assert route.kind == "recommendation"
    assert route.recall_policy == "recall_first"


def source(text):
    return Hit(content=text, score=1, source_file="fixture", chunk_id="one",
               metadata={"source_events": [{"text": text, "context_only": False}]})


def table(question, *values):
    return AnswerTable.model_validate({"slots": [{"target": question, "values": [
        {"value": value, "supports": [{"source_id": 1, "quote": quote}]}
        for value, quote in values]}]})


def test_question_word_in_named_heading_remains_usable():
    # Label handling is exercised by the extraction validator; this fixture
    # checks that the final answer still accepts the exact listed title.
    text = "Workshop topics:\n- How to restore wood\n- Painting basics"
    question = "Which workshop topic covers restoration?"
    answer = table(question, ("How to restore wood", "How to restore wood"))
    validate_table(answer, [source(text)], [question], "text")


def test_single_quantity_rejects_duplicate_derivations_without_silently_dropping_values():
    question = "How many hours do I work each week?"
    text = "I work up to 36 hours a week."
    contract = AnswerContract(outputs=[AnswerOutput(output_id="hours", question_span=question,
        operator="direct", subjects=["user"])])
    answer = BoundAnswer(slots=[BoundSlot(output_id="hours", values=[
        BoundValue(value="up to 36 hours per week", subjects=["user"], supports=[Support(source_id=1, quote=text)]),
        BoundValue(value="36 hours per week", subjects=["user"], supports=[Support(source_id=1, quote=text)]),
    ])])
    result = check_and_execute(contract, answer, [source(text)])
    assert result.values["hours"] == ["up to 36 hours per week"]
    assert len(answer.slots[0].values) == 2


def test_different_requested_quantities_are_not_collapsed():
    question = "How many hours did I work before and after the move?"
    text = "I worked 30 hours before and 36 hours after the move."
    answer = table(question, ("30 hours before", text), ("36 hours after", text))
    validate_table(answer, [source(text)], [question], "duration")
    assert render_table(answer) == "30 hours before; 36 hours after"


def test_context_slots_preserve_proof_without_rendering_extra_answers():
    question = "How many hours do I work each week?"
    text = "I work up to 36 hours a week."
    contract = AnswerContract(outputs=[AnswerOutput(output_id="hours", question_span=question,
        operator="direct", subjects=["user"])])
    answer = BoundAnswer(slots=[BoundSlot(output_id="hours", values=[
        BoundValue(value="up to 36 hours per week", subjects=["user"], supports=[Support(source_id=1, quote=text)])
    ])])
    result = check_and_execute(contract, answer, [source(text)])
    assert result.render() == "up to 36 hours per week"
