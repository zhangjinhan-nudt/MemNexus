"""Semantic regression fixtures, not benchmark observations or gold rules."""


from agent_memory.models import Hit
from agent_memory.reasoning import classify_question
from agent_memory.runtime import _reasoning_query
from agent_memory.synthesis import AnswerTable, render_table, validate_table


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
    return AnswerTable.model_validate({"slots": [{"derivation": "Fixture audit of the cited evidence.", "target": question, "values": [
        {"value": value, "supports": [{"source_id": 1, "quote": quote}]}
        for value, quote in values]}]})


def test_question_word_in_named_heading_remains_usable():
    # Label handling is exercised by the extraction validator; this fixture
    # checks that the final answer still accepts the exact listed title.
    text = "Workshop topics:\n- How to restore wood\n- Painting basics"
    question = "Which workshop topic covers restoration?"
    answer = table(question, ("How to restore wood", "How to restore wood"))
    validate_table(answer, [source(text)], [question], "text")


def test_different_requested_quantities_are_not_collapsed():
    question = "How many hours did I work before and after the move?"
    text = "I worked 30 hours before and 36 hours after the move."
    answer = table(question, ("30 hours before", text), ("36 hours after", text))
    validate_table(answer, [source(text)], [question], "duration")
    assert render_table(answer) == "30 hours before, 36 hours after"
