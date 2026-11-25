"""A partly covered conjunction is published as the answer *plus* the named gap.

LongMemEval's unanswerable items state the expected output: "The information
provided is not enough. You mentioned planting 5 plants for tomatoes but you did
not mention chili peppers." The answer table already records that gap; the
renderer used to drop it, so the declaration never reached the reader and a
partly-correct answer scored as a plain wrong one.

Publishing is conditional because the opposite failure is just as real: a note
about an incidental absence would lengthen an answer that was already complete,
which costs precision on list answers. The gate is measured -- LoCoMo 1 of 1000
rows, LongMemEval 7 of 275 -- and these tests pin its four conditions.
"""

from agent_memory.synthesis import AnswerTable, render_table


def table(value: str | None, gap: str = "", *, role: str = "answer") -> AnswerTable:
    slots = []
    if value is not None:
        slots.append(
            {"derivation": "Fixture audit of the cited evidence.", 
                "target": "target",
                "role": role,
                "values": [{"value": value, "supports": [{"source_id": 1, "quote": value}]}],
                "missing": gap,
            }
        )
    else:
        slots.append({"derivation": "Fixture audit of the cited evidence.", "target": "target", "role": role, "values": [], "missing": gap})
    return AnswerTable.model_validate({"slots": slots})


def test_partial_conjunction_publishes_the_answer_and_the_gap():
    rendered = render_table(
        table("5 tomato plants", "No evidence of chili pepper plants being planted."),
        question="How many plants did I initially plant for tomatoes and chili peppers?",
    )
    assert rendered.startswith("5 tomato plants")
    assert "chili pepper" in rendered


def test_a_complete_answer_is_left_alone():
    rendered = render_table(
        table("5 tomato plants"),
        question="How many plants did I initially plant for tomatoes and chili peppers?",
    )
    assert rendered == "5 tomato plants"


def test_published_gap_respects_sentence_punctuation():
    answer = table("I planted five tomato plants.", "No evidence of chili pepper plants being planted.")
    assert render_table(answer, question="How many tomato and chili pepper plants?") == (
        "I planted five tomato plants. No evidence of chili pepper plants being planted."
    )


def test_recommendation_sentences_keep_their_own_punctuation():
    answer = AnswerTable.model_validate({"slots": [{
        "target": "advice", "derivation": "Advice based on the user's hiking plans.",
        "values": [{"value": "Bring a map.", "basis": "recommendation"},
                   {"value": "Check the weather!", "basis": "recommendation"}],
    }]})
    assert render_table(answer) == "Bring a map. Check the weather!"


def test_a_gap_about_something_the_question_did_not_ask_is_not_published():
    # The question coordinates two objects, but the gap is about a doctor's name
    # the question never mentions -- an incidental absence, not an answer gap.
    rendered = render_table(
        table("4 engineers", "No evidence of Dr. Johnson being seen at the clinic."),
        question="How many engineers did I lead when I started my new role as manager?",
    )
    assert rendered == "4 engineers"


def test_a_single_fact_question_does_not_gain_a_gap():
    rendered = render_table(
        table("blue", "No evidence of the other colours."),
        question="What colour is the car?",
    )
    assert rendered == "blue"


def test_abstention_stays_a_single_statement():
    rendered = render_table(
        table(None, "No evidence of any plants being planted."),
        question="How many plants did I plant for tomatoes and chili peppers?",
    )
    assert rendered == "Not mentioned in the conversation"


def test_a_gap_without_an_absence_assertion_is_not_published():
    # A premise correction is not a gap: the answer is still complete.
    rendered = render_table(
        table("5", "The role title in the question is Manager; the sources state Engineer."),
        question="How many people did I lead as Manager and as Engineer?",
    )
    assert rendered == "5"
