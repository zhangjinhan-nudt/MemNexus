"""Speech-act boundaries must not erase affirmative facts or promote questions."""

from datetime import UTC, datetime

import pytest

from agent_memory.evidence_policy import question_only
from agent_memory.grounding import _source_event
from agent_memory.models import Event, Hit, Turn
from agent_memory.synthesis import AnswerTable, validate_table


@pytest.mark.parametrize("text", [
    "When I started my new role in January, I led four engineers.",
    "when i arrived at the lab, i called my colleague.",
    "When Alex arrived, Rachel opened the door.",
    "When the semester ended, the students returned home.",
    "When I'm ready, we'll leave.",
    "Where I live, the shops close early.",
    "Where Rachel works, she leads a small team.",
    "The colleague who called yesterday leads the team.",
    "My route, which crosses the river, takes twenty minutes.",
    "I know who leads the team and which route she takes.",
    "When I arrived, I called Alex. Did Rachel arrive too?",
])
def test_affirmative_main_clause_remains_evidence(text):
    assert question_only(text) is False


@pytest.mark.parametrize("text", [
    "When did I start my new role",
    "Where does Rachel work",
    "Who leads the team",
    "Which engineer leads the team",
    "Whose team won the award",
    "Whom did Rachel call",
    "What happened when I arrived",
    "How many people said that I led the team",
    "When I started, did I lead four engineers",
    "When I arrived, who opened the door",
    "Where I work, which team does Rachel lead",
    "When did I arrive, before or after Rachel",
    "Where in Denver did I park, near the hotel",
    "When you say that I led four engineers, does that establish my job title",
    "Who said that when I arrived, I led the team",
    "Which claim is true, I led the team or Rachel did",
    "When I arrived, I called Alex?",
    "When I arrived, I called Alex？",
    # Ambiguous fragments do not gain assertion status from missing punctuation.
    "When I arrived at the lab",
    "Where Rachel works",
])
def test_real_or_ambiguous_wh_questions_remain_context(text):
    assert question_only(text) is True


@pytest.mark.parametrize("question", [False, True])
def test_source_grade_and_synthesis_share_the_speech_act_boundary(question):
    text = ("When I started, did I lead four engineers" if question
            else "When I started, I led four engineers.")
    turn = Turn(turn_id="t", request_id="r", session_id="s", sequence=1,
                events=[Event(role="user", speaker="Alex", content=text,
                              occurred_at=datetime(2025, 1, 1, tzinfo=UTC))])
    event = _source_event(turn, 0, text, 0)
    assert event["is_question"] is question
    assert event["context_only"] is question
    assert event["evidence_grade"] == ("none" if question else "primary")
    source = Hit(content=text, score=0.9, source_file="source", chunk_id="t",
                 metadata={"source_events": [event]})
    table = AnswerTable.model_validate({"slots": [{
        "target": "How many engineers did Alex lead when starting?", "derivation": "Use the explicit statement.",
        "values": [{"value": "four engineers", "supports": [{"source_id": 1, "quote": text}]}],
    }]})
    if question:
        with pytest.raises(ValueError, match="question alone"):
            validate_table(table, [source], [table.slots[0].target], "count")
    else:
        validate_table(table, [source], [table.slots[0].target], "count")
