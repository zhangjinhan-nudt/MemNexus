"""The benchmark may discard only candidates supported solely by questions."""

from agent_memory.models import Candidate, Event, Evidence, ExtractionResult, ExtractionWindow, Session, Turn
from agent_memory.providers import _drop_invalid_candidates


def test_local_filter_keeps_valid_sibling_and_preserves_exact_quote():
    invalid = Candidate(
        content="Caroline adopted a child",
        subject="Caroline",
        predicate="adopted",
        value="a child",
        evidence=[Evidence(turn_id="q", event_index=0, quote="Did Caroline adopt a child?")],
    )
    valid = Candidate(
        content="Caroline researched adoption agencies",
        subject="Caroline",
        predicate="research",
        value="adoption agencies",
        evidence=[Evidence(turn_id="a", event_index=0, quote="Researching adoption agencies")],
    )
    result = ExtractionResult(candidates=[invalid, valid])
    window = ExtractionWindow(
        session=Session(session_id="s"),
        new_turns=[
            Turn(turn_id="q", session_id="s", sequence=1, request_id="q",
                 events=[Event(role="user", content="Did Caroline adopt a child?")]),
            Turn(turn_id="a", session_id="s", sequence=2, request_id="a",
                 events=[Event(role="user", content="Researching adoption agencies")]),
        ],
        context_turns=[],
    )
    rejected = _drop_invalid_candidates(window, result)
    assert sum(rejected.values()) == 1
    assert result.candidates == [valid]
    assert result.candidates[0].evidence[0].quote == "Researching adoption agencies"
