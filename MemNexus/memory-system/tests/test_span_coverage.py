"""Visibility is a literal original-span property, not semantic entailment."""
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from agent_memory.grounding import _records_already_visible
from agent_memory.models import Event, Hit, Turn


def original(identifier="turn-a", *, text="ABCDEFGHIJ", second="ABCDEFGHIJ"):
    return Turn(turn_id=identifier, session_id="s", request_id=identifier, sequence=1,
                events=[Event(role="user", content=text, occurred_at=datetime(2025, 1, 1, tzinfo=UTC)),
                        Event(role="assistant", content=second, occurred_at=datetime(2025, 1, 1, tzinfo=UTC))])


def record(turn, start, end, index=0):
    return {"turn_id": turn.turn_id, "event_index": index, "start": start, "end": end,
            "text": turn.events[index].content[start:end]}


def hits(*records):
    return [Hit(content="original evidence", score=.8, source_file="s", chunk_id=str(i),
                metadata={"source_events": [r]}) for i, r in enumerate(records)]


def covered(candidate, *visible, turns=None):
    known = turns or [original()]
    return _records_already_visible(candidate, hits(*visible), {t.turn_id: t for t in known})


def test_whole_original_span_covers_subspan():
    t = original()
    assert covered([record(t, 2, 7)], record(t, 0, 10))


def test_subspan_does_not_cover_whole_or_disjoint_part():
    t = original()
    assert not covered([record(t, 0, 10)], record(t, 2, 7))
    assert not covered([record(t, 7, 10)], record(t, 0, 7))


def test_partial_intersection_is_not_full_visibility():
    t = original()
    assert not covered([record(t, 3, 9)], record(t, 0, 6))


def test_complementary_and_overlapping_spans_jointly_cover():
    t = original()
    assert covered([record(t, 0, 10)], record(t, 0, 4), record(t, 4, 10))
    assert covered([record(t, 0, 10)], record(t, 0, 6), record(t, 4, 10))


def test_a_one_character_gap_prevents_coverage():
    t = original()
    assert not covered([record(t, 0, 10)], record(t, 0, 4), record(t, 5, 10))


def test_identical_text_in_another_turn_is_not_the_same_original():
    a, b = original(), original("turn-b")
    assert not covered([record(a, 0, 10)], record(b, 0, 10), turns=[a, b])


def test_identical_text_in_another_event_or_speaker_is_not_coverage():
    t = original()
    assert not covered([record(t, 0, 10, 0)], record(t, 0, 10, 1))


@pytest.mark.parametrize("field", ["start", "end", "turn_id", "event_index"])
def test_unknown_identity_or_offset_is_not_assumed_redundant(field):
    t = original()
    candidate = record(t, 0, 10)
    candidate.pop(field)
    assert not covered([candidate], record(t, 0, 10))
    assert not covered([record(t, 0, 10)], candidate)


@pytest.mark.parametrize("change", [
    {"start": 2, "end": 5},  # right length, wrong literal position
    {"start": 0, "end": 5},  # end/length inconsistency
    {"start": -1, "end": 2},
    {"start": True, "end": 4},
])
def test_wrong_offsets_never_prove_coverage(change):
    t = original()
    malformed = {**record(t, 0, 3), **change}
    assert not covered([malformed], record(t, 0, 10))
    assert not covered([record(t, 0, 3)], malformed)


def test_every_candidate_record_must_be_visible():
    t = original()
    assert not covered([record(t, 0, 3), record(t, 7, 10)], record(t, 0, 3))
    assert covered([record(t, 0, 3), record(t, 7, 10)], record(t, 0, 3), record(t, 7, 10))


def test_unresolved_original_and_empty_record_are_conservative():
    t = original()
    r = record(t, 0, 3)
    assert not _records_already_visible([r], hits(r), {})
    assert not covered([], record(t, 0, 10))
    assert not covered([record(t, 0, 0)], record(t, 0, 10))


def test_visibility_check_is_pure_and_does_not_expand_or_rewrite_evidence():
    t = original()
    candidate, visible = [record(t, 2, 7)], hits(record(t, 0, 10))
    previous = deepcopy((candidate, visible))
    assert _records_already_visible(candidate, visible, {t.turn_id: t})
    assert (candidate, visible) == previous
