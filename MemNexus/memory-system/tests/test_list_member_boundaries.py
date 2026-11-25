"""Packing preserves member ownership without inventing pronoun links."""

import pytest
from test_query_evidence_packing import source, turn

from agent_memory.grounding import _chunks, _cost, _list_member_ranges, _local_spans, close_evidence_bundles
from agent_memory.models import QueryPlan
from agent_memory.prompt_codec import evidence_catalog


def assert_exact_partition(text, chunks, limit=700):
    assert "".join(piece for _, piece in chunks) == text
    cursor = 0
    for start, piece in chunks:
        assert start == cursor
        assert text[start:start + len(piece)] == piece
        assert 0 < len(piece) <= limit
        cursor += len(piece)


@pytest.mark.parametrize("marker_a,marker_b", [("1.", "2."), ("1)", "2)"), ("-", "-"), ("•", "•")])
def test_intro_cannot_force_a_fitting_members_description_into_the_next_member(marker_a, marker_b):
    intro = "Background details. " * 24 + "\n\n"
    first = f"{marker_a} Aster makes custom rings. " + "Careful design. " * 11 + "She works with unusual gems.\n"
    second = f"{marker_b} Selene uses recycled gold. She makes minimal bands.\n"
    text = intro + first + second
    chunks = _chunks(text)
    assert_exact_partition(text, chunks)
    owner = [piece for _, piece in chunks if "She works with unusual gems." in piece]
    assert len(owner) == 1 and first in owner[0]
    assert not any(piece.startswith("She works") for _, piece in chunks)
    local = _local_spans(text)
    assert any(piece == first for _, piece in local)
    assert not any("She works with unusual gems." in piece and "Aster" not in piece for _, piece in local)


def test_swapping_member_order_preserves_the_same_ownership():
    first = "1. Selene uses recycled gold. She makes minimal bands.\n"
    second = "2. Aster makes custom rings. She works with unusual gems.\n"
    text = first + second
    assert_exact_partition(text, _chunks(text, 80), 80)
    assert (len(first), second) in _local_spans(text)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_nested_members_stay_with_parent_and_postscript_is_not_its_description(newline):
    parent = "1. Aster design:\n   - Gold finish.\n   - Silver finish.\n\n   Both are handmade.\n"
    sibling = "2. Selene design:\n   - Bronze finish.\n\n"
    postscript = "Independent closing paragraph.\n"
    parent, sibling, postscript = (piece.replace("\n", newline) for piece in (parent, sibling, postscript))
    text = f"Options:{newline}" + parent + sibling + postscript
    ranges = _list_member_ranges(text)
    assert [text[left:right] for left, right in ranges] == [parent, sibling]
    assert all(right <= ranges[i + 1][0] for i, (_, right) in enumerate(ranges[:-1]))
    assert_exact_partition(text, _chunks(text))
    spans = _local_spans(text)
    assert any(piece == parent for _, piece in spans)
    assert not any("Selene" in piece and "Independent closing" in piece for _, piece in spans)


def test_oversized_member_is_bounded_and_its_tail_does_not_join_next_title():
    first = "1. Aster design. " + "Long description. " * 90 + "She prefers unusual gems.\n"
    second = "2. Selene design. She prefers plain bands.\n"
    text = first + second
    chunks = _chunks(text)
    assert_exact_partition(text, chunks)
    assert not any("She prefers unusual gems." in piece and "Selene" in piece for _, piece in chunks)
    assert any(start == len(first) and piece == second for start, piece in chunks)


def test_prose_dates_and_versions_are_not_list_markers():
    text = "2023.05.12: release v1.2\n3.14 is pi.\nVersion 2.0 has shipped.\n"
    assert _list_member_ranges(text) == []
    assert _chunks("One. Two. Three.", 10) == [(0, "One. Two. "), (10, "Three.")]
    assert_exact_partition(text, _chunks(text, 25), 25)
    assert _chunks("") == []
    with pytest.raises(ValueError):
        _chunks(text, 0)


@pytest.mark.parametrize("heading", ["Components:", "### Components", "**Components:**"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_heading_stays_with_first_fitting_list_item_at_chunk_boundary(heading, newline):
    prefix = ("General background. " * 27 + "\n\n").replace("\n", newline)
    item = "1. A brass frame with carefully fitted joints and a polished finish." + newline
    text = prefix + heading + newline * 2 + item + "2. A glass panel." + newline
    limit = len(prefix) + len(heading) + len(newline) * 2 + 10
    chunks = _chunks(text, limit)
    assert_exact_partition(text, chunks, limit)
    matching = [piece for _, piece in chunks if heading in piece]
    assert len(matching) == 1 and item in matching[0]
    assert not matching[0].endswith(heading + newline * 2)


def test_oversized_heading_item_pair_keeps_existing_bound_and_member_identity():
    text = "".join("Introduction. " for _ in range(6)) + "\nParts:\n1. " + "detail " * 80 + "\n2. Final.\n"
    assert_exact_partition(text, _chunks(text, 90), 90)
    ranges = _list_member_ranges(text)
    assert all(text[start:].startswith(("1. ", "2. ")) for start, _ in ranges)


def test_fact_closure_preserves_member_binding_inside_a_short_complete_source():
    text = "Designers:\n1. Aster makes rings. She works with unusual gemstones.\n2. Selene makes plain bands.\n"
    original = turn("designers", text)
    seed = source(original, "Designers:\n")
    budget = _cost(seed) + 550
    result = close_evidence_bundles([seed], [original], session_id="s", blocked_events=set(),
        budget=budget, query="Who works with unusual gemstones?", plan=QueryPlan(query_type="fact"))
    texts = [e["text"] for s in evidence_catalog(result)["sources"] for e in s["source_events"]]
    assert any("Aster" in piece and "unusual gemstones" in piece for piece in texts)
    assert sum(_cost(hit) for hit in result) <= budget
    assert seed.content in " ".join(texts)
    for hit in result:
        for event in hit.metadata["source_events"]:
            assert text[event["start"]:event["end"]] == event["text"]
