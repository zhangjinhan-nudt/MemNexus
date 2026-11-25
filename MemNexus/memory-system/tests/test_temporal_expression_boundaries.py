"""An explicit offset owns its date operand, not the rest of the sentence."""

import pytest

from agent_memory.temporal import derive_times, resolve_time_references


def test_independent_offsets_in_one_sentence_are_both_preserved():
    text = "A was two days before March 10, 2023 and B was three days after April 4, 2023."
    refs = resolve_time_references(text)
    assert [(ref.expression, ref.start) for ref in refs] == [
        ("two days before March 10, 2023", "2023-03-08"),
        ("three days after April 4, 2023", "2023-04-07"),
    ]
    assert all(text[ref.span_start:ref.span_end] == ref.expression for ref in refs)


def test_offset_does_not_hide_a_later_independent_calendar_date():
    text = "A was two days before March 10, 2023 and the receipt was dated April 4, 2023."
    refs = resolve_time_references(text)
    assert [(ref.expression, ref.start) for ref in refs] == [
        ("two days before March 10, 2023", "2023-03-08"),
        ("April 4, 2023", "2023-04-04"),
    ]


def test_unknown_event_anchor_cannot_borrow_an_unrelated_later_date():
    text = "I moved two days before graduation and the unrelated receipt was dated April 4, 2023."
    refs = resolve_time_references(text)
    assert [(ref.expression, ref.start) for ref in refs] == [("April 4, 2023", "2023-04-04")]
    assert not any(ref.rule == "offset_from_written_date" for ref in refs)


@pytest.mark.parametrize("expression", [
    "one week before 15th of March 2021",
    "one week before the 15th of March 2021",
    "one week prior to March 15, 2021",
    "one week before 2021-03-15",
])
def test_direct_date_operands_remain_resolvable_without_source_clock(expression):
    refs = derive_times(expression, None)
    assert len(refs) == 1
    assert refs[0].start == "2021-03-08"
    assert refs[0].expression == expression


def test_approximate_operand_keeps_approximation_and_does_not_absorb_next_fact():
    text = "About one week before March 15, 2021 and the invoice on April 4, 2023."
    refs = resolve_time_references(text)
    assert refs[0].expression == "About one week before March 15, 2021"
    assert refs[0].start is None and refs[0].precision == "approximate"
    assert refs[1].start == "2023-04-04"
