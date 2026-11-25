from test_answer_synthesis import table

from agent_memory.synthesis import render_table


def test_explicit_requested_gap_does_not_require_english_conjunction():
    answer = table("番茄5株")
    slot = answer.slots[0]
    slot.coverage = "partial"
    slot.missing = "辣椒数量未知"
    slot.missing_kind = "requested_part"
    rendered = render_table(answer, question="分别种了多少蔬菜？")
    assert "番茄5株" in rendered
    assert "辣椒数量未知" in rendered


def test_optional_detail_does_not_pollute_supported_answer():
    answer = table("red")
    slot = answer.slots[0]
    slot.coverage = "partial"
    slot.missing = "Blue was not mentioned"
    slot.missing_kind = "optional_detail"
    assert render_table(answer, question="red and blue") == "red"


def test_legacy_table_keeps_default_scope():
    assert table().slots[0].missing_kind == "unspecified"
