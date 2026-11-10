"""Invariants for the SimpleMem paper protocol.

These lock the three mistakes that made the same data produce three different
conclusions on 2026-10-20: including adversarial items in the Average, averaging
per question instead of per reasoning type, and comparing unpaired item sets.
"""

from __future__ import annotations

import statistics

import pytest

from analyze_optimized_comparison import locomo_score
from paper_protocol import (
    REASONING_TYPES,
    adversarial_success,
    bleu1_of,
    common_subset,
    f1_of,
    paper_table,
)


def row(category_id, prediction, gold, system="MemNexus", qa=0, sample="conv-x"):
    return {
        "system": system,
        "sample_id": sample,
        "qa_index": qa,
        "category_id": category_id,
        "category": {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}[category_id],
        "question": f"q{qa}",
        "gold_answer": gold,
        "prediction": prediction,
        "model_calls": {"input_tokens": 100, "output_tokens": 10, "cost_rmb": 0.01},
    }


def test_average_is_unweighted_over_four_types():
    # 1 perfect multi-hop, 3 perfect temporal, 10 bad single-hop, no open-domain.
    rows = [row(1, "Sweden", "Sweden", qa=0)]
    rows += [row(2, "2022", "2022", qa=10 + i) for i in range(3)]
    rows += [row(4, "wrong", "right", qa=20 + i) for i in range(10)]
    table = paper_table(rows, systems=("MemNexus",))
    stats = table["systems"]["MemNexus"]
    # OpenDomain absent -> three observed types, equal weight each.
    assert set(stats["f1_by_type"]) == {"MultiHop", "Temporal", "SingleHop"}
    assert stats["average_f1"] == pytest.approx(statistics.mean([1.0, 1.0, 0.0]))
    # A per-question mean would be 4/14 = 0.286; equal weight is 0.667.
    assert stats["average_f1"] != pytest.approx(4 / 14)


def test_adversarial_is_reported_separately_and_excluded_from_average():
    rows = [
        row(1, "Sweden", "Sweden", qa=0),
        row(2, "2022", "2022", qa=1),
        row(3, "counseling", "counseling", qa=2),
        row(4, "wrong", "right", qa=3),
    ]
    rows += [row(5, "Not mentioned in the conversation", "Not mentioned in the conversation", qa=10 + i) for i in range(4)]
    stats = paper_table(rows, systems=("MemNexus",))["systems"]["MemNexus"]
    assert stats["adversarial_sr"] == 1.0
    assert stats["adversarial_n"] == 4
    # Average covers exactly the four reasoning types: three perfect, single-hop 0.
    assert stats["average_f1"] == pytest.approx(0.75)
    assert set(stats["f1_by_type"]) == set(REASONING_TYPES.values())


@pytest.mark.parametrize("category_id", [1, 2, 3, 4])
def test_f1_matches_official_locomo_scorer(category_id):
    rows = [
        row(category_id, "counseling and mental health", "Psychology, counseling certification", qa=0),
        row(category_id, "Sweden", "Sweden", qa=1),
    ]
    for r in rows:
        assert f1_of(r) == pytest.approx(locomo_score(r))


def test_bleu1_bounds():
    assert bleu1_of(row(2, "May 7 2023", "May 7 2023")) == pytest.approx(1.0)
    assert bleu1_of(row(2, "something else entirely", "May 7 2023")) == pytest.approx(0.0)
    partial = bleu1_of(row(2, "May 7 2023 in the morning", "May 7 2023"))
    assert 0.0 < partial < 1.0


def test_adversarial_success_requires_an_explicit_decline():
    assert adversarial_success(row(5, "Not mentioned in the conversation", "x"))
    assert adversarial_success(row(5, "A. Not mentioned in the conversation", "x"))
    assert adversarial_success(row(5, "I do not have enough information in my memory.", "x"))
    # Answering the distractor is a failure even if hedged.
    assert not adversarial_success(row(5, "B. The blue teapot", "x"))
    assert not adversarial_success(row(5, "Her sister's name is Dana", "x"))


def test_common_subset_keeps_only_questions_both_systems_answered():
    rows = [
        row(1, "a", "a", system="MemNexus", qa=0),
        row(1, "a", "a", system="SimpleMem", qa=0),
        row(1, "b", "b", system="MemNexus", qa=1),
        row(1, "c", "c", system="SimpleMem", qa=2),
    ]
    keep = common_subset(rows)
    assert len(keep) == 2
    assert {r["qa_index"] for r in keep} == {0}
    assert {r["system"] for r in keep} == {"MemNexus", "SimpleMem"}


def test_token_cost_excludes_build_calls():
    rows = [row(1, "a", "a", qa=0), row(2, "b", "b", qa=1)]
    stats = paper_table(rows, systems=("MemNexus",))["systems"]["MemNexus"]
    assert stats["token_cost_per_question"] == pytest.approx(110.0)
