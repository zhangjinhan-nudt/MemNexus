"""Question routing labels must never abort a run from inside persistence.

Regression for the 2026-10-20 crash: ``classify_question`` emits
``kind="recommendation"`` (its own ``recall_policy`` branch lists that label),
but ``QueryPlan.query_type`` is a ``Literal`` that did not include it. The value
travelled through ``model_copy(update=...)`` — which does not validate — and
only raised inside ``SQLiteStore.save_answer`` when the plan was re-validated as
part of ``AnswerContext``. That took down the whole benchmark run instead of one
question.
"""

import pytest
from pydantic import ValidationError

from agent_memory.models import AnswerContext, QueryPlan


def test_router_labels_accepted_by_query_plan():
    """Every label the deterministic router can emit must be constructible."""
    for label in ("fact", "temporal", "relation", "causal", "set", "count",
                  "inference", "recommendation", "adversarial"):
        assert QueryPlan(query_type=label).query_type == label


def test_recommendation_survives_answer_context_roundtrip():
    """The exact path that crashed: plan -> model_dump -> AnswerContext."""
    plan = QueryPlan(query_type="recommendation")
    context = AnswerContext.model_validate({"plan": plan.model_dump()})
    assert context.plan is not None
    assert context.plan.query_type == "recommendation"


@pytest.mark.parametrize("label", ["suggestion", "opinion", "", None, "RECOMMENDATION"])
def test_unknown_router_label_degrades_instead_of_raising(label):
    """An unseen label must not abort the run; it falls back to the generic path."""
    assert QueryPlan(query_type=label).query_type == "fact"


def test_truly_invalid_type_still_rejected():
    """Degradation is for strings only — a non-string remains a contract error."""
    with pytest.raises(ValidationError):
        QueryPlan(query_type=123)
