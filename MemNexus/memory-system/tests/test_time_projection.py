"""Direction and interval precision for returning one of two named events."""
import pytest

from agent_memory.models import Hit
from agent_memory.synthesis import AnswerTable, binding_audit, render_table, validate_table


def project(operation, dates, order=(0, 1), declared=""):
    names = ["copper delivery", "steel delivery"]
    texts = [f"The {name} happened {when}." for name, when in zip(names, dates, strict=True)]
    sources = [
        Hit(content=text, score=1, source_file="fixture", chunk_id=str(i), metadata={
            "source_events": [{"role":"user", "speaker":"Rhea", "text":text, "evidence_grade":"primary",
                               "occurred_at":"2031-06-01T12:00:00+00:00"}]
        }) for i, text in enumerate(texts)
    ]
    bound = [{"identity": names[i], "statement": texts[i], "supports":[{"source_id": i+1, "quote":texts[i]}],
              "time_ref":{"source_id":i+1, "source_event":1, "reference":1}} for i in order]
    table = AnswerTable.model_validate({"slots":[{"target":"deliveries", "derivation":"Compare delivery dates.",
             "values":[{"operation":operation, "inputs":bound, "value":declared, "basis":"inference"}]}]})
    validate_table(table, sources, ["deliveries"], "text")
    rendered = render_table(table)
    validate_table(table, sources, ["deliveries"], "text")
    assert render_table(table) == rendered
    item = binding_audit(table)["computed_temporal"][0]
    assert item["declared"] == declared and item["computed"] == rendered
    return rendered


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("op,expected,contradiction", [
    ("time_earlier", "steel delivery", "copper delivery"), ("time_later", "copper delivery", "steel delivery"),
])
def test_compute_declared_projection_in_either_order(op, expected, contradiction, order):
    assert project(op, ["on 2030-07-12", "in June 2030"], order, contradiction) == expected


@pytest.mark.parametrize("op", ["time_compare","time_earlier","time_later"])
@pytest.mark.parametrize("dates,outcome", [
    (["on 2030-04-05"]*2, "order within that day is unknown"),
    (["on 2030-04-05", "in April 2030"], "order is not determined"),
])
def test_ambiguous_intervals_preserve_both_choices(op, dates, outcome):
    answer = project(op, dates)
    assert outcome in answer and "copper delivery" in answer and "steel delivery" in answer
