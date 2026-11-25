"""Set answers need source member coverage, without relaxing provenance fences."""
from test_query_evidence_packing import source, turn

from agent_memory.grounding import _cost, close_evidence_bundles
from agent_memory.models import QueryPlan
from agent_memory.prompt_codec import evidence_catalog


def visible(result):
    return "\n".join(e["text"] for s in evidence_catalog(result)["sources"] for e in s["source_events"])


def test_set_closure_keeps_all_adjacent_members_with_heading_already_visible():
    text = ("Project overview.\nDeliverables:\n"
            "1. Prepare the survey instrument.\n"
            "2. Recruit participants from community groups.\n"
            "3. Publish the anonymized observations.\n\n"
            "Methods: Prepare the survey instrument with survey preparation guidance.\n")
    t = turn("project", text)
    seed = source(t, text[:text.index("1.")])
    budget = _cost(seed) + 900
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(), budget=budget,
        query="What deliverables did we agree for the project?",
        plan=QueryPlan(query_type="set", required_info=["project deliverables"]))
    assert all(item in visible(result) for item in ["1. Prepare", "2. Recruit", "3. Publish"])
    assert sum(_cost(h) for h in result) <= budget
    assert any(event["start"] <= 0 and event["end"] >= len(seed.content)
               for event in result[0].metadata["source_events"])
    for event in result[0].metadata["source_events"]:
        assert text[event["start"]:event["end"]] == event["text"]


def test_over_budget_list_falls_back_to_member_without_evicting_original():
    text = "Saved options:\n" + "".join(f"{i}. Option {i} uses a separate configuration.\n" for i in range(1, 80))
    t = turn("options", text)
    seed = source(t, "Saved options:\n")
    budget = _cost(seed) + 450
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(), budget=budget,
        query="List the saved options.", plan=QueryPlan(query_type="set"))
    assert sum(_cost(h) for h in result) <= budget
    assert "1. Option 1" in visible(result)
    assert "79. Option 79" not in visible(result)
    assert seed.metadata["source_events"][0] in result[0].metadata["source_events"]


def test_ordinal_set_plan_does_not_expand_unrequested_members():
    text = "Options:\n1. Alpha configuration.\n2. Beta configuration.\n3. Gamma configuration.\n"
    t = turn("ordinal", text)
    seed = source(t, "Options:\n")
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events=set(), budget=5000,
        query="What was the 2nd option?", plan=QueryPlan(query_type="set"))
    assert "2. Beta configuration." in visible(result)
    assert "3. Gamma configuration." not in visible(result)


def test_list_expansion_cannot_cross_blocked_source_boundary():
    text = "Requirements:\n1. Private identity.\n2. Private address.\n"
    t = turn("blocked", text)
    seed = source(t, "Requirements:\n")
    result = close_evidence_bundles([seed], [t], session_id="s", blocked_events={(t.turn_id, 0)},
        budget=5000, query="List the requirements.", plan=QueryPlan(query_type="set"))
    assert result == [seed]
