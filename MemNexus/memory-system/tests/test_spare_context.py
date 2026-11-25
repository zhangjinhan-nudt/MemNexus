"""Spare-capacity recovery never evicts selected evidence."""
from test_reading_units import make_hit, make_turn, visible
from agent_memory.grounding import ground_evidence
from agent_memory.models import QueryPlan, AnswerContract, AnswerOutput


def read(budget):
    query = 'Which helmet did you recommend?'
    turns = [make_turn('parent', 'I own a folding bicycle. Which helmet should I choose?',
                       role='user', sequence=1),
             make_turn('reply', 'Try the Aero model for it.', sequence=2),
             make_turn('other', 'The helmet strap should fit securely.', sequence=20)]
    hits = [make_hit(t, t.events[0].content, rank=i+1) for i,t in enumerate(turns[1:])]
    plan = QueryPlan(route='none', answer_contract=AnswerContract(outputs=[
        AnswerOutput(output_id='helmet', question_span=query)]))
    return ground_evidence(query, plan, hits, session_id='s', current_turns=[],
        resolved_turns={t.turn_id:t for t in turns}, blocked_events=set(), budget=budget)


def test_final_catalog_restores_context_without_evicting_other_evidence():
    result = read(5000)
    assert 'folding bicycle' in visible(result.sources)
    assert 'strap should fit securely' in visible(result.sources)
    assert any(d['reason']=='context_restored_with_spare_budget'
               for d in result.trace.selection_decisions)


def test_tight_budget_preserves_selected_sources_without_overflow():
    full = read(5000)
    compact_cost = sum(d['cost'] for d in full.trace.selection_decisions
                       if (d['candidate_id']=='reply' and d['variant']=='compact')
                       or (d['candidate_id']=='other' and d['selected']))
    result = read(compact_cost)
    assert {h.chunk_id for h in result.sources} == {'reply','other'}
    assert result.trace.context_serialized_chars <= compact_cost
