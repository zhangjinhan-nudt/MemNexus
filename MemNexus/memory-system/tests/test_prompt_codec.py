import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from agent_memory.grounding import _source_event, reread_evidence, source_payload
from agent_memory.models import Event, Hit, QueryPlan, Turn
from agent_memory.prompt_codec import compact_schema, evidence_catalog
from agent_memory.providers import CallableModel
from agent_memory.reasoning import classify_question
from agent_memory.synthesis import BoundAnswer, BoundSlot, BoundValue, EvidenceGap, ExecutionResult, Support


def source(n, text, speaker="Inez"):
    turn = Turn(turn_id=f"t{n}", request_id=f"r{n}", session_id="s", sequence=n,
                events=[Event(role="user", speaker=speaker, content=text, occurred_at=datetime(2025,1,1,tzinfo=UTC))])
    return turn, Hit(content=text, score=.8, source_file="s", chunk_id=f"c{n}",
                     metadata={"source_events": [_source_event(turn,0,text,0)]})


def test_catalog_lossless_membership_and_speaker_boundary():
    _, a = source(1, "I own a red kite.")
    _, b = source(2, "I own a red kite.", "Nora")
    c = a.model_copy(deep=True)
    payload = evidence_catalog([a,b,c])
    assert "events" not in payload
    assert payload["sources"][0]["source_events"] == payload["sources"][2]["source_events"]
    assert payload["sources"][1]["source_events"] != payload["sources"][0]["source_events"]
    for h, row in zip([a,b,c], payload['sources'], strict=True):
        assert row["source_events"] == source_payload(h)["source_events"]


def test_schema_removes_annotations_not_properties_or_enum_values():
    schema = {'type':'object','title':'Doc','description':'docs','properties':{'title':{'type':'string','title':'Title','enum':['title','description']},'value':{'type':'number','minimum':2}},'required':['title'],'additionalProperties':False}
    old = deepcopy(schema)
    result = compact_schema(schema)
    assert schema == old
    assert result['properties']['title']['enum'] == ['title','description']
    assert result['properties']['value']['minimum'] == 2
    assert result['required'] == ['title'] and result['additionalProperties'] is False


@pytest.mark.parametrize('q', ['Why does Inez sometimes delay exercise?', 'What does Nora sometimes read?'])
def test_times_substring_is_not_a_count_route(q):
    assert classify_question(q).kind != 'count'


def test_existing_evidence_reread_needs_no_new_event():
    t,h = source(1, "The parcel was delivered on Friday.")
    gap = EvidenceGap(output_id='arrival', code='candidate_unresolved', cue='parcel delivered Friday', action='reread')
    bindings = BoundAnswer(slots=[BoundSlot(output_id='arrival', values=[
        BoundValue(value='Friday', supports=[Support(source_id=1,quote=h.content)])])])
    sources, receipt = reread_evidence([h], turns=[t], session_id='s',blocked_events=set(),budget=0,
                                      query='When was the parcel delivered?',gaps=[gap],bindings=bindings)
    assert sources == [h]
    assert receipt['read_source_ids'] == [1]
    assert receipt['added_context_chars'] == 0


def test_render_set_keeps_every_value_without_rephrasing():
    result = ExecutionResult(values={'names':['Ada','Bela','Chen']},operations=[{'output_id':'names','operator':'set'}])
    assert result.render() == 'Ada, Bela, Chen'


def test_factual_planner_expands_into_existing_contract():
    async def scenario():
        prompts=[]
        def chat(prompt):
            prompts.append(prompt)
            return json.dumps({'route':'both','semantic_queries':['Inez arrival'],'required_info':['arrival date'],
                               'answer_contract':{'outputs':[{'output_id':'arrival','question_span':'When did Inez arrive?','operator':'time_point'}]}})
        result = await CallableModel(chat).plan('When did Inez arrive?',{})
        assert isinstance(result, QueryPlan)
        assert result.response_intent == 'direct_answer'
        assert result.answer_contract.outputs[0].operator == 'time_point'
        assert 'action_template' not in prompts[0]
    asyncio.run(scenario())


def test_catalog_only_sends_selected_ids():
    _,a = source(1,'Old supported fact.')
    _,b = source(2,'Other unrelated fact.')
    payload = evidence_catalog([a,b],source_ids=[1])
    assert [s['source_id'] for s in payload['sources']] == [1]
    assert all(e['text'] != b.content for e in payload['sources'][0]['source_events'])
