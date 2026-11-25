from datetime import datetime,timezone
from agent_memory.models import Turn,Event,Hit,QueryPlan
from agent_memory.grounding import close_evidence_bundles,_source_event

def test_assistant_context_keeps_the_question_it_answers():
    now=datetime(2025,1,1,tzinfo=timezone.utc)
    texts=[('user','I own a small bicycle. Which helmet should I choose?'),('assistant','Try helmets that fit securely.'),('user','Which helmets are easiest to wash?'),('assistant','Some helmets have washable padding.')]
    turns=[Turn(turn_id=str(i),request_id='r',session_id='s',sequence=i,events=[Event(role=role,content=text,occurred_at=now)]) for i,(role,text) in enumerate(texts)]
    hit=Hit(content=texts[1][1],score=1,source_file='s',chunk_id='c',metadata={'source_events':[_source_event(turns[1],0,texts[1][1],0)]})
    output=close_evidence_bundles([hit],turns,session_id='s',blocked_events=set(),budget=5000,query='What helmet should I choose?',plan=QueryPlan())
    assert any(e['turn_id']=='0' for e in output[0].metadata['source_events'])


def test_blocked_parent_does_not_gain_expansion_permission():
    now=datetime(2025,1,1,tzinfo=timezone.utc)
    turns=[Turn(turn_id=str(i),request_id='r',session_id='s',sequence=i,events=[Event(role=role,content=text,occurred_at=now)]) for i,(role,text) in enumerate([('user','My private bicycle details.'),('assistant','Try a helmet that fits securely.')])]
    hit=Hit(content=turns[1].events[0].content,score=1,source_file='s',chunk_id='c',metadata={'source_events':[_source_event(turns[1],0,turns[1].events[0].content,0)]})
    output=close_evidence_bundles([hit],turns,session_id='s',blocked_events={('0',0)},budget=5000,query='What helmet should I choose?',plan=QueryPlan())
    assert all(e['turn_id']!='0' for e in output[0].metadata['source_events'])
