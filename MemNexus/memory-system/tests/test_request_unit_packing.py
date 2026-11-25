"""Request-scoped novelty is a packing hint, never evidence sufficiency."""

from copy import deepcopy

from test_collection_source_limit import candidate

from agent_memory.grounding import _request_relation_units, _request_units, _source_event, ground_evidence
from agent_memory.models import AnswerContract, AnswerOutput, Evidence, FactMember, QueryPlan
from agent_memory.reasoning import indexed_memory_hits
from agent_memory.retrieval import Retriever, _Match


def plan(unit="value"):
    question = "Which activities did Alex attend?"
    return QueryPlan(query_type="set", recall_policy="recall_first", query_entities=["Alex"],
                     answer_contract=AnswerContract(outputs=[AnswerOutput(
                         output_id="activities", question_span=question, operator="set",
                         subjects=["Alex"], relation="activities", count_unit=unit)]))


def source(index=1, value="race", identity=None):
    memory, turn = candidate(index)
    memory.value = value
    memory.semantics.event_identity = identity
    hit = Retriever._hit(_Match(memory=memory, score=.8), set())
    hit.metadata["source_events"] = [_source_event(turn, 0, turn.events[0].content, 0)]
    return hit, turn


def test_distinct_members_of_same_relation_gain_coverage_not_duplicate_mentions():
    first, _ = source(1, "cycling")
    duplicate, _ = source(2, "cycling")
    other, _ = source(3, "running")
    request = plan()
    assert _request_units(first, request) == _request_units(duplicate, request)
    assert not _request_units(first, request) & _request_units(other, request)


def test_no_reward_for_unrelated_subject_or_relation_but_no_filter():
    hit, turn = source()
    hit.metadata["predicate"] = "favorite_food"
    request = plan()
    assert not _request_units(hit, request)
    result = ground_evidence("Which activities did Alex attend?", request, [hit],
                             session_id="session", current_turns=[], resolved_turns={turn.turn_id: turn},
                             blocked_events=set(), max_sources=2, budget=10000)
    assert result.sources and result.trace.binding_units_selected == 0
    hit.metadata["predicate"] = "activities"
    hit.metadata["subject"] = "Blair"
    assert not _request_units(hit, request)


def test_compiled_relation_alias_contributes_without_rebuilding_exact_collection():
    hit, _ = source()
    request = plan()
    output = request.answer_contract.outputs[0]
    output.relation = "participated in activities"
    output.qualifier = "before a later event"
    assert request.collection_query is None and not _request_units(hit, request)
    output.retrieval_relations = [("Alex", "activities")]
    assert len(_request_units(hit, request)) == 1
    output.subjects = ["Blair"]
    assert not _request_units(hit, request)  # A catalog alias cannot substitute an actor.


def test_indexed_recall_keeps_relevant_third_party_fact_with_object_subject():
    memory, turn = candidate(1)
    memory.subject = "community_project"
    memory.content = "Alex volunteered at the charity event."
    memory.evidence[0].quote = memory.content
    turn.events[0].content = memory.content
    hits = indexed_memory_hits("Which charity events did Alex volunteer at?",
                               QueryPlan(query_entities=["Alex"]), [memory], [], limit=4)
    assert [hit.metadata["memory_id"] for hit in hits] == [memory.memory_id]


def test_source_clock_event_address_and_category_are_not_event_identity():
    hit, _ = source()
    assert hit.metadata["semantics"]["event_key"]
    assert not _request_units(hit, plan("event"))
    hit.metadata["semantics"]["event_identity"] = "annual-race-2024"
    clone = hit.model_copy(deep=True)
    clone.metadata["source_events"][0]["occurred_at"] = "2025-01-01"
    assert _request_units(hit, plan("event")) == _request_units(clone, plan("event"))


def test_only_visible_member_spans_contribute_and_unknown_is_not_zero():
    hit, turn = source()
    hit.metadata["semantics"]["cardinality"] = "many"
    hit.metadata["semantics"]["members"] = [FactMember(value="race", identity="race-one", evidence=[
        Evidence(turn_id=turn.turn_id, event_index=0, quote="event 1")]).model_dump(mode="json")]
    assert len(_request_units(hit, plan("entity"))) == 1
    hit.metadata["source_events"][0]["text"] = "Alex"
    assert not _request_units(hit, plan("entity"))
    assert not _request_units(hit, QueryPlan())


def test_conflicting_state_remains_a_distinct_hint_not_silently_deduplicated():
    hit, _ = source()
    withdrawn = hit.model_copy(deep=True)
    withdrawn.metadata["semantics"]["operation"] = "remove"
    # Scalar hints have no member operation; use a sourced collection member.
    for item in (hit, withdrawn):
        item.metadata["semantics"]["members"] = [{"value": "race", "identity": "race", "evidence":
                                                  deepcopy(item.metadata["evidence"])}]
    assert not _request_units(hit, plan()) & _request_units(withdrawn, plan())


def test_value_units_do_not_collapse_distinct_properties_of_same_entity():
    hit, _ = source()
    hit.metadata["semantics"]["members"] = [
        {"value": value, "identity": "same-event", "evidence": deepcopy(hit.metadata["evidence"])}
        for value in ("running", "cycling")
    ]
    assert len(_request_units(hit, plan("value"))) == 2
    assert len(_request_units(hit, plan("entity"))) == 1


def test_seed_reservation_recomputes_member_gain_and_obeys_max_sources():
    pairs = [source(index, value) for index, value in ((1, "run"), (2, "run"), (3, "bike"))]
    # Equal relevance and query-term overlap; only requested member novelty
    # distinguishes the third fact from the repeated account of the first.
    for hit, _ in pairs:
        hit.content = "Alex attended an activity."
    request = plan()
    before = [hit.model_dump(mode="json") for hit, _ in pairs]
    result = ground_evidence("Which activities did Alex attend?", request, [hit for hit, _ in pairs],
                             session_id="session", current_turns=[],
                             resolved_turns={turn.turn_id: turn for _, turn in pairs}, blocked_events=set(),
                             max_sources=2, budget=100000)
    assert [hit.metadata["value"] for hit in result.sources] == ["run", "bike"]
    assert result.trace.binding_units_selected == 2
    assert result.trace.binding_units_available == 2
    assert [hit.model_dump(mode="json") for hit, _ in pairs] == before


def test_catalog_fields_survive_lexical_distractors_and_reporter_subject():
    question = "Put Lena and Omar's milestones in chronological order."
    request = QueryPlan(query_entities=["Lena", "Omar"], answer_contract=AnswerContract(outputs=[
        AnswerOutput(output_id="order", question_span=question, operator="comparison",
                     subjects=["Lena", "Omar"], subject_mode="joint", relation="milestones",
                     retrieval_relations=[("user", "attended Lena's ceremony"), ("Omar", "degree awarded")])]))
    memories=[];turns={}
    for i,(subject,predicate) in enumerate([
        ("Lena", "chronological order milestones discussion"),
        ("user", "attended Lena's ceremony"), ("Omar", "degree awarded")],1):
        memory,turn=candidate(i)
        memory.subject=subject;memory.predicate=predicate
        memory.content=f"{subject} {predicate}."
        memory.value=f"event {i}"
        memory.evidence[0].quote=memory.content
        turn.events[0].content=memory.content
        memories.append(memory);turns[turn.turn_id]=turn
    hits=indexed_memory_hits(question,request,memories,[],limit=2)
    assert {h.metadata['memory_id'] for h in hits}=={m.memory_id for m in memories[1:]}
    result=ground_evidence(question,request,hits,
        session_id="session",current_turns=[],resolved_turns=turns,blocked_events=set(),
        max_sources=2,budget=10000)
    assert {h.metadata['predicate'] for h in result.sources}=={"attended Lena's ceremony","degree awarded"}
    assert all(h.metadata['source_support_status']=="resolved" for h in result.sources)


def test_relation_coverage_packs_scalar_fields_without_claiming_member_completeness():
    question="How many specimens are there in total?"
    request=QueryPlan(query_type="count",recall_policy="recall_first",answer_contract=AnswerContract(outputs=[
        AnswerOutput(output_id="total",question_span=question,operator="count",subjects=["Alex"],
                     relation="specimens",count_unit="entity",
                     retrieval_relations=[("Alex","mineral inventory"),("Alex","shell inventory")])]))
    pairs=[source(i) for i in range(1,4)]
    for i,(hit,_) in enumerate(pairs):
        hit.content="Alex has specimens in a collection."
        hit.metadata['predicate']='mineral inventory' if i<2 else 'shell inventory'
        hit.metadata['origin']='indexed_sweep'
    before=[h.model_dump(mode='json') for h,_ in pairs]
    result=ground_evidence(question,request,[h for h,_ in pairs],session_id='session',current_turns=[],
        resolved_turns={t.turn_id:t for _,t in pairs},blocked_events=set(),max_sources=2,budget=10000)
    assert {h.metadata['predicate'] for h in result.sources}=={'mineral inventory','shell inventory'}
    assert result.trace.binding_units_selected==0  # Scalar counts do not invent counted members.
    assert [h.model_dump(mode='json') for h,_ in pairs]==before
    assert result.trace.context_tokens_estimate<=10000


def test_relation_hint_requires_visible_evidence_and_does_not_become_truth():
    hit,_=source()
    request=plan()
    request.answer_contract.outputs[0].retrieval_relations=[('Alex','activities')]
    assert _request_relation_units(hit,request)
    hit.metadata['source_events'][0]['text']='unrelated visible text'
    assert not _request_relation_units(hit,request)
    hit.metadata['source_events']=[]
    assert not _request_relation_units(hit,request)
