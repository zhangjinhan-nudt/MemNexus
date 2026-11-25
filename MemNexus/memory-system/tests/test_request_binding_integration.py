"""Existing planner/retriever/reader interfaces, driven only by local model stubs."""

import asyncio
import json
from copy import deepcopy

from agent_memory.models import AnswerContract, AnswerOutput, Evidence, Hit, Memory, QueryPlan, Session
from agent_memory.providers import CallableModel
from agent_memory.retrieval import Retriever
from agent_memory.settings import Settings
from agent_memory.synthesis import render_table


class ScriptedChat:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    async def __call__(self, prompt):
        self.prompts.append(prompt)
        assert self.responses, "Unexpected additional model call"
        return json.dumps(self.responses.pop(0))


def prompt_data(prompt):
    return json.JSONDecoder().raw_decode(prompt.split("\n输入数据：\n", 1)[1])[0]


def memory(identifier, subject, text):
    return Memory(memory_id=identifier, session_id="session", scope_id="session",
                  subject=subject, predicate="hobbies", value=text, content=text,
                  evidence=[Evidence(turn_id=identifier, event_index=0, quote=text)])


def source(text):
    return Hit(content=text, score=1, source_file="fixture", chunk_id="one", metadata={
        "source_events": [{"text": text, "speaker": "Inez", "role": "user", "occurred_at": None,
                           "context_only": False, "evidence_grade": "primary"}],
    })


def answer(question, quote, value, binding=None):
    item = {"value": value, "supports": [{"source_id": 1, "quote": quote}]}
    if binding is not None:
        item["request_binding"] = binding
    return {"slots": [{"target": question, "derivation": "Use the requested sourced fact.",
                       "values": [item]}]}


def search(question, response, memories, *, known_speakers=None):
    chat = ScriptedChat(response)
    retriever = Retriever(Settings(planner_mode="model"), CallableModel(chat))
    try:
        result = asyncio.run(retriever.search(question, Session(session_id="session"), memories, [],
                                              known_speakers=known_speakers))
    finally:
        retriever.close()
    assert len(chat.prompts) == 1
    return result, chat.prompts[0]


def test_planner_wire_uses_request_outputs_without_legacy_background_inputs():
    question = "Which activities does Inez enjoy on holiday?"
    chat = ScriptedChat({"route": "both", "answer_contract": {"outputs": [{
        "output_id": "activities", "question_span": question, "operator": "set",
        "subjects": ["Inez"], "relation": "hobbies", "qualifier": "on holiday",
        "relation_refs": [1], "count_unit": "value",
    }]}})
    plan = asyncio.run(CallableModel(chat).plan(question, {
        "relation_catalog": [("Inez", "hobbies", "many", None)],
    }))
    assert len(chat.prompts) == 1
    assert '"required_inputs"' not in chat.prompts[0]
    assert '"retrieval_relations"' not in chat.prompts[0]
    assert '"relation_refs"' in chat.prompts[0]
    assert prompt_data(chat.prompts[0])["context"]["relation_catalog"] == [["Inez", "hobbies", "many", None]]
    output = plan.answer_contract.outputs[0]
    assert output.question_span == question and output.qualifier == "on holiday"
    assert output.required_inputs == [] and output.relation_refs == [1]


def test_final_retrieval_plan_cannot_recreate_discarded_wrong_subject_collection():
    question = "Which hobbies does Inez enjoy?"
    response = {"route": "both", "subject": "Kai", "predicate": "hobbies",
                "semantic_queries": ["Kai hobbies"],
                "collection_query": {"subjects": ["Kai"], "predicate": "hobbies"},
                "answer_contract": {"outputs": [{
                    "output_id": "hobbies", "question_span": question, "operator": "set",
                    "subjects": ["Kai"], "relation": "hobbies", "count_unit": "value", "relation_refs": [2],
                }]}}
    result, _ = search(question, response, [
        memory("inez", "Inez", "Inez enjoys hiking."),
        memory("kai", "Kai", "Kai says Inez enjoys climbing."),
    ], known_speakers=["Inez", "Kai"])
    assert result.plan.subject is None and result.plan.collection_query is None
    assert result.plan.query_entities == ["Inez"]
    assert result.plan.semantic_queries == []
    output = result.plan.answer_contract.outputs[0]
    assert output.subjects == [] and output.relation_refs == []
    assert output.question_span == question and output.relation == "hobbies"
    assert result.plan.required_info == [question]
    assert {hit.metadata["memory_id"] for hit in result.results} == {"inez", "kai"}


def test_contract_created_collection_reaches_final_route_and_subject_hints():
    question = "How many hobbies do I have?"
    result, _ = search(question, {"route": "both", "answer_contract": {"outputs": [{
        "output_id": "total", "question_span": question, "operator": "count",
        "subjects": ["Inez"], "relation": "hobbies", "count_unit": "value", "relation_refs": [1],
    }]}}, [memory("inez", "Inez", "Inez enjoys hiking and climbing.")])
    assert result.plan.collection_query.operation == "count"
    assert result.plan.query_type == "count" and result.plan.recall_policy == "recall_first"
    assert result.plan.query_entities == ["Inez"]
    assert result.plan.required_info == [question]


def test_dropped_collection_does_not_leave_its_subject_in_final_route():
    question = "Which activities does Inez enjoy on holiday?"
    result, _ = search(question, {
        "route": "both", "collection_query": {"subjects": ["Kai"], "predicate": "hobbies"},
        "answer_contract": {"outputs": [{
            "output_id": "activities", "question_span": question, "operator": "set",
            "relation": "hobbies", "qualifier": "on holiday", "count_unit": "value",
        }]},
    }, [memory("inez", "Inez", "Inez enjoys hiking on holiday."),
        memory("kai", "Kai", "Kai enjoys cycling.")], known_speakers=["Inez"])
    assert result.plan.collection_query is None
    assert result.plan.query_entities == ["Inez"]
    assert result.plan.required_info == [question]
    assert result.plan.answer_contract.outputs[0].qualifier == "on holiday"


def test_reader_receives_minimal_anchored_contract_and_unmodified_question_sources():
    question, quote = "Which color does Inez prefer for a hiking bag?", "I prefer blue for my hiking bag."
    plan = QueryPlan(subject="Kai", semantic_queries=["irrelevant search phrase"],
                     required_info=["invented background premise"], answer_contract=AnswerContract(outputs=[
                         AnswerOutput(output_id="color", question_span=question, subjects=["Inez"],
                                      relation="preferred color", qualifier="for a hiking bag", relation_refs=[1],
                                      retrieval_relations=[("Inez", "preferred color")],
                                      required_inputs=["Date of first purchase"]),
                     ]))
    before = plan.model_dump()
    sources = [source(quote)]
    source_before = [hit.model_dump() for hit in sources]
    chat = ScriptedChat(answer(question, quote, "blue", {"request_id": "color", "match": "matched"}))
    table = asyncio.run(CallableModel(chat).synthesize(question, plan, sources))
    data = prompt_data(chat.prompts[0])
    assert len(chat.prompts) == 1 and render_table(table) == "blue"
    assert data["question"] == question and data["targets"] == [question]
    output = data["request_contract"]["outputs"][0]
    assert output["subjects"] == ["Inez"] and output["qualifier"] == "for a hiking bag"
    assert not {"relation_refs", "retrieval_relations", "required_inputs"} & output.keys()
    assert "plan" not in data and "required_info" not in data and "collection_query" not in data
    assert data["sources"][0]["source_events"][0]["text"] == quote
    assert plan.model_dump() == before and [hit.model_dump() for hit in sources] == source_before


def test_unanchored_contract_falls_back_to_original_question_answer_without_extra_call():
    question, quote = "Which color does Inez prefer?", "I prefer blue."
    plan = QueryPlan(answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="wrong", question_span="Which color does Kai prefer?", subjects=["Kai"], relation="color",
    )]))
    chat = ScriptedChat(answer(question, quote, "blue"))
    table = asyncio.run(CallableModel(chat).synthesize(question, plan, [source(quote)]))
    data = prompt_data(chat.prompts[0])
    assert "request_contract" not in data and data["question"] == question
    assert render_table(table) == "blue" and len(chat.prompts) == 1
    assert plan.answer_contract.outputs[0].output_id == "wrong"


def test_executed_operand_role_conflict_uses_the_existing_single_repair_from_original_evidence():
    question = "How many people were on Inez's team while Inez was a senior engineer?"
    quote = "As a manager Inez led nine people; as a senior engineer Inez worked in a team of four."
    plan = QueryPlan(answer_contract=AnswerContract(outputs=[AnswerOutput(
        output_id="team", question_span=question, operator="count", subjects=["Inez"],
        relation="team size", qualifier="while a senior engineer", count_unit="entity",
    )]))
    initial = answer(question, quote, "", {
        "request_id": "team", "subject": "Inez", "relation": "team size",
        "qualifier": "as a senior engineer", "match": "unresolved",
    })
    initial["slots"][0]["values"][0].update(operation="sum", unit="people", inputs=[{
        "identity": "manager team", "statement": "As a manager Inez led nine people.", "amount": "9",
        "supports": [{"source_id": 1, "quote": "As a manager Inez led nine people"}],
        "qualification": {"subject": "Inez", "relation": "team size", "qualifier": "as a manager",
                          "match": "conflict",
                          "reason": "The manager role differs from the requested senior engineer role."},
    }])
    repaired = deepcopy(initial)
    repaired["slots"][0]["values"][0].update(request_binding={
        "request_id": "team", "subject": "Inez", "relation": "team size",
        "qualifier": "as a senior engineer", "match": "matched",
    }, inputs=[{
        "identity": "senior engineer team", "statement": "As a senior engineer Inez worked in a team of four.",
        "amount": "4", "supports": [{"source_id": 1, "quote": "as a senior engineer Inez worked in a team of four"}],
        "qualification": {"subject": "Inez", "relation": "team size", "qualifier": "as a senior engineer",
                          "match": "matched"},
    }])
    chat = ScriptedChat(initial, repaired)
    table = asyncio.run(CallableModel(chat).synthesize(question, plan, [source(quote)]))
    assert len(chat.prompts) == 2 and render_table(table) == "4 people"
    assert "previous_response" in chat.prompts[1] and "explicitly declares conflict" in chat.prompts[1]
    assert "input 'manager team'" in chat.prompts[1]
    assert "do not merely relabel conflict as matched" in chat.prompts[1]
    assert prompt_data(chat.prompts[0]) == prompt_data(chat.prompts[1])
