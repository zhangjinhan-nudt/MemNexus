"""Offline protocol regression tests; synthetic fixtures are NOT benchmark results."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_mixed200 as mixed
from run_locomo_pilot import evaluation_query, raw_turns


@pytest.fixture(scope="module")
def corpus():
    return mixed.build_units()


def test_full_history_and_no_gold_leakage(corpus):
    units, cohort, selected = corpus
    assert len(units) == 110
    assert sum(len(u["questions"]) for u in units) == 200
    by_id = {r["question_id"]: r for r in selected}
    for unit in units:
        turns = raw_turns(unit["sample"], unit["max_session"])
        assert len(turns) == unit["sample"]["expected_raw_turns"]
        if unit["dataset"] == "LoCoMo":
            assert unit["max_session"] == cohort["max_session"] == 5
            continue
        row = by_id[unit["questions"][0]["question_id"]]
        expected = [m for session in row["haystack_sessions"] for m in session if m["content"].strip()]
        assert [t["text"] for t in turns] == [m["content"] for m in expected]
        assert [t["speaker"] for t in turns] == [m["role"] for m in expected]
        assert unit["max_session"] == len(row["haystack_sessions"])
        assert "has_answer" not in json.dumps(unit["sample"])
        assert "answer_session_ids" not in json.dumps(unit["sample"])
        query, _ = evaluation_query(unit["questions"][0])
        assert row["question_date"] in query


def test_full_history_catches_legacy_two_session_default(corpus):
    unit = corpus[0][10]
    assert len(raw_turns(unit["sample"], 2)) < unit["sample"]["expected_raw_turns"]


def test_budget_includes_failed_and_nested_attempts(tmp_path):
    mixed.save(tmp_path / "manifest.json", {"prior_spend_rmb": 4})
    mixed.save(tmp_path / "attempts/a/usage.json", {"cost_rmb": 2, "reserved_rmb": 1})
    mixed.save(tmp_path / "units/failed/simplemem_usage.json", {"cost_rmb": 3})
    assert mixed.remaining_budget(tmp_path, 12) == 2
    with pytest.raises(RuntimeError, match="exhausted"):
        mixed.remaining_budget(tmp_path, 10)


def test_budget_malformed_ledger_is_not_ignored(tmp_path):
    mixed.save(tmp_path / "usage.json", {"cost_rmb": "not a number"})
    with pytest.raises(ValueError):
        mixed.retained_spend(tmp_path)


def test_source_mismatch_and_wrong_message_count_reject_resume():
    unit = {"raw_sha256": "raw", "sample": {"expected_raw_turns": 50}, "questions": []}
    data = {"signature": "v1", "raw_sha256": "raw", "stage": {
        "system": "SimpleMem", "raw_turns": 2, "memory_count": 1}, "rows": []}
    with pytest.raises(ValueError, match="ingest"):
        mixed.check_stage(data, unit, "SimpleMem", "v1")
    with pytest.raises(ValueError, match="mismatch"):
        mixed.check_stage(data, unit, "SimpleMem", "v2")


def test_completed_stage_never_repeats_calls(tmp_path, monkeypatch):
    import asyncio
    unit = {"unit_id": "fixture", "raw_sha256": "raw", "dataset": "LoCoMo",
            "sample": {"expected_raw_turns": 3}, "questions": [{"item_id": "q", "question": "Q"}]}
    mixed.save(tmp_path / "run.json", {"signature": "s", "stages": {}})
    mixed.save(tmp_path / "manifest.json", {})
    for system in mixed.SYSTEMS:
        mixed.save(tmp_path / f"attempts/fixture_{system.lower()}__002/stage_result.json", {
            "signature": "s", "raw_sha256": "raw",
            "stage": {"system": system, "raw_turns": 3, "memory_count": 1},
            "rows": [{"item_id": "q", "unit_id": "fixture", "system": system,
                      "question": "Q", "prediction": "P"}]})
    monkeypatch.setattr(mixed, "load_runner", lambda: None)
    monkeypatch.setattr(mixed, "judge_checkpoint", lambda *a: None)
    _, rows, stages = asyncio.run(mixed.run_units(tmp_path, [unit], 1))
    assert len(rows) == len(stages) == 2
    assert len(list((tmp_path / "attempts").iterdir())) == 2


def test_f1_multiset_overlap():
    assert mixed.token_f1("red red blue", "red green") == pytest.approx(.4)
    assert mixed.token_f1("", "one") == 0
    assert mixed.token_f1("Two!", "two") == 1


def test_long_message_not_split(corpus):
    longest = max((t["text"] for u in corpus[0] if u["dataset"] == "LongMemEval-S"
                   for t in raw_turns(u["sample"], u["max_session"])), key=len)
    from agent_memory.models import Event
    assert len(longest) > 16000
    assert Event(role="user", content=longest).content == longest


def test_small_cohort_outcome_blind_and_full_history(corpus):
    from collections import Counter
    units = mixed.select_small_units(corpus[0])
    assert units == mixed.select_small_units(corpus[0])
    assert sum(len(u['questions']) for u in units) == 26
    assert Counter(q['category'] for u in units if u['dataset']=='LoCoMo' for q in u['questions']) == {1:4,2:4,3:4,4:4,5:4}
    assert len([u for u in units if u['dataset']=='LongMemEval-S']) == 6
    original = {u['unit_id']: u for u in corpus[0]}
    for unit in units:
        assert unit['raw_sha256'] == original[unit['unit_id']]['raw_sha256']
        assert unit['sample'] == original[unit['unit_id']]['sample']
        assert unit['max_session'] == original[unit['unit_id']]['max_session']


def test_reused_control_retains_original_signature_and_rejects_mutation(tmp_path, monkeypatch):
    import asyncio
    unit={'unit_id':'fixture','raw_sha256':'raw','dataset':'LoCoMo',
          'sample':{'expected_raw_turns':3},'questions':[{'item_id':'q','question':'Q'}]}
    old_path=tmp_path/'old/stage_result.json'
    data={'signature':'old','raw_sha256':'raw','stage':{'system':'SimpleMem','raw_turns':3,'memory_count':1},
          'rows':[{'item_id':'q','unit_id':'fixture','system':'SimpleMem','question':'Q','prediction':'P'}]}
    mixed.save(old_path,data)
    ref={'stage_result':str(old_path),'signature':'old','source_run':str(old_path.parent),
         'artifact_sha256':{str(old_path):mixed.digest(old_path)}}
    folder=tmp_path/'new'
    mixed.save(folder/'run.json',{'signature':'new','stages':{}})
    mixed.save(folder/'manifest.json',{'baseline_reuse':{'fixture_simplemem':ref}})
    new_data={'signature':'new','raw_sha256':'raw','stage':{'system':'MemNexus','raw_turns':3,'memory_count':1},
              'rows':[{'item_id':'q','unit_id':'fixture','system':'MemNexus','question':'Q','prediction':'P2'}]}
    mixed.save(folder/'attempts/fixture_memnexus__001/stage_result.json',new_data)
    monkeypatch.setattr(mixed,'load_runner',lambda:None)
    monkeypatch.setattr(mixed,'judge_checkpoint',lambda *args:None)
    _,rows,stages=asyncio.run(mixed.run_units(folder,[unit],1))
    assert len(rows)==len(stages)==2
    assert mixed.read(old_path)==data  # No re-signing or fabricated new provider run.
    assert mixed.read(folder/'run.json')['stages']['fixture_simplemem']==str(old_path)
    data['rows'][0]['prediction']='changed'
    mixed.save(old_path,data)
    with pytest.raises(ValueError,match='changed after freeze'):
        mixed.load_reused_baseline(ref,unit)


def test_reuse_rejects_changed_baseline_implementation_before_any_calls(tmp_path):
    old={'source_sha256':{'SimpleMem/main.py':'old'},'settings':{},'dataset_overrides':{'LoCoMo':{}},
         'datasets':{'LoCoMo':{'sha256':'raw'}}}
    mixed.save(tmp_path/'manifest.json',old)
    mixed.save(tmp_path/'run.json',{'signature':'old'})
    with pytest.raises(ValueError,match='implementation changed'):
        mixed.prepare_baseline_reuse([], {**old,'source_sha256':{'SimpleMem/main.py':'new'}}, {'LoCoMo':tmp_path})
