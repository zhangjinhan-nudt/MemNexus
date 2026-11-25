"""Crash, budget and data identity regressions; no external model calls."""
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

import lme_paired_protocol as batch


@pytest.fixture
def simulated(tmp_path, monkeypatch):
    account = tmp_path / "account"
    sys.path.insert(0, str(batch.ACCOUNT))
    import budget_account
    import run_locomo_pilot as native
    monkeypatch.setattr(batch, "ACCOUNT", account)
    monkeypatch.setattr(budget_account, "HERE", account)
    batch.write(account/"authorization.json", {"forward_allowance_rmb": 100})
    batch.write(account/"admissions.json", [])
    questions = [dict(sample_id=f"case_{i}_S", sample_index=i, category="multi-session",
                      category_id=0, qa_index=0, question=f"question {i}", setting="S", evidence=["private"])
                 for i in range(3)]
    samples = {}
    for q in questions:
        text = q["question"]
        samples[q["sample_index"]] = dict(sample_id=q["sample_id"], dataset="LongMemEval-S", setting="S",
            question_type="multi-session", qa=[dict(question=text, answer="gold canary", question_date="2024-05-04",
                                                   dataset="LongMemEval-S", category=4)], conversation={
            "session_1": [dict(speaker="user", text="first session", dia_id="s1-t1")],
            "session_1_date_time": "2024-01-01",
            "session_3": [dict(speaker="assistant", text="last session", dia_id="s3-t1")],
            "session_3_date_time": "2024-02-03"})
    detail = batch.inspect_inputs(questions,samples)
    calls = []

    def trace(folder, cost, arm):
        usage = dict(cost_rmb=cost,reserved_rmb=0,provider_attempts=1,input_tokens=10,output_tokens=2)
        batch.write(folder/"usage.json", usage)
        events = [dict(event="request",body=dict(model=batch.MODEL,temperature=0,max_tokens=512 if arm=="judge" else 32768,
                                                 thinking={"type":"disabled"})),
                  dict(event="response",body=dict(model=batch.MODEL))]
        (folder/"usage.calls.jsonl").write_text("\n".join(map(json.dumps,events)),encoding="utf-8")
        return usage

    async def system(arm,sample,qa,items,folder,cap,maximum,snapshots):
        sid = sample["sample_id"]
        calls.append((sid,arm))
        # Same gold-independent construction data for both arms. Real adapters
        # implement construction before accessing qa/evaluation_query.
        turns = native.raw_turns(sample,maximum)
        assert [t["text"] for t in turns] == ["first session", "last session"]
        assert "gold canary" not in json.dumps(turns) and maximum == 3
        assert snapshots == (folder/"snapshots" if arm=="MemNexus" else None)
        usage = trace(folder, 1, arm)
        stage = dict(raw_turns=2,internal_session_count=2,session_layout="dataset_sessions",
                     memory_base_replayed=False,build_usage=usage,build_ms=10,total_usage=usage)
        rows = [dict(sample_id=sid,system=arm,question=qa[0]["question"],prediction="generated")]
        if arm == "MemNexus":
            access = {"source_read_scope": "events", "authorized_source_sessions": ["past"]}
            stage.update(**access, history_session_ids=["past", "current"], qa_session_id="current")
            rows[0].update(**access, reasoning=dict(access))
        batch.write(folder/"stage_result.json",dict(rows=rows,stage=stage))

    def judge(rows,cap,usage_ledger):
        calls.append((rows[0]["sample_id"],"judge"))
        for r in rows:
            r["judge_result"]="CORRECT"
        return trace(usage_ledger.parent, .02, "judge")

    monkeypatch.setattr(batch,"run_unit",system)
    monkeypatch.setattr(native,"judge_rows",judge)
    out = tmp_path/"test-output"
    def run(count=3, budget=30):
        return asyncio.run(batch.run_registered(out,"unit",questions[:count],samples,detail,budget,SimpleNamespace()))
    return SimpleNamespace(run=run,out=out,account=account,calls=calls,samples=samples,questions=questions,
                           detail=detail,system=system)


def test_same_prefix_and_budget_stop(simulated):
    s = simulated
    assert s.run(budget=20.09) == 2
    assert len(s.calls)==3
    status=batch.read(s.out/"status.json")
    assert status["completed_pairs"]==1 and status["next_id"]=="case_1_S"
    assert status["spent_or_reserved_rmb"]==2.02
    assert s.run(budget=20.09)==2 and len(s.calls)==3


def test_completed_components_resume_without_requests(simulated):
    s=simulated
    assert s.run(count=2)==0
    calls=[("case_0_S","MemNexus"),("case_0_S","SimpleMem"),("case_0_S","judge"),
           ("case_1_S","SimpleMem"),("case_1_S","MemNexus"),("case_1_S","judge")]
    assert s.calls==calls and s.run(count=2)==0 and s.calls==calls
    assert len(batch.read(s.account/"admissions.json"))==6
    assert len(batch.read(s.out/"rows_judged.json"))==4


@pytest.mark.parametrize("target", ["stage", "row", "grant", "judge"])
def test_completed_lme_components_cannot_mix_old_quote_permissions(simulated, target):
    s = simulated
    assert s.run(count=1) == 0
    calls = list(s.calls)
    arm = "judge" if target == "judge" else "MemNexus"
    path = s.account / "components" / f"paired_unit_{arm}" / "case_0_S" / "stage_result.json"
    result = batch.read(path)
    row = next(r for r in result["rows"] if r["system"] == "MemNexus")
    if target == "stage":
        result["stage"].pop("source_read_scope")
    elif target == "grant":
        row["reasoning"]["authorized_source_sessions"] = ["another-session"]
    else:
        row.pop("source_read_scope")
        row["reasoning"].pop("source_read_scope")
    batch.write(path, result)
    unchanged = path.read_bytes()
    with pytest.raises(RuntimeError, match="history scope"):
        s.run(count=1)
    assert s.calls == calls and path.read_bytes() == unchanged


def test_uncertain_partial_request_is_not_reissued(simulated,monkeypatch):
    s=simulated
    async def interrupted(arm,*args):
        if arm=="SimpleMem":
            folder=args[3]
            batch.write(folder/"usage.json",dict(cost_rmb=.5,reserved_rmb=.1))
            raise TimeoutError("simulated")
        return await s.system(arm,*args)
    monkeypatch.setattr(batch,"run_unit",interrupted)
    with pytest.raises(TimeoutError):
        s.run(count=1)
    with pytest.raises(RuntimeError,match="never reissued"):
        s.run(count=1)
    assert s.calls==[("case_0_S","MemNexus")]
    assert batch.read(s.account/"budget.json")["reserved_rmb"]==10


def test_refuse_oracle_and_unregistered_model(simulated):
    s=simulated
    q=s.questions[0]
    with pytest.raises(ValueError,match="LME-S"):
        batch.inspect_inputs([{**q,"setting":"O"}],s.samples)
    assert s.run(count=1)==0
    trace=s.account/"components"/"paired_unit_MemNexus"/q["sample_id"]/"usage.calls.jsonl"
    trace.write_text(trace.read_text(encoding="utf-8").replace(batch.MODEL,"another-model"),encoding="utf-8")
    with pytest.raises(ValueError,match="model"):
        s.run(count=1)


def test_shell_ablation_cannot_change_native_method(monkeypatch):
    clean=batch.effective_method(batch.runtime_config())
    for key,value in {"MEMORY_PROMOTION_ENABLED":"false","MEMORY_GROUNDING_ENABLED":"false",
                      "MEMORY_GROUNDING_SEMANTIC":"true","MEMORY_PLANNER_MODE":"rules",
                      "MEMORY_INDEXED_SWEEP_LIMIT":"0","MEMORY_EMBEDDING_ENCODING":"plain",
                      "EXPERIMENT_MEMNEXUS_MAX_TOKENS":"100","EXPERIMENT_TRUNCATION_MAX_TOKENS":"0",
                      "EXPERIMENT_DISABLE_PLAN":"1","SIMPLEMEM_CFG":"old_flat_config",
                      "ENABLE_PLANNING":"false","ENABLE_REFLECTION":"false"}.items():
        monkeypatch.setenv(key,value)
    actual=batch.effective_method(batch.runtime_config())
    assert actual==clean and actual["MemNexus"]["indexed_sweep_limit"]==32
    assert actual["SimpleMem"]["ENABLE_REFLECTION"]
    assert actual["qa_history_access"]["MemNexus"]["source_read_scope"] == "events"
    assert "api_key" not in json.dumps(actual).lower()
