"""Contract tests for the offline LoCoMo-1000 finalizer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import finalize_locomo1000 as final


def make_row(system: str, qa_index: int, category_id: int = 4, *, sample: str = "conv-x",
             prediction: str = "alpha", gold: str = "alpha", question: str | None = None) -> dict:
    return {
        "system": system, "sample_id": sample, "qa_index": qa_index,
        "category_id": category_id, "category": final.CATEGORIES[category_id],
        "question": question if question is not None else f"question-{qa_index}",
        "gold_answer": gold, "prediction": prediction, "retrieved_sources": ["alpha evidence"],
        "retrieval_count": 1, "latency_ms": 12,
        "model_calls": {"input_tokens": 100, "output_tokens": 10},
        "reasoning": {"answer_path": "test"}, "plan": {"route": "test"},
        "grounding": {"supplemental_rounds": 0}, "warnings": [],
    }


def make_cohort(questions: list[dict]) -> dict:
    return {
        "questions": questions, "question_count": len(questions),
        "conversation_count": len({q["sample_id"] for q in questions}),
        "categories": dict(Counter(q["category"] for q in questions)),
        "max_session": 30, "raw_turns": {"conv-x": 1, "conv-y": 1},
    }


def questions_for(count: int = 5, *, sample: str = "conv-x") -> list[dict]:
    result = []
    for qa_index in range(count):
        cid = qa_index + 1 if count >= 5 else 4
        result.append({"sample_id": sample, "sample_index": 0, "qa_index": qa_index,
                       "category_id": cid, "category": final.CATEGORIES[cid],
                       "question": f"question-{qa_index}"})
    return result


def stage_record(system: str, rows: list[dict], *, sample: str = "conv-x") -> dict:
    return {
        "unit": f"{sample}_{system.lower()}", "path": Path(f"{sample}_{system.lower()}/stage_result.json"),
        "payload": {"stage": {"system": system, "sample_id": sample, "raw_turns": 1,
                              "memory_snapshot_sha256": "snapshot-sha", "build_usage": {}},
                    "rows": rows},
    }


def write_snapshot(snapshot_dir: Path, sample: str = "conv-x") -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / f"{sample}.json").write_text(json.dumps({
        "sample_id": sample, "sha256": "snapshot-sha", "raw_turns": 1,
        "build_usage": {"input_tokens": 5, "output_tokens": 2, "cost_rmb": 0.001},
    }), encoding="utf-8")


def small_gate_inputs() -> tuple[dict, list[dict]]:
    cohort = make_cohort(questions_for())
    rows = []
    for question in cohort["questions"]:
        for system in final.SYSTEMS:
            prediction = "not mentioned" if question["category_id"] == 5 else "alpha"
            rows.append(make_row(system, question["qa_index"], question["category_id"],
                                 prediction=prediction, question=question["question"]))
    records = [stage_record(system, [row for row in rows if row["system"] == system])
               for system in final.SYSTEMS]
    return cohort, records


def validate_small(cohort: dict, records: list[dict], snapshot_dir: Path) -> dict:
    return final.validate_gate(
        cohort, records, [], expected_raw_turns={"conv-x": 1},
        run_summary={"max_session": 30}, snapshot_dir=snapshot_dir, require_full=False,
    )


def test_gate_duplicate_key_is_rejected(tmp_path: Path):
    cohort, records = small_gate_inputs()
    write_snapshot(tmp_path)
    records[0]["payload"]["rows"].append(dict(records[0]["payload"]["rows"][0]))
    audit = validate_small(cohort, records, tmp_path)
    assert not audit["gate_passed"]
    assert "duplicate_row_key" in {failure["code"] for failure in audit["failures"]}


def test_gate_missing_question_is_rejected(tmp_path: Path):
    cohort, records = small_gate_inputs()
    write_snapshot(tmp_path)
    records[1]["payload"]["rows"].pop()
    audit = validate_small(cohort, records, tmp_path)
    assert not audit["gate_passed"]
    assert "key_misalignment" in {failure["code"] for failure in audit["failures"]}


def test_identical_text_with_distinct_qa_index_remains_two_questions():
    mn = [make_row("MemNexus", qa, question="same text") for qa in (7, 8)]
    sm = [make_row("SimpleMem", qa, question="same text") for qa in (7, 8)]
    pairs = final.paired_rows(mn + sm)
    items, _ = final.build_paired_items(pairs)
    assert len(items) == 2
    assert {item["qa_index"] for item in items} == {7, 8}


def complete_pairs(*, sample: str = "conv-x") -> list[tuple[dict, dict]]:
    pairs = []
    for cid in range(1, 6):
        if cid == 5:
            mn_prediction, sm_prediction, gold = "distractor", "not mentioned", "ignored"
        else:
            mn_prediction, sm_prediction, gold = "alpha", "wrong", "alpha"
        mn = make_row("MemNexus", cid, cid, sample=sample, prediction=mn_prediction, gold=gold)
        sm = make_row("SimpleMem", cid, cid, sample=sample, prediction=sm_prediction, gold=gold)
        pairs.append((mn, sm))
    return pairs


def test_average_is_four_category_equal_weight_and_excludes_adversarial():
    pairs = complete_pairs()
    items, alignment = final.build_paired_items(pairs)
    metrics = final.compute_metrics(items, pairs, 10, alignment, False)
    mn = metrics["paper_table"]["systems"]["MemNexus"]
    sm = metrics["paper_table"]["systems"]["SimpleMem"]
    assert mn["average_f1"] == pytest.approx(1.0)
    assert sm["average_f1"] == pytest.approx(0.0)
    assert mn["adversarial_sr"] == 0.0
    assert sm["adversarial_sr"] == 1.0
    assert metrics["pooled_f1_diagnostic"]["MemNexus"]["n"] == 4


def test_cluster_bootstrap_is_reproducible_and_uses_complete_category_draws():
    pairs = complete_pairs(sample="conv-x") + complete_pairs(sample="conv-y")
    items, _ = final.build_paired_items(pairs)
    first = final.cluster_bootstrap(items, replicates=30, seed=12345)
    second = final.cluster_bootstrap(items, replicates=30, seed=12345)
    assert first == second
    assert first["n_clusters"] == 2
    assert first["valid_four_category_replicates"] == 30
    for metric in ("Average", "category:multi-hop", "category:temporal",
                   "category:open-domain", "category:single-hop", "AdvSR"):
        assert first["valid_replicates_by_metric"][metric] == 30
        assert first["ci95"][metric] is not None


def test_judge_error_and_stale_prediction_do_not_enter_denominator():
    pairs = complete_pairs()[:1]
    mn, sm = pairs[0]
    judged = [
        {**mn, "judge_result": "error"},
        {**sm, "judge_result": "CORRECT"},
        {**mn, "prediction": "old prediction", "judge_result": "CORRECT"},
    ]
    items, alignment = final.build_paired_items(pairs, judged)
    result = final.judge_metrics(items)
    assert result["systems"]["MemNexus"]["denominator"] == 0
    assert result["systems"]["MemNexus"]["error_or_invalid_excluded"] == 1
    assert result["systems"]["SimpleMem"]["denominator"] == 1
    assert result["systems"]["SimpleMem"]["accuracy"] == 1.0
    assert alignment["stale_or_unmatched"] == 1


def test_judge_ledger_audit_includes_per_item_usage_filename(tmp_path: Path):
    judgments = tmp_path / "judgments"
    judgments.mkdir()
    events = [
        {"event": "request", "attempt": 1, "body": {"model": "judge", "messages": []}},
        {"event": "response", "attempt": 1, "body": {"usage": {
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
        }}},
    ]
    (judgments / "item.usage.calls.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8",
    )
    result = final.judge_ledger_audit(tmp_path)
    assert result["status"] == "available"
    assert result["ledger_count"] == 1
    assert result["totals"]["requests"] == 1
    assert result["totals"]["total_tokens"] == 110


def test_connector_diagnostic_is_symmetric_for_both_systems():
    mn = make_row("MemNexus", 0, 1, prediction="alpha, beta", gold="alpha, beta")
    sm = make_row("SimpleMem", 0, 1, prediction="alpha, beta", gold="alpha, beta")
    diagnostic = final.connector_diagnostic([(mn, sm)])
    assert diagnostic["symmetric_transform"].startswith("both systems")
    assert diagnostic["systems"]["MemNexus"]["changed_item_scores"] == 1
    assert diagnostic["systems"]["SimpleMem"]["changed_item_scores"] == 1
    assert (diagnostic["systems"]["MemNexus"]["semicolon"]["average_f1"]
            == diagnostic["systems"]["SimpleMem"]["semicolon"]["average_f1"])


def write_run_fixture(root: Path) -> tuple[Path, Path, Path]:
    run = root / "run"
    cohort_path = root / "cohort.json"
    snapshots = root / "snapshots"
    cohort, records = small_gate_inputs()
    # Omit one SimpleMem adversarial answer: four reasoning categories remain paired.
    records[1]["payload"]["rows"] = [row for row in records[1]["payload"]["rows"]
                                     if row["category_id"] != 5]
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    (run / "attempts").mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"max_session": 30}), encoding="utf-8")
    for record in records:
        unit = run / "attempts" / record["unit"]
        unit.mkdir()
        (unit / "stage_result.json").write_text(json.dumps(record["payload"]), encoding="utf-8")
        (unit / "usage.calls.jsonl").write_text("", encoding="utf-8")
    write_snapshot(snapshots)
    return run, cohort_path, snapshots


def test_default_gate_failure_writes_only_audit(tmp_path: Path):
    run, cohort, snapshots = write_run_fixture(tmp_path)
    out = tmp_path / "default-out"
    code = final.execute(run, cohort, out, allow_partial=False, bootstrap=5, snapshot_dir=snapshots)
    assert code == 2
    assert (out / "audit.json").exists()
    assert not (out / "REPORT.md").exists()
    assert not (out / "metrics.json").exists()


def test_allow_partial_marks_every_human_report_and_generates_all_files(tmp_path: Path):
    run, cohort, snapshots = write_run_fixture(tmp_path)
    out = tmp_path / "partial-out"
    code = final.execute(run, cohort, out, allow_partial=True, bootstrap=5, snapshot_dir=snapshots)
    assert code == 0
    expected = {"audit.json", "metrics.json", "paired_items.jsonl", "paired_items.csv",
                "mn_losses.md", "token_audit.json", "token_audit.md", "REPORT.md", "DASHBOARD.html"}
    assert expected <= {path.name for path in out.iterdir()}
    assert not json.loads((out / "audit.json").read_text(encoding="utf-8"))["gate_passed"]
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["status"] == "PARTIAL"
    assert "PARTIAL" in (out / "REPORT.md").read_text(encoding="utf-8")
    dashboard = (out / "DASHBOARD.html").read_text(encoding="utf-8")
    assert "PARTIAL" in dashboard
    assert "<svg" in dashboard
    assert "<script" not in dashboard.casefold()
    token = json.loads((out / "token_audit.json").read_text(encoding="utf-8"))
    assert token["pricing"] == final.tb.PRICING
    assert token["gross_accepted_additive"] is False
    assert token["judge_cost"]["status"] == "unavailable"
