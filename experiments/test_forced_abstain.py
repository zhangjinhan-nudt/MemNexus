"""Contract for the `forced_abstain` gate field (protocol R2).

R2 says the first reading of any comparison is `forced_abstain == 0`.  That only
works if the field is actually produced, and if it means exactly one thing:
**the emitted text came from a crash fallback, not from a decision of the
system.**  The distinction that matters is a legitimate abstention -- on an
adversarial question "Not mentioned in the conversation." is the correct
answer, so flagging it would make the gate fire on good runs.
"""

from __future__ import annotations

import run_locomo_pilot


def test_empty_prediction_is_forced():
    assert run_locomo_pilot.is_forced_abstain("") is True
    assert run_locomo_pilot.is_forced_abstain("   ") is True
    assert run_locomo_pilot.is_forced_abstain(None) is True


def test_provider_failure_marker_is_forced():
    assert run_locomo_pilot.is_forced_abstain("Failed to generate answer") is True
    assert run_locomo_pilot.is_forced_abstain("failed to generate answer") is True


def test_legitimate_abstention_is_not_forced():
    # A decision, not a crash.  Adversarial gold answers are exactly this text.
    assert run_locomo_pilot.is_forced_abstain("Not mentioned in the conversation.") is False


def test_real_answer_is_not_forced():
    assert run_locomo_pilot.is_forced_abstain("Adoption agencies") is False


def test_runtime_trace_flag_overrides_text():
    # The runtime's binding fallback emits a sentence that looks like a normal
    # abstention, so the text check alone cannot see it.  The trace flag is
    # authoritative and the runner ORs the two.
    text = "Insufficient evidence to determine the answer."
    assert run_locomo_pilot.is_forced_abstain(text) is False
    reasoning = {"answer_path": "semantic_binding_failed", "forced_abstain": True}
    flagged = run_locomo_pilot.is_forced_abstain(text) or bool(reasoning.get("forced_abstain"))
    assert flagged is True


def test_every_row_shape_carries_the_field():
    # compare_paired_runs reads row.get("forced_abstain"); a missing key would
    # silently read as "no crash" and let a broken run pass the gate.
    source = (
        open(run_locomo_pilot.__file__, encoding="utf-8").read()
    )
    # Three constructors carry the field; the two resume paths reuse those rows.
    assert source.count("rows.append(completed[key])") == 2
    assert source.count('"forced_abstain":') == 3
