"""Generated evidence only: deny widening the single zero-call OPS reissue."""
from copy import deepcopy

import pytest

from benchmarks.performance import qwen38_accuracy_diagnostic as diagnostic
from benchmarks.performance import qwen38_accuracy_reissue as reissue
from qa.system.test_qwen38_accuracy_reissue import proof_fixture, rehash
from qa.system.test_qwen38_accuracy_diagnostic import completed, retained, terminals


def test_control_two_keeps_spent_history_and_carries_same_namespace_into_remaining(tmp_path, monkeypatch):
    protocol, actual, old, proof = proof_fixture(tmp_path, monkeypatch)
    before = diagnostic.digest(old)
    pilot = diagnostic.bind_plan(protocol, actual, control_reissue=proof)
    remaining = diagnostic.bind_plan(protocol, actual, stage="remaining", retained=retained(), control_reissue=proof)
    assert diagnostic.digest(old) == before
    assert pilot["maximum_private_model_requests"] == 24
    assert remaining["maximum_private_model_requests"] == 184
    assert pilot["maximum_reserved_model_slots_including_failed_control_per_arm"] == 232
    assert pilot["maximum_actual_model_posts_per_arm"] == 208
    assert pilot["historical_control_slots_consumed"] == 24
    assert pilot["source_attempts"] == remaining["source_attempts"]
    assert len(pilot["source_attempts"]) == 6
    assert not ({row["attempt_id"] for row in pilot["attempts"]} & {row["attempt_id"] for row in remaining["attempts"]})
    assert diagnostic.validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))


@pytest.mark.parametrize("artifact,field", [("prior_reconciliation", "actual_model_posts"),
    ("prior_settlement", "actual_model_posts"), ("prior_settlement", "actual_video_attempts")])
def test_boolean_false_is_not_an_actual_zero_call_counter(tmp_path, monkeypatch, artifact, field):
    _, _, _, proof = proof_fixture(tmp_path, monkeypatch)
    proof["artifacts"][artifact]["document"][field] = False
    rehash(proof, artifact)
    with pytest.raises(ValueError):
        reissue.validate_control_reissue(proof)


def test_boolean_false_cannot_replace_source_model_posts_counter(tmp_path, monkeypatch):
    _, _, _, proof = proof_fixture(tmp_path, monkeypatch)
    proof["artifacts"]["prior_source_terminals"]["document"]["records"][4]["model_posts"] = False
    rehash(proof, "prior_source_terminals")
    with pytest.raises(ValueError):
        reissue.validate_control_reissue(proof)


@pytest.mark.parametrize("field,value", [("operational_control_attempt", 3),
    ("historical_control_slots_consumed", 0),
    ("maximum_reserved_model_slots_including_failed_control_per_arm", 208),
    ("maximum_actual_model_posts_per_arm", 232), ("maximum_actual_source_pipeline_attempts_per_case", 2)])
def test_plan_cannot_refund_relabel_widen_or_create_third_control(tmp_path, monkeypatch, field, value):
    protocol, actual, _, proof = proof_fixture(tmp_path, monkeypatch)
    plan = diagnostic.bind_plan(protocol, actual, control_reissue=proof)
    plan[field] = value
    with pytest.raises(ValueError):
        diagnostic.validate_diagnostic_plan(plan)


def test_reissue_cannot_reuse_old_spent_model_slot_identity(tmp_path, monkeypatch):
    protocol, actual, old, proof = proof_fixture(tmp_path, monkeypatch)
    plan = diagnostic.bind_plan(protocol, actual, control_reissue=proof)
    plan["attempts"][0]["attempt_id"] = old["attempts"][0]["attempt_id"]
    with pytest.raises(ValueError):
        diagnostic.validate_diagnostic_plan(plan)


@pytest.mark.parametrize("field,value", [("source_pipeline_started", True), ("model_posts", 1),
    ("complete_result", {"verdict": "no_violation"}), ("server_receipt_verified", True)])
def test_prior_private_activity_or_verified_result_is_never_control_only(tmp_path, monkeypatch, field, value):
    _, _, _, proof = proof_fixture(tmp_path, monkeypatch)
    proof["artifacts"]["prior_source_terminals"]["document"]["records"][4][field] = deepcopy(value)
    rehash(proof, "prior_source_terminals")
    with pytest.raises(ValueError):
        reissue.validate_control_reissue(proof)


@pytest.mark.parametrize("artifact", ["prior_settlement", "root_authorization", "prior_reconciliation"])
def test_missing_original_spent_settlement_or_root_authority_fails_closed(tmp_path, monkeypatch, artifact):
    _, _, _, proof = proof_fixture(tmp_path, monkeypatch)
    proof["artifacts"].pop(artifact)
    with pytest.raises(ValueError):
        reissue.validate_control_reissue(proof)
