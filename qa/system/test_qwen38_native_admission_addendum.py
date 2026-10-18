"""Generated metadata and runtime-state booleans only; no runtime/model calls."""
from copy import deepcopy

import pytest

from benchmarks.performance import qwen38_accuracy_diagnostic as core
from benchmarks.performance import qwen38_native_admission_addendum as admission
from benchmarks.performance import qwen38_native_fullsix as native
from qa.system.test_qwen38_native_fullsix import fixture


def prepared(monkeypatch):
    parent, selection = fixture(monkeypatch)
    protocol = native.freeze_protocol(parent, selection, policy_raw_sha256="a"*64)
    monkeypatch.setattr(admission, "PROTOCOL_RAW_SHA256", "b"*64)
    monkeypatch.setattr(admission, "PROTOCOL_SHA256", core.digest(protocol))
    addendum = admission.freeze_addendum(protocol, protocol_raw_sha256="b"*64)
    monkeypatch.setattr(admission, "ADDENDUM_RAW_SHA256", "c"*64)
    monkeypatch.setattr(admission, "ADDENDUM_SHA256", core.digest(addendum))
    return protocol, addendum


def test_addendum_preserves_protocol_selection_and_never_grants_execution(monkeypatch):
    protocol, addendum = prepared(monkeypatch)
    original = core.digest(protocol)
    admission.validate_admission_addendum(addendum, protocol, protocol_raw_sha256="b"*64)
    assert core.digest(protocol) == original
    assert addendum["source_attempts_required_per_arm"] == 6 and addendum["maximum_model_requests_per_arm"] == 208
    assert addendum["execution_order"] == [5,1,2,3,4,6]
    assert addendum["selection_criterion_sha256"] == core.digest(protocol["selection"])
    assert addendum["global_invalid_gate_requires_proven_process_death"] is False
    assert addendum["automatic_source_or_model_retry_authorized"] is False


@pytest.mark.parametrize("changes,expected", [
    ({}, "healthy_same_runtime"),
    ({"admission_healthy": False}, "runtime_admission_closed_drain_unknown"),
    ({"admission_healthy": None}, "runtime_admission_closed_drain_unknown"),
    ({"server_drain_complete": None}, "runtime_admission_closed_drain_unknown"),
    ({"server_drain_complete": False}, "runtime_admission_closed_drain_unknown"),
    ({"live_same_identity": None}, "runtime_admission_closed_drain_unknown"),
    ({"live_same_identity": False}, "runtime_admission_closed_drain_unknown"),
    ({"admission_healthy": 1}, "runtime_admission_closed_drain_unknown"),
    ({"confirmed_process_failure": True}, "confirmed_global_runtime_process_failure"),
    ({"confirmed_process_failure": 1}, "runtime_admission_closed_drain_unknown"),
])
def test_closed_or_unknown_drain_halts_without_inventing_process_death(changes, expected):
    state = {"admission_healthy": True, "server_drain_complete": True, "live_same_identity": True}
    state.update(changes)
    assert admission.classify_runtime_gate(**state) == expected


@pytest.mark.parametrize("field,value", [
    ("closed_admission_or_unknown_server_drain_is_global_runtime_gate_invalid", False),
    ("global_invalid_gate_requires_proven_process_death", True),
    ("one_source_timeout_alone_proves_process_death", True),
    ("continue_source_local_failures_only_if_positive_live_same_identity_admission_healthy", False),
    ("fully_finished_drained_invalid_semantics_is_source_local_if_gate_healthy", False),
    ("automatic_admission_restore_authorized", True), ("automatic_restart_authorized", True),
    ("automatic_source_or_model_retry_authorized", True),
    ("root_stop_verify_and_new_explicit_release_required_to_restore", False),
    ("all_failed_and_unprocessed_cases_stay_in_six_denominator", False),
    ("maximum_model_requests_per_arm", 24), ("maximum_source_pipeline_attempts_per_case", 2),
    ("source_attempts_required_per_arm", 1), ("parent_protocol_bytes_modified", True),
    ("ranking_or_labels_or_population_modified", True), ("final_check_used", True),
])
def test_addendum_rejects_widened_restart_retry_scope_and_changed_continuation(monkeypatch, field, value):
    protocol, addendum = prepared(monkeypatch)
    addendum[field] = value
    with pytest.raises(ValueError):
        admission.validate_admission_addendum(addendum, protocol, protocol_raw_sha256="b"*64)


def test_plan_and_root_release_bind_both_genuine_addendum_digests(monkeypatch):
    protocol, addendum = prepared(monkeypatch)
    plan = {"native_protocol": protocol, "native_protocol_file_sha256": "b"*64,
        "native_protocol_sha256": core.digest(protocol), "native_admission_addendum": addendum,
        "native_admission_addendum_sha256": core.digest(addendum), "native_admission_addendum_file_sha256": "c"*64}
    release = {key: plan[key] for key in ("native_admission_addendum_sha256", "native_admission_addendum_file_sha256")}
    assert admission.validate_plan_addendum(plan, release) is True
    changed = deepcopy(release); changed["native_admission_addendum_file_sha256"] = "d"*64
    with pytest.raises(ValueError, match="root_addendum_binding"):
        admission.validate_plan_addendum(plan, changed)
    changed = deepcopy(plan); changed.pop("native_admission_addendum_file_sha256")
    with pytest.raises(ValueError, match="addendum_unbound"):
        admission.validate_plan_addendum(changed, release)
    changed = deepcopy(plan); changed["native_protocol"]["selection"]["source_population_denominator"] = 5
    with pytest.raises(ValueError):
        admission.validate_plan_addendum(changed, release)


def test_echoed_new_file_digest_or_changed_timestamp_cannot_replace_frozen_addendum(monkeypatch):
    protocol, addendum = prepared(monkeypatch)
    plan = {"native_protocol": protocol, "native_protocol_file_sha256": "b"*64,
        "native_protocol_sha256": core.digest(protocol), "native_admission_addendum": addendum,
        "native_admission_addendum_sha256": core.digest(addendum), "native_admission_addendum_file_sha256": "d"*64}
    release = {key: plan[key] for key in ("native_admission_addendum_sha256", "native_admission_addendum_file_sha256")}
    with pytest.raises(ValueError, match="addendum_unbound"):
        admission.validate_plan_addendum(plan, release)
    plan["native_admission_addendum_file_sha256"] = "c"*64
    plan["native_admission_addendum"]["frozen_at_utc"] = "2099-01-01T00:00:00+00:00"
    plan["native_admission_addendum_sha256"] = core.digest(plan["native_admission_addendum"])
    release = {key: plan[key] for key in ("native_admission_addendum_sha256", "native_admission_addendum_file_sha256")}
    with pytest.raises(ValueError, match="addendum_unbound"):
        admission.validate_plan_addendum(plan, release)
