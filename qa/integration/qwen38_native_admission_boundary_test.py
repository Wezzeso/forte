"""Genuine pre-output addendum hashes and pure state classification only."""
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path

import pytest

from benchmarks.performance import qwen38_native_admission_addendum as admission

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks/private/qwen38/native-bf16-fullsix-a12"
RAW = "ade38909b0dbfec6987ecd6a7e69fd291bef1f7e592e4565dd8d5c9e2a754670"
SEMANTIC = "86fb6845a066b0744c11e25a7515cc2495e433685a14a139dbdce979289b714e"


def actual():
    protocol_raw = (BASE / "protocol.frozen.json").read_bytes()
    addendum_raw = (BASE / "runtime-admission-addendum.frozen.json").read_bytes()
    assert hashlib.sha256(protocol_raw).hexdigest() == admission.PROTOCOL_RAW_SHA256
    assert hashlib.sha256(addendum_raw).hexdigest() == RAW
    protocol, addendum = json.loads(protocol_raw), json.loads(addendum_raw)
    core = admission.native_contract().parent_contract()
    assert core.digest(addendum) == SEMANTIC
    return protocol, addendum


def bound():
    protocol, addendum = actual()
    plan = {"native_protocol": protocol, "native_protocol_file_sha256": admission.PROTOCOL_RAW_SHA256,
        "native_protocol_sha256": admission.PROTOCOL_SHA256, "native_admission_addendum": addendum,
        "native_admission_addendum_sha256": SEMANTIC, "native_admission_addendum_file_sha256": RAW}
    release = {key: plan[key] for key in ("native_admission_addendum_sha256", "native_admission_addendum_file_sha256")}
    return plan, release


def test_actual_addendum_preserves_original_protocol_and_has_no_runtime_or_retry_authority():
    plan, release = bound()
    before = (BASE / "protocol.frozen.json").read_bytes()
    assert admission.validate_plan_addendum(plan, release) is True
    data = plan["native_admission_addendum"]
    assert data["runtime_gate_measured_or_verified_by_this_addendum"] is False
    assert data["automatic_source_or_model_retry_authorized"] is False
    assert data["automatic_restart_authorized"] is False
    assert data["global_invalid_gate_requires_proven_process_death"] is False
    assert data["source_attempts_required_per_arm"] == 6 and data["maximum_model_requests_per_arm"] == 208
    assert (BASE / "protocol.frozen.json").read_bytes() == before


@pytest.mark.parametrize("admit,drain,identity", list(product((False, True), repeat=3)))
def test_all_three_positive_observations_are_required(admit, drain, identity):
    result = admission.classify_runtime_gate(admission_healthy=admit, server_drain_complete=drain,
        live_same_identity=identity)
    assert result == ("healthy_same_runtime" if admit and drain and identity else "runtime_admission_closed_drain_unknown")


@pytest.mark.parametrize("field", ["admission_healthy", "server_drain_complete", "live_same_identity"])
@pytest.mark.parametrize("value", [None, 1, "true", {}, []])
def test_truthy_or_missing_values_never_establish_a_positive_runtime_gate(field, value):
    state = {"admission_healthy": True, "server_drain_complete": True, "live_same_identity": True}
    state[field] = value
    assert admission.classify_runtime_gate(**state) == "runtime_admission_closed_drain_unknown"


@pytest.mark.parametrize("value", [None, 0, 1, "false", []])
def test_untyped_process_failure_does_not_invent_process_death(value):
    assert admission.classify_runtime_gate(admission_healthy=True, server_drain_complete=True,
        live_same_identity=True, confirmed_process_failure=value) == "runtime_admission_closed_drain_unknown"


@pytest.mark.parametrize("change", ["raw_echo", "timestamp_echo", "semantic_echo", "release_raw", "parent"])
def test_self_consistent_echoes_cannot_replace_the_actual_frozen_documents(change):
    plan, release = bound()
    plan = deepcopy(plan)
    if change == "raw_echo":
        plan["native_admission_addendum_file_sha256"] = release["native_admission_addendum_file_sha256"] = "a" * 64
    elif change == "timestamp_echo":
        plan["native_admission_addendum"]["frozen_at_utc"] = "2099-01-01T00:00:00Z"
        digest = admission.native_contract().parent_contract().digest(plan["native_admission_addendum"])
        plan["native_admission_addendum_sha256"] = release["native_admission_addendum_sha256"] = digest
    elif change == "semantic_echo":
        plan["native_admission_addendum_sha256"] = release["native_admission_addendum_sha256"] = "b" * 64
    elif change == "release_raw":
        release["native_admission_addendum_file_sha256"] = "c" * 64
    else:
        plan["native_protocol"]["maximum_source_pipeline_attempts_per_case"] = 2
    with pytest.raises(ValueError):
        admission.validate_plan_addendum(plan, release)
