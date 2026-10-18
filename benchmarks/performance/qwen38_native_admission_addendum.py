"""Immutable pre-output clarification of the native full-six runtime gate.

Admission closed/unknown drain is globally invalid even without process death.
This grants no restart, retry, source request, release or restoration authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

SCHEMA = "forte.qwen38-native-fullsix-admission-addendum.v1"
PROTOCOL_RAW_SHA256 = "2b5e22c90040084f694eab9d0dab675c9efdede8c200cdbf985a461160ef9ea3"
PROTOCOL_SHA256 = "f7566a354e514f29cbb75803d1ba11fe1d2b2ff30b5b7caa29823a23e3585477"
ADDENDUM_RAW_SHA256 = "ade38909b0dbfec6987ecd6a7e69fd291bef1f7e592e4565dd8d5c9e2a754670"
ADDENDUM_SHA256 = "86fb6845a066b0744c11e25a7515cc2495e433685a14a139dbdce979289b714e"


def native_contract():
    try:
        import native_fullsix_contract as module
    except ModuleNotFoundError as error:
        if error.name != "native_fullsix_contract":
            raise
        from benchmarks.performance import qwen38_native_fullsix as module
    return module


def freeze_addendum(protocol, *, protocol_raw_sha256):
    native = native_contract()
    core = native.parent_contract()
    native.validate_protocol(protocol)
    core.require(protocol_raw_sha256 == PROTOCOL_RAW_SHA256 and core.digest(protocol) == PROTOCOL_SHA256,
        "native_admission_original_frozen_protocol_required")
    addendum = {"schema_version": SCHEMA, "parent_protocol_raw_sha256": PROTOCOL_RAW_SHA256,
        "parent_protocol_sha256": PROTOCOL_SHA256, "kind": "runtime_admission_gate_clarification",
        "selection_criterion_sha256": core.digest(protocol["selection"]),
        "root_authorized_before_native_private_outputs": True,
        "native_private_outputs_observed_during_prepare": False,
        "closed_admission_or_unknown_server_drain_is_global_runtime_gate_invalid": True,
        "global_invalid_gate_requires_proven_process_death": False,
        "confirmed_process_failure_remains_global_halt": True,
        "global_invalid_gate_terminal_status": "runtime_admission_closed_drain_unknown",
        "unknown_drain_must_not_be_claimed_completed": True,
        "continue_source_local_failures_only_if_positive_live_same_identity_admission_healthy": True,
        "fully_finished_drained_invalid_semantics_is_source_local_if_gate_healthy": True,
        "one_source_timeout_alone_proves_process_death": False,
        "automatic_admission_restore_authorized": False, "automatic_restart_authorized": False,
        "automatic_source_or_model_retry_authorized": False,
        "halt_and_preserve_current_failure_and_unprocessed_sources": True,
        "root_stop_verify_and_new_explicit_release_required_to_restore": True,
        "restore_requires_genuine_drain_and_live_runtime_identity_verification": True,
        "new_release_must_account_all_prior_source_attempts_and_reserved_slots": True,
        "restoration_itself_authorizes_execution_or_reused_ids": False,
        "parent_protocol_bytes_modified": False, "ranking_or_labels_or_population_modified": False,
        "source_attempts_required_per_arm": 6, "maximum_model_requests_per_arm": 208,
        "maximum_execution_seconds_per_arm": 8220, "maximum_source_pipeline_attempts_per_case": 1,
        "execution_order": [5, 1, 2, 3, 4, 6], "all_failed_and_unprocessed_cases_stay_in_six_denominator": True,
        "ordinary_source_failures_continue_while_runtime_gate_healthy": True,
        "final_check_used": False, "private_model_calls_during_prepare": 0,
        "runtime_gate_measured_or_verified_by_this_addendum": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    validate_admission_addendum(addendum, protocol, protocol_raw_sha256=protocol_raw_sha256)
    return addendum


def validate_admission_addendum(addendum, protocol, *, protocol_raw_sha256):
    native = native_contract()
    core = native.parent_contract()
    native.validate_protocol(protocol)
    core.require(protocol_raw_sha256 == PROTOCOL_RAW_SHA256 and core.digest(protocol) == PROTOCOL_SHA256
        and addendum.get("schema_version") == SCHEMA and addendum.get("parent_protocol_raw_sha256") == PROTOCOL_RAW_SHA256
        and addendum.get("parent_protocol_sha256") == PROTOCOL_SHA256
        and addendum.get("selection_criterion_sha256") == core.digest(protocol["selection"]),
        "native_admission_original_protocol_or_selection_changed")
    exact = {"kind": "runtime_admission_gate_clarification", "root_authorized_before_native_private_outputs": True,
        "native_private_outputs_observed_during_prepare": False,
        "closed_admission_or_unknown_server_drain_is_global_runtime_gate_invalid": True,
        "global_invalid_gate_requires_proven_process_death": False,
        "confirmed_process_failure_remains_global_halt": True,
        "global_invalid_gate_terminal_status": "runtime_admission_closed_drain_unknown",
        "unknown_drain_must_not_be_claimed_completed": True,
        "continue_source_local_failures_only_if_positive_live_same_identity_admission_healthy": True,
        "fully_finished_drained_invalid_semantics_is_source_local_if_gate_healthy": True,
        "one_source_timeout_alone_proves_process_death": False,
        "automatic_admission_restore_authorized": False, "automatic_restart_authorized": False,
        "automatic_source_or_model_retry_authorized": False,
        "halt_and_preserve_current_failure_and_unprocessed_sources": True,
        "root_stop_verify_and_new_explicit_release_required_to_restore": True,
        "restore_requires_genuine_drain_and_live_runtime_identity_verification": True,
        "new_release_must_account_all_prior_source_attempts_and_reserved_slots": True,
        "restoration_itself_authorizes_execution_or_reused_ids": False,
        "parent_protocol_bytes_modified": False, "ranking_or_labels_or_population_modified": False,
        "source_attempts_required_per_arm": 6, "maximum_model_requests_per_arm": 208,
        "maximum_execution_seconds_per_arm": 8220, "maximum_source_pipeline_attempts_per_case": 1,
        "execution_order": [5, 1, 2, 3, 4, 6], "all_failed_and_unprocessed_cases_stay_in_six_denominator": True,
        "ordinary_source_failures_continue_while_runtime_gate_healthy": True,
        "final_check_used": False, "private_model_calls_during_prepare": 0,
        "runtime_gate_measured_or_verified_by_this_addendum": False}
    core.require(all(addendum.get(key) == value and type(addendum.get(key)) is type(value) for key, value in exact.items()),
        "native_admission_addendum_scope_or_restore_policy_changed")
    return addendum


def classify_runtime_gate(*, admission_healthy, server_drain_complete, live_same_identity,
                          confirmed_process_failure=False):
    """Pure classification of genuine caller observations, never a readiness proof."""
    if type(confirmed_process_failure) is not bool:
        return "runtime_admission_closed_drain_unknown"
    if confirmed_process_failure is True:
        return "confirmed_global_runtime_process_failure"
    if admission_healthy is True and server_drain_complete is True and live_same_identity is True:
        return "healthy_same_runtime"
    return "runtime_admission_closed_drain_unknown"


def validate_plan_addendum(plan, release=None):
    core = native_contract().parent_contract()
    protocol = plan["native_protocol"]
    addendum = plan["native_admission_addendum"]
    validate_admission_addendum(addendum, protocol, protocol_raw_sha256=plan["native_protocol_file_sha256"])
    core.require(plan.get("native_protocol_sha256") == PROTOCOL_SHA256
        and plan.get("native_admission_addendum_sha256") == core.digest(addendum) == ADDENDUM_SHA256
        and plan.get("native_admission_addendum_file_sha256") == ADDENDUM_RAW_SHA256,
        "native_admission_plan_addendum_unbound")
    if release is not None:
        core.require(release.get("native_admission_addendum_sha256") == plan["native_admission_addendum_sha256"]
            and release.get("native_admission_addendum_file_sha256") == plan["native_admission_addendum_file_sha256"],
            "native_admission_exact_root_addendum_binding_required")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    core = native_contract().parent_contract()
    try:
        addendum = freeze_addendum(core.read_json(args.protocol), protocol_raw_sha256=core.file_digest(args.protocol))
        core.write_new(args.output, addendum)
        print('{"status":"native_admission_addendum_frozen_no_runtime_authority","model_calls_made":0}')
        return 0
    except Exception:
        print('{"status":"native_admission_addendum_invalid","model_calls_made":0}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
