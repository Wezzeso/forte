"""Separate native BF16 full-six protocol, metadata only and never authority.

No runtime placeholders, providers, media, labels, uploads or inference. The
historical HF/NF4 pilots and their failures remain separate immutable records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone

SCHEMA = "forte.qwen38-native-bf16-fullsix-protocol.v1"
KIND = "native_bf16_fullsix_accuracy"
ARMS = ["qwen38_untuned", "qwen38_tuned"]
ORDER = [5, 1, 2, 3, 4, 6]
CAPS = [1980, 1080, 1080, 1080, 960, 2040]
PARENT_PROTOCOL_SHA256 = "850fdb4837d549845104fee7984d44588196965396f4a0e034e72e3c8fcdfd9c"
SELECTION_POLICY_RAW_SHA256 = "7dd16f74490bb1d4076d67b6bf99336d976839e32344ff4fec941a71cbf16944"
SELECTION_POLICY_SHA256 = "71f963b2bfe74165b34b0815e63c6961ce67e3ee83dab34d25a85213c5aebc05"
METRIC_DEFINITIONS_SHA256 = "917253b155cfafe04825aec504305ab9d4e34e9b6b6bacc25e9efd6d047c7bbc"
RANKING = ["failure_inclusive_correct_over_six", "failure_inclusive_positive_recall", "precision",
    "complete_evidence_valid_case_count", "median_pipeline_start_to_result_seconds", "exact_remaining_tie"]


def parent_contract():
    try:
        import accuracy_diagnostic_contract as module
    except ModuleNotFoundError as error:
        if error.name != "accuracy_diagnostic_contract":
            raise
        from benchmarks.performance import qwen38_accuracy_diagnostic as module
    return module


def validate_selection(selection):
    contract = parent_contract()
    ranking = selection["ranking_lexicographic"]
    contract.require([row.get("metric") for row in ranking] == RANKING
        and [row.get("direction") for row in ranking[:5]] == ["higher", "higher", "higher", "higher", "lower"]
        and ranking[1].get("denominator") == 3
        and ranking[2].get("undefined_rule") == "Keep null; ranks below any defined numeric value for this tie-break."
        and ranking[4].get("available_only_if") == "Both arms have all six genuine complete/evidence-valid cases and valid monotonic timings for all six; otherwise omit this tie-break."
        and ranking[5].get("winner") == "qwen38_untuned", "native_fullsix_preoutcome_selection_changed")
    contract.require(selection["source_population_denominator"] == 6
        and selection["provided_positive_denominator"] == 3
        and selection["threshold_or_source_label_changes_allowed"] is False
        and selection["automatic_promotion_authorized"] is False,
        "native_fullsix_selection_denominator_or_tuning_changed")
    contract.require(contract.digest(selection["metrics"]) == METRIC_DEFINITIONS_SHA256,
        "native_fullsix_metric_definitions_changed")


def freeze_protocol(parent, policy, *, policy_raw_sha256):
    contract = parent_contract()
    contract.validate_protocol(parent)
    contract.require(policy.get("schema_version") == "forte.qwen38-accuracy-selection-policy.v2"
        and policy.get("comparison_arms") == ARMS and policy.get("source_population_denominator") == 6
        and policy.get("diagnostic_protocol_semantic_sha256") == contract.digest(parent)
        and policy.get("no_model_outputs_or_verdicts_read_for_policy_freeze") is True
        and policy.get("final_check_used") is False and contract._sha(policy_raw_sha256),
        "native_fullsix_authoritative_selection_reference_invalid")
    contract.require(contract.digest(parent) == PARENT_PROTOCOL_SHA256
        and policy_raw_sha256 == SELECTION_POLICY_RAW_SHA256 and contract.digest(policy) == SELECTION_POLICY_SHA256,
        "native_fullsix_exact_preexisting_population_and_selection_required")
    selection = {"ranking_lexicographic": deepcopy(policy["ranking_lexicographic"]),
        "metrics": deepcopy(policy["metrics"]), "source_population_denominator": 6,
        "provided_positive_denominator": 3, "threshold_or_source_label_changes_allowed": False,
        "automatic_promotion_authorized": False}
    protocol = {"schema_version": SCHEMA, "name": "native-bf16-full-six-v1", "kind": KIND,
        "phase": "evaluation", "parent_population_protocol": deepcopy(parent),
        "parent_population_protocol_sha256": contract.digest(parent),
        "population": deepcopy(parent["population"]), "provided_labels_sha256": parent["provided_labels_sha256"],
        "coarse_anchors_sha256": parent["coarse_anchors_sha256"],
        "historical_family_sha256": parent["historical_family_sha256"],
        "historical_recipe": deepcopy(parent["historical_recipe"]),
        "comparison_arms": ARMS.copy(), "arm_display_names": {"qwen38_untuned": "native BF16 base", "qwen38_tuned": "native BF16 FIT"},
        "execution_order": ORDER.copy(), "deadlines_by_original_ordinal_seconds": CAPS.copy(),
        "source_attempts_required_per_arm": 6, "source_attempts_required_total": 12,
        "maximum_model_requests_per_arm": 208, "maximum_execution_seconds_per_arm": 8220,
        "minimum_initial_generation_requests_per_arm": 56, "provider_max_attempts": 2,
        "maximum_source_pipeline_attempts_per_case": 1,
        "release_scope": "all_six_validation_sources_per_arm",
        "run_all_sources_after_source_failure": True, "pilot_advancement_required": False,
        "reference_or_verdict_agreement_used_for_execution_order_or_continuation": False,
        "ordinary_source_failures_continue": ["source_timeout", "invalid_model_semantics", "incomplete_coverage",
            "evidence_or_publication_failure", "source_decode_failure", "source_request_failure"],
        "global_halt_conditions": ["exact_root_release_or_reservation_expired_or_invalid", "hard_rental_stop",
            "full_arm_wall_or_reserved_post_limit_reached", "actual_runtime_identity_or_config_or_profile_changed",
            "confirmed_global_runtime_process_failure"],
        "single_source_failure_is_global_runtime_failure": False,
        "native_arithmetic_policy": {"base_weight_precision": "bfloat16", "base_quantization": "none",
            "base_and_fit_actual_profile_must_match": True, "actual_adapter_identity_bound_separately": True,
            "reuse_nf4_runtime_identity_or_receipt_forbidden": True},
        "runtime_binding_status": "pending_actual_native_GPU6_and_HTTP_gates",
        "runtime_prerequisites": ["actual_native_generated_GPU6_gate_passed", "actual_native_HTTP_gate_passed",
            "genuine_matched_base_and_FIT_native_runtime_profiles", "actual_model_adapter_source_image_config_and_base_pins",
            "new_exact_root_fullsix_release_per_arm", "fresh_A09_role_preflight_and_exact208_reservation"],
        "selection": selection, "selection_reference_raw_sha256": policy_raw_sha256,
        "selection_reference_semantic_sha256": contract.digest(policy),
        "frozen_before_any_native_private_outputs": True,
        "native_private_outputs_observed_during_prepare": False,
        "all_failed_missing_invalid_or_halted_sources_remain_in_six_denominator": True,
        "historical_production_deadline_seconds": 285, "record_actual_production_deadline_misses": True,
        "upload_or_usable_UI_latency_claim": False, "historical_HF_NF4_protocols_or_failures_modified": False,
        "is_HF_pilot_expansion_or_retry": False, "validation_is_reused_development": True,
        "final_check_used": False, "final_check_authorized": False, "media_or_labels_read_during_prepare": False,
        "private_model_calls_during_prepare": 0, "execution_authorized_by_protocol": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    contract = parent_contract()
    parent = contract.validate_protocol(protocol["parent_population_protocol"])
    contract.require(protocol.get("schema_version") == SCHEMA and protocol.get("name") == "native-bf16-full-six-v1"
        and protocol.get("kind") == KIND and protocol.get("phase") == "evaluation"
        and protocol["parent_population_protocol_sha256"] == contract.digest(parent)
        and protocol["parent_population_protocol_sha256"] == PARENT_PROTOCOL_SHA256
        and protocol["population"] == parent["population"]
        and protocol["provided_labels_sha256"] == parent["provided_labels_sha256"]
        and protocol["coarse_anchors_sha256"] == parent["coarse_anchors_sha256"]
        and protocol["historical_family_sha256"] == parent["historical_family_sha256"]
        and protocol["historical_recipe"] == parent["historical_recipe"], "native_fullsix_parent_population_or_recipe_changed")
    exact = {"comparison_arms": ARMS, "execution_order": ORDER,
        "deadlines_by_original_ordinal_seconds": CAPS, "source_attempts_required_per_arm": 6,
        "source_attempts_required_total": 12, "maximum_model_requests_per_arm": 208,
        "maximum_execution_seconds_per_arm": 8220, "minimum_initial_generation_requests_per_arm": 56,
        "provider_max_attempts": 2, "maximum_source_pipeline_attempts_per_case": 1,
        "release_scope": "all_six_validation_sources_per_arm", "run_all_sources_after_source_failure": True,
        "pilot_advancement_required": False,
        "reference_or_verdict_agreement_used_for_execution_order_or_continuation": False,
        "single_source_failure_is_global_runtime_failure": False,
        "runtime_binding_status": "pending_actual_native_GPU6_and_HTTP_gates",
        "frozen_before_any_native_private_outputs": True, "native_private_outputs_observed_during_prepare": False,
        "all_failed_missing_invalid_or_halted_sources_remain_in_six_denominator": True,
        "historical_production_deadline_seconds": 285, "record_actual_production_deadline_misses": True,
        "upload_or_usable_UI_latency_claim": False, "historical_HF_NF4_protocols_or_failures_modified": False,
        "is_HF_pilot_expansion_or_retry": False, "validation_is_reused_development": True,
        "final_check_used": False, "final_check_authorized": False, "media_or_labels_read_during_prepare": False,
        "private_model_calls_during_prepare": 0, "execution_authorized_by_protocol": False}
    contract.require(all(protocol.get(key) == value and type(protocol.get(key)) is type(value) for key, value in exact.items()),
        "native_fullsix_scope_budget_continuation_or_claim_changed")
    contract.require(protocol["native_arithmetic_policy"] == {"base_weight_precision": "bfloat16", "base_quantization": "none",
        "base_and_fit_actual_profile_must_match": True, "actual_adapter_identity_bound_separately": True,
        "reuse_nf4_runtime_identity_or_receipt_forbidden": True}, "native_fullsix_distinct_matched_BF16_policy_required")
    contract.require(protocol["ordinary_source_failures_continue"] == ["source_timeout", "invalid_model_semantics", "incomplete_coverage",
        "evidence_or_publication_failure", "source_decode_failure", "source_request_failure"]
        and protocol["global_halt_conditions"] == ["exact_root_release_or_reservation_expired_or_invalid", "hard_rental_stop",
            "full_arm_wall_or_reserved_post_limit_reached", "actual_runtime_identity_or_config_or_profile_changed",
            "confirmed_global_runtime_process_failure"], "native_fullsix_global_halt_policy_changed")
    contract.require(protocol["runtime_prerequisites"] == ["actual_native_generated_GPU6_gate_passed", "actual_native_HTTP_gate_passed",
        "genuine_matched_base_and_FIT_native_runtime_profiles", "actual_model_adapter_source_image_config_and_base_pins",
        "new_exact_root_fullsix_release_per_arm", "fresh_A09_role_preflight_and_exact208_reservation"]
        and protocol["selection_reference_raw_sha256"] == SELECTION_POLICY_RAW_SHA256
        and protocol["selection_reference_semantic_sha256"] == SELECTION_POLICY_SHA256
        and not any(key in protocol for key in ("runtime_bindings", "expected_model_identity", "native_runtime_profile_sha256")),
        "native_fullsix_actual_gate_prerequisites_or_placeholder_runtime_invalid")
    validate_selection(protocol["selection"])
    return protocol


def case_budgets(protocol):
    validate_protocol(protocol)
    parent = parent_contract()
    return [{"original_case_ordinal": ordinal, "initial_requests": parent.WINDOWS[ordinal - 1],
        "maximum_pipeline_inferences": parent.WINDOWS[ordinal - 1] + 8,
        "maximum_model_requests": (parent.WINDOWS[ordinal - 1] + 8) * 2,
        "diagnostic_deadline_seconds": CAPS[ordinal - 1]} for ordinal in ORDER]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-protocol", "selection-policy", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    contract = parent_contract()
    try:
        protocol = freeze_protocol(contract.read_json(args.parent_protocol), contract.read_json(args.selection_policy),
            policy_raw_sha256=contract.file_digest(args.selection_policy))
        contract.write_new(args.output, protocol)
        print('{"status":"native_fullsix_protocol_frozen_runtime_pending","model_calls_made":0,"private_values_printed":false}')
        return 0
    except Exception:
        print('{"status":"native_fullsix_metadata_invalid","model_calls_made":0,"private_values_printed":false}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
