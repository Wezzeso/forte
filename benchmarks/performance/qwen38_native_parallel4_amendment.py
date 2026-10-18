"""Root's immutable pre-output parallel-four policy; no profile selection yet."""
from __future__ import annotations

RAW_SHA256 = "d72c3fc0f1f88ea5cb607d7a9025949dab4110b3f4b1a8c0493c429f5f06d870"
SHA256 = "b72cb8c97b7a5656c3644ca693dc9aa29879a1763fb408facd5c316984831c18"


def validate_amendment(document, *, raw_sha256, protocol, addendum):
    try:
        import native_fullsix_contract as native
        import native_admission_contract as admission
    except ModuleNotFoundError:
        from benchmarks.performance import qwen38_native_fullsix as native
        from benchmarks.performance import qwen38_native_admission_addendum as admission
    core = native.parent_contract()
    native.validate_protocol(protocol)
    admission.validate_admission_addendum(addendum, protocol, protocol_raw_sha256=admission.PROTOCOL_RAW_SHA256)
    core.require(raw_sha256 == RAW_SHA256 and core.digest(document) == SHA256
        and document.get("schema_version") == "forte.qwen38-native-fullsix-profile-amendment.v1"
        and document.get("released_by_root") is True and document.get("frozen_before_any_native_private_outputs") is True
        and document.get("native_private_outputs_observed") is False
        and document.get("native_protocol_file_sha256") == admission.PROTOCOL_RAW_SHA256
        and document.get("native_protocol_sha256") == core.digest(protocol)
        and document.get("native_admission_addendum_file_sha256") == admission.ADDENDUM_RAW_SHA256
        and document.get("native_admission_addendum_sha256") == core.digest(addendum)
        and document.get("selection_criterion_sha256") == core.digest(protocol["selection"]),
        "native_parallel4_exact_preoutput_amendment_required")
    core.require(document.get("maximum_coarse_inflight") == 4 and document.get("maximum_refinement_inflight") == 1
        and document.get("coarse_result_order") == "original_window_index"
        and document.get("source_attempts_required_per_arm") == 6 and document.get("maximum_model_requests_per_arm") == 208
        and document.get("maximum_execution_seconds_per_arm") == 8220 and document.get("execution_order") == [5,1,2,3,4,6]
        and document.get("public_benchmark_then_explicit_root_profile_selection_required") is True
        and document.get("profile_selected_for_private_execution") is False
        and document.get("execution_authorized_by_amendment") is False
        and document.get("serial_or_offline_V2_proof_substitution_allowed") is False
        and document.get("automatic_retry_authorized") is False and document.get("final_check_used") is False,
        "native_parallel4_amendment_execution_or_scope_widening")
    return document
