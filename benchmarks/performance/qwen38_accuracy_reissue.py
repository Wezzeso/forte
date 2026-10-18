"""One root-authorized control reissue, only after proven pre-pipeline 0 POSTs.

Pure standard-library metadata. Old exposure is never refunded; this is not a
model/source retry. Actual plan/AST creation must run in the selected CPU image.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

SCHEMA = "forte.qwen38-accuracy-control-reissue.v1"
FAILURE_CODE = "quality_equivalence_timeout_proof_changed"
ARTIFACTS = {"prior_plan", "prior_source_terminals", "prior_model_slot_terminals",
    "prior_reconciliation", "prior_settlement", "root_authorization"}


def contract():
    try:
        import accuracy_diagnostic_contract as module
    except ModuleNotFoundError as error:
        if error.name != "accuracy_diagnostic_contract":
            raise
        from benchmarks.performance import qwen38_accuracy_diagnostic as module
    return module


def _require(value, code):
    if not value:
        raise ValueError(code)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _integers(document, expected):
    return all(type(document.get(key)) is int and document[key] == value for key, value in expected.items())


def prepare_control_reissue(**paths):
    """Hash real preserved artifacts; no model, media or remote operation."""
    _require(sys.version_info[:2] == (3, 12), "control_reissue_creation_requires_actual_cpu_python312")
    _require(set(paths) == ARTIFACTS, "control_reissue_exact_original_artifacts_required")
    artifacts = {}
    for name, path in paths.items():
        path = Path(path)
        _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024 * 1024,
            "control_reissue_original_metadata_unavailable")
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8-sig"))
        artifacts[name] = {"raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_utf8": raw.decode("utf-8"), "semantic_sha256": _digest(document), "document": document}
    proof = {"schema_version": SCHEMA, "operational_control_attempt": 2,
        "prepared_python_version": "3.12",
        "prior_control_attempt": 1, "old_exposure_consumed_without_refund": True,
        "maximum_actual_source_pipeline_attempts": 1, "automatic_retry_authorized": False,
        "artifacts": artifacts}
    validate_control_reissue(proof)
    return proof


def validate_control_reissue(proof):
    _require(proof.get("schema_version") == SCHEMA and type(proof.get("operational_control_attempt")) is int
        and proof.get("prepared_python_version") == "3.12"
        and proof["operational_control_attempt"] == 2 and proof.get("prior_control_attempt") == 1
        and proof.get("old_exposure_consumed_without_refund") is True
        and proof.get("maximum_actual_source_pipeline_attempts") == 1
        and proof.get("automatic_retry_authorized") is False and set(proof["artifacts"]) == ARTIFACTS,
        "control_reissue_only_one_preexecution_reissue_allowed")
    _require(_integers(proof, {"operational_control_attempt": 2, "prior_control_attempt": 1,
        "maximum_actual_source_pipeline_attempts": 1}), "control_reissue_exact_integer_counters_required")
    artifacts = proof["artifacts"]
    for value in artifacts.values():
        _require(set(value) == {"raw_sha256", "raw_utf8", "semantic_sha256", "document"}
            and isinstance(value["raw_utf8"], str) and len(value["raw_utf8"].encode("utf-8")) <= 32 * 1024 * 1024
            and _sha(value["raw_sha256"]) and hashlib.sha256(value["raw_utf8"].encode("utf-8")).hexdigest() == value["raw_sha256"]
            and json.loads(value["raw_utf8"].lstrip("\ufeff")) == value["document"]
            and value["semantic_sha256"] == _digest(value["document"]),
            "control_reissue_artifact_digest_invalid")
    old = artifacts["prior_plan"]["document"]
    _require(type(old.get("operational_control_attempt", 1)) is int and old.get("operational_control_attempt", 1) == 1 and "control_reissue" not in old
        and old.get("diagnostic_stage") == "pilot", "control_reissue_prior_must_be_original_pilot")
    contract().validate_diagnostic_plan(old)
    old_sha = artifacts["prior_plan"]["raw_sha256"]
    sources = artifacts["prior_source_terminals"]["document"]["records"]
    slots = artifacts["prior_model_slot_terminals"]["document"]["records"]
    _require(len(sources) == 6 and len({row["attempt_id"] for row in sources}) == 6
        and {row["attempt_id"] for row in sources} == {row["attempt_id"] for row in old["source_attempts"]},
        "control_reissue_old_six_terminal_denominator_required")
    by_id = {row["attempt_id"]: row for row in old["source_attempts"]}
    pilot_id = old["cases"][0]["attempt_id"]
    for row in sources:
        _require(all(row.get(key) == value for key, value in by_id[row["attempt_id"]].items())
            and _integers(row, {"model_posts": 0}) and row.get("source_pipeline_started") is False
            and row.get("public_contract_validated") is False and row.get("server_receipt_verified") is False
            and all(row.get(key) is None for key in ("job_id", "model_identity", "complete_result", "technical_completion",
                "server_receipt_sha256", "pipeline_start_to_result_ms")), "control_reissue_prior_pipeline_or_model_execution_forbidden")
        if row["attempt_id"] == pilot_id:
            _require(row.get("status") == "failed" and row.get("failure_stage") == "pre_execution_validation"
                and row.get("static_failure_code") == FAILURE_CODE and row.get("error_type") == "ValueError"
                and row.get("release_state") == "released", "control_reissue_exact_preexecution_failure_required")
        else:
            _require(row.get("status") == "not_attempted" and row.get("release_state") == "not_released",
                "control_reissue_unreleased_source_was_attempted")
    _require(slots == [{**slot, "status": "not_attempted"} for slot in old["attempts"]] and len(slots) == 24,
        "control_reissue_zero_post_slots_required")
    summary = artifacts["prior_reconciliation"]["document"]
    _require(_integers(summary, {"actual_model_posts": 0, "source_failed_control": 1, "source_not_released": 5,
        "source_attempts_required": 6, "model_slots": 24}), "control_reissue_exact_integer_counters_required")
    _require(summary.get("schema_version") == "forte.qwen38-accuracy-control-failure-reconciliation.v1"
        and summary.get("status") == "terminal_control_failure_before_pipeline"
        and summary.get("plan_raw_sha256") == old_sha and summary.get("actual_model_posts") == 0
        and summary.get("source_pipeline_started") is False and summary.get("source_failed_control") == 1
        and summary.get("source_not_released") == 5 and summary.get("source_attempts_required") == 6
        and summary.get("static_failure_code") == FAILURE_CODE and summary.get("model_slots") == 24
        and summary.get("model_slot_terminal_counts") == {"not_attempted": 24}
        and summary.get("unused_slots_refunded") is False and summary.get("past_stage_witness_inferred") is False
        and summary.get("existing_claims_or_results_modified") is False
        and summary.get("source_terminals_raw_sha256") == artifacts["prior_source_terminals"]["raw_sha256"]
        and summary.get("model_slot_terminals_raw_sha256") == artifacts["prior_model_slot_terminals"]["raw_sha256"],
        "control_reissue_actual_zero_post_reconciliation_required")
    settled = artifacts["prior_settlement"]["document"]
    state = settled.get("actual_helper_terminal_state", {})
    reconciliation = settled.get("reconciliation", {})
    hashes = settled.get("host_original_metadata_sha256", {})
    _require(_integers(settled, {"reserved_attempts": 24, "actual_model_posts": 0, "actual_video_attempts": 0,
        "source_failed_control": 1, "source_not_released": 5, "source_population_denominator": 6})
        and _integers(state, {"ExitCode": 1})
        and _integers(reconciliation, {"reserved_attempts": 24, "recorded_attempts": 24, "unresolved_attempts": 0}),
        "control_reissue_exact_integer_counters_required")
    _require(settled.get("schema_version") == "forte.qwen38-accuracy-pilot-control-settlement.v1"
        and settled.get("status") == "spent_control_failed_before_pipeline"
        and settled.get("attempt_plan_sha256") == old_sha and settled.get("reserved_attempts") == 24
        and settled.get("actual_model_posts") == settled.get("actual_video_attempts") == 0
        and settled.get("source_pipeline_started") is False and settled.get("all_slots_consumed") is True
        and settled.get("exposure_refunded") is False and settled.get("retry_authorized") is False
        and settled.get("new_attempt_authority") is False and settled.get("source_failed_control") == 1
        and settled.get("source_not_released") == 5 and settled.get("source_population_denominator") == 6
        and settled.get("static_failure_code") == FAILURE_CODE and settled.get("outcomes") == slots
        and state.get("Running") is False and state.get("Status") == "exited" and state.get("ExitCode") == 1
        and state.get("OOMKilled") is False and reconciliation.get("complete") is True
        and reconciliation.get("reserved_attempts") == reconciliation.get("recorded_attempts") == 24
        and reconciliation.get("unresolved_attempts") == 0 and reconciliation.get("status_counts") == {"not_attempted": 24}
        and reconciliation.get("exposure_reservations_refunded") is False and reconciliation.get("retry_authorized") is False
        and settled.get("attempt_reservation_sha256") == summary.get("reservation_raw_sha256")
        and settled.get("root_release_sha256") == summary.get("release_raw_sha256")
        and hashes.get("old-helper.summary.json.original") == artifacts["prior_reconciliation"]["raw_sha256"]
        and hashes.get("old-helper.source-terminals.json.original") == artifacts["prior_source_terminals"]["raw_sha256"]
        and hashes.get("old-helper.model-slot-terminals.json.original") == artifacts["prior_model_slot_terminals"]["raw_sha256"],
        "control_reissue_old24_spent_settlement_required")
    authority = artifacts["root_authorization"]["document"]
    _require(_integers(authority, {"operational_control_attempt": 2, "maximum_new_model_requests": 24,
        "maximum_execution_seconds": 960, "maximum_operational_reissues": 1, "actual_source_pipeline_attempts_authorized": 1}),
        "control_reissue_exact_integer_counters_required")
    _require(authority.get("schema_version") == "forte.qwen38-accuracy-control-reissue-authorization.v1"
        and authority.get("released_by_root") is True and authority.get("operational_control_attempt") == 2
        and authority.get("prior_plan_sha256") == old_sha
        and authority.get("prior_reconciliation_sha256") == artifacts["prior_reconciliation"]["raw_sha256"]
        and authority.get("prior_settlement_sha256") == artifacts["prior_settlement"]["raw_sha256"]
        and authority.get("maximum_new_model_requests") == 24 and authority.get("maximum_execution_seconds") == 960
        and authority.get("maximum_operational_reissues") == 1
        and authority.get("actual_source_pipeline_attempts_authorized") == 1,
        "control_reissue_exact_root_one_reissue_authorization_required")
    return _digest(proof)


def _without_python_ast_receipt(value):
    """Only operational AST receipt metadata; never normalize model results."""
    result = deepcopy(value)
    result.pop("receipt_sha256", None)
    result.pop("source_change_disclosure", None)
    for row in result.get("audited_common_source_deltas", []):
        row.get("proof", {}).pop("unchanged_ast_sha256", None)
    return result


def validate_reissue_plan_binding(plan):
    proof = plan["control_reissue"]
    proof_sha = validate_control_reissue(proof)
    _require(plan.get("operational_control_attempt") == 2 and plan.get("control_reissue_sha256") == proof_sha,
        "control_reissue_plan_namespace_unbound")
    old = proof["artifacts"]["prior_plan"]["document"]
    keys = ("arm", "diagnostic_protocol_sha256", "recipe_sha256", "expected_model_identity", "runtime_bindings",
        "provider", "worker", "role_policy", "comparison_runtime_profile", "comparison_runtime_profile_sha256",
        "effective_client_request_timeout_seconds", "evaluator_source_archive_sha256", "serving_source_archive_sha256")
    _require(all((key in plan) == (key in old) and plan.get(key) == old.get(key) for key in keys),
        "control_reissue_actual_runtime_or_protocol_changed")
    source_keys = ("arm", "sample_id", "content_sha256", "split", "repeat_index", "owned_path", "source_bytes", "duration_sec",
        "coarse_windows", "maximum_pipeline_inferences", "maximum_model_requests", "original_case_ordinal", "diagnostic_deadline_seconds")
    _require(len(plan["protocol_cases"]) == 6 and all(all(current[key] == previous[key] for key in source_keys)
        for current, previous in zip(plan["protocol_cases"], old["protocol_cases"])), "control_reissue_source_population_or_geometry_changed")
    _require(not ({row["attempt_id"] for row in plan["source_attempts"]} & {row["attempt_id"] for row in old["source_attempts"]})
        and not ({row["attempt_id"] for row in plan["attempts"]} & {row["attempt_id"] for row in old["attempts"]}),
        "control_reissue_spent_identity_reused")
    if "source_equivalence" in old or "source_equivalence" in plan:
        _require(_without_python_ast_receipt(plan["source_equivalence"]) == _without_python_ast_receipt(old["source_equivalence"]),
            "control_reissue_source_receipt_changed_beyond_python_ast_metadata")
    _require(_without_python_ast_receipt({"audited_common_source_deltas": plan.get("audited_common_source_deltas", [])})
        == _without_python_ast_receipt({"audited_common_source_deltas": old.get("audited_common_source_deltas", [])}),
        "control_reissue_audited_source_delta_summary_changed")
    _require(_integers(plan, {"historical_control_slots_consumed": 24, "maximum_actual_model_posts_per_arm": 208,
        "maximum_reserved_model_slots_including_failed_control_per_arm": 232,
        "maximum_actual_source_pipeline_attempts_per_case": 1}), "control_reissue_exact_integer_counters_required")
    _require(plan.get("historical_control_slots_consumed") == 24
        and plan.get("maximum_actual_model_posts_per_arm") == 208
        and plan.get("maximum_reserved_model_slots_including_failed_control_per_arm") == 232
        and plan.get("maximum_actual_source_pipeline_attempts_per_case") == 1,
        "control_reissue_budget_disclosure_required")
    return proof_sha


def validate_reissue_release(plan, release):
    if plan.get("operational_control_attempt", 1) == 1:
        _require(not any(key in release for key in ("operational_control_reissue_authorized_by_root", "operational_control_attempt",
            "prior_settlement_sha256", "control_reissue_sha256")), "control_reissue_release_without_reissue_plan")
        return True
    proof_sha = validate_reissue_plan_binding(plan)
    _require(release.get("operational_control_reissue_authorized_by_root") is True
        and release.get("operational_control_attempt") == 2
        and release.get("prior_settlement_sha256") == plan["control_reissue"]["artifacts"]["prior_settlement"]["raw_sha256"]
        and release.get("control_reissue_sha256") == proof_sha,
        "control_reissue_fresh_exact_root_release_required")
    return True
