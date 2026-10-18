"""Metadata-only, separately versioned accuracy protocol; never opens media/models.

The historical 285-second family remains immutable. A pilot reserves only its
24 POST slots; the other five cases stay in the declared six-case denominator.
This module grants no execution authority. Exact root/A09 gates remain required.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys

PROTOCOL_SCHEMA = "forte.qwen38-accuracy-diagnostic-protocol.v1"
PLAN_SCHEMA = "forte.qwen38-attempt-plan.v1"
ORDER = [5, 1, 2, 3, 4, 6]
WINDOWS = [18, 5, 5, 5, 4, 19]
DURATIONS = [72.7, 20.866667, 23.8, 22.6, 17.3, 78.733021]
DEADLINES = [1980, 1080, 1080, 1080, 960, 2040]
IDENTITY_FIELDS = ("arm", "sample_id", "content_sha256", "split", "repeat_index")
PIN_FIELDS = ("source_archive_sha256", "runtime_image_sha256", "base_manifest_sha256")
MODEL_FIELDS = ("model_id", "model_revision", "adapter_sha256")
ARMS = {"baseline_8b", "qwen38_untuned", "qwen38_tuned"}
ROLE_PINS = ("expected_assignments_sha256", "expected_source_manifest_sha256",
    "expected_automated_split_sha256", "expected_scenario_groups_sha256")


def require(value, code):
    if not value:
        raise ValueError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_records(path):
    value = read_json(path)
    rows = value.get("records") if isinstance(value, dict) else value
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), "diagnostic_terminal_records_required")
    return rows


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8", newline="\n") as target:
        path.chmod(0o600)
        json.dump(value, target, ensure_ascii=False, sort_keys=True, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _recipe(protocol):
    return {**deepcopy(protocol["historical_recipe"]), "accuracy_diagnostic": {
        "protocol_sha256": digest(protocol), "execution_order": ORDER.copy(),
        "deadlines_by_original_ordinal_seconds": DEADLINES.copy(),
        "effective_job_timeout_policy": "explicit_per_case_diagnostic_deadline_seconds",
        "source_attempts_per_case": 1, "advance_policy": "technical_completion_independent_of_verdict_agreement",
        "speed_benchmark_eligible": False}}


def _worker_geometry(worker, case, ordinal):
    require(worker["window_seconds"] == 8 and worker["stride_seconds"] == 4
        and worker["sample_fps"] == 2 and worker["max_frames_per_window"] == 16
        and worker["max_refinement_passes"] == 8 and worker["job_timeout_seconds"] == 285
        and worker["max_concurrent_jobs"] == worker["max_concurrent_clips"] == worker["max_inflight_per_job"] == 1
        and worker["staged_screening"] is False, "diagnostic_historical_worker_changed")
    require(type(case["duration_sec"]) in {float, int} and math.isfinite(case["duration_sec"])
        and abs(case["duration_sec"] - DURATIONS[ordinal - 1]) < 1e-6,
        "diagnostic_original_duration_or_order_changed")
    # Fixed durations are all >8 seconds. End-aligned tail yields ceil((D-8)/4)+1.
    # Tests compare this metadata formula against the unchanged frozen sampler.
    count = math.ceil((case["duration_sec"] - 8.) / 4.) + 1
    require(case["coarse_windows"] == count == WINDOWS[ordinal - 1], "diagnostic_window_geometry_changed")


def freeze_protocol(family, historical_plan):
    """Freeze the same population/reference pins and geometry without media access."""
    # Creation runs in the development/CPU-image environment. Host-side plan
    # validation deliberately imports no runtime dependencies.
    from evaluation.qwen38_quality_family import validate_family
    records, _ = validate_family(family)
    require(historical_plan["comparison_plan_sha256"] == digest(family)
        and historical_plan["recipe"] == family["recipe"]
        and historical_plan["recipe_sha256"] == family["recipe_sha256"]
        and historical_plan["worker"] == family["recipe"]["worker_settings"]
        and len(historical_plan["cases"]) == 6, "diagnostic_historical_family_changed")
    population = []
    for ordinal, case in enumerate(historical_plan["cases"], 1):
        record = records[case["sample_id"]]
        require(case["content_sha256"] == record["content_sha256"] and case["split"] == record["split"] == "validation"
            and case["repeat_index"] == 0, "diagnostic_historical_source_changed")
        _worker_geometry(historical_plan["worker"], case, ordinal)
        population.append({key: case[key] for key in ("sample_id", "content_sha256", "split", "duration_sec", "coarse_windows", "source_bytes")})
    require({row["sample_id"] for row in population} == set(records), "diagnostic_exact_six_population_required")
    protocol = {"schema_version": PROTOCOL_SCHEMA, "name": "accuracy-diagnostic-v1",
        "historical_family": deepcopy(family),
        "historical_family_sha256": digest(family), "historical_recipe_sha256": family["recipe_sha256"],
        "historical_recipe": deepcopy(family["recipe"]), "historical_base_models": deepcopy(family["base_models"]),
        "provided_labels_sha256": family["provided_labels_sha256"],
        "coarse_anchors_sha256": family.get("coarse_anchors_sha256"), "population": population,
        "execution_order": ORDER.copy(), "deadlines_by_original_ordinal_seconds": DEADLINES.copy(),
        "source_attempts_required": 6, "maximum_model_requests_per_arm": 208,
        "maximum_execution_seconds_per_arm": 8220, "provider_max_attempts": 2,
        "maximum_source_attempts_per_case": 1, "pilot_original_case_ordinal": 5,
        "pilot_maximum_model_requests": 24, "pilot_maximum_execution_seconds": 960,
        "remaining_maximum_model_requests": 184, "remaining_maximum_execution_seconds": 7260,
        "advance_requires_technical_completion": True, "advance_uses_provided_verdict_agreement": False,
        "same_geometry_rules_prompt_full_19_facts_generation_evidence_required": True,
        "historical_speed_family_modified": False, "speed_benchmark_eligible": False,
        "final_check_used": False, "validation_is_reused_development": True,
        "media_or_labels_read_during_prepare": False, "model_calls_during_prepare": 0}
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA and protocol.get("name") == "accuracy-diagnostic-v1"
        and protocol.get("execution_order") == ORDER
        and protocol.get("deadlines_by_original_ordinal_seconds") == DEADLINES,
        "diagnostic_protocol_order_or_deadlines_invalid")
    exact = {"source_attempts_required": 6, "maximum_model_requests_per_arm": 208,
        "maximum_execution_seconds_per_arm": 8220, "provider_max_attempts": 2,
        "maximum_source_attempts_per_case": 1, "pilot_original_case_ordinal": 5,
        "pilot_maximum_model_requests": 24, "pilot_maximum_execution_seconds": 960,
        "remaining_maximum_model_requests": 184, "remaining_maximum_execution_seconds": 7260,
        "advance_requires_technical_completion": True, "advance_uses_provided_verdict_agreement": False,
        "same_geometry_rules_prompt_full_19_facts_generation_evidence_required": True,
        "historical_speed_family_modified": False, "speed_benchmark_eligible": False,
        "final_check_used": False, "validation_is_reused_development": True,
        "media_or_labels_read_during_prepare": False, "model_calls_during_prepare": 0}
    require(all(protocol.get(key) == value and type(protocol.get(key)) is type(value) for key, value in exact.items()),
        "diagnostic_protocol_scope_or_budget_changed")
    require(all(_sha(protocol.get(key)) for key in ("historical_family_sha256", "historical_recipe_sha256", "provided_labels_sha256"))
        and protocol["historical_recipe_sha256"] == digest(protocol["historical_recipe"]), "diagnostic_parent_pins_invalid")
    family = protocol["historical_family"]
    require(family.get("schema_version") == "forte.qwen38-quality-family.v1"
        and isinstance(family.get("records"), list) and len(family["records"]) == 6,
        "diagnostic_embedded_family_population_invalid")
    records = {row["sample_id"]: row for row in family["records"]}
    require(len(records) == len({row["content_sha256"] for row in records.values()}) == 6
        and all(row["split"] == "validation" and _sha(row["content_sha256"]) for row in records.values())
        and set(family["base_models"]) == ARMS, "diagnostic_embedded_family_roles_invalid")
    attempts = family["attempts"]
    require(len(attempts) == len({row["attempt_id"] for row in attempts}) == 18,
        "diagnostic_embedded_family_attempts_invalid")
    for arm in ARMS:
        planned = [row for row in attempts if row["arm"] == arm]
        require(len(planned) == 6 and {row["sample_id"] for row in planned} == set(records)
            and all(row["repeat_index"] == 0 and row["split"] == "validation"
                and row["content_sha256"] == records[row["sample_id"]]["content_sha256"] for row in planned),
            "diagnostic_embedded_family_attempts_invalid")
    require(digest(family) == protocol["historical_family_sha256"] and family["recipe"] == protocol["historical_recipe"]
        and family["recipe_sha256"] == protocol["historical_recipe_sha256"]
        and family["base_models"] == protocol["historical_base_models"]
        and family["provided_labels_sha256"] == protocol["provided_labels_sha256"]
        and family.get("coarse_anchors_sha256") == protocol["coarse_anchors_sha256"], "diagnostic_historical_family_attribution_changed")
    recipe = protocol["historical_recipe"]
    require(recipe.get("provider_max_attempts") == 2 and recipe["generation"]["max_new_tokens"] == 4096
        and recipe["generation"]["seed"] == 20261017 and recipe["context_tokens"] == 32768
        and all(_sha(recipe.get(key)) for key in ("source_archive_sha256", "rules_sha256", "prompt_source_sha256")),
        "diagnostic_generation_or_context_changed")
    population = protocol["population"]
    require(len(population) == len({row["sample_id"] for row in population}) == len({row["content_sha256"] for row in population}) == 6,
        "diagnostic_exact_six_population_required")
    for ordinal, row in enumerate(population, 1):
        require(row["split"] == "validation" and _sha(row["content_sha256"])
            and row["sample_id"] in records and records[row["sample_id"]]["content_sha256"] == row["content_sha256"]
            and type(row["source_bytes"]) is int and row["source_bytes"] > 0, "diagnostic_source_metadata_invalid")
        _worker_geometry(recipe["worker_settings"], row, ordinal)
    return protocol


def _reissue_module():
    try:
        import accuracy_reissue_contract as module
    except ModuleNotFoundError as error:
        if error.name != "accuracy_reissue_contract":
            raise
        from benchmarks.performance import qwen38_accuracy_reissue as module
    return module


def _source_id(protocol, arm, ordinal, reissue_sha256=None):
    original = f"qwen38-accuracy-v1-{digest(protocol)[:20]}-{arm}-case-{ordinal:02d}-r0"
    return original if reissue_sha256 is None else original + "-control2-" + reissue_sha256[:16]


def role_policy_from_config(config):
    policy = {key: config[key] for key in ROLE_PINS}
    policy.update(allowed_splits=config.get("allowed_splits", ["validation"]),
        max_snapshot_age_sec=config.get("max_snapshot_age_sec", 300), final_check_used=config.get("final_check_used", False))
    _validate_role_policy(policy)
    return policy


def _validate_role_policy(policy):
    require(isinstance(policy, dict) and set(policy) == {*ROLE_PINS, "allowed_splits", "max_snapshot_age_sec", "final_check_used"}
        and all(_sha(policy.get(key)) for key in ROLE_PINS)
        and policy.get("allowed_splits") == ["validation"] and policy.get("final_check_used") is False
        and type(policy.get("max_snapshot_age_sec")) in {float, int}
        and math.isfinite(policy["max_snapshot_age_sec"]) and 0 < policy["max_snapshot_age_sec"] <= 300,
        "diagnostic_immutable_role_policy_invalid")


def bind_plan(protocol, actual_arm_plan, *, stage="pilot", retained=None, role_config=None, control_reissue=None):
    """Bind genuine arm/runtime metadata; no placeholder identity is produced."""
    validate_protocol(protocol)
    reissue_sha = None
    if control_reissue is not None:
        require(sys.version_info[:2] == (3, 12), "control_reissue_plan_creation_requires_actual_cpu_python312")
        reissue_sha = _reissue_module().validate_control_reissue(control_reissue)
    require(actual_arm_plan["comparison_plan_sha256"] == protocol["historical_family_sha256"]
        and actual_arm_plan["recipe"] == protocol["historical_recipe"]
        and actual_arm_plan["recipe_sha256"] == protocol["historical_recipe_sha256"]
        and actual_arm_plan["worker"] == protocol["historical_recipe"]["worker_settings"], "diagnostic_actual_parent_changed")
    actual_source = actual_arm_plan["runtime_bindings"]["source_archive_sha256"]
    if actual_source != protocol["historical_recipe"]["source_archive_sha256"]:
        from evaluation.qwen38_quality_equivalence import validate_equivalence
        validate_equivalence(protocol["historical_family"], actual_arm_plan["source_equivalence"], actual_source)
    arm = actual_arm_plan["arm"]
    require(arm in protocol["historical_base_models"], "diagnostic_arm_invalid")
    original = {row["sample_id"]: row for row in actual_arm_plan["cases"]}
    require(set(original) == {row["sample_id"] for row in protocol["population"]}, "diagnostic_actual_population_changed")
    protocol_cases = []
    for ordinal, row in enumerate(protocol["population"], 1):
        source = original[row["sample_id"]]
        require(all(source.get(key) == value for key, value in row.items()) and source["repeat_index"] == 0,
            "diagnostic_actual_source_metadata_changed")
        aid = _source_id(protocol, arm, ordinal, reissue_sha)
        protocol_cases.append({**deepcopy(source), "attempt_id": aid, "case_id": aid,
            "original_case_ordinal": ordinal, "diagnostic_deadline_seconds": DEADLINES[ordinal - 1]})
    require(stage in {"pilot", "remaining"}, "diagnostic_stage_invalid")
    released = [5] if stage == "pilot" else [1, 2, 3, 4, 6]
    cases = [protocol_cases[ordinal - 1] for ordinal in released]
    slots = [{"attempt_id": case["attempt_id"] + f"-model-{index:04d}", "source_attempt_id": case["attempt_id"],
        "case_id": case["case_id"], "content_sha256": case["content_sha256"], "split": "validation", "slot_index": index}
        for case in cases for index in range(case["maximum_model_requests"])]
    copied = ("provider", "runtime_bindings", "expected_model_identity", "role_config_sha256", "source_equivalence",
        "source_equivalence_sha256", "evaluator_source_archive_sha256", "serving_source_archive_sha256",
        "comparison_runtime_profile", "comparison_runtime_profile_sha256", "audited_common_source_deltas",
        "effective_client_request_timeout_seconds")
    plan = {key: deepcopy(actual_arm_plan[key]) for key in copied if key in actual_arm_plan}
    policy = role_policy_from_config(role_config) if role_config is not None else deepcopy(actual_arm_plan.get("role_policy"))
    _validate_role_policy(policy)
    plan["role_policy"] = policy
    recipe = _recipe(protocol)
    plan.update(schema_version=PLAN_SCHEMA, kind="accuracy_diagnostic", phase="evaluation", arm=arm,
        diagnostic_stage=stage, diagnostic_protocol=deepcopy(protocol), diagnostic_protocol_sha256=digest(protocol),
        comparison_plan_sha256=digest(protocol), historical_family_sha256=protocol["historical_family_sha256"],
        historical_recipe_sha256=protocol["historical_recipe_sha256"], recipe=recipe, recipe_sha256=digest(recipe),
        actual_arm_plan_sha256=digest(actual_arm_plan), worker=deepcopy(actual_arm_plan["worker"]),
        protocol_cases=protocol_cases, cases=cases, attempts=slots,
        source_attempts=[{key: case[key] for key in ("attempt_id", *IDENTITY_FIELDS)} for case in protocol_cases],
        released_source_attempts=[case["attempt_id"] for case in cases], execution_order=ORDER.copy(),
        source_attempts_required=6, maximum_private_model_requests=len(slots),
        maximum_execution_seconds=sum(case["diagnostic_deadline_seconds"] for case in cases),
        minimum_initial_generation_requests=sum(case["coarse_windows"] for case in cases),
        source_hash_recheck_required_by_executor_before_every_source=True,
        quality_model_calls_during_prepare=0, source_media_read_during_plan_binding=False,
        upload_start_or_browser_latency_measured=False, final_check_used=False,
        speed_benchmark_eligible=False, prepared_at_utc=datetime.now(timezone.utc).isoformat())
    if retained:
        require(set(retained) == {"retained_pilot_plan_path", "retained_pilot_plan_sha256",
            "retained_pilot_source_terminals_path", "retained_pilot_source_terminals_sha256", "retained_pilot_role_config_sha256"},
            "diagnostic_only_immutable_retained_bindings_allowed")
        plan.update(retained)
    if control_reissue is not None:
        plan.update(operational_control_attempt=2, control_reissue=deepcopy(control_reissue), control_reissue_sha256=reissue_sha,
            historical_control_slots_consumed=24, maximum_actual_model_posts_per_arm=208,
            maximum_reserved_model_slots_including_failed_control_per_arm=232,
            maximum_actual_source_pipeline_attempts_per_case=1)
    validate_diagnostic_plan(plan)
    return plan


def validate_diagnostic_plan(plan):
    require(plan.get("schema_version") == PLAN_SCHEMA and plan.get("kind") == "accuracy_diagnostic"
        and plan.get("phase") == "evaluation" and plan.get("diagnostic_stage") in {"pilot", "remaining"}, "diagnostic_plan_scope_invalid")
    protocol = validate_protocol(plan["diagnostic_protocol"])
    _validate_role_policy(plan.get("role_policy"))
    reissue_sha = None
    if plan.get("operational_control_attempt", 1) == 1:
        require(not any(key in plan for key in ("control_reissue", "control_reissue_sha256", "historical_control_slots_consumed",
            "maximum_actual_model_posts_per_arm", "maximum_reserved_model_slots_including_failed_control_per_arm",
            "maximum_actual_source_pipeline_attempts_per_case")), "control_reissue_proof_without_control2_forbidden")
    else:
        require(plan.get("operational_control_attempt") == 2 and isinstance(plan.get("control_reissue"), dict),
            "control_reissue_only_control2_allowed")
        reissue_sha = _reissue_module().validate_control_reissue(plan["control_reissue"])
        require(plan.get("control_reissue_sha256") == reissue_sha, "control_reissue_plan_proof_changed")
    require(plan["diagnostic_protocol_sha256"] == plan["comparison_plan_sha256"] == digest(protocol)
        and plan["historical_family_sha256"] == protocol["historical_family_sha256"]
        and plan["historical_recipe_sha256"] == protocol["historical_recipe_sha256"]
        and plan["recipe"] == _recipe(protocol) and plan["recipe_sha256"] == digest(plan["recipe"])
        and plan["worker"] == protocol["historical_recipe"]["worker_settings"], "diagnostic_recipe_or_parent_changed")
    arm, identity = plan["arm"], plan["expected_model_identity"]
    require(arm in protocol["historical_base_models"] and set(identity) == set(MODEL_FIELDS)
        and all(identity.get(key) == value for key, value in protocol["historical_base_models"][arm].items())
        and (_sha(identity["adapter_sha256"]) if arm == "qwen38_tuned" else identity["adapter_sha256"] is None),
        "diagnostic_actual_model_identity_invalid")
    require(all(_sha(plan["runtime_bindings"].get(key)) for key in PIN_FIELDS)
        and _sha(plan.get("role_config_sha256")) and _sha(plan.get("actual_arm_plan_sha256"))
        and isinstance(plan["provider"], dict), "diagnostic_actual_runtime_pins_missing")
    if identity["model_id"] == "Qwen/Qwen3.8-27B":
        require(plan.get("effective_client_request_timeout_seconds") == 240
            and _sha(plan["provider"].get("launch_config_file_sha256")),
            "diagnostic_actual_candidate_client_timeout_not_240")
    actual_source = plan["runtime_bindings"]["source_archive_sha256"]
    if actual_source != protocol["historical_recipe"]["source_archive_sha256"]:
        # The host verifies metadata hashes only. The separate CPU-image runner
        # must invoke the frozen AST/source-equivalence verifier before inference.
        receipt = plan["source_equivalence"]
        require(receipt.get("schema_version") == "forte.qwen38-quality-source-equivalence.v1"
            and receipt.get("family_sha256") == protocol["historical_family_sha256"]
            and receipt.get("evaluator_source_archive_sha256") == protocol["historical_recipe"]["source_archive_sha256"]
            and receipt.get("actual_serving_source_archive_sha256") == actual_source
            and receipt.get("receipt_sha256") == digest({key: value for key, value in receipt.items() if key != "receipt_sha256"})
            and plan.get("source_equivalence_sha256") == digest(receipt)
            and plan.get("evaluator_source_archive_sha256") == protocol["historical_recipe"]["source_archive_sha256"]
            and plan.get("serving_source_archive_sha256") == actual_source
            and isinstance(plan.get("comparison_runtime_profile"), dict)
            and plan.get("comparison_runtime_profile_sha256") == digest(plan["comparison_runtime_profile"]),
            "diagnostic_actual_equivalence_or_profile_changed")
    protocol_cases = plan["protocol_cases"]
    require(len(protocol_cases) == 6, "diagnostic_exact_six_cases_required")
    for ordinal, (case, population) in enumerate(zip(protocol_cases, protocol["population"]), 1):
        aid = _source_id(protocol, arm, ordinal, reissue_sha)
        require(all(case.get(key) == value for key, value in population.items())
            and case.get("original_case_ordinal") == ordinal and case.get("attempt_id") == case.get("case_id") == aid
            and case.get("arm") == arm and case.get("repeat_index") == 0
            and case.get("diagnostic_deadline_seconds") == DEADLINES[ordinal - 1]
            and case.get("maximum_pipeline_inferences") == WINDOWS[ordinal - 1] + 8
            and case.get("maximum_model_requests") == (WINDOWS[ordinal - 1] + 8) * 2
            and isinstance(case.get("owned_path"), str)
            and (Path(case["owned_path"]).is_absolute() or PurePosixPath(case["owned_path"]).is_absolute()),
            "diagnostic_case_binding_or_budget_changed")
    require(plan["source_attempts"] == [{key: case[key] for key in ("attempt_id", *IDENTITY_FIELDS)} for case in protocol_cases],
        "diagnostic_six_source_denominator_changed")
    pilot = plan["diagnostic_stage"] == "pilot"
    expected_cases = [protocol_cases[index - 1] for index in ([5] if pilot else [1, 2, 3, 4, 6])]
    require(plan["cases"] == expected_cases and plan["released_source_attempts"] == [row["attempt_id"] for row in expected_cases]
        and plan["execution_order"] == ORDER, "diagnostic_released_subset_or_order_changed")
    expected_slots = [{"attempt_id": case["attempt_id"] + f"-model-{index:04d}", "source_attempt_id": case["attempt_id"],
        "case_id": case["case_id"], "content_sha256": case["content_sha256"], "split": "validation", "slot_index": index}
        for case in expected_cases for index in range(case["maximum_model_requests"])]
    require(plan["attempts"] == expected_slots and plan["maximum_private_model_requests"] == (24 if pilot else 184)
        and plan["maximum_execution_seconds"] == (960 if pilot else 7260)
        and plan["minimum_initial_generation_requests"] == (4 if pilot else 52), "diagnostic_exact_released_slot_or_wall_cap_changed")
    require(plan["source_attempts_required"] == 6 and plan["source_hash_recheck_required_by_executor_before_every_source"] is True
        and plan["source_media_read_during_plan_binding"] is False and plan["quality_model_calls_during_prepare"] == 0
        and plan["upload_start_or_browser_latency_measured"] is False and plan["final_check_used"] is False
        and plan["speed_benchmark_eligible"] is False, "diagnostic_claim_or_denominator_changed")
    retained_paths = ("retained_pilot_source_terminals_path", "retained_pilot_plan_path")
    retained_hashes = ("retained_pilot_source_terminals_sha256", "retained_pilot_plan_sha256", "retained_pilot_role_config_sha256")
    retained_keys = (*retained_paths, *retained_hashes)
    if pilot:
        require(not any(key in plan for key in retained_keys), "diagnostic_pilot_cannot_retain_or_retry_prior_result")
    else:
        require(all(isinstance(plan.get(key), str)
            and (Path(plan[key]).is_absolute() or PurePosixPath(plan[key]).is_absolute()) for key in retained_paths)
            and all(_sha(plan.get(key)) for key in retained_hashes), "diagnostic_remaining_requires_immutable_pilot")
    if reissue_sha is not None:
        _reissue_module().validate_reissue_plan_binding(plan)
    return plan


def validate_reserved_role_policy(plan, reservation):
    """Match immutable policy to the actual central role-preflight receipt.

    A09's standard reservation creation separately verifies freshness, split
    ancestry and the central chain. This check does not grant a reservation or
    substitute for its exact plan/release/runtime/expiry validation.
    """
    policy = plan["role_policy"]
    _validate_role_policy(policy)
    gate = reservation.get("exposure_gate", {})
    count = len({row["content_sha256"] for row in plan["cases"]})
    require(gate.get("schema_version") == "forte.qwen38-role-preflight.v1" and gate.get("ok") is True
        and gate.get("assignments_sha256") == policy["expected_assignments_sha256"]
        and gate.get("requested_contents") == count
        and gate.get("allowed_split_counts") == {"validation": count}
        and all(gate.get("input_checksums", {}).get(name) == policy["expected_" + name + "_sha256"]
            for name in ("source_manifest", "automated_split", "scenario_groups"))
        and gate.get("human_verified") is False and gate.get("scenario_independence_verified") is False
        and gate.get("compute_authorized") is False and gate.get("media_opened") is False
        and gate.get("model_calls_made") is False, "diagnostic_reserved_role_policy_mismatch")
    return True


def validate_retained_pilot(plan, pilot_plan, terminal_rows):
    validate_diagnostic_plan(plan)
    validate_diagnostic_plan(pilot_plan)
    require(plan["diagnostic_stage"] == "remaining" and pilot_plan["diagnostic_stage"] == "pilot"
        and all(plan[key] == pilot_plan[key] for key in ("arm", "diagnostic_protocol_sha256", "recipe_sha256",
            "runtime_bindings", "expected_model_identity", "protocol_cases", "source_attempts",
            "provider", "role_policy"))
        and plan["retained_pilot_role_config_sha256"] == pilot_plan["role_config_sha256"]
        and all((key in plan) == (key in pilot_plan) and plan.get(key) == pilot_plan.get(key) for key in (
            "effective_client_request_timeout_seconds", "comparison_runtime_profile", "comparison_runtime_profile_sha256",
            "source_equivalence", "source_equivalence_sha256", "evaluator_source_archive_sha256",
            "serving_source_archive_sha256", "audited_common_source_deltas", "operational_control_attempt",
            "control_reissue", "control_reissue_sha256")), "diagnostic_retained_pilot_plan_mismatch")
    expected = pilot_plan["source_attempts"][4]
    rows = [row for row in terminal_rows if row.get("attempt_id") == expected["attempt_id"]]
    require(len(rows) == 1 and len(terminal_rows) == len({row.get("attempt_id") for row in terminal_rows}) == 6
        and {row.get("attempt_id") for row in terminal_rows} == {row["attempt_id"] for row in pilot_plan["source_attempts"]},
        "diagnostic_retained_terminal_population_invalid")
    planned_by_id = {row["attempt_id"]: row for row in pilot_plan["source_attempts"]}
    require(all(all(row.get(key) == value for key, value in planned_by_id[row["attempt_id"]].items())
        and (row["attempt_id"] == expected["attempt_id"] or row.get("status") in {"not_attempted", "not_released"})
        for row in terminal_rows), "diagnostic_unreleased_source_was_attempted")
    row = rows[0]
    require(all(row.get(key) == value for key, value in expected.items())
        and row.get("comparison_plan_sha256") == plan["comparison_plan_sha256"]
        and row.get("recipe_sha256") == plan["recipe_sha256"] and row.get("status") == "done"
        and row.get("public_contract_validated") is True and row.get("server_receipt_verified") is True
        and row.get("model_identity") == plan["expected_model_identity"], "diagnostic_pilot_technical_completion_required")
    completion = row.get("technical_completion", {})
    require(completion.get("schema_version") == "forte.qwen38-accuracy-technical-completion.v1"
        and all(completion.get(key) is True for key in ("complete_source_coverage", "strict_publication",
            "native_source_complete", "returned_evidence_matches_native_source", "fresh_stage_witness"))
        and completion.get("reference_or_verdict_agreement_used") is False
        and _sha(completion.get("server_receipt_sha256")), "diagnostic_pilot_actual_technical_receipt_required")
    # Verdict is deliberately never read or used to permit remaining sources.
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--family", type=Path, required=True)
    protocol.add_argument("--historical-arm-plan", type=Path, required=True)
    bind = sub.add_parser("bind")
    bind.add_argument("--protocol", type=Path, required=True)
    bind.add_argument("--actual-arm-plan", type=Path, required=True)
    bind.add_argument("--role-config", type=Path)
    bind.add_argument("--control-reissue", type=Path)
    bind.add_argument("--stage", choices=("pilot", "remaining"), default="pilot")
    bind.add_argument("--retained-pilot-plan", type=Path)
    bind.add_argument("--retained-source-terminals", type=Path)
    for command in (protocol, bind):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "protocol":
            output = freeze_protocol(read_json(args.family), read_json(args.historical_arm_plan))
        else:
            retained = None
            require((args.stage == "remaining") == bool(args.retained_pilot_plan and args.retained_source_terminals)
                and (args.stage == "remaining" or not (args.retained_pilot_plan or args.retained_source_terminals)),
                "diagnostic_retained_inputs_stage_mismatch")
            if args.stage == "remaining":
                retained = {"retained_pilot_source_terminals_path": str(args.retained_source_terminals.resolve()),
                    "retained_pilot_source_terminals_sha256": file_digest(args.retained_source_terminals),
                    "retained_pilot_plan_path": str(args.retained_pilot_plan.resolve()),
                    "retained_pilot_plan_sha256": file_digest(args.retained_pilot_plan),
                    "retained_pilot_role_config_sha256": read_json(args.retained_pilot_plan)["role_config_sha256"]}
            actual_plan = read_json(args.actual_arm_plan)
            if args.role_config:
                require(file_digest(args.role_config) == actual_plan["role_config_sha256"], "diagnostic_actual_role_config_bytes_mismatch")
            output = bind_plan(read_json(args.protocol), actual_plan, stage=args.stage, retained=retained,
                role_config=read_json(args.role_config) if args.role_config else None,
                control_reissue=read_json(args.control_reissue) if args.control_reissue else None)
            if retained:
                validate_retained_pilot(output, read_json(args.retained_pilot_plan), read_records(args.retained_source_terminals))
        write_new(args.output, output)
        print('{"status":"metadata_prepared_not_executed","model_calls_made":0,"private_values_printed":false}')
        return 0
    except Exception:
        print('{"status":"diagnostic_metadata_invalid","model_calls_made":0,"private_values_printed":false}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
