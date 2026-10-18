"""CPU-only native full-six plan binding from actual raw receipts.

The factory issues no release, reservation, GET, POST, decode or model call.
Freshness is checked at admission, separately from long-running plan integrity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path, PurePosixPath

PROTOCOL_HELPER_SHA256 = "f8591adf6fae3cb66d90c2982f175ead7ed301e181a32eff71823239298d0ac9"
ADMISSION_HELPER_SHA256 = "0df592c2a575bb02695818b9cdb079e24fc2b123634189344a6d3960c86d73a1"
GATE_PINS = ("native_generated_gpu_gate_receipt_sha256", "native_generated_http_gate_receipt_sha256", "native_ready_receipt_sha256")
ARTIFACT_NAMES = {"native_config", "role_config", "role_preflight", "accepted_inventory", "exposure_status",
    "generated_gpu_gate", "generated_http_gate", "ready_capture", "consumer_prepare", "consumer_source_proof"}
ROLES = {"qwen38_untuned": "base_diagnostic", "qwen38_tuned": "fit_candidate"}
ROLE_POLICY_PINS = {"expected_assignments_sha256": "206b779487a100ae51ed0502fcd91713db82914b1a8b74536d6918826c1c532c",
    "expected_source_manifest_sha256": "400ab2322ad4e0c83f5736d66280f215cf9f60d0d7200731718b4b8e2964bc8f",
    "expected_automated_split_sha256": "0101f79778d69b165a72a59448bd4b7ac2bda356fea00c5661fc199d650a96be",
    "expected_scenario_groups_sha256": "4c3af79f4333678d67d568a47f224fceccb7a32dcc2f80e63a68667b900e3c5e"}


def protocol_contract():
    try:
        import native_fullsix_contract as module
    except ModuleNotFoundError as error:
        if error.name != "native_fullsix_contract": raise
        from benchmarks.performance import qwen38_native_fullsix as module
    return module


def admission_contract():
    try:
        import native_admission_contract as module
    except ModuleNotFoundError as error:
        if error.name != "native_admission_contract": raise
        from benchmarks.performance import qwen38_native_admission_addendum as module
    return module


def parent_contract():
    return protocol_contract().parent_contract()


def artifact(path):
    """Read bounded exact metadata bytes; never accept a caller's claimed hash."""
    core = parent_contract()
    path = Path(path)
    core.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 * 1024 * 1024,
        "native_plan_actual_metadata_file_required")
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    import hashlib
    return {"raw_utf8": raw.decode("utf-8"), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "semantic_sha256": core.digest(document), "document": document}


def validate_artifact(value):
    import hashlib
    core = parent_contract()
    core.require(isinstance(value, dict) and set(value) == {"raw_utf8", "raw_sha256", "semantic_sha256", "document"}
        and isinstance(value["raw_utf8"], str) and len(value["raw_utf8"].encode("utf-8")) <= 64 * 1024 * 1024
        and hashlib.sha256(value["raw_utf8"].encode("utf-8")).hexdigest() == value["raw_sha256"]
        and core.digest(json.loads(value["raw_utf8"].lstrip("\ufeff"))) == value["semantic_sha256"] == core.digest(value["document"]),
        "native_plan_raw_artifact_binding_invalid")
    return value["document"]


def _utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    parent_contract().require(result.tzinfo is not None, "native_plan_receipt_timezone_required")
    return result.astimezone(timezone.utc)


def _ints(document, expected):
    return all(type(document.get(key)) is int and document[key] == value for key,value in expected.items())


def _runtime(config):
    """Select genuine installed APIs, never project parallel readiness as serial."""
    if config.get("backend") == "vllm_bf16_native":
        identity = importlib.import_module("serving.qwen38_vllm_identity")
        profile = importlib.import_module("serving.qwen38_vllm_profile")
        identity.validate_native_config(config)
        return {"variant": "serial_native_v1", "expected": identity.expected_native_identity,
            "ready": identity.validate_native_readiness, "profile": profile.native_runtime_profile(config),
            "profile_sha256": profile.native_runtime_profile_sha256(config), "pipeline": profile.PIPELINE,
            "provider_module": "serving.qwen38_vllm_provider", "identity_module": "serving.qwen38_vllm_identity",
            "profile_module": "serving.qwen38_vllm_profile", "readiness_module": "serving.qwen38_vllm_identity"}
    if config.get("backend") == "vllm_bf16_native_parallel4":
        profile = importlib.import_module("experimental_parallel4.profile")
        admission = importlib.import_module("experimental_parallel4.admission")
        profile.validate_parallel_config(config)
        return {"variant": "parallel4_v1", "expected": profile.expected_parallel_identity,
            "ready": admission.validate_parallel_readiness, "profile": profile.parallel_runtime_profile(config),
            "profile_sha256": profile.parallel_runtime_profile_sha256(config), "pipeline": profile.PIPELINE,
            "provider_module": "experimental_parallel4.provider_pool", "identity_module": "experimental_parallel4.profile",
            "profile_module": "experimental_parallel4.profile", "readiness_module": "experimental_parallel4.admission"}
    parent_contract().require(False, "native_plan_real_native_backend_required")


def _gpu_chain(document):
    try:
        import native_gpu_chain_contract as module
    except ModuleNotFoundError as error:
        if error.name != "native_gpu_chain_contract": raise
        from benchmarks.performance import qwen38_native_gpu_chain as module
    return module.validate_gpu_gate_chain(document)


def _profile_amendment(plan, runtime):
    core = parent_contract()
    if runtime["variant"] == "serial_native_v1":
        core.require("native_profile_amendment" not in plan, "native_plan_serial_amendment_unexpected")
        return
    amendment_artifact = plan.get("native_profile_amendment")
    core.require(amendment_artifact is not None, "native_plan_parallel4_root_profile_amendment_required")
    amendment = validate_artifact(amendment_artifact)
    try:
        import native_parallel4_amendment_contract as module
    except ModuleNotFoundError as error:
        if error.name != "native_parallel4_amendment_contract": raise
        from benchmarks.performance import qwen38_native_parallel4_amendment as module
    module.validate_amendment(amendment, raw_sha256=amendment_artifact["raw_sha256"],
        protocol=plan["native_protocol"], addendum=plan["native_admission_addendum"])
    core.require(amendment["pipeline_conditions"] == runtime["pipeline"]
        and amendment["engine_conditions"] == plan["native_runtime_profile"]["engine"],
        "native_plan_parallel4_profile_amendment_invalid")
    core.require(plan.get("native_profile_amendment_sha256") == amendment_artifact["raw_sha256"],
        "native_plan_parallel4_amendment_digest_changed")


def _validate_role(plan, *, admission_now=None):
    core = parent_contract()
    artifacts = plan["artifacts"]
    config = artifacts["role_config"]["document"]
    policy = core.role_policy_from_config(config)
    core.require(policy == plan["role_policy"] and plan["role_config_sha256"] == artifacts["role_config"]["raw_sha256"],
        "native_plan_actual_role_policy_or_config_changed")
    core.require(all(policy.get(key) == value for key,value in ROLE_POLICY_PINS.items()),
        "native_plan_original_immutable_role_assignments_changed")
    gate = artifacts["role_preflight"]["document"]
    core.validate_reserved_role_policy(plan, {"exposure_gate": gate})
    inventory = artifacts["accepted_inventory"]["document"]
    status = artifacts["exposure_status"]["document"]
    core.require(gate.get("exposure_coverage_complete_snapshot") is True
        and type(gate.get("exposure_snapshot_age_sec")) in {float, int}
        and math.isfinite(gate["exposure_snapshot_age_sec"]) and 0 <= gate["exposure_snapshot_age_sec"] <= policy["max_snapshot_age_sec"]
        and gate.get("exposure_inventory_status_sha256") == artifacts["exposure_status"]["raw_sha256"]
        and core._sha(gate.get("exposure_registry_tip_sha256")) and type(gate.get("exposure_registry_event_count")) is int
        and gate["exposure_registry_event_count"] > 0
        and status.get("schema_version") == "forte.exposure-coverage.v1" and status.get("coverage_state") == "complete_snapshot"
        and status.get("scope") == "all_accepted_jobs_since_service_start"
        and status.get("source_inventory_sha256") == artifacts["accepted_inventory"]["raw_sha256"]
        and inventory.get("schema_version") == "forte.service-content-exposure-inventory.v1"
        and inventory.get("coverage_state") == "complete_snapshot"
        and status.get("captured_at_utc") == inventory.get("captured_at_utc"), "native_plan_actual_exposure_snapshot_binding_invalid")
    if admission_now is not None:
        age = (admission_now - _utc(inventory["captured_at_utc"])).total_seconds()
        core.require(0 <= age <= policy["max_snapshot_age_sec"], "native_plan_role_snapshot_not_fresh_at_admission")


def _validate_consumer(plan, config, runtime):
    core = parent_contract()
    binding = plan["consumer_runtime_bindings"]
    proof = plan["artifacts"]["consumer_source_proof"]["document"]
    receipt = plan["artifacts"]["consumer_prepare"]["document"]
    core.require(all(core._sha(binding.get(key)) for key in ("source_archive_sha256", "runtime_image_sha256"))
        and proof.get("schema_version") == "forte.qwen38-native-consumer-source-proof.v1"
        and proof.get("source_archive_sha256") == binding["source_archive_sha256"]
        and proof.get("runtime_image_sha256") == binding["runtime_image_sha256"]
        and proof.get("actual_imports_match_source_archive") is True
        and _ints(proof, {"network_calls":0,"model_calls":0,"gpu_operations":0}),
        "native_plan_actual_consumer_source_or_image_proof_invalid")
    modules = proof.get("loaded_module_sha256", {})
    required = {"forte.pipeline", "forte.model.adapter", "forte.video.sampling", "forte.video.decode",
        "forte.decision.engine", "forte.evidence.assembly", runtime["provider_module"], runtime["identity_module"],
        runtime["profile_module"], runtime["readiness_module"]}
    core.require(required <= modules.keys() and all(core._sha(modules[name]) for name in required)
        and binding.get("actual_module_sha256") == modules
        and binding.get("consumer_source_proof_sha256") == plan["artifacts"]["consumer_source_proof"]["raw_sha256"]
        and binding.get("consumer_prepare_receipt_sha256") == plan["artifacts"]["consumer_prepare"]["raw_sha256"],
        "native_plan_consumer_actual_module_inventory_incomplete")
    core.require(receipt.get("schema_version") == "forte.qwen38.vllm-bf16.consumer-prepare.v1"
        and receipt.get("status") == "prepared" and receipt.get("source_archive_sha256") == binding["source_archive_sha256"]
        and receipt.get("selected_worker_image_sha256") == binding["runtime_image_sha256"]
        and receipt.get("config_file_sha256") == plan["artifacts"]["native_config"]["raw_sha256"]
        and receipt.get("private_media_read") is False and receipt.get("actual_model_readiness_verified") is False
        and all(type(receipt.get(key)) is int and receipt[key] == 0 for key in ("network_calls", "model_calls", "gpu_operations"))
        and all(receipt.get("loaded_module_sha256", {}).get(name) == value for name, value in modules.items()),
        "native_plan_genuine_consumer_preparation_missing")
    for role in ROLES.values():
        row = receipt.get("consumers", {}).get(role, {})
        core.require(row.get("expected_identity") == runtime["expected"](config, role)
            and _ints(row,{"model_calls":0}) and row.get("http_timeout_seconds") == 240 and row.get("job_timeout_seconds") == 285,
            "native_plan_consumer_both_actual_roles_unverified")


def _validate_gates(plan, config, runtime, *, admission_now=None):
    core = parent_contract()
    artifacts = plan["artifacts"]
    capture = artifacts["ready_capture"]["document"]
    core.require(capture.get("schema_version") == "forte.qwen38-native-ready-capture.v1"
        and capture.get("config_file_sha256") == artifacts["native_config"]["raw_sha256"]
        and capture.get("runtime_profile_sha256") == runtime["profile_sha256"]
        and capture.get("engine_instance_id") == plan["native_engine_instance_id"]
        and capture.get("read_only_ready_observation") is True and _ints(capture,{"model_posts":0})
        and set(capture.get("ready_by_role", {})) == set(ROLES.values()), "native_plan_actual_same_engine_ready_capture_missing")
    http = artifacts["generated_http_gate"]["document"]
    for role, ready in capture["ready_by_role"].items():
        runtime["ready"](ready, config, role)
        core.require(ready["engine_instance_id"] == plan["native_engine_instance_id"]
            and ready["generated_http_gate"] == http, "native_plan_same_engine_HTTP6_or_ready_changed")
    if admission_now is not None:
        age = (admission_now - _utc(capture["captured_at_utc"])).total_seconds()
        core.require(0 <= age <= 300, "native_plan_ready_capture_not_fresh_at_admission")
    historical = _gpu_chain(artifacts["generated_gpu_gate"]["document"])
    core.require(historical.get("status") == "passed" and historical.get("scope") == "historical_offline_sequence_one"
        and historical.get("generated_calls") == 6 and historical.get("original_tensors_verified") == 992
        and historical.get("logical_targets_verified") == 496 and historical.get("physical_wrappers_verified") == 304
        and historical.get("base_dtype") == "BF16" and historical.get("base_quantization", "missing") is None
        and historical.get("does_not_prove_current_HTTP_engine") is True
        and historical.get("runtime_image_sha256") == config["runtime_image_sha256"]
        and historical.get("model_manifest_sha256") == config["base_manifest"]["sha256"]
        and historical.get("adapter_sha256") == config["roles"]["fit_candidate"]["adapter_sha256"],
        "native_plan_historical_GPU6_chain_incomplete_or_wrong_arithmetic")
    # Current parallel identity requires its own concurrent same-engine HTTP6,
    # actual four-slot/own-drain ledger. Historical sequence-one is only lineage.
    if runtime["variant"] == "parallel4_v1":
        core.require(http.get("profile_variant") == "parallel4_v1", "native_plan_parallel4_actual_HTTP_gate_required")
        parallel = validate_artifact(plan.get("native_parallel4_public_gate", {}))
        core.require(parallel.get("schema_version") == "forte.qwen38-native-parallel4-public-gate.v1"
            and parallel.get("status") == "passed" and parallel.get("generated_only") is True
            and _ints(parallel,{"private_model_calls":0,"peak_actual_engine_requests":4}) and parallel.get("native_variant") == "parallel4_v1"
            and parallel.get("config_file_sha256") == artifacts["native_config"]["raw_sha256"]
            and parallel.get("runtime_profile_sha256") == runtime["profile_sha256"]
            and parallel.get("model_source_archive_sha256") == config["source_archive"]["sha256"]
            and parallel.get("runtime_image_sha256") == config["runtime_image_sha256"]
            and parallel.get("parallel_GPU_four_live_requests_verified") is True
            and parallel.get("peak_actual_engine_requests") == 4
            and parallel.get("same_engine_HTTP6_sha256") == artifacts["generated_http_gate"]["raw_sha256"]
            and parallel.get("engine_instance_id") == plan["native_engine_instance_id"]
            and parallel.get("own_ID_drains_verified") is True and parallel.get("healthy_sibling_completion_verified") is True
            and parallel.get("unknown_drain_closes_and_aborts_siblings_verified") is True
            and parallel.get("no_replay_verified") is True and parallel.get("all992_exact_adapter_verified") is True
            and parallel.get("unknown_drain_abort_proof_scope") == "cpu_source_transport_only"
            and parallel.get("unknown_drain_abort_actual_GPU_fault_injection") is False,
            "native_plan_parallel4_actual_public_GPU_HTTP_sibling_abort_gate_required")
        failure = validate_artifact(parallel.get("unknown_drain_abort_cpu_receipt",{}))
        core.require(failure.get("schema_version") == "forte.qwen38-native-parallel4-failure-policy-cpu.v1"
            and failure.get("status") == "passed" and failure.get("scope") == "cpu_source_transport_only"
            and failure.get("actual_GPU_fault_injection") is False
            and failure.get("model_source_archive_sha256") == config["source_archive"]["sha256"]
            and failure.get("runtime_profile_sha256") == runtime["profile_sha256"]
            and core._sha(failure.get("test_report_sha256"))
            and failure.get("unknown_drain_global_close_verified") is True
            and failure.get("known_sibling_abort_verified") is True and failure.get("no_replay_verified") is True,
            "native_plan_parallel4_failure_policy_CPU_scope_receipt_required")
        core.require(plan.get("native_parallel4_public_gate_receipt_sha256") == plan["native_parallel4_public_gate"]["raw_sha256"],
            "native_plan_parallel4_public_gate_pin_changed")


def _recipe(plan):
    result = deepcopy(plan["native_protocol"]["historical_recipe"])
    result["worker_settings"] = deepcopy(plan["worker"])
    result["native_fullsix"] = {key: plan[key] for key in ("native_protocol_sha256", "native_admission_addendum_sha256",
        "native_runtime_profile_sha256", "native_variant")}
    result["native_fullsix"].update(model_source_archive_sha256=plan["runtime_bindings"]["source_archive_sha256"],
        consumer_source_archive_sha256=plan["consumer_runtime_bindings"]["source_archive_sha256"],
        consumer_runtime_image_sha256=plan["consumer_runtime_bindings"]["runtime_image_sha256"],
        actual_provider_max_attempts=1, effective_deadline_policy="explicit_case_accuracy_caps_not_historical285",
        historical_NF4_runtime_equivalence_claim=False)
    if "native_profile_amendment_sha256" in plan:
        result["native_fullsix"]["native_profile_amendment_sha256"] = plan["native_profile_amendment_sha256"]
    return result


def _cases(protocol, runtime, arm, sources):
    core = parent_contract()
    by_id = {row["sample_id"]: row for row in sources}
    core.require(len(sources) == len(by_id) == 6 and set(by_id) == {row["sample_id"] for row in protocol["population"]},
        "native_plan_exact_six_source_map_required")
    cases = []
    for ordinal in protocol["execution_order"]:
        metadata = protocol["population"][ordinal - 1]
        source = by_id[metadata["sample_id"]]
        core.require(all(source.get(key) == value for key, value in metadata.items())
            and isinstance(source.get("owned_path"), str) and PurePosixPath(source["owned_path"]).is_absolute()
            and ".." not in PurePosixPath(source["owned_path"]).parts, "native_plan_source_metadata_or_absolute_path_changed")
        aid = f"qwen38-native-six-{core.digest(protocol)[:16]}-{runtime['profile_sha256'][:16]}-{arm}-{ordinal:02d}-r0"
        logical = metadata["coarse_windows"] + 8
        cases.append({**deepcopy(metadata), "owned_path": source["owned_path"], "arm": arm, "repeat_index": 0,
            "attempt_id": aid, "case_id": aid, "original_case_ordinal": ordinal,
            "diagnostic_deadline_seconds": protocol["deadlines_by_original_ordinal_seconds"][ordinal - 1],
            "maximum_pipeline_inferences": logical, "maximum_model_requests": logical * 2})
    return cases


def bind_native_plan(protocol, actual_arm_metadata, role_config=None, *, now=None):
    """Bind actual file-backed receipts; output still grants no execution."""
    native, admission, core = protocol_contract(), admission_contract(), parent_contract()
    native.validate_protocol(protocol)
    metadata = deepcopy(actual_arm_metadata)
    core.require(set(metadata["artifacts"]) == ARTIFACT_NAMES, "native_plan_exact_actual_artifacts_required")
    for value in metadata["artifacts"].values(): validate_artifact(value)
    if role_config is not None:
        core.require(core.digest(role_config) == metadata["artifacts"]["role_config"]["semantic_sha256"],
            "native_plan_actual_role_config_argument_mismatch")
    config = metadata["artifacts"]["native_config"]["document"]
    runtime = _runtime(config)
    arm = metadata["arm"]; core.require(arm in ROLES, "native_plan_arm_invalid")
    role = ROLES[arm]
    cases = _cases(protocol, runtime, arm, metadata["sources"])
    identity = runtime["expected"](config, role)
    plan = {"schema_version": "forte.qwen38-attempt-plan.v1", "kind": "native_bf16_fullsix_accuracy", "phase": "evaluation",
        "arm": arm, "native_role": role, "native_variant": runtime["variant"],
        "native_protocol": deepcopy(protocol), "native_protocol_sha256": core.digest(protocol),
        "native_protocol_file_sha256": admission.PROTOCOL_RAW_SHA256,
        "native_admission_addendum": metadata["native_admission_addendum"],
        "native_admission_addendum_sha256": admission.ADDENDUM_SHA256,
        "native_admission_addendum_file_sha256": admission.ADDENDUM_RAW_SHA256,
        "native_expected_runtime_identity": identity,
        "expected_model_identity": {"model_id": config["model_id"], "model_revision": config["model_revision"], "adapter_sha256": identity["adapter_sha256"]},
        "native_runtime_profile": runtime["profile"], "native_runtime_profile_sha256": runtime["profile_sha256"],
        "native_engine_instance_id": metadata["artifacts"]["ready_capture"]["document"]["engine_instance_id"],
        "provider": {"launch_config": metadata["launch_config_path"],
            "launch_config_file_sha256": metadata["artifacts"]["native_config"]["raw_sha256"],
            "base_url": metadata["base_url"], "served_model_name": config["roles"][role]["served_model_name"],
            "actual_max_attempts": 1, "request_timeout_seconds": 240},
        "runtime_bindings": {"source_archive_sha256": config["source_archive"]["sha256"],
            "runtime_image_sha256": config["runtime_image_sha256"], "base_manifest_sha256": config["base_manifest"]["sha256"]},
        "consumer_runtime_bindings": metadata["consumer_runtime_bindings"], "artifacts": metadata["artifacts"],
        "native_provider_module": runtime["provider_module"], "native_identity_module": runtime["identity_module"],
        "native_profile_module": runtime["profile_module"], "native_readiness_module": runtime["readiness_module"],
        "role_policy": core.role_policy_from_config(metadata["artifacts"]["role_config"]["document"]),
        "role_config_sha256": metadata["artifacts"]["role_config"]["raw_sha256"],
        "worker": {**deepcopy(protocol["historical_recipe"]["worker_settings"]), **runtime["pipeline"]},
        "cases": cases, "source_attempts": [{key: case[key] for key in ("attempt_id", *core.IDENTITY_FIELDS)} for case in cases],
        "source_attempts_required": 6, "released_source_attempts": [case["attempt_id"] for case in cases],
        "maximum_private_model_requests": 208, "maximum_execution_seconds": 8220, "minimum_initial_generation_requests": 56,
        "comparison_plan_sha256": core.digest(protocol), "execution_order": [5,1,2,3,4,6],
        "prepared_at_utc": (now or datetime.now(timezone.utc)).isoformat(), "model_calls_during_prepare": 0,
        "source_media_read_during_plan_binding": False, "upload_start_or_browser_latency_measured": False,
        "final_check_used": False, "execution_authorized_by_plan": False}
    if "native_profile_amendment" in metadata:
        plan["native_profile_amendment"] = metadata["native_profile_amendment"]
        plan["native_profile_amendment_sha256"] = metadata["native_profile_amendment"]["raw_sha256"]
    if "native_parallel4_public_gate" in metadata:
        plan["native_parallel4_public_gate"] = metadata["native_parallel4_public_gate"]
        plan["native_parallel4_public_gate_receipt_sha256"] = metadata["native_parallel4_public_gate"]["raw_sha256"]
    plan["attempts"] = [{"attempt_id": case["attempt_id"] + f"-model-{index:04d}", "source_attempt_id": case["attempt_id"],
        "case_id": case["case_id"], "content_sha256": case["content_sha256"], "split": "validation", "slot_index": index}
        for case in cases for index in range(case["maximum_model_requests"])]
    for name, artifact_name in zip(GATE_PINS, ("generated_gpu_gate", "generated_http_gate", "ready_capture")):
        plan[name] = metadata["artifacts"][artifact_name]["raw_sha256"]
    plan["recipe"] = _recipe(plan); plan["recipe_sha256"] = core.digest(plan["recipe"])
    validate_native_admission(plan, now=now)
    return plan


def validate_native_plan(plan):
    """Stable integrity checks, safe throughout an admitted 8220-second arm."""
    core, native, admission = parent_contract(), protocol_contract(), admission_contract()
    core.require(core.file_digest(native.__file__) == PROTOCOL_HELPER_SHA256
        and core.file_digest(admission.__file__) == ADMISSION_HELPER_SHA256,
        "native_plan_exact_frozen_protocol_helpers_required")
    core.require(plan.get("schema_version") == "forte.qwen38-attempt-plan.v1" and plan.get("kind") == "native_bf16_fullsix_accuracy"
        and plan.get("phase") == "evaluation" and plan.get("arm") in ROLES, "native_plan_scope_invalid")
    native.validate_protocol(plan["native_protocol"]); admission.validate_plan_addendum(plan)
    core.require(set(plan["artifacts"]) == ARTIFACT_NAMES, "native_plan_exact_actual_artifacts_required")
    for value in plan["artifacts"].values(): validate_artifact(value)
    config = plan["artifacts"]["native_config"]["document"]
    runtime = _runtime(config); role = ROLES[plan["arm"]]
    identity = runtime["expected"](config, role)
    core.require(plan["native_role"] == role and plan["native_variant"] == runtime["variant"]
        and plan["native_expected_runtime_identity"] == identity and plan["native_runtime_profile"] == runtime["profile"]
        and plan["native_runtime_profile_sha256"] == runtime["profile_sha256"]
        and plan["expected_model_identity"] == {"model_id": config["model_id"], "model_revision": config["model_revision"], "adapter_sha256": identity["adapter_sha256"]}
        and plan["runtime_bindings"] == {"source_archive_sha256": config["source_archive"]["sha256"],
            "runtime_image_sha256": config["runtime_image_sha256"], "base_manifest_sha256": config["base_manifest"]["sha256"]},
        "native_plan_actual_model_profile_or_source_binding_changed")
    _profile_amendment(plan, runtime)
    provider = plan["provider"]
    core.require(provider.get("launch_config_file_sha256") == plan["artifacts"]["native_config"]["raw_sha256"]
        and PurePosixPath(provider["launch_config"]).is_absolute() and provider["served_model_name"] == config["roles"][role]["served_model_name"]
        and isinstance(provider.get("base_url"), str) and provider["base_url"].startswith("http://")
        and _ints(provider,{"actual_max_attempts":1,"request_timeout_seconds":240})
        and all(plan.get("native_" + name + "_module") == runtime[name + "_module"] for name in ("provider", "identity", "profile", "readiness")),
        "native_plan_actual_provider_implementation_or_config_changed")
    cases = _cases(plan["native_protocol"], runtime, plan["arm"], plan["cases"])
    expected_slots = [{"attempt_id": case["attempt_id"] + f"-model-{index:04d}", "source_attempt_id": case["attempt_id"],
        "case_id": case["case_id"], "content_sha256": case["content_sha256"], "split": "validation", "slot_index": index}
        for case in cases for index in range(case["maximum_model_requests"])]
    core.require(plan["cases"] == cases and plan["attempts"] == expected_slots
        and plan["source_attempts"] == [{key: case[key] for key in ("attempt_id", *core.IDENTITY_FIELDS)} for case in cases]
        and plan["released_source_attempts"] == [case["attempt_id"] for case in cases]
        and plan["worker"] == {**plan["native_protocol"]["historical_recipe"]["worker_settings"], **runtime["pipeline"]}
        and plan["recipe"] == _recipe(plan) and plan["recipe_sha256"] == core.digest(plan["recipe"])
        and plan["comparison_plan_sha256"] == plan["native_protocol_sha256"], "native_plan_frozen_cases_slots_worker_or_recipe_changed")
    exact = {"source_attempts_required": 6, "maximum_private_model_requests": 208, "maximum_execution_seconds": 8220,
        "minimum_initial_generation_requests": 56, "execution_order": [5,1,2,3,4,6], "model_calls_during_prepare": 0,
        "source_media_read_during_plan_binding": False, "upload_start_or_browser_latency_measured": False,
        "final_check_used": False, "execution_authorized_by_plan": False}
    core.require(all(plan.get(key) == value and type(plan.get(key)) is type(value) for key,value in exact.items()),
        "native_plan_denominator_budget_or_authority_changed")
    for name, artifact_name in zip(GATE_PINS, ("generated_gpu_gate", "generated_http_gate", "ready_capture")):
        core.require(plan.get(name) == plan["artifacts"][artifact_name]["raw_sha256"], "native_plan_actual_gate_receipt_pin_changed")
    _validate_role(plan); _validate_consumer(plan, config, runtime); _validate_gates(plan, config, runtime)
    return plan


def validate_native_admission(plan, *, now=None):
    """Call once before claim/launch, never as a renewed per-POST role clock."""
    validate_native_plan(plan)
    now = now or datetime.now(timezone.utc)
    config = plan["artifacts"]["native_config"]["document"]
    _validate_role(plan, admission_now=now)
    _validate_gates(plan, config, _runtime(config), admission_now=now)
    return True


def validate_native_release_bindings(plan, release):
    core = parent_contract()
    admission_contract().validate_plan_addendum(plan, release)
    core.require(all(release.get(name) == plan[name] for name in (*GATE_PINS, "native_runtime_profile_sha256", "native_engine_instance_id"))
        and release.get("producer_source_archive_sha256") == plan["consumer_runtime_bindings"]["source_archive_sha256"]
        and release.get("producer_runtime_image_sha256") == plan["consumer_runtime_bindings"]["runtime_image_sha256"]
        and release.get("consumer_prepare_receipt_sha256") == plan["artifacts"]["consumer_prepare"]["raw_sha256"]
        and release.get("consumer_source_proof_sha256") == plan["artifacts"]["consumer_source_proof"]["raw_sha256"],
        "native_plan_exact_gate_and_separate_consumer_release_pins_required")
    if plan["native_variant"] == "parallel4_v1":
        core.require(release.get("native_profile_amendment_sha256") == plan["native_profile_amendment_sha256"]
            and release.get("native_parallel4_public_gate_receipt_sha256") == plan["native_parallel4_public_gate_receipt_sha256"]
            and release.get("native_profile_selected_after_public_benchmark") is True
            and release.get("native_variant") == "parallel4_v1",
            "native_plan_parallel4_release_amendment_missing")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--metadata", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    core = parent_contract()
    try:
        plan = bind_native_plan(core.read_json(args.protocol), core.read_json(args.metadata))
        core.write_new(args.output, plan)
        print('{"status":"native_fullsix_actual_plan_bound_not_released","source_attempts":6,"reserved_slot_scope":208,"model_calls":0}')
        return 0
    except Exception as error:
        print(json.dumps({"status":"native_fullsix_actual_plan_binding_failed","error_type":type(error).__name__,"model_calls":0}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
