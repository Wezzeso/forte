"""Generated plan packets tested through the actual staged native validators."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from benchmarks.performance import qwen38_accuracy_diagnostic as core
from benchmarks.performance import qwen38_native_plan as factory
from qa.system.test_qwen38_native_admission_addendum import prepared

STAGED = Path(__file__).resolve().parents[2] / "training/private/qwen38-vllm-integration-a10/prepared"


def _native_fixtures():
    spec = importlib.util.spec_from_file_location("generated_native_provider_fixtures", STAGED/"tests/test_qwen38_vllm_provider.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.synthetic_config, module.synthetic_ready


def packet(tmp_path, monkeypatch):
    protocol, addendum = prepared(monkeypatch)
    synthetic_config, synthetic_ready = _native_fixtures()
    config = synthetic_config(); now = datetime(2026,10,17,20,0,tzinfo=timezone.utc)
    ready = {role:synthetic_ready(config,role) for role in factory.ROLES.values()}
    artifacts = {}
    def put(name, document):
        path = tmp_path/(name+".json"); path.write_text(json.dumps(document), encoding="utf-8")
        artifacts[name] = factory.artifact(path)
    put("native_config", config)
    policy = {"expected_assignments_sha256":"a"*64,"expected_source_manifest_sha256":"b"*64,
        "expected_automated_split_sha256":"c"*64,"expected_scenario_groups_sha256":"d"*64}
    monkeypatch.setattr(factory,"ROLE_POLICY_PINS",policy)
    put("role_config",{**policy,"max_snapshot_age_sec":300,"allowed_splits":["validation"],"final_check_used":False})
    captured=(now-timedelta(seconds=10)).isoformat()
    put("accepted_inventory",{"schema_version":"forte.service-content-exposure-inventory.v1",
        "coverage_state":"complete_snapshot","captured_at_utc":captured})
    put("exposure_status",{"schema_version":"forte.exposure-coverage.v1","coverage_state":"complete_snapshot",
        "scope":"all_accepted_jobs_since_service_start","source_inventory_sha256":artifacts["accepted_inventory"]["raw_sha256"],
        "captured_at_utc":captured})
    put("role_preflight",{"schema_version":"forte.qwen38-role-preflight.v1","ok":True,"requested_contents":6,
        "allowed_split_counts":{"validation":6},"assignments_sha256":policy["expected_assignments_sha256"],
        "input_checksums":{name:policy["expected_"+name+"_sha256"] for name in ("source_manifest","automated_split","scenario_groups")},
        "human_verified":False,"scenario_independence_verified":False,"compute_authorized":False,"media_opened":False,
        "model_calls_made":False,"exposure_coverage_complete_snapshot":True,"exposure_snapshot_age_sec":10,
        "exposure_inventory_status_sha256":artifacts["exposure_status"]["raw_sha256"],"exposure_registry_tip_sha256":"e"*64,
        "exposure_registry_event_count":1})
    put("ready_capture",{"schema_version":"forte.qwen38-native-ready-capture.v1","captured_at_utc":captured,
        "config_file_sha256":artifacts["native_config"]["raw_sha256"],"runtime_profile_sha256":config["runtime_profile_sha256"],
        "engine_instance_id":ready["base_diagnostic"]["engine_instance_id"],"read_only_ready_observation":True,"model_posts":0,"ready_by_role":ready})
    put("generated_http_gate",ready["base_diagnostic"]["generated_http_gate"])
    gpu={"status":"passed","scope":"historical_offline_sequence_one","generated_calls":6,"original_tensors_verified":992,
        "logical_targets_verified":496,"physical_wrappers_verified":304,"base_dtype":"BF16","base_quantization":None,
        "does_not_prove_current_HTTP_engine":True,"runtime_image_sha256":config["runtime_image_sha256"],
        "model_manifest_sha256":config["base_manifest"]["sha256"],"adapter_sha256":config["roles"]["fit_candidate"]["adapter_sha256"]}
    put("generated_gpu_gate",{"generated_unit_chain_only":True,"normalized":gpu})
    monkeypatch.setattr(factory,"_gpu_chain",lambda document:document["normalized"])
    modules={name:"f"*64 for name in ("forte.pipeline","forte.model.adapter","forte.video.sampling","forte.video.decode",
        "forte.decision.engine","forte.evidence.assembly","serving.qwen38_vllm_provider","serving.qwen38_vllm_identity","serving.qwen38_vllm_profile")}
    consumer_source="9"*64;consumer_image="8"*64
    put("consumer_source_proof",{"schema_version":"forte.qwen38-native-consumer-source-proof.v1","source_archive_sha256":consumer_source,
        "runtime_image_sha256":consumer_image,"actual_imports_match_source_archive":True,"network_calls":0,"model_calls":0,"gpu_operations":0,
        "loaded_module_sha256":modules})
    from serving.qwen38_vllm_identity import expected_native_identity
    put("consumer_prepare",{"schema_version":"forte.qwen38.vllm-bf16.consumer-prepare.v1","status":"prepared",
        "source_archive_sha256":consumer_source,"selected_worker_image_sha256":consumer_image,
        "config_file_sha256":artifacts["native_config"]["raw_sha256"],"private_media_read":False,"actual_model_readiness_verified":False,
        "network_calls":0,"model_calls":0,"gpu_operations":0,"loaded_module_sha256":modules,
        "consumers":{role:{"expected_identity":expected_native_identity(config,role),"model_calls":0,"http_timeout_seconds":240,"job_timeout_seconds":285}
            for role in factory.ROLES.values()}})
    metadata={"arm":"qwen38_untuned","artifacts":artifacts,"native_admission_addendum":addendum,
        "launch_config_path":"/run/configs/native.json","base_url":"http://native-model:8000/v1",
        "consumer_runtime_bindings":{"source_archive_sha256":consumer_source,"runtime_image_sha256":consumer_image,
            "actual_module_sha256":modules,"consumer_source_proof_sha256":artifacts["consumer_source_proof"]["raw_sha256"],
            "consumer_prepare_receipt_sha256":artifacts["consumer_prepare"]["raw_sha256"]},
        "sources":[{**row,"owned_path":"/data/forte-training-sources/"+row["content_sha256"]+".mov"} for row in protocol["population"]]}
    return protocol,metadata,now


def changed_artifact(plan,name,mutate):
    document=deepcopy(plan["artifacts"][name]["document"]);mutate(document)
    raw=json.dumps(document)
    import hashlib
    plan["artifacts"][name]={"document":document,"raw_utf8":raw,"raw_sha256":hashlib.sha256(raw.encode()).hexdigest(),"semantic_sha256":core.digest(document)}


def test_actual_serial_validators_bind_all_six_and_keep_model_consumer_sources_separate(tmp_path,monkeypatch):
    protocol,metadata,now=packet(tmp_path,monkeypatch)
    plan=factory.bind_native_plan(protocol,metadata,now=now)
    assert len(plan["attempts"])==208 and len(plan["cases"])==len(plan["source_attempts"])==6
    assert [row["original_case_ordinal"] for row in plan["cases"]]==[5,1,2,3,4,6]
    assert plan["runtime_bindings"]["source_archive_sha256"]!=plan["consumer_runtime_bindings"]["source_archive_sha256"]
    assert plan["provider"]["actual_max_attempts"]==1 and plan["native_variant"]=="serial_native_v1"
    assert plan["worker"]["job_timeout_seconds"]==285 and plan["maximum_execution_seconds"]==8220
    # Integrity remains valid after300s; fresh role/ready is an admission-only rule.
    assert factory.validate_native_plan(plan) is plan
    with pytest.raises(ValueError,match="not_fresh_at_admission"):
        factory.validate_native_admission(plan,now=now+timedelta(seconds=301))


@pytest.mark.parametrize("mutation",[
    lambda p:p["attempts"].pop(),lambda p:p["source_attempts"].pop(),
    lambda p:p["cases"].reverse(),lambda p:p["cases"][0].__setitem__("diagnostic_deadline_seconds",285),
    lambda p:p["runtime_bindings"].__setitem__("source_archive_sha256",p["consumer_runtime_bindings"]["source_archive_sha256"]),
    lambda p:p["provider"].__setitem__("actual_max_attempts",2),
    lambda p:p.__setitem__("native_variant","parallel4_v1"),
    lambda p:p.__setitem__("native_engine_instance_id","changed_engine_instance_0001"),
    lambda p:p.__setitem__("native_generated_http_gate_receipt_sha256","0"*64),
    lambda p:p["consumer_runtime_bindings"]["actual_module_sha256"].pop("forte.pipeline"),
    lambda p:p.__setitem__("execution_authorized_by_plan",True),
])
def test_plan_denies_scope_runtime_engine_provider_and_module_switches(tmp_path,monkeypatch,mutation):
    protocol,metadata,now=packet(tmp_path,monkeypatch);plan=factory.bind_native_plan(protocol,metadata,now=now)
    mutation(plan)
    with pytest.raises((ValueError,RuntimeError)):
        factory.validate_native_plan(plan)


@pytest.mark.parametrize("mutate",[
    lambda r:r["ready_by_role"]["base_diagnostic"].__setitem__("draining",True),
    lambda r:r["ready_by_role"]["fit_candidate"]["loaded_engine"]["adapter_load"].__setitem__("original_tensors_verified",990),
    lambda r:r["ready_by_role"]["fit_candidate"]["live_admission"].__setitem__("drain_state","unknown"),
    lambda r:r["ready_by_role"]["fit_candidate"].__setitem__("engine_instance_id","other_engine_instance_0002"),
    lambda r:r["ready_by_role"]["fit_candidate"]["generated_http_gate"].__setitem__("attempts",5),
])
def test_finished_actual_runtime_contract_rejects_990_tensors_unknown_drain_old_engine_or_partial_HTTP6(tmp_path,monkeypatch,mutate):
    protocol,metadata,now=packet(tmp_path,monkeypatch)
    # Re-encode honest changed bytes, so denial tests the native contract rather than JSON hash mismatch.
    holder={"artifacts":metadata["artifacts"]};changed_artifact(holder,"ready_capture",mutate)
    with pytest.raises((ValueError,RuntimeError)):
        factory.bind_native_plan(protocol,metadata,now=now)


def test_historical_GPU6_does_not_replace_actual_HTTP_and_ready_proofs(tmp_path,monkeypatch):
    protocol,metadata,now=packet(tmp_path,monkeypatch)
    metadata["artifacts"].pop("ready_capture")
    with pytest.raises(ValueError,match="exact_actual_artifacts"):
        factory.bind_native_plan(protocol,metadata,now=now)
