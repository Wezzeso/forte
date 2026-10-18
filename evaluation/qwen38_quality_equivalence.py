"""Bind genuine later serving archives to an unchanged frozen quality family.

The evaluator archive remains the baseline's original archive. Actual model
serving/training archives have separate pins. The only optional common-source
delta is an independently checked candidate-only timeout forwarding keyword.
This module makes no model, media, training, or network calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

from evaluation.io import digest, read_json, read_records
from evaluation.qwen38_comparison import (ARMS, IDENTITY_FIELDS, MODEL_FIELDS,
    _confusion, _count, _distribution, _index, _normalize, _references, _summary)
from evaluation.qwen38_pipeline import file_sha
from evaluation.qwen38_quality_family import arm_plan, require, validate_family
from evaluation.qwen38_server_export import write_private_new
from evaluation.weak_supervision import provided_label_digest
from serving.qwen38_comparison_profile import runtime_profile, runtime_profile_sha256

SCHEMA = "forte.qwen38-quality-source-equivalence.v1"
TIMEOUT_PATH = "src/forte/pipeline.py"
TIMEOUT_POLICY = "candidate_request_timeout_forwarding.v1"
INFERENCE_PATH = "serving/qwen38_service.py"
INFERENCE_POLICY = "candidate_peft_inference_activation.v1"
REVIEWED_INFERENCE_SERVICE_SHA256 = "513460f34644edd9876db3294ec611477c0cb5c44f6b0a8ebbb69eb759585d7f"
INFERENCE_STEP_ALLOWED = {INFERENCE_PATH,"serving/qwen38_source_lineage.py",
    "evaluation/qwen38_quality_equivalence.py","serving/tests/test_qwen38_inference_lineage.py"}
PREFIXES = ("src/forte/", "rules/", "annotations/")
EXACT = {"models/manifest.json", "configs/serving/qwen38/public-candidate.manifest.json",
    "evaluation/io.py", "evaluation/metrics.py", "evaluation/weak_supervision.py",
    "evaluation/qwen38_comparison.py", "evaluation/qwen38_pipeline.py",
    "evaluation/qwen38_quality_family.py", "evaluation/qwen38_server_export.py",
    "training/qwen38_preflight.py", "training/guards.py", "scripts/audit_data.py",
    "scripts/build_reviewed_split.py", "scripts/exposure_registry.py", "scripts/split_gate.py"}


def archive_manifest(path: Path) -> dict:
    files = {}
    total = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and ".." not in name.parts
                and (member.isfile() or member.isdir()), "quality_equivalence_archive_member_invalid")
            if not member.isfile():
                continue
            key = str(name)
            total += member.size
            require(key not in files and 0 <= member.size <= 64 * 1024 * 1024
                and total <= 256 * 1024 * 1024, "quality_equivalence_archive_limits")
            with archive.extractfile(member) as stream:
                files[key] = hashlib.sha256(stream.read()).hexdigest()
    return files


def common_paths(files: dict) -> set:
    return {name for name in files if name.startswith(PREFIXES)} | (EXACT & files.keys())


def _member_bytes(archive_path: Path, wanted: str) -> bytes:
    with tarfile.open(archive_path, "r:*") as archive:
        matches = [member for member in archive if str(PurePosixPath(member.name)) == wanted]
        require(len(matches) == 1 and matches[0].isfile() and matches[0].size <= 1024 * 1024,
            "quality_equivalence_public_source_member_invalid")
        with archive.extractfile(matches[0]) as stream:
            return stream.read()


def _verify_common_deltas(receipt: dict, old: dict, new: dict, paths: set) -> set:
    require(not (common_paths(new) - paths), "quality_equivalence_added_common_source")
    deltas = receipt.get("audited_common_source_deltas", [])
    require(isinstance(deltas, list) and len(deltas) <= 1, "quality_equivalence_unreviewed_common_delta")
    allowed = set()
    for row in deltas:
        require(set(row) == {"path", "policy", "evaluator_source_utf8", "actual_serving_source_utf8", "proof"}
            and row["path"] == TIMEOUT_PATH and row["policy"] == TIMEOUT_POLICY and row["path"] in paths,
            "quality_equivalence_unreviewed_common_delta")
        old_bytes, new_bytes = row["evaluator_source_utf8"].encode(), row["actual_serving_source_utf8"].encode()
        require(hashlib.sha256(old_bytes).hexdigest() == old.get(TIMEOUT_PATH)
            and hashlib.sha256(new_bytes).hexdigest() == new.get(TIMEOUT_PATH),
            "quality_equivalence_timeout_source_hash_mismatch")
        from serving.qwen38_source_lineage import verify_candidate_timeout_forwarding
        proof = verify_candidate_timeout_forwarding(old_bytes, new_bytes)
        require(row["proof"] == proof, "quality_equivalence_timeout_proof_changed")
        allowed.add(TIMEOUT_PATH)
    changed = {name for name in paths if old.get(name) != new.get(name)}
    require(changed == allowed, "quality_equivalence_common_files_changed")
    identical = {name: old[name] for name in sorted(paths - allowed)}
    if allowed or "byte_identical_common_file_sha256" in receipt:
        require(receipt.get("byte_identical_common_file_sha256") == identical
            and receipt.get("protected_common_file_count") == len(paths)
            and receipt.get("byte_identical_common_file_count") == len(identical)
            and receipt.get("audited_common_source_delta_count") == len(allowed)
            and receipt.get("all_common_files_byte_identical") is (not allowed),
            "quality_equivalence_common_file_counts_changed")
    return allowed


def _archive_step(before,after,before_sha,after_sha):
    changed=sorted(name for name in before.keys()|after.keys() if before.get(name)!=after.get(name))
    protected=common_paths(before)|common_paths(after)
    equal=sum(name in before and name in after and before[name]==after[name] for name in protected)
    return {"from_archive_sha256":before_sha,"to_archive_sha256":after_sha,
        "from_file_count":len(before),"to_file_count":len(after),
        "byte_identical_file_count":sum(before[name]==after[name] for name in before.keys()&after.keys()),
        "changed_existing_file_count":sum(name in before and name in after for name in changed),
        "added_file_count":sum(name not in before for name in changed),
        "removed_file_count":sum(name not in after for name in changed),
        "protected_common_path_count":len(protected),"byte_identical_protected_common_path_count":equal,
        "changed_protected_common_path_count":len(protected)-equal,
        "changed_members":[{"path":name,"before_sha256":before.get(name),"after_sha256":after.get(name)} for name in changed]}


def _verify_serving_deltas(receipt: dict, actual: dict) -> None:
    rows=receipt.get("audited_serving_source_deltas", [])
    require(isinstance(rows,list) and len(rows)<=1,"quality_equivalence_unreviewed_serving_delta")
    if actual.get(INFERENCE_PATH)==REVIEWED_INFERENCE_SERVICE_SHA256:
        require(len(rows)==1,"quality_equivalence_reviewed_serving_proof_required")
    for row in rows:
        require(set(row)=={"path","policy","predecessor_source_archive_sha256","predecessor_archive_files",
            "predecessor_source_utf8","actual_serving_source_utf8","proof"}
            and row["path"]==INFERENCE_PATH and row["policy"]==INFERENCE_POLICY
            and isinstance(row["predecessor_source_archive_sha256"],str)
            and re.fullmatch(r"[a-f0-9]{64}",row["predecessor_source_archive_sha256"]),
            "quality_equivalence_unreviewed_serving_delta")
        previous=row["predecessor_archive_files"]
        require(isinstance(previous,dict),"quality_equivalence_serving_predecessor_inventory_invalid")
        changed={name for name in previous.keys()|actual.keys() if previous.get(name)!=actual.get(name)}
        require(INFERENCE_PATH in changed and changed<=INFERENCE_STEP_ALLOWED,
            "quality_equivalence_unreviewed_predecessor_member_change")
        require(isinstance(previous,dict) and common_paths(previous)==common_paths(actual)
            and all(previous[name]==actual[name] for name in common_paths(previous)),
            "quality_equivalence_serving_predecessor_common_changed")
        old_bytes,new_bytes=row["predecessor_source_utf8"].encode(),row["actual_serving_source_utf8"].encode()
        require(hashlib.sha256(old_bytes).hexdigest()==previous.get(INFERENCE_PATH)
            and hashlib.sha256(new_bytes).hexdigest()==actual.get(INFERENCE_PATH),
            "quality_equivalence_serving_source_hash_mismatch")
        from serving.qwen38_source_lineage import verify_candidate_inference_mode
        require(row["proof"]==verify_candidate_inference_mode(old_bytes,new_bytes),
            "quality_equivalence_serving_inference_proof_changed")
    if rows or "audited_serving_source_delta_count" in receipt:
        require(receipt.get("audited_serving_source_delta_count")==len(rows),
            "quality_equivalence_serving_delta_count_changed")
    chain=receipt.get("training_serving_archive_chain")
    if actual.get(INFERENCE_PATH)==REVIEWED_INFERENCE_SERVICE_SHA256:
        require(isinstance(chain,dict),"quality_equivalence_measured_training_serving_chain_required")
    if chain is not None:
        require(len(rows)==1 and set(chain)=={"training_source_archive_sha256","training_archive_files","steps"}
            and isinstance(chain["training_source_archive_sha256"],str)
            and re.fullmatch(r"[a-f0-9]{64}",chain["training_source_archive_sha256"]),
            "quality_equivalence_training_serving_chain_invalid")
        previous=rows[0]["predecessor_archive_files"];training=chain["training_archive_files"]
        training_sha=chain["training_source_archive_sha256"];previous_sha=rows[0]["predecessor_source_archive_sha256"]
        actual_sha=receipt["actual_serving_source_archive_sha256"]
        require(chain["steps"]==[_archive_step(training,previous,training_sha,previous_sha),
            _archive_step(previous,actual,previous_sha,actual_sha),_archive_step(training,actual,training_sha,actual_sha)],
            "quality_equivalence_archive_step_counts_or_members_changed")


def validate_equivalence(family: dict, receipt: dict, actual_source: str) -> None:
    validate_family(family)
    require(receipt.get("schema_version") == SCHEMA and receipt.get("family_sha256") == digest(family)
        and receipt.get("evaluator_source_archive_sha256") == family["recipe"]["source_archive_sha256"]
        and receipt.get("actual_serving_source_archive_sha256") == actual_source,
        "quality_equivalence_archive_attribution_mismatch")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    require(receipt.get("receipt_sha256") == digest(body), "quality_equivalence_receipt_changed")
    old, new = receipt["evaluator_archive_files"], receipt["actual_serving_archive_files"]
    paths = common_paths(old)
    require(paths and receipt.get("common_file_sha256") == {name: old[name] for name in sorted(paths)},
        "quality_equivalence_common_inventory_changed")
    allowed_deltas = _verify_common_deltas(receipt, old, new, paths)
    _verify_serving_deltas(receipt,new)
    pins = {**family["recipe"].get("prompt_implementation_pins", {}),
        "rules/catalog.json": family["recipe"]["rules_sha256"]}
    if family["recipe"].get("taxonomy_sha256"):
        pins["annotations/taxonomy.v1.json"] = family["recipe"]["taxonomy_sha256"]
    require(all(old.get(name) == value and (new.get(name) == value or name in allowed_deltas)
        for name, value in pins.items()),
        "quality_equivalence_frozen_prompt_rule_or_taxonomy_changed")
    expected_changes = [{"path": name, "evaluator_sha256": old.get(name), "actual_serving_sha256": new.get(name),
        "scope": "audited_candidate_client_timeout" if name in allowed_deltas else
            "model_serving" if name.startswith("serving/") else "training" if name.startswith("training/") else "other_source"}
        for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name)]
    require(receipt.get("source_changes") == expected_changes
        and isinstance(receipt.get("source_change_disclosure"), str) and receipt["source_change_disclosure"].strip(),
        "quality_equivalence_source_changes_not_disclosed")


def prove(family: dict, evaluator_archive: Path, actual_archive: Path, *, disclosure: str,
          allow_candidate_timeout_forwarding: bool = False,
          serving_predecessor_archive: Path|None = None,serving_predecessor_sha256: str|None = None,
          training_source_archive: Path|None = None,training_source_sha256: str|None = None) -> dict:
    validate_family(family)
    require(file_sha(evaluator_archive) == family["recipe"]["source_archive_sha256"],
        "quality_equivalence_wrong_frozen_evaluator_archive")
    old, new = archive_manifest(evaluator_archive), archive_manifest(actual_archive)
    paths = common_paths(old)
    require(not (common_paths(new) - paths), "quality_equivalence_added_common_source")
    changed_common = {name for name in paths if old.get(name) != new.get(name)}
    deltas = []
    if changed_common:
        require(allow_candidate_timeout_forwarding and changed_common == {TIMEOUT_PATH},
            "quality_equivalence_common_files_changed")
        from serving.qwen38_source_lineage import verify_candidate_timeout_forwarding
        old_bytes, new_bytes = _member_bytes(evaluator_archive, TIMEOUT_PATH), _member_bytes(actual_archive, TIMEOUT_PATH)
        deltas.append({"path": TIMEOUT_PATH, "policy": TIMEOUT_POLICY,
            "evaluator_source_utf8": old_bytes.decode("utf-8"), "actual_serving_source_utf8": new_bytes.decode("utf-8"),
            "proof": verify_candidate_timeout_forwarding(old_bytes, new_bytes)})
    receipt = {"schema_version": SCHEMA, "family_sha256": digest(family),
        "evaluator_source_archive_sha256": file_sha(evaluator_archive),
        "actual_serving_source_archive_sha256": file_sha(actual_archive),
        "evaluator_archive_files": old, "actual_serving_archive_files": new,
        "common_file_sha256": {name: old[name] for name in sorted(common_paths(old))},
        "protected_common_file_count": len(paths), "byte_identical_common_file_count": len(paths - changed_common),
        "byte_identical_common_file_sha256": {name: old[name] for name in sorted(paths - changed_common)},
        "audited_common_source_delta_count": len(deltas), "audited_common_source_deltas": deltas,
        "all_common_files_byte_identical": not deltas,
        "source_changes": [{"path": name, "evaluator_sha256": old.get(name), "actual_serving_sha256": new.get(name),
            "scope": "audited_candidate_client_timeout" if name in changed_common else
                "model_serving" if name.startswith("serving/") else "training" if name.startswith("training/") else "other_source"}
            for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name)],
        "source_change_disclosure": disclosure, "actual_archive_bytes_read_and_hashed": True,
        "frozen_family_or_baseline_records_modified": False, "model_calls": 0}
    require((serving_predecessor_archive is None)==(serving_predecessor_sha256 is None),
        "quality_equivalence_exact_serving_predecessor_required")
    if serving_predecessor_archive is not None:
        require(file_sha(serving_predecessor_archive)==serving_predecessor_sha256,
            "quality_equivalence_serving_predecessor_archive_mismatch")
        previous=archive_manifest(serving_predecessor_archive)
        old_service,new_service=_member_bytes(serving_predecessor_archive,INFERENCE_PATH),_member_bytes(actual_archive,INFERENCE_PATH)
        from serving.qwen38_source_lineage import verify_candidate_inference_mode
        receipt["audited_serving_source_deltas"]=[{"path":INFERENCE_PATH,"policy":INFERENCE_POLICY,
            "predecessor_source_archive_sha256":serving_predecessor_sha256,"predecessor_archive_files":previous,
            "predecessor_source_utf8":old_service.decode(),"actual_serving_source_utf8":new_service.decode(),
            "proof":verify_candidate_inference_mode(old_service,new_service)}]
        receipt["audited_serving_source_delta_count"]=1
    require((training_source_archive is None)==(training_source_sha256 is None),
        "quality_equivalence_exact_training_archive_required")
    if training_source_archive is not None:
        require(serving_predecessor_archive is not None and file_sha(training_source_archive)==training_source_sha256,
            "quality_equivalence_training_archive_mismatch")
        training=archive_manifest(training_source_archive)
        receipt["training_serving_archive_chain"]={"training_source_archive_sha256":training_source_sha256,
            "training_archive_files":training,"steps":[_archive_step(training,previous,training_source_sha256,serving_predecessor_sha256),
                _archive_step(previous,new,serving_predecessor_sha256,file_sha(actual_archive)),
                _archive_step(training,new,training_source_sha256,file_sha(actual_archive))]}
    receipt["receipt_sha256"] = digest(receipt)
    validate_equivalence(family, receipt, file_sha(actual_archive))
    return receipt


def bind_arm(family: dict, spec: dict, receipt: dict, config: dict, *, role_config_sha256: str) -> dict:
    require(spec["arm"] in {"qwen38_untuned", "qwen38_tuned"}, "quality_equivalence_future_27b_arm_required")
    actual_source = spec["runtime_bindings"]["source_archive_sha256"]
    validate_equivalence(family, receipt, actual_source)
    identity = spec["expected_model_identity"]
    require(config["source_archive"]["sha256"] == actual_source
        and config["runtime_image_sha256"] == spec["runtime_bindings"]["runtime_image_sha256"]
        and config["base_manifest"]["sha256"] == spec["runtime_bindings"]["base_manifest_sha256"]
        and all(config.get(key) == identity[key] for key in MODEL_FIELDS), "quality_equivalence_actual_runtime_config_mismatch")
    recipe = family["recipe"]
    require(all(config["generation"].get(key) == value for key, value in recipe["generation"].items())
        and config["limits"]["max_images"] == recipe["worker_settings"]["max_frames_per_window"]
        and config["limits"]["max_context_tokens"] == recipe["context_tokens"],
        "quality_equivalence_runtime_generation_or_frame_context_changed")
    profile = runtime_profile_sha256(config)
    require(config.get("runtime_profile_sha256", profile) == profile, "quality_equivalence_runtime_profile_changed")
    shared_spec = deepcopy(spec)
    shared_spec["runtime_bindings"]["source_archive_sha256"] = recipe["source_archive_sha256"]
    plan = arm_plan(family, shared_spec, role_config_sha256=role_config_sha256)
    # This is a new future-arm plan. Its execution binds genuine serving bytes;
    # the original baseline plan and frozen family are never altered.
    plan["runtime_bindings"] = deepcopy(spec["runtime_bindings"])
    plan.update(evaluator_source_archive_sha256=recipe["source_archive_sha256"],
        serving_source_archive_sha256=actual_source, source_equivalence=receipt,
        source_equivalence_sha256=digest(receipt), comparison_runtime_profile=runtime_profile(config),
        comparison_runtime_profile_sha256=profile)
    plan["audited_common_source_deltas"] = [{key: value for key, value in row.items()
        if key not in {"evaluator_source_utf8", "actual_serving_source_utf8"}}
        for row in receipt.get("audited_common_source_deltas", [])]
    plan["effective_client_request_timeout_seconds"] = config["limits"]["request_timeout_seconds"]
    for case in plan["cases"]:
        case["case_id"] = case["attempt_id"]
    for slot in plan["attempts"]:
        slot["case_id"] = slot["source_attempt_id"]
    return plan


def comparable_runtime_profiles(first: dict, second: dict) -> dict:
    """Keep genuine identities; permit only the paired load-only serving fix."""
    left,right=first["comparison_runtime_profile"],second["comparison_runtime_profile"]
    details={"actual_runtime_profile_sha256":[digest(left),digest(right)],
        "actual_source_archive_sha256":[left["source_archive_sha256"],right["source_archive_sha256"]],
        "pure_model_only_causal_claim":False}
    if left==right:return {**details,"exact_profile_identity":True,"audited_serving_control_variant":False}
    require({k:v for k,v in left.items() if k!="source_archive_sha256"}
        =={k:v for k,v in right.items() if k!="source_archive_sha256"},
        "quality_equivalence_27b_runtime_profiles_differ")
    for newer,previous in ((first,second),(second,first)):
        receipt=newer.get("source_equivalence",{});older=previous.get("source_equivalence",{})
        for row in receipt.get("audited_serving_source_deltas",[]):
            if row["predecessor_source_archive_sha256"]!=previous["runtime_bindings"]["source_archive_sha256"]:continue
            require(row["predecessor_archive_files"]==older.get("actual_serving_archive_files")
                and newer["comparison_runtime_profile"]["source_archive_sha256"]==receipt.get("actual_serving_source_archive_sha256")
                and previous["comparison_runtime_profile"]["source_archive_sha256"]==older.get("actual_serving_source_archive_sha256"),
                "quality_equivalence_serving_profile_predecessor_unbound")
            _verify_serving_deltas(receipt,receipt["actual_serving_archive_files"])
            return {**details,"exact_profile_identity":False,"identical_non_source_profile_fields":True,
                "audited_serving_control_variant":True,"serving_control_policy":INFERENCE_POLICY,
                "actual_old_and_new_source_identities_preserved":True}
    require(False,"quality_equivalence_27b_serving_profile_variant_unproven")


def compare(family: dict, plans: list[dict], terminals: list[dict], labels: list[dict], *, metadata: dict) -> dict:
    records, expected = validate_family(family)
    actual = _index(terminals, "attempt_id")
    require(not actual.keys() - expected.keys(), "quality_equivalence_unplanned_source_result")
    arms = {plan["arm"]: plan for plan in plans}
    require(len(plans) == 3 and set(arms) == set(ARMS), "quality_equivalence_all_three_actual_arm_plans_required")
    references = _references(labels, records)
    require(provided_label_digest(labels) == family["provided_labels_sha256"], "quality_equivalence_reference_pin_changed")
    family_sha, blockers, reports = digest(family), Counter(), {}
    for arm in ARMS:
        plan = arms[arm]
        require(plan["comparison_plan_sha256"] == family_sha and plan["recipe"] == family["recipe"]
            and plan["recipe_sha256"] == family["recipe_sha256"]
            and plan["worker"] == family["recipe"]["worker_settings"], "quality_equivalence_common_recipe_changed")
        source = plan["runtime_bindings"]["source_archive_sha256"]
        if arm == "baseline_8b":
            require(source == family["recipe"]["source_archive_sha256"], "quality_equivalence_original_baseline_source_changed")
        else:
            validate_equivalence(family, plan["source_equivalence"], source)
            require(plan.get("source_equivalence_sha256") == digest(plan["source_equivalence"])
                and plan.get("evaluator_source_archive_sha256") == family["recipe"]["source_archive_sha256"]
                and plan.get("serving_source_archive_sha256") == source
                and plan.get("comparison_runtime_profile_sha256") == digest(plan["comparison_runtime_profile"]),
                "quality_equivalence_bound_proof_or_profile_changed")
            profile = plan["comparison_runtime_profile"]
            require(profile["source_archive_sha256"] == source
                and profile["runtime_image_sha256"] == plan["runtime_bindings"]["runtime_image_sha256"]
                and profile["base_manifest_sha256"] == plan["runtime_bindings"]["base_manifest_sha256"],
                "quality_equivalence_profile_actual_artifact_mismatch")
            expected_deltas = [{key: value for key, value in row.items()
                if key not in {"evaluator_source_utf8", "actual_serving_source_utf8"}}
                for row in plan["source_equivalence"].get("audited_common_source_deltas", [])]
            require(plan.get("audited_common_source_deltas", []) == expected_deltas
                and plan.get("effective_client_request_timeout_seconds", profile["limits"]["request_timeout_seconds"])
                    == profile["limits"]["request_timeout_seconds"],
                "quality_equivalence_effective_timeout_or_audit_disclosure_changed")
        expected_arm = {aid: row for aid, row in expected.items() if row["arm"] == arm}
        require({row["attempt_id"]: row for row in plan["source_attempts"]} == expected_arm,
            "quality_equivalence_source_denominator_changed")
        identity = plan["expected_model_identity"]
        require(set(identity) == set(MODEL_FIELDS)
            and {key: identity[key] for key in MODEL_FIELDS[:2]} == family["base_models"][arm]
            and (identity["adapter_sha256"] is None if arm != "qwen38_tuned" else isinstance(identity["adapter_sha256"], str)
                and re.fullmatch(r"[a-f0-9]{64}",identity["adapter_sha256"])), "quality_equivalence_actual_model_identity_invalid")
        confusion, failures, timings, completed = _confusion(), Counter(), [], 0
        for aid, planned in expected_arm.items():
            row = actual.get(aid)
            if row is not None:
                require(all(row.get(key) == planned[key] for key in IDENTITY_FIELDS)
                    and row.get("comparison_plan_sha256") == family_sha
                    and row.get("recipe_sha256") == family["recipe_sha256"], "quality_equivalence_original_terminal_binding_changed")
                require(row.get("model_identity" if row.get("status") == "done" else "expected_model_identity") == identity,
                    "quality_equivalence_terminal_model_identity_changed")
            normalized = _normalize(row, records[planned["sample_id"]])
            completed += normalized["completed"]
            if not normalized["completed"]:
                failures[str(normalized["failure"])] += 1
            reference = references.get(planned["content_sha256"])
            if reference is not None:
                _count(confusion, reference["provided_video_label"], normalized)
            if row is None:
                blockers["missing_source_terminal"] += 1
            if normalized["completed"]:
                if row.get("server_receipt_verified") is not True:
                    blockers["fresh_native_evidence_receipt_unverified"] += 1
                value = row.get("pipeline_start_to_result_ms")
                if type(value) in (int, float) and value >= 0:
                    timings.append(value)
        reports[arm] = {"actual_model_identity": identity, "original_arm_plan_sha256": digest(plan),
            "actual_runtime_bindings": plan["runtime_bindings"],
            "evaluator_source_archive_sha256": family["recipe"]["source_archive_sha256"],
            "source_equivalence_sha256": plan.get("source_equivalence_sha256"),
            "comparison_runtime_profile_sha256": plan.get("comparison_runtime_profile_sha256"),
            "actual_runtime_config_file_sha256":plan.get("provider",{}).get("launch_config_file_sha256"),
            "audited_serving_control_delta_count":plan.get("source_equivalence",{}).get("audited_serving_source_delta_count",0),
            "effective_client_request_timeout_seconds": plan.get("effective_client_request_timeout_seconds",
                family["recipe"]["worker_settings"]["job_timeout_seconds"] if arm == "baseline_8b" else
                plan["comparison_runtime_profile"]["limits"]["request_timeout_seconds"]),
            "audited_common_source_deltas": plan.get("audited_common_source_deltas", []),
            "completion": {"required_attempts": 6, "completed": completed, "failed_or_unprocessed": 6-completed,
                "failure_counts": dict(failures)}, "provided_binary": _summary(confusion),
            "pipeline_start_to_result_ms_not_upload_or_ui": _distribution(timings)}
    profile_comparability=comparable_runtime_profiles(arms["qwen38_untuned"],arms["qwen38_tuned"])
    for key in ("family_frozen_before_baseline", "all_arm_plans_frozen_before_respective_execution",
                "actual_root_releases_and_exposure_reservations_verified", "validation_reuse_disclosed"):
        if metadata.get(key) is not True:
            blockers[key + "_unverified"] += 1
    if metadata.get("final_check_used") is not False:
        blockers["final_check_exclusion_unverified"] += 1
    if len(references) != 6:
        blockers["provided_binary_reference_incomplete"] += 1
    report = {"schema_version": "forte.qwen38-quality-equivalent-source-comparison.v1",
        "status": "SYNTHETIC_ONLY" if metadata.get("inference_kind") == "synthetic" else
            "DEVELOPMENT_DIAGNOSTIC_READY" if metadata.get("inference_kind") == "real_local" and not blockers
            else "PARTIAL_OR_PROVENANCE_INCOMPLETE", "family_sha256": family_sha,
        "recipe_sha256": family["recipe_sha256"], "arms": reports,
        "runtime_profile_comparability":profile_comparability,
        "source_equivalence_receipts": {arm: arms[arm]["source_equivalence"] for arm in ARMS[1:]},
        "original_terminal_records_sha256": digest(terminals), "original_arm_plans_sha256": digest(plans),
        "provided_labels_sha256": provided_label_digest(labels), "blocker_counts": dict(blockers),
        "metadata_sha256": digest(metadata), "frozen_family_baseline_or_terminal_records_modified": False,
        "claim_limits": {"validation_is_reused_development": True, "final_check_read_or_evaluated": False,
            "heldout_or_universal_accuracy_claim": False, "event_type_fact_support_legal_boundary_gold": False,
            "upload_or_browser_latency_measured": False, "model_promotion_authorized": False,
            "pure_model_only_causal_comparison": False},
        "condition_differences": "Baseline architecture/processor and caller configuration are historical comparison conditions. Both 27B arms require identical non-source runtime profile fields and client request timeout. A source identity difference is permitted only for the separately paired byte/AST-proven PEFT inference-mode activation fix. Actual source/config/profile hashes are preserved. Candidate-only timeout forwarding is separately audited; protected common and serving-only changes have distinct measured counts. Failed cases remain unpublished failures, never negative predictions. No pure model-only causal claim is made."}
    report["report_sha256"] = digest(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    proof = commands.add_parser("prove")
    for name in ("family", "evaluator-archive", "actual-archive", "output"):
        proof.add_argument("--"+name, type=Path, required=True)
    proof.add_argument("--disclosure", required=True)
    proof.add_argument("--allow-candidate-timeout-forwarding", action="store_true")
    proof.add_argument("--serving-predecessor-archive",type=Path)
    proof.add_argument("--serving-predecessor-sha256")
    proof.add_argument("--training-source-archive",type=Path)
    proof.add_argument("--training-source-sha256")
    bind = commands.add_parser("bind-arm")
    for name in ("family", "spec", "equivalence", "runtime-config", "role-config", "output"):
        bind.add_argument("--"+name, type=Path, required=True)
    report = commands.add_parser("compare")
    for name in ("family", "labels", "metadata", "output"):
        report.add_argument("--"+name, type=Path, required=True)
    for name in ("arm-plan", "terminals"):
        report.add_argument("--"+name, type=Path, required=True, action="append")
    args = parser.parse_args(argv)
    try:
        family = read_json(args.family)
        if args.command == "prove":
            result = prove(family, args.evaluator_archive, args.actual_archive, disclosure=args.disclosure,
                allow_candidate_timeout_forwarding=args.allow_candidate_timeout_forwarding,
                serving_predecessor_archive=args.serving_predecessor_archive,
                serving_predecessor_sha256=args.serving_predecessor_sha256,
                training_source_archive=args.training_source_archive,training_source_sha256=args.training_source_sha256)
        elif args.command == "bind-arm":
            result = bind_arm(family, read_json(args.spec), read_json(args.equivalence),
                read_json(args.runtime_config), role_config_sha256=file_sha(args.role_config))
        else:
            result = compare(family, [read_json(path) for path in args.arm_plan],
                [row for path in args.terminals for row in read_records(path)], read_records(args.labels),
                metadata=read_json(args.metadata))
            result["original_input_file_sha256"] = {"family": file_sha(args.family),
                "arm_plans": [file_sha(path) for path in args.arm_plan],
                "terminals": [file_sha(path) for path in args.terminals],
                "provided_labels": file_sha(args.labels), "metadata": file_sha(args.metadata)}
            result["report_sha256"] = digest({key:value for key,value in result.items() if key!="report_sha256"})
        write_private_new(args.output, result)
        print(json.dumps({"status": args.command+"_prepared", "model_calls": 0,
            "frozen_family_or_baseline_modified": False, "raw_private_values_printed": False}))
        return 0
    except Exception as error:
        print(json.dumps({"status": "quality_equivalence_input_invalid", "error_type": type(error).__name__,
            "raw_private_values_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
