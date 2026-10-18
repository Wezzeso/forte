"""Synthetic-only isolation, provenance and attempt-reservation checks."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_automated_split import build_automated_candidate
from scripts.exposure_registry import append_exposures, file_digest, read_registry, register_accepted_inventory
from training.guards import TrainingGateError
from training.qwen38_preflight import (FIXED_STOP_UTC, reconcile_attempt_outcomes, reserve_attempt_plan, semantic_recipe_sha256, validate_attempt_reservation,
                                       validate_role_metadata, validate_runtime_relocation, validate_silver_metadata)


def write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    records, workbook, groups = [], [], []
    for index in range(20):
        content = hashlib.sha256(f"generated-{index}".encode()).hexdigest()
        records.append({"recording_id": f"record-{index}", "content_hash": content,
                        "group_id": f"group-{index}", "workbook_row_ids": [index + 2], "duration_sec": 10})
        workbook.append({"row_id": index + 2, "has_violation": bool(index % 2)})
        groups.append({"proposed_group_id": f"proposal-{index}", "recording_ids": [f"record-{index}"],
                       "method": "exact_content_only", "confidence_status": "heuristic_unvalidated"})
    manifest = private / "recordings.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    registry = private / "exposure.jsonl"
    append_exposures(registry, records, {records[0]["content_hash"]}, trial_id="generated-seed", phase="baseline", purpose="benchmark_attempt")
    inventory = write(private / "accepted.json", {"schema_version": "forte.service-content-exposure-inventory.v1",
        "coverage_state": "complete_snapshot", "scope": "all_accepted_jobs_since_service_start",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "accepted_job_count": 1,
        "unique_content_count": 1, "active_job_count": 0, "records": [{"content_hash": records[0]["content_hash"]}]})
    status = private / "status.json"
    register_accepted_inventory(inventory, registry, manifest, status)
    proposal = {"schema_version": "forte.automatic-group-proposals.v1", "proposal_origin": "local_model_and_metadata",
                "method": "generated-label-blind", "config_sha256": "a" * 64,
                "source_manifest_sha256": file_digest(manifest), "groups": groups}
    split, scenario, _, _ = build_automated_candidate(records, workbook, proposal, read_registry(registry, records),
                                                     manifest_sha256=file_digest(manifest), workbook_sha256="b" * 64)
    paths = {"source_manifest": manifest, "automated_split": write(private / "split.json", split),
             "scenario_groups": write(private / "scenario.json", scenario)}
    config = {key: str(value) for key, value in paths.items()}
    config.update({"expected_" + key + "_sha256": file_digest(value) for key, value in paths.items()})
    config.update(exposure_registry=str(registry), exposure_inventory_status=str(status),
                  source_inventory_path_override=str(inventory), expected_assignments_sha256=split["assignments_sha256"])
    return config, split, private, records


def content(split, role):
    return next(row["content_hash"] for row in split["records"] if row["split"] == role)


def test_separate_role_gate_has_no_compute_or_gold_authority(inputs):
    config, split, _, _ = inputs
    result = validate_role_metadata(config, requested_contents={content(split, "train")})
    assert result["ok"] and result["allowed_split_counts"] == {"train": 1}
    assert result["compute_authorized"] is False and result["human_verified"] is False
    assert result["media_opened"] is False and result["exposure_coverage_complete_snapshot"]


def test_source_pin_and_final_check_fail_closed(inputs):
    config, split, _, _ = inputs
    wrong = {**config, "expected_source_manifest_sha256": "0" * 64}
    with pytest.raises(TrainingGateError, match="source_manifest_pin_mismatch"):
        validate_role_metadata(wrong, requested_contents={content(split, "train")})
    with pytest.raises(TrainingGateError, match="outside_allowed_roles"):
        validate_role_metadata(config, requested_contents={content(split, "final_check")})
    with pytest.raises(TrainingGateError, match="final_check_forbidden"):
        validate_role_metadata(config, requested_contents={content(split, "final_check")}, allowed_splits={"final_check"})


def test_snapshot_age_cannot_be_relaxed_and_stale_snapshot_blocks(inputs):
    config, split, _, _ = inputs
    with pytest.raises(TrainingGateError, match="snapshot_age_limit_invalid"):
        validate_role_metadata({**config, "max_snapshot_age_sec": 10000}, requested_contents={content(split, "train")})
    inventory = Path(config["source_inventory_path_override"])
    payload = json.loads(inventory.read_text())
    payload["captured_at_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    write(inventory, payload)
    register_accepted_inventory(inventory, Path(config["exposure_registry"]), Path(config["source_manifest"]),
                                Path(config["exposure_inventory_status"]))
    with pytest.raises(TrainingGateError, match="snapshot_stale"):
        validate_role_metadata(config, requested_contents={content(split, "train")})


@pytest.fixture
def silver_inputs(inputs):
    config, split, private, _ = inputs
    selected = content(split, "train")
    teacher = {"model_id": "generated-private-teacher", "model_revision": "c" * 40,
               "processor_revision": "c" * 40, "model_manifest_sha256": "d" * 64}
    provenance = {**teacher, "external_ai_used": False, "actual_snapshot_verified": True,
                  "execution_location": "team_local_service"}
    row = {"content_sha256": selected, "split": "train", "human_reviewed": False,
           "source_pts_identity_checked": True, "partition_binding": {"assignments_sha256": config["expected_assignments_sha256"],
           "split_file_sha256": config["expected_automated_split_sha256"]}, "provenance": provenance,
           "windows": [{"provenance": provenance}]}
    annotations = private / "machine.jsonl"
    annotations.write_text(json.dumps(row) + "\n")
    proof = write(private / "proof.json", {"file_sha256": file_digest(annotations), "all_train": True,
                                           "human_reviewed": 0, "records": 1, "windows": 1, "schema_errors": 0})
    media = write(private / "media.json", {"assignments_sha256": config["expected_assignments_sha256"],
                                           "frames": [{"content_sha256": selected, "split": "train"}]})
    taxonomy = write(private / "taxonomy.json", {})
    policy = write(private / "policy.json", {"generated": True})
    config.update(schema_version="forte.qwen38.silver-config.v1", expected_teacher=teacher,
                  machine_annotations=str(annotations), export_proof=str(proof), training_media_manifest=str(media), taxonomy_path=str(taxonomy))
    config.update(anchor_question_policy=str(policy), expected_anchor_question_policy_sha256=file_digest(policy))
    for key in ("machine_annotations", "export_proof", "training_media_manifest"):
        config["expected_" + key + "_sha256"] = file_digest(Path(config[key]))
    config["expected_taxonomy_sha256"] = file_digest(taxonomy)
    return config, split, private, selected, media


def test_silver_export_teacher_and_media_pins_are_separate(silver_inputs):
    config, split, private, selected, media = silver_inputs
    assert validate_silver_metadata(config)["teacher_windows"] == 1
    bad = deepcopy(config)
    bad["expected_teacher"]["model_revision"] = "f" * 40
    with pytest.raises(TrainingGateError, match="teacher_runtime_pin_mismatch"):
        validate_silver_metadata(bad)
    write(media, {"assignments_sha256": config["expected_assignments_sha256"],
                  "frames": [{"content_sha256": content(split, "final_check"), "split": "final_check"}]})
    config["expected_training_media_manifest_sha256"] = file_digest(media)
    with pytest.raises(TrainingGateError, match="media_outside_silver_train"):
        validate_silver_metadata(config)


def test_single_compute_release_positive_reserve_and_runner_validation(silver_inputs):
    from training.qwen38_silver import validate_attempt_accounting, validate_compute_release
    config, split, private, selected, _ = silver_inputs
    plan = write(private / "smoke.plan.json", {"schema_version": "forte.qwen38-attempt-plan.v1", "phase": "smoke",
        "attempts": [{"attempt_id": f"generated-step-{index}", "content_sha256": selected, "split": "train"} for index in range(4)]})
    release_path, reservation_path = private / "compute-release.json", private / "reservation.json"
    now = datetime.now(timezone.utc)
    # This test is deterministic after the dated rental window too.
    if now >= FIXED_STOP_UTC - timedelta(hours=1):
        now = FIXED_STOP_UTC - timedelta(hours=2)
    config.update(compute_release=str(release_path), attempt_reservation=str(reservation_path),
        attempt_plan=str(plan), expected_attempt_plan_sha256=file_digest(plan), max_steps=2,
        stop_utc=FIXED_STOP_UTC.isoformat(), max_wall_seconds=30,
        runtime_image_sha256="8" * 64, code_archive_sha256="9" * 64, seed=20261017,
        gradient_accumulation_steps=1, model_manifest=str(write(private / "base.json", {"generated": True})))
    bindings = {"source_manifest_sha256": config["expected_source_manifest_sha256"],
                "model_manifest_sha256": file_digest(Path(config["model_manifest"]))}
    release = {"schema_version": "forte.qwen38.compute-release.v1", "authorized": True, "released_by": "root",
        "experiment_id": "QWEN38-SILVER", "mode": "smoke", "config_sha256": semantic_recipe_sha256(config),
        "input_bindings": bindings, "runtime_image_sha256": config["runtime_image_sha256"],
        "code_archive_sha256": config["code_archive_sha256"], "maximum_attempts": 4, "maximum_optimizer_steps": 2,
        "attempt_plan_sha256": file_digest(plan), "exclusive_idle_gpu_handoff_verified": True,
        "idle_handoff_receipt_sha256": "6" * 64, "train_only_mount_verified": True,
        "issued_at": (now - timedelta(seconds=5)).isoformat(), "expires_at": (now + timedelta(minutes=10)).isoformat(),
        "max_wall_seconds": 30}
    write(release_path, release)
    config["expected_compute_release_sha256"] = file_digest(release_path)
    assert semantic_recipe_sha256(config) == release["config_sha256"]
    validate_compute_release(config, bindings, smoke=True, now=now)
    result = reserve_attempt_plan(config, config_sha256="0" * 64, plan_path=plan, release_path=release_path,
                                  output_path=reservation_path, now=now)
    assert result["reserved_attempts"] == 4 and result["model_calls_made"] is False
    receipt = json.loads(reservation_path.read_text())
    assert receipt["config_sha256"] == release["config_sha256"]
    assert receipt["release_sha256"] == config["expected_compute_release_sha256"]
    assert receipt["source_archive_sha256"] == config["code_archive_sha256"]
    assert receipt["base_manifest_sha256"] == bindings["model_manifest_sha256"]
    validate_attempt_reservation(receipt, json.loads(plan.read_text()), plan_sha256=file_digest(plan), phase="smoke", now=now)
    validate_compute_release(config, bindings, smoke=True, now=now)
    rows = [{"content_sha256": selected}, {"content_sha256": selected}]
    validate_attempt_accounting(config, rows, smoke=True, now=now)
    for key in ("runtime_image_sha256", "source_archive_sha256", "base_manifest_sha256", "config_sha256"):
        write(reservation_path, {**receipt, key: "0" * 64})
        with pytest.raises(TrainingGateError, match="attempt_runtime_binding_mismatch"):
            validate_attempt_accounting(config, rows, smoke=True, now=now)
    write(reservation_path, receipt)
    with pytest.raises(TrainingGateError, match="already_reserved"):
        reserve_attempt_plan(config, config_sha256="0" * 64, plan_path=plan, release_path=release_path,
                             output_path=private / "duplicate-reservation.json", now=now)
    for key in ("runtime_image_sha256", "code_archive_sha256"):
        with pytest.raises(TrainingGateError, match="root_release_binding_mismatch|runtime_release_pin_mismatch"):
            validate_compute_release({**config, key: "0" * 64}, bindings, smoke=True, now=now)
    with pytest.raises(TrainingGateError, match="root_release_binding_mismatch"):
        validate_compute_release(config, {**bindings, "model_manifest_sha256": "0" * 64}, smoke=True, now=now)
    # Root reserves the exact Linux recipe into the local authoritative ledger.
    local_cfg = dict(config)
    second_plan = deepcopy(json.loads(plan.read_text()))
    for item in second_plan["attempts"]:
        item["attempt_id"] += "-relocated"
    second_plan_path = write(private / "relocated.plan.json", second_plan)
    local_cfg.update(attempt_plan=str(second_plan_path), expected_attempt_plan_sha256=file_digest(second_plan_path),
                     compute_release=str(private / "relocated.release.json"),
                     attempt_reservation=str(private / "relocated.reservation.json"))
    remote_cfg = dict(local_cfg)
    for key in ("source_manifest", "automated_split", "training_media_manifest", "compute_release", "attempt_plan", "attempt_reservation"):
        remote_cfg[key] = "/private/generated/" + Path(local_cfg[key]).name
    validate_runtime_relocation(local_cfg, remote_cfg)
    relocated_release = {**release, "config_sha256": semantic_recipe_sha256(remote_cfg),
                         "attempt_plan_sha256": file_digest(second_plan_path)}
    write(Path(local_cfg["compute_release"]), relocated_release)
    local_cfg["expected_compute_release_sha256"] = file_digest(Path(local_cfg["compute_release"]))
    remote_cfg["expected_compute_release_sha256"] = local_cfg["expected_compute_release_sha256"]
    reserve_attempt_plan(local_cfg, config_sha256="0" * 64, plan_path=second_plan_path,
        release_path=Path(local_cfg["compute_release"]), output_path=Path(local_cfg["attempt_reservation"]),
        runtime_config=remote_cfg, now=now)
    relocated_receipt = json.loads(Path(local_cfg["attempt_reservation"]).read_text())
    assert relocated_receipt["config_sha256"] == semantic_recipe_sha256(remote_cfg)
    assert relocated_receipt["local_metadata_view_sha256"] == semantic_recipe_sha256(local_cfg)
    for key in ("max_steps", "expected_machine_annotations_sha256", "expected_training_media_manifest_sha256"):
        with pytest.raises(TrainingGateError, match="runtime_relocation_changes_recipe_or_pins"):
            validate_runtime_relocation(local_cfg, {**remote_cfg, key: "changed"})


def test_acceptance_reservation_exact_slots_expiry_and_runtime_pins():
    now = FIXED_STOP_UTC - timedelta(hours=3)
    plan = {"schema_version": "forte.qwen38-acceptance-plan.v1", "cases": [
        {"case_id": "generated", "content_sha256": "a" * 64, "split": "train"}],
        "attempts": [{"attempt_id": f"attempt-{index}", "case_id": "generated", "index": index} for index in range(3)]}
    attempts = [{"attempt_id": f"attempt-{index}", "content_sha256": "a" * 64, "split": "train"} for index in range(3)]
    receipt = {"schema_version": "forte.qwen38-reservation.v1", "status": "reserved", "phase": "acceptance",
        "attempt_plan_sha256": "b" * 64, "requested_attempts": 3, "attempts": attempts,
        "released_by_root": True, "release_sha256": "c" * 64, "registry_tip_sha256": "d" * 64,
        "expires_at_utc": (now + timedelta(hours=1)).isoformat(), **{key: "e" * 64 for key in
        ("config_sha256", "source_archive_sha256", "runtime_image_sha256", "base_manifest_sha256")}}
    assert validate_attempt_reservation(receipt, plan, plan_sha256="b" * 64, phase="acceptance", now=now)["reserved_attempts"] == 3
    with pytest.raises(TrainingGateError, match="expired"):
        validate_attempt_reservation(receipt, plan, plan_sha256="b" * 64, now=FIXED_STOP_UTC)
    with pytest.raises(TrainingGateError, match="scope_invalid"):
        validate_attempt_reservation({**receipt, "attempts": attempts[:2]}, plan, plan_sha256="b" * 64, now=now)
    with pytest.raises(TrainingGateError, match="runtime_pins_missing"):
        validate_attempt_reservation({**receipt, "runtime_image_sha256": None}, plan, plan_sha256="b" * 64, now=now)


def test_missing_root_release_does_not_reserve_anything(inputs):
    config, split, private, records = inputs
    plan = write(private / "plan.json", {"schema_version": "forte.qwen38-attempt-plan.v1", "phase": "evaluation",
        "attempts": [{"attempt_id": "generated", "content_sha256": content(split, "validation"), "split": "validation"}]})
    release = write(private / "release.json", {"released_by_root": False})
    before = read_registry(Path(config["exposure_registry"]), records)["tip_sha256"]
    with pytest.raises(TrainingGateError, match="exact_root_release_missing"):
        reserve_attempt_plan(config, config_sha256="a" * 64, plan_path=plan, release_path=release,
                             output_path=private / "reservation.json")
    assert read_registry(Path(config["exposure_registry"]), records)["tip_sha256"] == before


def test_failed_cancelled_and_unresolved_slots_stay_in_denominator():
    receipt = {"schema_version": "forte.qwen38-reservation.v1", "status": "reserved", "requested_attempts": 3,
               "attempts": [{"attempt_id": value} for value in ("generated-a", "generated-b", "generated-c")]}
    outcomes = [{"attempt_id": "generated-a", "status": "failed"}, {"attempt_id": "generated-b", "status": "cancelled"}]
    pending = reconcile_attempt_outcomes(receipt, outcomes)
    assert pending["complete"] is False and pending["reserved_attempts"] == 3 and pending["unresolved_attempts"] == 1
    complete = reconcile_attempt_outcomes(receipt, [*outcomes, {"attempt_id": "generated-c", "status": "not_attempted"}])
    assert complete["complete"] and complete["status_counts"] == {"cancelled": 1, "failed": 1, "not_attempted": 1}
    assert complete["retry_authorized"] is False and complete["exposure_reservations_refunded"] is False
    with pytest.raises(TrainingGateError, match="reconciliation_attempt_invalid"):
        reconcile_attempt_outcomes(receipt, [*outcomes, outcomes[0]])
