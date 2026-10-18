"""Fail-closed Qwen3.8 metadata and private attempt accounting.

This module never opens footage, tensors or model weights and never calls a
model. Metadata validation grants no compute authority. Reservations require
a separate exact-plan root release and add to the existing central registry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.audit_data import check_split, json_bytes
from scripts.build_reviewed_split import exposure_coverage
from scripts.exposure_registry import (canonical, file_digest, manifest_identity,
                                       read_registry, register_content_manifest, valid_hash)
from training.guards import TrainingGateError, canonical_sha256, require


FIXED_STOP_UTC = datetime(2026, 10, 18, 6, 1, 25, tzinfo=timezone.utc)
SILVER_SCHEMA = "forte.qwen38.silver-config.v1"
RESERVATION_SCHEMA = "forte.qwen38-reservation.v1"
PLAN_SCHEMAS = {"forte.qwen38-attempt-plan.v1", "forte.qwen38_comparison_plan.v1",
                "forte.qwen38-acceptance-plan.v1"}
RELOCATABLE_PATH_KEYS = frozenset({"source_manifest", "automated_split", "scenario_groups", "machine_annotations",
    "export_proof", "taxonomy_path", "anchor_question_policy", "question_catalog", "training_media_manifest",
    "media_root", "model_dir", "processor_dir", "model_manifest", "source_frames_root", "source_frames_original_root",
    "exposure_registry", "exposure_inventory_status", "source_inventory_path_override", "runtime_lock",
    "compute_release", "attempt_plan", "attempt_reservation", "tiny_smoke_receipt", "two_step_smoke_receipt",
    "nf4_bias_initialization_proof", "silver_balance_plan", "output_dir"})


def semantic_recipe_sha256(config: dict) -> str:
    """Bind settings/paths/pins, excluding only self-referential receipt hashes.

    The release binds this digest. The reservation separately binds the exact
    release bytes and every attempt; the executor compares those again.
    """
    excluded = {"expected_compute_release_sha256", "expected_attempt_reservation_sha256",
                "config_sha256", "semantic_recipe_sha256"}
    return canonical_sha256({key: value for key, value in config.items() if key not in excluded})


def validate_runtime_relocation(local_config: dict, runtime_config: dict) -> None:
    """A local metadata view may relocate paths, never pins/settings/roles.

    This permits the local central ledger to reserve a Linux-container recipe
    whose private files have identical hashes under Windows preparation paths.
    The release still binds the exact container recipe's semantic digest.
    """
    excluded = RELOCATABLE_PATH_KEYS | {"expected_compute_release_sha256", "expected_attempt_reservation_sha256",
                                       "config_sha256", "semantic_recipe_sha256"}
    require({key: value for key, value in local_config.items() if key not in excluded}
            == {key: value for key, value in runtime_config.items() if key not in excluded},
            "qwen38_runtime_relocation_changes_recipe_or_pins")
    for key in RELOCATABLE_PATH_KEYS & (local_config.keys() | runtime_config.keys()):
        require(isinstance(local_config.get(key), str) and isinstance(runtime_config.get(key), str)
                and local_config[key] and runtime_config[key], "qwen38_runtime_relocation_path_missing")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _private(path: Path) -> None:
    require("private" in path.resolve().parts, "qwen38_metadata_must_remain_private")


def _pin(config: dict, key: str, pin_key: str | None = None) -> Path:
    path = Path(config.get(key, "")).resolve()
    expected = config.get(pin_key or "expected_" + key + "_sha256")
    require(valid_hash(expected) and path.is_file() and file_digest(path) == expected,
            "qwen38_" + key + "_pin_mismatch")
    return path


def validate_role_metadata(config: dict, *, requested_contents: set[str],
                           requested_recording_ids: set[str] | None = None,
                           allowed_splits: set[str] | None = None) -> dict:
    """Check frozen identities, central exposure chain and fresh accepted jobs.

    Callers supply independently pinned config bytes through their release.
    This route is separate from E1 human-gold and organizer-binary-MIL gates.
    Only train/development-validation contents can be requested.
    """
    allowed = {"train"} if allowed_splits is None else set(allowed_splits)
    require(allowed and allowed <= {"train", "validation"}, "qwen38_final_check_forbidden")
    require(requested_contents and all(valid_hash(value) for value in requested_contents),
            "qwen38_requested_contents_invalid")
    paths = {name: _pin(config, name) for name in ("source_manifest", "automated_split", "scenario_groups")}
    records, split, scenario = _rows(paths["source_manifest"]), _json(paths["automated_split"]), _json(paths["scenario_groups"])
    by_id, by_hash = manifest_identity(records)
    expected_assignment = config.get("expected_assignments_sha256")
    require(valid_hash(expected_assignment) and split.get("assignments_sha256") == expected_assignment
            and hashlib.sha256(json_bytes(split.get("records", []))).hexdigest() == expected_assignment,
            "qwen38_frozen_assignments_pin_mismatch")
    require(split.get("source_manifest_sha256") == file_digest(paths["source_manifest"]),
            "qwen38_split_source_binding_mismatch")
    require(split.get("state") == "automated_provisional" and split.get("locked") is False
            and split.get("human_reviewed") is False and split.get("scenario_review_complete") is False
            and scenario.get("human_reviewed") is False and scenario.get("review_state") == "machine_provisional",
            "qwen38_provisional_provenance_invalid")
    registry_path, status_path = Path(config["exposure_registry"]).resolve(), Path(config["exposure_inventory_status"]).resolve()
    _private(registry_path)
    _private(status_path)
    registry = read_registry(registry_path, records)
    integrity = check_split(split, records, scenario=scenario, excluded_final_hashes=registry["content_hashes"])
    require(integrity.get("ok") is True, "qwen38_split_integrity_failed")
    old_count, old_tip = split.get("exposure_registry_event_count"), split.get("exposure_registry_tip_sha256")
    require(type(old_count) is int and 0 < old_count <= registry["event_count"]
            and registry["events"][old_count - 1]["event_sha256"] == old_tip,
            "qwen38_frozen_exposure_snapshot_not_ancestor")
    old_hashes = sorted({event["content_hash"] for event in registry["events"][:old_count]})
    require(split.get("exposure_content_set_sha256") == hashlib.sha256(canonical(old_hashes)).hexdigest(),
            "qwen38_frozen_exposure_content_binding_mismatch")
    max_age = config.get("max_snapshot_age_sec", 300)
    require(type(max_age) in {int, float} and 0 < max_age <= 300, "qwen38_snapshot_age_limit_invalid")
    override = config.get("source_inventory_path_override")
    coverage, errors = exposure_coverage(status_path, records, registry, max_age,
                                         source_inventory_path_override=Path(override).resolve() if override else None)
    require(not errors, "qwen38_fresh_exposure_gate_failed:" + ",".join(sorted(errors)))
    roles: dict[str, set[str]] = defaultdict(set)
    for row in split["records"]:
        roles[row["content_hash"]].add(row["split"])
    for content in requested_contents:
        require(content in by_hash and len(roles[content]) == 1 and roles[content] <= allowed,
                "qwen38_requested_content_outside_allowed_roles")
    for recording_id in requested_recording_ids or set():
        require(by_id.get(recording_id) in requested_contents, "qwen38_requested_recording_mismatch")
    counts = Counter(next(iter(roles[content])) for content in requested_contents)
    return {"schema_version": "forte.qwen38-role-preflight.v1", "ok": True,
            "requested_contents": len(requested_contents), "allowed_split_counts": dict(sorted(counts.items())),
            "assignments_sha256": expected_assignment, "input_checksums": {key: file_digest(value) for key, value in paths.items()},
            "exposure_registry_tip_sha256": registry["tip_sha256"], "exposure_registry_event_count": registry["event_count"],
            "exposure_inventory_status_sha256": file_digest(status_path), **coverage,
            "human_verified": False, "scenario_independence_verified": False,
            "compute_authorized": False, "media_opened": False, "model_calls_made": False}


def validate_silver_metadata(config: dict, requested_contents: set[str] | None = None,
                             requested_recording_ids: set[str] | None = None) -> dict:
    """Validate exact W2 teacher identity and TRAIN-only visual-cache metadata.

    A04's row loader separately verifies raw observation/support joins and
    constructs unknown/conflict masks. No full-fact or boundary truth is
    inferred here from the presence of a teacher receipt.
    """
    require(config.get("schema_version") == SILVER_SCHEMA, "qwen38_silver_config_schema_invalid")
    paths = {"machine_annotations": _pin(config, "machine_annotations"),
             "export_proof": _pin(config, "export_proof"),
             "anchor_question_policy": _pin(config, "anchor_question_policy"),
             "taxonomy_path": _pin(config, "taxonomy_path", "expected_taxonomy_sha256"),
             "training_media_manifest": _pin(config, "training_media_manifest")}
    _private(paths["machine_annotations"])
    _private(paths["training_media_manifest"])
    rows, proof = _rows(paths["machine_annotations"]), _json(paths["export_proof"])
    require(rows and proof.get("file_sha256") == file_digest(paths["machine_annotations"])
            and proof.get("all_train") is True and proof.get("human_reviewed") == 0
            and proof.get("records") == len(rows) and proof.get("schema_errors") == 0,
            "qwen38_teacher_export_binding_invalid")
    expected_teacher = config.get("expected_teacher", {})
    teacher_keys = ("model_id", "model_revision", "processor_revision", "model_manifest_sha256")
    require(all(isinstance(expected_teacher.get(key), str) and expected_teacher[key] for key in teacher_keys)
            and valid_hash(expected_teacher["model_manifest_sha256"]), "qwen38_teacher_pins_missing")
    contents, windows = set(), 0
    for row in rows:
        content = row.get("content_sha256")
        require(valid_hash(content) and content not in contents and row.get("split") == "train"
                and row.get("human_reviewed") is False and row.get("source_pts_identity_checked") is True,
                "qwen38_teacher_source_provenance_invalid")
        contents.add(content)
        binding = row.get("partition_binding", {})
        require(binding.get("assignments_sha256") == config.get("expected_assignments_sha256")
                and binding.get("split_file_sha256") == config.get("expected_automated_split_sha256"),
                "qwen38_teacher_split_binding_mismatch")
        for provenance in [row.get("provenance", {}), *[window.get("provenance", {}) for window in row.get("windows", [])]]:
            require(all(provenance.get(key) == expected_teacher[key] for key in teacher_keys)
                    and provenance.get("external_ai_used") is False and provenance.get("actual_snapshot_verified") is True
                    and provenance.get("execution_location") in {"team_deployed_service", "team_local_service"},
                    "qwen38_teacher_runtime_pin_mismatch")
        windows += len(row.get("windows", []))
    selected = contents if requested_contents is None else set(requested_contents)
    require(selected and selected <= contents, "qwen38_requested_silver_contents_missing")
    role = validate_role_metadata(config, requested_contents=selected, requested_recording_ids=requested_recording_ids)
    media = _json(paths["training_media_manifest"])
    require(isinstance(media, dict) and media.get("assignments_sha256") == config["expected_assignments_sha256"],
            "qwen38_media_assignment_binding_mismatch")
    media_rows = media.get("records", media.get("entries", media.get("frames")))
    require(isinstance(media_rows, list) and media_rows, "qwen38_training_media_manifest_invalid")
    media_contents = set()
    for item in media_rows:
        require(isinstance(item, dict) and item.get("split") == "train"
                and item.get("content_sha256", item.get("content_hash")) in contents,
                "qwen38_media_outside_silver_train")
        media_contents.add(item.get("content_sha256", item.get("content_hash")))
    require(selected <= media_contents, "qwen38_training_media_coverage_missing")
    require(proof.get("windows") == windows, "qwen38_teacher_window_count_mismatch")
    return {**role, "schema_version": "forte.qwen38-silver-preflight.v1", "teacher_records": len(rows),
            "teacher_windows": windows, "teacher_is_human_gold": False,
            "input_checksums": {**role["input_checksums"], **{key: file_digest(value) for key, value in paths.items()}},
            "teacher_identity": expected_teacher, "target_facts_validated_by_this_gate": False,
            "training_authorized_by_this_gate": False, "feature_tensors_opened": False}


def reconcile_attempt_outcomes(receipt: dict, outcomes: list[dict]) -> dict:
    """Account every reserved slot; no failed/cancelled attempt disappears.

    This is an aggregate reconciliation of an executor's immutable attempt
    records. It cannot refund exposure or authorize a retry/new attempt.
    """
    require(receipt.get("schema_version") == RESERVATION_SCHEMA and receipt.get("status") == "reserved",
            "qwen38_reconciliation_reservation_invalid")
    expected = {item["attempt_id"] for item in receipt.get("attempts", [])}
    require(expected and len(expected) == receipt.get("requested_attempts"), "qwen38_reconciliation_scope_invalid")
    seen, statuses = set(), Counter()
    allowed = {"done", "success", "failed", "cancelled", "rejected", "not_attempted"}
    for item in outcomes:
        identity, status = item.get("attempt_id"), item.get("status")
        require(identity in expected and identity not in seen and status in allowed,
                "qwen38_reconciliation_attempt_invalid")
        seen.add(identity)
        statuses[status] += 1
    return {"schema_version": "forte.qwen38-attempt-reconciliation.v1", "complete": seen == expected,
            "reserved_attempts": len(expected), "recorded_attempts": len(seen),
            "unresolved_attempts": len(expected - seen), "status_counts": dict(sorted(statuses.items())),
            "exposure_reservations_refunded": False, "retry_authorized": False}


def _plan_attempts(plan: dict) -> list[dict]:
    require(plan.get("schema_version") in PLAN_SCHEMAS, "qwen38_attempt_plan_schema_invalid")
    attempts = plan.get("attempts")
    require(isinstance(attempts, list) and attempts, "qwen38_attempts_missing")
    cases = {row["case_id"]: row for row in plan.get("cases", [])}
    ids, normalized = set(), []
    for row in attempts:
        require(isinstance(row, dict) and isinstance(row.get("attempt_id"), str) and row["attempt_id"]
                and row["attempt_id"] not in ids, "qwen38_attempt_identity_invalid")
        ids.add(row["attempt_id"])
        source = cases.get(row.get("case_id"), row)
        content, split = source.get("content_sha256"), source.get("split")
        require(valid_hash(content) and split in {"train", "validation"}, "qwen38_attempt_role_invalid")
        require(row.get("content_sha256", content) == content and row.get("split", split) == split,
                "qwen38_attempt_source_mismatch")
        normalized.append({"attempt_id": row["attempt_id"], "content_sha256": content, "split": split})
    return normalized


def validate_attempt_reservation(receipt: dict, plan: dict, *, plan_sha256: str,
                                 phase: str | None = None, now: datetime | None = None) -> dict:
    """Pure validator for executors; every attempted/failed call consumes its slot."""
    attempts = _plan_attempts(plan)
    require(receipt.get("schema_version") == RESERVATION_SCHEMA and receipt.get("status") == "reserved"
            and receipt.get("attempt_plan_sha256") == plan_sha256 and valid_hash(plan_sha256),
            "qwen38_attempt_reservation_binding_invalid")
    require(receipt.get("attempts") == attempts and receipt.get("requested_attempts") == len(attempts)
            and receipt.get("released_by_root") is True and valid_hash(receipt.get("release_sha256"))
            and valid_hash(receipt.get("registry_tip_sha256")), "qwen38_attempt_reservation_scope_invalid")
    if phase is not None:
        require(receipt.get("phase") == phase, "qwen38_attempt_reservation_phase_mismatch")
    try:
        expiry = datetime.fromisoformat(receipt["expires_at_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        raise TrainingGateError("qwen38_attempt_reservation_expiry_invalid") from None
    require(expiry.tzinfo is not None and (now or datetime.now(timezone.utc)) < expiry <= FIXED_STOP_UTC,
            "qwen38_attempt_reservation_expired")
    require(all(valid_hash(receipt.get(key)) for key in ("config_sha256", "source_archive_sha256",
            "runtime_image_sha256", "base_manifest_sha256")), "qwen38_attempt_reservation_runtime_pins_missing")
    return {"ok": True, "reserved_attempts": len(attempts), "failures_and_cancellations_consume_slots": True}


def reserve_attempt_plan(config: dict, *, config_sha256: str, plan_path: Path,
                         release_path: Path, output_path: Path, now: datetime | None = None,
                         runtime_config: dict | None = None) -> dict:
    """Root-only exact release; does not issue any private model operation."""
    for path in (plan_path, release_path, output_path):
        _private(path)
    require(not output_path.exists(), "qwen38_reservation_output_already_exists")
    plan, release = _json(plan_path), _json(release_path)
    attempts = _plan_attempts(plan)
    phase = plan.get("phase", "evaluation" if "comparison" in plan.get("schema_version", "") else "acceptance")
    require(phase in {"smoke", "fit", "evaluation", "acceptance"}, "qwen38_reservation_phase_invalid")
    roles = {row["split"] for row in attempts}
    require(phase not in {"smoke", "fit"} or roles == {"train"}, "qwen38_fitting_must_be_train_only")
    silver_release = release.get("schema_version") == "forte.qwen38.compute-release.v1"
    if silver_release:
        # Shared with qwen38_silver.validate_compute_release: no self-hash.
        if runtime_config is not None:
            validate_runtime_relocation(config, runtime_config)
        config_sha256 = semantic_recipe_sha256(runtime_config if runtime_config is not None else config)
    released = ((silver_release and release.get("released_by") == "root" and release.get("authorized") is True
                 and release.get("mode") == ("smoke" if phase == "smoke" else "train") and phase in {"smoke", "fit"})
                or (release.get("schema_version") == "forte.qwen38-root-release.v1"
                    and release.get("released_by_root") is True and release.get("phase") == phase))
    require(released
            and release.get("attempt_plan_sha256") == file_digest(plan_path)
            and release.get("config_sha256") == config_sha256 and valid_hash(config_sha256)
            and release.get("maximum_attempts", release.get("max_attempts")) == len(attempts), "qwen38_exact_root_release_missing")
    source_archive = release.get("source_archive_sha256", release.get("code_archive_sha256"))
    base_manifest = release.get("base_manifest_sha256", release.get("input_bindings", {}).get("model_manifest_sha256"))
    require(all(valid_hash(value) for value in (source_archive, release.get("runtime_image_sha256"), base_manifest)),
            "qwen38_release_runtime_pins_missing")
    now = now or datetime.now(timezone.utc)
    try:
        expiry = datetime.fromisoformat(release.get("expires_at_utc", release.get("expires_at", "")).replace("Z", "+00:00"))
    except (ValueError, KeyError, TypeError):
        raise TrainingGateError("qwen38_release_expiry_invalid") from None
    require(expiry.tzinfo is not None and now < expiry <= FIXED_STOP_UTC,
            "qwen38_release_expired_or_past_fixed_stop")
    require(release.get("exclusive_idle_gpu_handoff_verified") is True
            and valid_hash(release.get("idle_handoff_receipt_sha256")), "qwen38_idle_gpu_handoff_missing")
    wall_limit = release.get("maximum_execution_seconds", release.get("max_wall_seconds"))
    require(type(wall_limit) is int and 0 < wall_limit
            and now.timestamp() + wall_limit <= expiry.timestamp(),
            "qwen38_release_execution_bound_invalid")
    if phase in {"smoke", "fit"}:
        steps = release.get("maximum_optimizer_steps", release.get("max_optimizer_steps"))
        require(type(steps) is int and 0 < steps <= (2 if phase == "smoke" else 100),
                "qwen38_release_optimizer_bound_invalid")
    if config.get("silver_loss_policy") == "forte.silver.fact-state-content-anchor.v1":
        bindings = release.get("input_bindings", {})
        require(all(valid_hash(bindings.get(key)) for key in
                    ("effective_rows_sha256", "effective_fact_weights_sha256", "supervision_objective_sha256",
                     "balance_plan_semantic_sha256", "balance_plan_file_sha256")),
                "qwen38_balanced_supervision_release_pins_missing")
    contents = {row["content_sha256"] for row in attempts}
    gate = (validate_silver_metadata(config, requested_contents=contents) if phase in {"smoke", "fit"}
            else validate_role_metadata(config, requested_contents=contents, allowed_splits=roles))
    records = _rows(Path(config["source_manifest"]))
    trial_id = "qwen38-" + phase + "-" + file_digest(plan_path)[:20]
    require(not any(event["trial_id"] == trial_id for event in read_registry(Path(config["exposure_registry"]), records)["events"]),
            "qwen38_attempt_plan_already_reserved")
    worklist_path = output_path.with_suffix(".contents.json")
    require(not worklist_path.exists(), "qwen38_reservation_worklist_already_exists")
    worklist = {"assignments_sha256": config["expected_assignments_sha256"],
                "split_file_sha256": config["expected_automated_split_sha256"],
                "attempt_plan_sha256": file_digest(plan_path),
                "records": [{"content_sha256": content, "split": next(row["split"] for row in attempts
                              if row["content_sha256"] == content)} for content in sorted(contents)]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with worklist_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(worklist, sort_keys=True, indent=2) + "\n")
    # register_content_manifest independently rechecks frozen source-role pins.
    registration = register_content_manifest(worklist_path, Path(config["exposure_registry"]),
        Path(config["source_manifest"]), Path(config["automated_split"]),
        expected_assignments_sha256=config["expected_assignments_sha256"],
        trial_id=trial_id, purpose="local_model_attempt", allowed_splits=roles)
    receipt = {"schema_version": RESERVATION_SCHEMA, "status": "reserved", "phase": phase,
        "attempt_plan_sha256": file_digest(plan_path), "config_sha256": config_sha256,
        "release_sha256": file_digest(release_path), "released_by_root": True,
        "content_worklist_sha256": file_digest(worklist_path),
        "local_metadata_view_sha256": semantic_recipe_sha256(config),
        "requested_attempts": len(attempts), "attempts": attempts,
        "recorded_at_utc": now.isoformat(), "registry_tip_sha256": registration["registry_tip_sha256"],
        "source_archive_sha256": source_archive, "runtime_image_sha256": release["runtime_image_sha256"],
        "base_manifest_sha256": base_manifest, "expires_at_utc": expiry.isoformat(),
        "input_bindings": release.get("input_bindings", {}),
        "maximum_execution_seconds": wall_limit,
        "exposure_gate": gate, "failures_and_cancellations_consume_slots": True,
        "settlement_required_for_every_attempt": True, "model_calls_made": False}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"ok": True, "phase": phase, "reserved_attempts": len(attempts),
            "registered_contents": len(contents), "model_calls_made": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reserve-plan", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--runtime-config", type=Path, help="exact container recipe; local config may relocate only paths")
    args = parser.parse_args()
    try:
        config = _json(args.config)
        if args.reserve_plan:
            require(args.release is not None and args.output is not None, "qwen38_reservation_paths_missing")
            result = reserve_attempt_plan(config, config_sha256=file_digest(args.config), plan_path=args.reserve_plan,
                                          release_path=args.release, output_path=args.output,
                                          runtime_config=_json(args.runtime_config) if args.runtime_config else None)
        else:
            result = validate_silver_metadata(config)
            if args.output:
                _private(args.output)
                require(not args.output.exists(), "qwen38_preflight_output_already_exists")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc), "model_calls_made": False}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "qwen38_metadata_operation_failed",
                          "error_type": type(exc).__name__, "model_calls_made": False}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
