"""Importable fail-closed metadata gate for an independently approved lock.

Exposure is a disqualification from unseen final-check data, never training
permission. The caller must pin the trusted approved assignments digest and
separately validate reviewed targets; binary labels are not fact annotations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path

try:
    from .audit_data import check_split
    from .build_reviewed_split import exposure_coverage
    from .exposure_registry import canonical, read_registry, valid_hash
except ImportError:
    from audit_data import check_split
    from build_reviewed_split import exposure_coverage
    from exposure_registry import canonical, read_registry, valid_hash


def validate_locked_training_inputs(
    split: dict,
    records: list[dict],
    scenario: dict,
    *,
    expected_assignments_sha256: str,
    registry_path: Path,
    coverage_status_path: Path,
    requested_content_hashes: set[str] | None = None,
    requested_recording_ids: set[str] | None = None,
    max_snapshot_age_sec: float = 300,
    source_inventory_path_override: Path | None = None,
) -> dict:
    """Return aggregate diagnostics only; callers must reject `ok is not True`.

    `expected_assignments_sha256` must come from the independently approved,
    pinned run configuration, not be copied from an untrusted replacement
    split. Pinning prevents validation/final-check assignments being edited
    into training after approval. No target facts or media are read here.
    """
    errors: Counter = Counter()
    try:
        if not valid_hash(expected_assignments_sha256) or split.get("assignments_sha256") != expected_assignments_sha256:
            errors["approved_assignment_binding_mismatch"] += 1
        registry = read_registry(Path(registry_path), records)
        split_check = check_split(split, records, scenario=scenario, require_locked=True, excluded_final_hashes=registry["content_hashes"])
        errors.update(split_check["errors"])
        tip = split.get("exposure_registry_tip_sha256")
        snapshot_count = split.get("exposure_registry_event_count")
        event_index = {event["event_sha256"]: index + 1 for index, event in enumerate(registry["events"])}
        if tip not in event_index or not isinstance(snapshot_count, int) or snapshot_count != event_index.get(tip):
            errors["approved_exposure_snapshot_not_ancestor"] += 1
        else:
            original_hashes = sorted({event["content_hash"] for event in registry["events"][:snapshot_count]})
            original_set_digest = hashlib.sha256(canonical(original_hashes)).hexdigest()
            if split.get("exposure_content_set_sha256") != original_set_digest:
                errors["approved_exposure_content_set_binding_mismatch"] += 1
        coverage, coverage_errors = exposure_coverage(Path(coverage_status_path), records, registry, max_snapshot_age_sec, source_inventory_path_override=source_inventory_path_override)
        errors.update(coverage_errors)
        by_recording = {row.get("recording_id", row.get("sample_id")): row for row in split.get("records", [])}
        by_content = defaultdict(list)
        for row in split.get("records", []):
            by_content[row.get("content_hash", row.get("content_sha256"))].append(row)
        for recording_id in requested_recording_ids or set():
            row = by_recording.get(recording_id)
            if row is None or row.get("split") != "train":
                errors["requested_recording_not_approved_train"] += 1
        for content_hash in requested_content_hashes or set():
            content_rows = by_content.get(content_hash, [])
            if not valid_hash(content_hash) or not content_rows or any(row.get("split") != "train" for row in content_rows):
                errors["requested_content_not_approved_train"] += 1
        return {"ok": not errors, "errors": dict(sorted(errors.items())),
                "requested_training_contents": len(requested_content_hashes or set()),
                "requested_training_recordings": len(requested_recording_ids or set()),
                "exposure_events": registry["event_count"], "distinct_exposed_contents": len(registry["content_hashes"]),
                "exposure_coverage_complete_snapshot": coverage["exposure_coverage_complete_snapshot"],
                "locked_split_verified": not errors, "training_authority_inferred_from_exposure": False,
                "target_facts_validated": False}
    except Exception as exc:
        code = str(exc) if isinstance(exc, ValueError) and str(exc).replace("_", "").isalpha() else "training_exposure_gate_operation_failed"
        return {"ok": False, "errors": {code: 1}, "locked_split_verified": False,
                "training_authority_inferred_from_exposure": False, "target_facts_validated": False}


def validate_automated_experiment_inputs(
    split: dict,
    records: list[dict],
    scenario: dict,
    *,
    experiment_authorization: dict,
    registry_path: Path,
    coverage_status_path: Path,
    provided_label_records: list[dict],
    expected_provided_label_source_sha256: str,
    label_split: str = "train",
    requested_content_hashes: set[str] | None = None,
    requested_recording_ids: set[str] | None = None,
    max_snapshot_age_sec: float = 300,
    source_inventory_path_override: Path | None = None,
) -> dict:
    """Separate explicit weak experiment gate; does not relax the human gate.

    Use `label_split='validation'` only in evaluation code. Training passes
    train-only original binary records. Authorization/pins come from the
    independently trusted run config, never from the provisional split.
    """
    try:
        try:
            from .build_automated_split import AUTOMATED_ORIGINS, label_records_digest
            from .exposure_registry import file_digest
        except ImportError:
            from build_automated_split import AUTOMATED_ORIGINS, label_records_digest
            from exposure_registry import file_digest
        errors: Counter = Counter()
        authorization = experiment_authorization
        expected_digest = authorization.get("expected_assignments_sha256")
        if authorization.get("authorized") is not True or authorization.get("allow_provisional_groups") is not True or authorization.get("validation_assignments_frozen") is not True or authorization.get("allowed_target") != "video_binary_mil" or not isinstance(authorization.get("authorization_reference"), str) or not authorization["authorization_reference"].strip():
            errors["explicit_provisional_experiment_authorization_missing"] += 1
        if not valid_hash(expected_digest) or expected_digest != split.get("assignments_sha256"):
            errors["provisional_assignment_pin_mismatch"] += 1
        if not valid_hash(expected_provided_label_source_sha256) or authorization.get("expected_provided_label_source_sha256") != expected_provided_label_source_sha256 or split.get("provided_label_source_sha256") != expected_provided_label_source_sha256 or split.get("source_workbook_sha256") != expected_provided_label_source_sha256:
            errors["independent_provided_label_source_pin_mismatch"] += 1
        if split.get("state") != "automated_provisional" or split.get("locked") is not False or split.get("training_authorized") is not False or split.get("human_reviewed") is not False or split.get("scenario_review_complete") is not False or split.get("group_review_origin") not in AUTOMATED_ORIGINS:
            errors["automated_track_provenance_invalid"] += 1
        if scenario.get("review_state") != "machine_provisional" or scenario.get("human_reviewed") is not False:
            errors["automatic_scenario_provenance_invalid"] += 1
        provenance = split.get("grouping_provenance", {})
        if provenance.get("origin") != split.get("group_review_origin") or provenance.get("human_reviewed") is not False or provenance.get("scenario_independence_verified") is not False or not valid_hash(split.get("source_group_proposals_sha256")):
            errors["automatic_grouping_binding_invalid"] += 1
        registry = read_registry(Path(registry_path), records)
        integrity = check_split(split, records, scenario=scenario, excluded_final_hashes=registry["content_hashes"])
        errors.update(integrity["errors"])
        tip, event_count = split.get("exposure_registry_tip_sha256"), split.get("exposure_registry_event_count")
        event_index = {event["event_sha256"]: index + 1 for index, event in enumerate(registry["events"])}
        if tip not in event_index or event_count != event_index.get(tip):
            errors["provisional_exposure_snapshot_not_ancestor"] += 1
        else:
            old_hashes = sorted({event["content_hash"] for event in registry["events"][:event_count]})
            if split.get("exposure_content_set_sha256") != hashlib.sha256(canonical(old_hashes)).hexdigest():
                errors["provisional_exposure_content_binding_mismatch"] += 1
        coverage, coverage_errors = exposure_coverage(Path(coverage_status_path), records, registry, max_snapshot_age_sec, source_inventory_path_override=source_inventory_path_override)
        errors.update(coverage_errors)
        if label_split not in {"train", "validation", "final_check"}:
            errors["invalid_provided_label_partition"] += 1
        label_binding = split.get("provided_label_manifests", {}).get(label_split, {})
        if authorization.get("expected_provided_label_manifests_sha256", {}).get(label_split) != label_binding.get("sha256"):
            errors["independent_provided_label_manifest_pin_mismatch"] += 1
        if label_binding.get("sha256") != label_records_digest(provided_label_records) or label_binding.get("records") != len(provided_label_records):
            errors["provided_binary_manifest_binding_mismatch"] += 1
        by_content, by_recording = defaultdict(list), {}
        split_groups = defaultdict(set)
        for assignment in split.get("records", []):
            by_content[assignment.get("content_hash")].append(assignment)
            by_recording[assignment.get("recording_id")] = assignment
            split_groups[assignment.get("split")].add(assignment.get("scenario_group_id"))
        if not split_groups["train"] or not split_groups["validation"] or split_groups["train"] & split_groups["validation"]:
            errors["provisional_train_validation_groups_invalid"] += 1
        label_hashes = set()
        for label in provided_label_records:
            content_hash = label.get("content_sha256", label.get("content_hash"))
            assignments = by_content.get(content_hash, [])
            if content_hash in label_hashes:
                errors["duplicate_provided_binary_content"] += 1
            label_hashes.add(content_hash)
            if not assignments or any(row.get("split") != label_split or row.get("scenario_group_id") != label.get("group_id") for row in assignments) or label.get("split") != label_split:
                errors["provided_label_partition_or_group_mismatch"] += 1
            if label.get("content_hash", content_hash) != content_hash or type(label.get("provided_video_label")) is not bool or label.get("provided_label_origin") != "organizer_workbook" or label.get("provided_label_source_sha256") != expected_provided_label_source_sha256 or label.get("target_scope") != "video_binary_mil_only" or label.get("teacher_used_as_binary_ground_truth") is not False or any(key in label for key in ("facts", "events", "fact_targets", "event_targets")):
                errors["binary_label_used_as_unsupported_event_truth"] += 1
        for content_hash in requested_content_hashes or set():
            if content_hash not in label_hashes or not by_content.get(content_hash) or any(row.get("split") != label_split for row in by_content[content_hash]):
                errors["requested_content_outside_allowed_partition"] += 1
        for recording_id in requested_recording_ids or set():
            assignment = by_recording.get(recording_id)
            if assignment is None or assignment.get("split") != label_split or assignment.get("content_hash") not in label_hashes:
                errors["requested_recording_outside_allowed_partition"] += 1
        return {"ok": not errors, "errors": dict(sorted(errors.items())), "track": "automated_weak_supervision",
                "label_split": label_split, "provided_binary_records": len(provided_label_records),
                "expected_assignments_sha256": expected_digest, "provided_label_source_sha256": expected_provided_label_source_sha256,
                "provided_label_manifest_sha256": label_records_digest(provided_label_records),
                "exposure_registry_tip_sha256": registry["tip_sha256"], "coverage_snapshot_sha256": file_digest(Path(coverage_status_path)),
                "grouping_provenance": provenance, "human_reviewed": False, "human_verified_training_authorized": False,
                "training_authority_inferred_from_exposure": False, "target_scope": "video_binary_mil_only",
                "event_fact_ground_truth_validated": False, "scenario_independence_verified": False}
    except Exception as exc:
        code = str(exc) if isinstance(exc, ValueError) and str(exc).replace("_", "").isalpha() else "automated_experiment_gate_operation_failed"
        return {"ok": False, "errors": {code: 1}, "track": "automated_weak_supervision", "human_reviewed": False,
                "human_verified_training_authorized": False, "event_fact_ground_truth_validated": False}
