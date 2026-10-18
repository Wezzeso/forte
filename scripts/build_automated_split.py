#!/usr/bin/env python3
"""Separate provisional machine-group path; never impersonates human review.

Consumes only private opaque grouping proposals, source identity metadata and
independent organizer workbook binary labels. No media/model inference runs
here. Actual dataset generation requires an explicit orchestration release.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

try:
    from .audit_data import SPLITS, UnionFind, check_split, create_candidate, json_bytes, load_workbook_rows, read_jsonl, sha256_file, write_json, write_jsonl
    from .build_reviewed_split import exposure_coverage
    from .exposure_registry import DEFAULT_REGISTRY, manifest_identity, read_registry, valid_hash
except ImportError:
    from audit_data import SPLITS, UnionFind, check_split, create_candidate, json_bytes, load_workbook_rows, read_jsonl, sha256_file, write_json, write_jsonl
    from build_reviewed_split import exposure_coverage
    from exposure_registry import DEFAULT_REGISTRY, manifest_identity, read_registry, valid_hash

AUTOMATED_VERSION = "forte.automated-provisional-split.v1"
AUTOMATED_ORIGINS = {"local_model", "local_features", "local_model_and_metadata", "local_model_and_features"}


def label_records_digest(rows: list[dict]) -> str:
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    return hashlib.sha256(payload).hexdigest()


def automatic_scenario_groups(records: list[dict], proposal: dict, *, expected_manifest_sha256: str) -> tuple[dict, dict]:
    by_id, by_hash = manifest_identity(records)
    if proposal.get("schema_version") != "forte.automatic-group-proposals.v1" or proposal.get("proposal_origin") not in AUTOMATED_ORIGINS:
        raise ValueError("automatic_group_provenance_invalid")
    if proposal.get("source_manifest_sha256") != expected_manifest_sha256 or not valid_hash(proposal.get("config_sha256")):
        raise ValueError("automatic_group_source_binding_mismatch")
    if proposal.get("human_reviewed") is True or proposal.get("locked") is True:
        raise ValueError("automatic_groups_cannot_claim_human_approval")
    forbidden_labels = {"labels", "provided_labels", "has_violation", "provided_video_label", "ground_truth_label"}
    if forbidden_labels & set(proposal):
        raise ValueError("automatic_grouping_cannot_consume_provided_answers")
    groups = proposal.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("automatic_groups_missing")
    union = UnionFind(list(by_id))
    for members in by_hash.values():
        for member in members[1:]:
            union.union(members[0], member)
    seen_members, seen_groups = set(), set()
    method_counts = Counter()
    for group in groups:
        if forbidden_labels & set(group):
            raise ValueError("automatic_grouping_cannot_consume_provided_answers")
        group_id = group.get("proposed_group_id")
        members = group.get("recording_ids")
        if not isinstance(group_id, str) or not group_id.strip() or group_id in seen_groups:
            raise ValueError("automatic_group_identity_conflict")
        if not isinstance(members, list) or not members or any(not isinstance(member, str) or member not in by_id for member in members) or len(set(members)) != len(members) or set(members) & seen_members:
            raise ValueError("automatic_group_membership_conflict")
        seen_members.update(members)
        seen_groups.add(group_id)
        for member in members[1:]:
            union.union(members[0], member)
        method_counts[str(group.get("method", "unspecified"))] += 1
    if seen_members != set(by_id):
        raise ValueError("automatic_group_coverage_incomplete")
    merged = defaultdict(list)
    for recording_id in sorted(by_id):
        merged[union.find(recording_id)].append(recording_id)
    scenario = {"schema_version": AUTOMATED_VERSION, "review_state": "machine_provisional", "human_reviewed": False,
                "group_review_origin": proposal["proposal_origin"], "locked": False, "approved_by": None,
                "groups": [{"scenario_group_id": "auto_scenario_" + hashlib.sha256("|".join(members).encode()).hexdigest()[:24],
                            "recording_ids": members, "review_state": "machine_provisional", "confidence_status": "heuristic_unvalidated"}
                           for _, members in sorted(merged.items())],
                "limitations": ["Machine/feature proposals are not human-reviewed scenario identity or proof of scenario independence.",
                                "Exact-content-only groups may contain unrecognized related scenarios."]}
    return scenario, {"input_proposal_groups": len(groups), "output_provisional_groups": len(merged),
                      "method_group_counts": dict(sorted(method_counts.items())), "human_reviewed": False}


def independent_binary_labels(records: list[dict], workbook_rows: list[dict], source_sha256: str) -> dict[str, dict]:
    if not valid_hash(source_sha256):
        raise ValueError("provided_label_source_digest_invalid")
    _, by_hash = manifest_identity(records)
    by_row = {}
    for row in workbook_rows:
        if row.get("row_id") in by_row:
            raise ValueError("provided_label_row_identity_conflict")
        by_row[row.get("row_id")] = row
    labels = {}
    for content_hash, aliases in sorted(by_hash.items()):
        source_records = [record for record in records if record["content_hash"] == content_hash]
        row_ids = sorted(row_id for record in source_records for row_id in record.get("workbook_row_ids", []))
        if not row_ids or any(row_id not in by_row for row_id in row_ids):
            raise ValueError("provided_binary_label_missing")
        states = {by_row[row_id].get("has_violation") for row_id in row_ids}
        if len(states) != 1 or next(iter(states)) not in (True, False) or any(type(by_row[row_id].get("has_violation")) is not bool for row_id in row_ids):
            raise ValueError("provided_binary_label_conflict_or_unknown")
        labels[content_hash] = {"content_sha256": content_hash, "content_hash": content_hash,
                                "recording_id": aliases[0], "recording_ids": aliases,
                                "provided_video_label": next(iter(states)), "provided_label_source_sha256": source_sha256,
                                "provided_label_origin": "organizer_workbook", "provided_label_row_ids": row_ids,
                                "target_scope": "video_binary_mil_only", "human_annotation_reviewed": False,
                                "teacher_used_as_binary_ground_truth": False}
    return labels


def build_automated_candidate(records: list[dict], workbook_rows: list[dict], proposal: dict, registry: dict, *, manifest_sha256: str, workbook_sha256: str, seed: int = 20261017) -> tuple[dict, dict, dict[str, list[dict]], dict]:
    scenario, diagnostics = automatic_scenario_groups(records, proposal, expected_manifest_sha256=manifest_sha256)
    provided_labels = independent_binary_labels(records, workbook_rows, workbook_sha256)
    candidate = create_candidate(records, workbook_rows, seed, [0.7, 0.15, 0.15], scenario, registry["content_hashes"])
    integrity = check_split(candidate, records, scenario=scenario, excluded_final_hashes=registry["content_hashes"])
    if not integrity["ok"]:
        raise ValueError("automated_candidate_integrity_failed")
    by_content = defaultdict(list)
    for assignment in candidate["records"]:
        by_content[assignment["content_hash"]].append(assignment)
    label_partitions = {split: [] for split in SPLITS}
    for content_hash, label in sorted(provided_labels.items()):
        assignments = by_content[content_hash]
        split_values = {row["split"] for row in assignments}
        scenario_ids = {row["scenario_group_id"] for row in assignments}
        if len(split_values) != 1 or len(scenario_ids) != 1:
            raise ValueError("automated_duplicate_assignment_conflict")
        assignment = next(row for row in assignments if row["recording_id"] == label["recording_id"])
        split = assignment["split"]
        label_partitions[split].append({**label, "group_id": assignment["scenario_group_id"],
                                       "scenario_group_id": assignment["scenario_group_id"],
                                       "recording_group_id": assignment["group_id"], "split": split})
    if not label_partitions["train"] or not label_partitions["validation"]:
        raise ValueError("insufficient_provisional_groups_for_train_validation")
    candidate.update({"schema_version": AUTOMATED_VERSION, "state": "automated_provisional", "locked": False,
                      "training_authorized": False, "approved_by": None, "scenario_review_complete": False,
                      "group_review_origin": proposal["proposal_origin"], "human_reviewed": False,
                      "grouping_provenance": {"origin": proposal["proposal_origin"], "human_reviewed": False,
                                              "config_sha256": proposal["config_sha256"],
                                              "confidence_status": "heuristic_unvalidated", "scenario_independence_verified": False},
                      "source_manifest_sha256": manifest_sha256, "source_workbook_sha256": workbook_sha256,
                      "source_group_proposals_sha256": hashlib.sha256(json_bytes(proposal)).hexdigest(),
                      "provided_label_source_sha256": workbook_sha256,
                      "provided_label_manifests": {split: {"sha256": label_records_digest(rows), "records": len(rows)} for split, rows in label_partitions.items()},
                      "exposure_registry_tip_sha256": registry["tip_sha256"], "exposure_registry_event_count": registry["event_count"],
                      "exposure_content_set_sha256": registry["content_set_sha256"],
                      "weak_supervision": {"target": "video_binary_mil", "binary_label_origin": "organizer_workbook",
                                           "teacher_as_event_ground_truth": False, "human_verified": False,
                                           "explicit_run_config_authorization_required": True, "validation_assignment_pin_required": True},
                      "final_check_independence_verified": False,
                      "limitations": ["Automated scenario proposals are provisional; related scenarios may remain across groups.",
                                      "Organizer binary labels support video-level targets only, not fact, event, timing or grounding truth.",
                                      "All exposed scenarios are excluded from final_check; remaining scenario independence is unverified.",
                                      "Separate explicit experiment authorization and frozen assignments are required; this is not a human-reviewed lock."]})
    diagnostics.update({"ok": True, "unique_binary_labeled_contents": len(provided_labels),
                        "split_unique_content_counts": {split: len(rows) for split, rows in label_partitions.items()},
                        "distinct_model_exposed_contents": len(registry["content_hashes"]), "exposure_events": registry["event_count"],
                        "human_reviewed": False, "locked": False, "training_authorized": False})
    return candidate, scenario, label_partitions, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    parser.add_argument("--workbook-rows", type=Path, default=Path("data/manifests/workbook_rows.jsonl"))
    parser.add_argument("--workbook", type=Path, default=Path("open_labels.xlsx"))
    parser.add_argument("--base-split", type=Path, default=Path("data/splits/candidate_split.json"))
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--coverage-status", type=Path, default=Path("benchmarks/private/exposure-inventory-status.json"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/automated_provisional_split.json"))
    parser.add_argument("--scenario-output", type=Path, default=Path("data/splits/scenario_groups.automated_provisional.json"))
    parser.add_argument("--label-output-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--execution-release-reference", required=True, help="Orchestrator-provided local metadata release reference; not GPU/network permission")
    parser.add_argument("--allow-stale-exposure-snapshot-for-preparation", action="store_true", help="Local metadata preparation only; training gate still requires a fresh snapshot")
    parser.add_argument("--experiment-authorization-reference", help="Explicit trusted orchestrator/user authorization for the separate provisional experiment")
    parser.add_argument("--experiment-config-output", type=Path, default=Path("data/splits/automated_experiment.run-config.json"))
    args = parser.parse_args()
    try:
        if not args.execution_release_reference.strip():
            raise ValueError("automated_generation_release_missing")
        records = read_jsonl(args.manifest)
        manifest_sha256, workbook_sha256 = sha256_file(args.manifest), sha256_file(args.workbook)
        base = json.loads(args.base_split.read_text(encoding="utf-8-sig"))
        if base.get("source_manifest_sha256") != manifest_sha256 or base.get("source_workbook_sha256") != workbook_sha256:
            raise ValueError("automated_source_binding_mismatch")
        registry = read_registry(args.exposure_registry, records)
        coverage, coverage_errors = exposure_coverage(args.coverage_status, records, registry, 300)
        stale_preparation = args.allow_stale_exposure_snapshot_for_preparation and set(coverage_errors) == {"exposure_inventory_snapshot_stale"}
        if coverage_errors and not stale_preparation:
            print(json.dumps({"ok": False, "errors": coverage_errors, "outputs_written": False, "training_authorized": False}))
            return 2
        proposal = json.loads(args.proposals.read_text(encoding="utf-8-sig"))
        original_rows = load_workbook_rows(args.workbook, "labels")
        candidate, scenario, labels, report = build_automated_candidate(records, original_rows, proposal, registry,
                                                                      manifest_sha256=manifest_sha256, workbook_sha256=workbook_sha256)
        outputs = [args.output, args.scenario_output, *[args.label_output_root / ("provided_binary_labels." + split + ".jsonl") for split in SPLITS]]
        if args.experiment_authorization_reference:
            outputs.append(args.experiment_config_output)
        for path in outputs:
            if path.exists():
                raise ValueError("automated_artifact_exists_use_new_isolated_output")
            allowed = Path("data/splits").resolve() if path.suffix == ".json" else Path("data/manifests").resolve()
            if not path.resolve().is_relative_to(allowed):
                raise ValueError("automated_artifacts_must_remain_private")
        candidate.update({"source_group_proposals_sha256": sha256_file(args.proposals), "scenario_groups_sha256": hashlib.sha256(json_bytes(scenario)).hexdigest(),
                          "exposure_coverage_status_sha256": sha256_file(args.coverage_status), "execution_release_reference": args.execution_release_reference,
                          "pre_run_coverage_refresh_required": stale_preparation,
                          "provided_label_scope": "Separate per-split private manifests; training must not load validation/final-check labels."})
        for split in SPLITS:
            write_jsonl(args.label_output_root / ("provided_binary_labels." + split + ".jsonl"), labels[split])
        write_json(args.scenario_output, scenario)
        write_json(args.output, candidate)
        if args.experiment_authorization_reference:
            authorization = {"authorized": True, "authorization_reference": args.experiment_authorization_reference,
                             "allow_provisional_groups": True, "validation_assignments_frozen": True, "allowed_target": "video_binary_mil",
                             "expected_assignments_sha256": candidate["assignments_sha256"],
                             "expected_provided_label_source_sha256": workbook_sha256,
                             "expected_provided_label_manifests_sha256": {split: candidate["provided_label_manifests"][split]["sha256"] for split in SPLITS}}
            config = {"schema_version": "forte.automated-experiment-run-config.v1", "track": "automated_weak_supervision",
                      "experiment_authorization": authorization, "execution_stage": "prepared_local_metadata_only",
                      "requires_explicit_gpu_window_release": True, "human_verified_training_authorized": False,
                      "split_path": str(args.output.resolve()), "split_manifest_sha256": sha256_file(args.output),
                      "scenario_path": str(args.scenario_output.resolve()), "scenario_manifest_sha256": sha256_file(args.scenario_output),
                      "registry_path": str(args.exposure_registry.resolve()), "coverage_status_path": str(args.coverage_status.resolve()),
                      "provided_label_paths": {split: str((args.label_output_root / ("provided_binary_labels." + split + ".jsonl")).resolve()) for split in SPLITS},
                      "pre_run_coverage_refresh_required": stale_preparation,
                      "limitation": "Automated groups are unvalidated; binary labels do not establish event/fact/grounding truth or measured accuracy."}
            write_json(args.experiment_config_output, config)
        report.update(outputs_written=True, exposure_coverage_complete_snapshot=not coverage_errors,
                      pre_run_coverage_refresh_required=stale_preparation, experiment_config_written=bool(args.experiment_authorization_reference),
                      assignments_sha256=candidate["assignments_sha256"], split_manifest_sha256=sha256_file(args.output))
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)) else "automated_split_operation_failed"
        print(json.dumps({"ok": False, "error_code": code, "error_type": type(exc).__name__, "outputs_written": False, "training_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
