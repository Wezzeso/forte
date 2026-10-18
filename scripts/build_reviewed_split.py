#!/usr/bin/env python3
"""Bridge explicit local human scenario proposals to an unlocked candidate.

Reads a consistent read-only SQLite snapshot with JSON projections of identity
and scenario-review metadata only. It does not read media or annotation facts.
Incomplete/conflicting review fails closed; output is aggregate JSON only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

try:
    from .audit_data import check_split, create_candidate, json_bytes, read_jsonl, sha256_file, write_json
    from .exposure_registry import DEFAULT_REGISTRY, accepted_inventory_hashes, manifest_identity, read_registry
except ImportError:
    from audit_data import check_split, create_candidate, json_bytes, read_jsonl, sha256_file, write_json
    from exposure_registry import DEFAULT_REGISTRY, accepted_inventory_hashes, manifest_identity, read_registry

BRIDGE_VERSION = "forte.human-scenario-bridge.v1"


def review_snapshot(database: Path) -> tuple[list[dict], list[dict]]:
    if not database.is_file():
        raise ValueError("human_review_database_missing")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=20)
    connection.row_factory = sqlite3.Row
    proposal_projection = "(SELECT json_group_array(json(item.value)) FROM json_each({payload},'$.corrections') AS item WHERE json_extract(item.value,'$.kind')='scenario_group_proposal')"
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        records = []
        query = "SELECT sample_id,content_hash,revision,json_extract(payload,'$.recording_ids') AS recording_ids,json_extract(payload,'$.review_origin') AS review_origin,json_extract(payload,'$.reviewer_id') AS reviewer_id,json_extract(payload,'$.review_state') AS review_state,json_extract(payload,'$.full_review_complete') AS full_review_complete," + proposal_projection.format(payload="payload") + " AS proposals FROM records ORDER BY sample_id"
        for row in connection.execute(query):
            record = dict(row)
            record["recording_ids"] = json.loads(record["recording_ids"] or "null")
            record["proposals"] = json.loads(record["proposals"] or "[]")
            record["full_review_complete"] = record["full_review_complete"] == 1
            records.append(record)
        audits = []
        query = "SELECT sample_id,revision,actor,action,changed_at,json_extract(after_payload,'$.review_origin') AS review_origin,json_extract(after_payload,'$.reviewer_id') AS reviewer_id," + proposal_projection.format(payload="after_payload") + " AS proposals FROM audit WHERE action='review_update' ORDER BY audit_id"
        for row in connection.execute(query):
            audit = dict(row)
            audit["proposals"] = json.loads(audit["proposals"] or "[]")
            audits.append(audit)
        return records, audits
    finally:
        connection.rollback()
        connection.close()


def proposal_signature(proposal: dict) -> tuple:
    return tuple(proposal.get(key) for key in ("kind", "proposed_group_id", "scenario_reviewed", "reviewer_id"))


def analyze_review(records: list[dict], review_records: list[dict], audits: list[dict]) -> tuple[dict, dict | None]:
    _, by_hash = manifest_identity(records)
    expected = set(by_hash)
    found: dict[str, dict] = {}
    errors: Counter = Counter()
    group_members: dict[str, list[str]] = defaultdict(list)
    group_reviewers: dict[str, set[str]] = defaultdict(set)
    confirmed = 0
    annotation_reviewed = 0
    proof_index = defaultdict(list)
    for audit in audits:
        proof_index[audit.get("sample_id")].append(audit)
    for record in review_records:
        content_hash = record.get("content_hash")
        if content_hash not in expected:
            errors["unknown_review_content"] += 1
            continue
        if content_hash in found:
            errors["duplicate_review_content"] += 1
            continue
        found[content_hash] = record
        members = record.get("recording_ids")
        if not isinstance(members, list) or sorted(members) != by_hash[content_hash]:
            errors["duplicate_recording_alias_coverage_mismatch"] += 1
        proposals = record.get("proposals", [])
        if not isinstance(proposals, list) or len(proposals) != 1:
            errors["missing_or_conflicting_scenario_proposal"] += 1
            continue
        proposal = proposals[0]
        if not isinstance(proposal, dict):
            errors["invalid_scenario_proposal"] += 1
            continue
        group_id = proposal.get("proposed_group_id")
        author = proposal.get("reviewer_id")
        if proposal.get("scenario_reviewed") is not True or not isinstance(group_id, str) or not group_id.strip() or group_id != group_id.strip() or len(group_id) > 128 or any(ord(character) < 32 for character in group_id):
            errors["scenario_confirmation_incomplete"] += 1
            continue
        if record.get("review_origin") != "human" or not isinstance(record.get("reviewer_id"), str) or not record["reviewer_id"].strip() or not isinstance(author, str) or not author.strip():
            errors["scenario_confirmation_not_human_attributed"] += 1
            continue
        # The original proposal author can differ from a later notes editor.
        # Find the saved human audit that actually confirmed this exact proposal.
        proof = any(audit.get("action") == "review_update" and audit.get("review_origin") == "human" and audit.get("actor") == author and audit.get("reviewer_id") == author and isinstance(audit.get("revision"), int) and 0 < audit["revision"] <= record.get("revision", 0) and any(isinstance(item, dict) and proposal_signature(item) == proposal_signature(proposal) for item in audit.get("proposals", [])) for audit in proof_index[record.get("sample_id")])
        if not proof:
            errors["scenario_confirmation_audit_missing"] += 1
            continue
        group_members[group_id].extend(by_hash[content_hash])
        group_reviewers[group_id].add(author)
        confirmed += 1
        annotation_reviewed += bool(record.get("review_state") == "reviewed" and record.get("full_review_complete") is True)
    errors["missing_review_content"] = len(expected - set(found))
    nonzero = {key: count for key, count in sorted(errors.items()) if count}
    report = {"ready": not nonzero and bool(expected), "unique_contents": len(expected), "review_units": len(review_records),
              "human_confirmed_contents": confirmed, "human_annotation_reviewed_contents": annotation_reviewed,
              "proposed_scenario_groups": len(group_members), "errors": nonzero,
              "training_authorized": False, "locked": False}
    if not report["ready"]:
        return report, None
    scenario = {"schema_version": BRIDGE_VERSION, "review_state": "human_confirmed_proposal", "approved_by": None, "locked": False,
                "provenance": "Explicit scenario confirmations attributed to saved local human review audit entries; pending A20 split approval.",
                "groups": [{"scenario_group_id": group_id, "recording_ids": sorted(members),
                            "review_state": "human_confirmed_proposal", "reviewer_ids": sorted(group_reviewers[group_id])}
                           for group_id, members in sorted(group_members.items())]}
    return report, scenario


def build_candidate(records: list[dict], rows: list[dict], review_records: list[dict], audits: list[dict], registry: dict, *, seed: int = 20261017, ratios: list[float] | None = None) -> tuple[dict, dict | None, dict | None]:
    report, scenario = analyze_review(records, review_records, audits)
    report["distinct_model_exposed_contents"] = len(registry["content_hashes"])
    report["exposure_events"] = registry["event_count"]
    if not registry["event_count"]:
        report["errors"]["exposure_inventory_empty"] = 1
        report["ready"] = False
    if not report["ready"]:
        return report, None, None
    candidate = create_candidate(records, rows, seed, ratios or [0.7, 0.15, 0.15], scenario, registry["content_hashes"])
    unconstrained = create_candidate(records, rows, seed, ratios or [0.7, 0.15, 0.15], scenario)
    desired_final_groups = len({row["scenario_group_id"] for row in unconstrained["records"] if row["split"] == "final_check"})
    available_final_groups = len({row["scenario_group_id"] for row in candidate["records"] if row["split"] == "final_check"})
    report["required_unexposed_final_check_groups"] = desired_final_groups
    report["allocated_unexposed_final_check_groups"] = available_final_groups
    if available_final_groups < desired_final_groups or desired_final_groups == 0:
        report["errors"]["insufficient_unexposed_scenario_groups"] = 1
        report["ready"] = False
        return report, None, None
    check = check_split(candidate, records, scenario=scenario, excluded_final_hashes=registry["content_hashes"])
    if not check["ok"]:
        raise ValueError("human_proposal_candidate_integrity_failed")
    candidate.update({"builder_version": BRIDGE_VERSION, "scenario_review_source": "local_human_confirmations_pending_split_approval",
                      "annotation_review_complete": report["human_annotation_reviewed_contents"] == len({record["content_hash"] for record in records}),
                      "exposure_registry_tip_sha256": registry["tip_sha256"], "exposure_registry_event_count": registry["event_count"],
                      "exposure_content_set_sha256": registry["content_set_sha256"], "ready_for_a20_review": True,
                      "limitations": ["Human scenario proposals are complete; candidate remains unlocked and is not training authorization."]})
    report["split_integrity"] = check
    report["ready_for_a20_review"] = True
    return report, candidate, scenario


def exposure_coverage(status_path: Path, records: list[dict], registry: dict, max_age_sec: float, *, source_inventory_path_override: Path | None = None) -> tuple[dict, dict]:
    """Verify a private accepted-job inventory snapshot, never query its results."""
    errors = {}
    aggregate = {"exposure_coverage_complete_snapshot": False}
    if not status_path.is_file():
        return aggregate, {"exposure_inventory_coverage_missing": 1}
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    if status.get("schema_version") != "forte.exposure-coverage.v1" or status.get("coverage_state") != "complete_snapshot" or status.get("scope") != "all_accepted_jobs_since_service_start":
        errors["exposure_inventory_coverage_incomplete"] = 1
    captured = dt.datetime.fromisoformat(status["captured_at_utc"].replace("Z", "+00:00"))
    if captured.tzinfo is None:
        errors["exposure_inventory_snapshot_time_invalid"] = 1
    else:
        age = (dt.datetime.now(dt.timezone.utc) - captured).total_seconds()
        aggregate["exposure_snapshot_age_sec"] = round(age, 3)
        if age < -30 or age > max_age_sec:
            errors["exposure_inventory_snapshot_stale"] = 1
    ancestors = {event["event_sha256"] for event in registry["events"]}
    if status.get("registry_tip_sha256") not in ancestors or not isinstance(status.get("registry_event_count"), int) or status["registry_event_count"] > registry["event_count"]:
        errors["exposure_coverage_registry_snapshot_mismatch"] = 1
    inventory_path = Path(source_inventory_path_override) if source_inventory_path_override is not None else Path(status["source_inventory_path"])
    if "private" not in inventory_path.resolve().parts or not inventory_path.is_file() or sha256_file(inventory_path) != status.get("source_inventory_sha256"):
        errors["accepted_job_inventory_digest_mismatch"] = 1
    else:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
        try:
            inventory_hashes = accepted_inventory_hashes(inventory)
        except ValueError:
            errors["accepted_job_inventory_invalid"] = 1
        else:
            known = {record["content_hash"] for record in records} & inventory_hashes
            if not known <= registry["content_hashes"] or status.get("known_corpus_contents") != len(known) or status.get("accepted_job_count") != inventory.get("accepted_job_count"):
                errors["accepted_known_content_not_registered"] = 1
            aggregate["accepted_job_count"] = inventory.get("accepted_job_count")
            aggregate["accepted_known_corpus_contents"] = len(known)
    aggregate["exposure_coverage_complete_snapshot"] = not errors
    return aggregate, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["readiness", "build"])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    parser.add_argument("--workbook-rows", type=Path, default=Path("data/manifests/workbook_rows.jsonl"))
    parser.add_argument("--workbook", type=Path, default=Path("open_labels.xlsx"))
    parser.add_argument("--base-split", type=Path, default=Path("data/splits/candidate_split.json"))
    parser.add_argument("--review-database", type=Path, default=Path("annotations/private/review.sqlite3"))
    parser.add_argument("--exposure-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--exposure-inventory-status", type=Path, default=Path("benchmarks/private/exposure-inventory-status.json"))
    parser.add_argument("--max-exposure-snapshot-age-sec", type=float, default=300)
    parser.add_argument("--candidate-output", type=Path, default=Path("data/splits/reviewed_candidate_split.json"))
    parser.add_argument("--scenario-output", type=Path, default=Path("data/splits/scenario_groups.human_proposal.json"))
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    try:
        records = read_jsonl(args.manifest)
        base = json.loads(args.base_split.read_text(encoding="utf-8-sig"))
        if base.get("source_manifest_sha256") != sha256_file(args.manifest) or base.get("source_workbook_sha256") != sha256_file(args.workbook):
            raise ValueError("bridge_source_input_digest_mismatch")
        registry = read_registry(args.exposure_registry, records)
        review_records, audits = review_snapshot(args.review_database)
        report, candidate, scenario = build_candidate(records, read_jsonl(args.workbook_rows), review_records, audits, registry, seed=args.seed)
        coverage, coverage_errors = exposure_coverage(args.exposure_inventory_status, records, registry, args.max_exposure_snapshot_age_sec)
        report.update(coverage)
        report["errors"].update(coverage_errors)
        report["ready"] = report["ready"] and not coverage_errors
        report["ready_for_a20_review"] = report["ready"]
        report["command"] = args.command
        report["outputs_written"] = False
        if args.command == "build" and report["ready"]:
            for path in (args.candidate_output, args.scenario_output):
                if not path.resolve().is_relative_to(Path("data/splits").resolve()):
                    raise ValueError("bridge_outputs_must_remain_private_splits")
                if path.resolve() in {args.base_split.resolve(), args.review_database.resolve(), args.exposure_registry.resolve(), Path("data/splits/scenario_groups.review.json").resolve()} or "locked" in path.name.casefold():
                    raise ValueError("bridge_cannot_overwrite_source_or_protected_split")
                if path.exists():
                    existing = json.loads(path.read_text(encoding="utf-8-sig"))
                    if existing.get("locked") or existing.get("state") == "locked":
                        raise ValueError("bridge_cannot_overwrite_source_or_protected_split")
            candidate.update({"source_manifest_sha256": sha256_file(args.manifest), "source_workbook_sha256": sha256_file(args.workbook),
                              "source_workbook_rows_sha256": sha256_file(args.workbook_rows),
                              "human_review_metadata_sha256": hashlib.sha256(json_bytes({"records": review_records, "audits": audits})).hexdigest(),
                              "exposure_coverage_status_sha256": sha256_file(args.exposure_inventory_status),
                              "exposure_snapshot_scope": "accepted-job inventory at build time; refresh before split lock while live uploads remain active",
                              "scenario_groups_sha256": hashlib.sha256(json_bytes(scenario)).hexdigest()})
            write_json(args.scenario_output, scenario)
            write_json(args.candidate_output, candidate)
            report["outputs_written"] = True
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ready"] else 2
    except Exception as exc:
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)) else "scenario_bridge_operation_failed"
        print(json.dumps({"ready": False, "error_type": type(exc).__name__, "error_code": code, "training_authorized": False, "locked": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
