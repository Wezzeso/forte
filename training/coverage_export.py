"""Normalize one bounded teacher run into a NEW partial machine-SILVER export.

Original proposal audits, old machine exports and original JPEGs stay intact.
This CPU utility checks source files and JPEG identities using the established
machine normalizer. It performs no inference and produces no human gold.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .guards import HEX64, TrainingGateError, canonical_sha256, file_sha256, load_annotations, load_json, require


def _eligible_window(window: dict, expected_teacher: dict) -> bool:
    provenance = window.get("provenance", {})
    attestation = provenance.get("serving_attestation", {})
    if not (window.get("model_output_complete") is True and provenance.get("actual_snapshot_verified") is True
            and all(provenance.get(key) == value for key, value in expected_teacher.items())
            and HEX64.fullmatch(str(provenance.get("prompt_sha256", ""))) is not None
            and attestation.get("model_files_verified") is True
            and all(attestation.get(key) == expected_teacher.get(key) for key in
                    ("model_revision", "processor_revision", "model_manifest_sha256"))):
        return False
    reviews = {(item.get("frame_id"), item.get("fact_id")): item for item in window.get("support_reviews", [])}
    for target in window.get("frame_fact_targets", []):
        review = reviews.get((target.get("frame_id"), target.get("fact_id")), {})
        if (target.get("state") in {"present", "absent"} and target.get("supervision_mask") is True
                and target.get("frame_id") in target.get("supporting_frame_ids", [])
                and review.get("supervision_mask") is True and review.get("supports_state") is True
                and review.get("teacher_state") == target["state"] and review.get("state") == target["state"]
                and review.get("supporting_frame_ids") == target["supporting_frame_ids"]
                and review.get("provenance") == target.get("provenance")):
            return True
    return False


def export_coverage(config: dict) -> dict[str, Any]:
    output, proof_path = Path(config["output"]), Path(config["proof_output"])
    require(not output.exists() and not proof_path.exists(), "coverage_export_destination_exists")
    paths = {key: Path(config[key]) for key in ("teacher_plan", "source_manifest", "split", "native_ledger", "machine_schema")}
    require(all(file_sha256(path) == config.get("expected_" + key + "_sha256") for key, path in paths.items()),
            "coverage_export_input_pin_mismatch")
    plan, split, ledger = (load_json(paths[key]) for key in ("teacher_plan", "split", "native_ledger"))
    from .teacher_sampling import validate_teacher_plan
    validate_teacher_plan(plan, split, split_sha256=file_sha256(paths["split"]))
    content_set = {record["content_sha256"] for record in plan["records"]}
    require(len(content_set) == 17 and all(record["split"] == "train" for record in plan["records"]),
            "coverage_export_exact_train_population_invalid")
    roles = {}
    for row in split["records"]:
        content = row.get("content_sha256", row.get("content_hash"))
        roles.setdefault(content, set()).add((row["split"], row.get("scenario_group_id", row.get("group_id"))))
    sources = {}
    for row in load_annotations(paths["source_manifest"]):
        sources.setdefault(row.get("content_sha256", row.get("content_hash")), set()).add(row["recording_id"])
    slots = {(item["content_sha256"], item["second_index"]): item for item in ledger["attempts"]}
    raw_root = Path(config["raw_teacher_root"]).resolve()
    audits = list(raw_root.glob("train/*/teacher.private.json"))
    require(len(audits) == 17, "coverage_export_requires_terminal_audit_for_every_content")
    from annotations.machine import summarize_record
    from .silver_combine import _validate_schema
    summaries, seen, audit_records, call_count = [], set(), [], 0
    excluded_sources, completed_windows, eligible_windows = [], 0, 0
    for path in sorted(audits):
        raw_digest = file_sha256(path)
        original = load_json(path)
        content = original.get("asset", {}).get("content_hash")
        require(content in content_set and content not in seen and roles.get(content) is not None
            and len(roles[content]) == 1 and next(iter(roles[content]))[0] == "train",
            "coverage_export_raw_content_or_role_mismatch")
        seen.add(content)
        require(original.get("schema_version") == "forte.local-proposal.v1"
            and original.get("phase") in {"partial", "done", "failed"}
            and original.get("annotation_origin") == "local_model" and original.get("human_reviewed") is False
            and original.get("organizer_labels_used_for_facts") is False,
            "coverage_export_raw_origin_invalid")
        provenance = original.get("provenance", {})
        require(provenance.get("plan_sha256") == plan["plan_sha256"]
            and all(provenance.get(key) == value for key, value in config["expected_teacher"].items()),
            "coverage_export_teacher_identity_mismatch")
        attempted = original.get("coverage", {}).get("attempted_model_requests")
        require(type(attempted) is int and 0 <= attempted <= 2, "coverage_export_attempt_count_invalid")
        require(len(original.get("window_decisions", [])) <= attempted
            and len({window.get("second_index") for window in original.get("window_decisions", [])})
            == len(original.get("window_decisions", [])), "coverage_export_completed_window_count_invalid")
        call_count += attempted
        for window in original.get("window_decisions", []):
            key = (content, window.get("second_index"))
            require(key in slots and window.get("frame_ids") == [frame["frame_id"] for frame in slots[key]["frames"]]
                and window.get("anchor_frame_ids") == [slots[key]["anchor_frame_id"]]
                and window.get("anchor_pts_sec") == [slots[key]["anchor_pts_sec"]],
                "coverage_export_raw_window_native_binding_changed")
        proposal = deepcopy(original)
        proposal.update(audit_sha256=raw_digest, audit_path=str(path))
        group = next(iter(roles[content]))[1]
        record = {"content_hash": content, "sample_id": "sample_" + content[:24],
            "recording_ids": sorted(sources[content]), "group_id": group, "split": "train",
            "source_role": "organizer_development", "local_model_proposals": [proposal]}
        summary = summarize_record(record, raw_root)
        completed_windows += len(original.get("window_decisions", []))
        kept = [] if summary is None else [window for window in summary.get("windows", [])
            if _eligible_window(window, config["expected_teacher"])]
        reason = ("normalizer_has_no_source_summary" if summary is None else
                  "source_or_jpeg_identity_unverified" if summary.get("source_pts_identity_checked") is not True else
                  "no_valid_attested_decisive_exact_frame_targets" if not kept else None)
        if reason is not None:
            excluded_sources.append({"content_sha256": content, "reason_code": reason,
                "attempted_calls": attempted, "completed_windows": len(original.get("window_decisions", []))})
        else:
            eligible_windows += len(kept)
            kept_ids = {window["window_id"] for window in kept}
            summary["windows"] = kept
            for second in summary.get("per_second", []):
                second["window_ids"] = [wid for wid in second.get("window_ids", []) if wid in kept_ids]
                if not second["window_ids"]:
                    second["state"], second["event_state"] = "source_frames_available_unanalyzed", "abstain"
            summary["group_id"] = group
            summary["partition_binding"] = {"split_file_sha256": config["expected_split_sha256"],
                "assignments_sha256": split["assignments_sha256"], "origin": split.get("state"),
                "scenario_independence_validated": False}
            summary["coverage_export_filter"] = {"completed_windows": len(original.get("window_decisions", [])),
                "retained_windows": len(kept), "scope": "attested_decisive_exact_frame_targets_only",
                "raw_audit_sha256": raw_digest, "missing_windows_are_not_negative": True}
            summary["summary_sha256"] = canonical_sha256({key: value for key, value in summary.items()
                                                          if key not in {"summary_sha256", "generated_at"}})
            summaries.append(summary)
        require(file_sha256(path) == raw_digest, "coverage_export_original_audit_changed")
        audit_records.append({"content_sha256": content, "path": str(path), "sha256": raw_digest,
            "phase": original["phase"], "attempted_calls": attempted,
            "exported": reason is None, "excluded_reason_code": reason,
            "completed_windows": len(original.get("window_decisions", [])),
            "retained_windows": len(kept) if reason is None else 0})
    require(seen == content_set and call_count <= 34, "coverage_export_population_or_attempt_budget_changed")
    _validate_schema(summaries, paths["machine_schema"])
    if summaries:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            for summary in sorted(summaries, key=lambda row: row["content_sha256"]):
                stream.write(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    result = {"schema_version": "forte.coverage-machine-export.v1",
        "status": "partial_silver_exported" if summaries else "no_eligible_targets",
        "file_sha256": file_sha256(output) if summaries else None,
        "records": len(summaries), "windows": sum(len(row["windows"]) for row in summaries),
        "planned_sources": 17, "attempted_source_receipts": len(audit_records),
        "excluded_source_count": len(excluded_sources), "excluded_sources": excluded_sources,
        "completed_teacher_windows": completed_windows, "eligible_teacher_windows": eligible_windows,
        "completed_windows_without_eligible_targets": completed_windows - eligible_windows,
        "invalid_or_incomplete_call_count": call_count - completed_windows,
        "attempted_teacher_calls": call_count, "maximum_teacher_calls": 34,
        "all_train": True, "human_reviewed": 0, "schema_errors": 0,
        "raw_audit_receipts": audit_records, "raw_audits_modified": False,
        "old_export_modified": False, "source_jpegs_modified": False,
        "whole_video_gold_created": False, "legal_or_boundary_gold_created": False,
        "native_ledger_sha256": config["expected_native_ledger_sha256"]}
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    with proof_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    require(bool(summaries), "coverage_export_no_eligible_targets")
    return {key: value for key, value in result.items() if key not in {"raw_audit_receipts", "excluded_sources"}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(export_coverage(load_json(args.config)), sort_keys=True))
        return 0
    except TrainingGateError as error:
        print(json.dumps({"status": "blocked", "reason_code": str(error)}))
        return 2
    except Exception as error:
        print(json.dumps({"status": "failed", "reason_code": "coverage_export_failed", "error_type": type(error).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
