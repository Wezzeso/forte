#!/usr/bin/env python3
"""Reconcile a private user table into frozen role-specific coarse anchors.

Reads identifiers/workbook metadata only, never video pixels or observations.
No event type, boundary, fact, evidence or legal truth is inferred.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

try:
    from .audit_data import SPLITS, alias_key, json_bytes, load_workbook_rows, read_jsonl, sha256_file, write_json, write_jsonl
    from .build_automated_split import label_records_digest
except ImportError:
    from audit_data import SPLITS, alias_key, json_bytes, load_workbook_rows, read_jsonl, sha256_file, write_json, write_jsonl
    from build_automated_split import label_records_digest


def build_anchor_exports(source: dict, records: list[dict], original_rows: list[dict], audited_rows: list[dict], split: dict,
                         *, source_sha256: str, workbook_sha256: str) -> tuple[dict, dict]:
    if source.get("schema_version") != "forte.user-coarse-anchor-input.v1" or source.get("remaining_rows_explicitly_negative") is not True:
        raise ValueError("coarse_anchor_source_scope_invalid")
    raw_map = source.get("positive_anchor_seconds", {})
    positive_map = {}
    for name, seconds in raw_map.items():
        key = alias_key(name)
        if key in positive_map or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("coarse_anchor_source_identity_or_time_invalid")
        positive_map[key] = float(seconds)
    if len(original_rows) != source.get("expected_rows") or len(positive_map) != source.get("expected_positive_rows"):
        raise ValueError("coarse_anchor_source_count_mismatch")
    by_audit = {row["row_id"]: row for row in audited_rows}
    by_recording = {record["recording_id"]: record for record in records}
    assignments = {row["recording_id"]: row for row in split["records"]}
    if hashlib.sha256(json_bytes(split["records"])).hexdigest() != split.get("assignments_sha256"):
        raise ValueError("coarse_anchor_frozen_assignment_digest_mismatch")
    anchors_by_content, matched_keys, source_name_keys = defaultdict(list), set(), set()
    duplicate_disagreements = 0
    for row in original_rows:
        key = alias_key(row["source_filename"])
        if key in source_name_keys:
            raise ValueError("coarse_anchor_workbook_confusable_filename_collision")
        source_name_keys.add(key)
        positive = key in positive_map
        if type(row.get("has_violation")) is not bool or row["has_violation"] != positive:
            raise ValueError("user_binary_table_disagrees_with_original_workbook")
        audited = by_audit.get(row["row_id"], {})
        if audited.get("join_state") != "matched" or alias_key(audited.get("source_filename", "")) != key:
            raise ValueError("coarse_anchor_audited_unicode_join_mismatch")
        recording_id = audited.get("recording_id")
        record = by_recording.get(recording_id)
        if record is None or recording_id not in assignments:
            raise ValueError("coarse_anchor_recording_not_in_frozen_split")
        anchor = positive_map.get(key)
        if positive:
            matched_keys.add(key)
            if row.get("timestamp_sec") is None or abs(row["timestamp_sec"] - anchor) > 1e-9:
                raise ValueError("user_anchor_disagrees_with_original_workbook")
            if record.get("duration_sec") is None or anchor > record["duration_sec"]:
                raise ValueError("user_anchor_outside_source_duration")
        anchors_by_content[record["content_hash"]].append({"recording_id": recording_id, "workbook_row_id": row["row_id"],
                                                          "provided_video_label": positive, "provided_anchor_sec": anchor})
    if matched_keys != set(positive_map):
        raise ValueError("coarse_anchor_unknown_filename_or_unicode_collision")
    outputs = {role: [] for role in SPLITS}
    for content_hash, entries in sorted(anchors_by_content.items()):
        aliases = sorted(entry["recording_id"] for entry in entries)
        states = {entry["provided_video_label"] for entry in entries}
        assigned = [assignments[recording_id] for recording_id in aliases]
        roles, scenario_ids = {row["split"] for row in assigned}, {row["scenario_group_id"] for row in assigned}
        if len(states) != 1 or len(roles) != 1 or len(scenario_ids) != 1:
            raise ValueError("coarse_anchor_duplicate_label_or_partition_conflict")
        role = next(iter(roles))
        points = sorted({entry["provided_anchor_sec"] for entry in entries if entry["provided_anchor_sec"] is not None})
        duplicate_disagreements += len(aliases) > 1 and len(points) > 1
        outputs[role].append({"schema_version": "forte.user-coarse-anchor.v1", "content_sha256": content_hash, "content_hash": content_hash,
                              "recording_id": aliases[0], "recording_ids": aliases, "group_id": next(iter(scenario_ids)),
                              "scenario_group_id": next(iter(scenario_ids)), "split": role,
                              "provided_video_label": next(iter(states)), "coarse_anchor_sec": points,
                              "source_recording_anchors": sorted(entries, key=lambda item: item["recording_id"]),
                              "anchor_origin": "user_provided_coarse_table", "anchor_source_sha256": source_sha256,
                              "provided_label_source_sha256": workbook_sha256, "assignments_sha256": split["assignments_sha256"],
                              "target_scope": "class_agnostic_approximate_event_anchor", "timing_precision": "coarse_seconds",
                              "anchor_semantics": "approximate_point_not_boundary", "uncertainty_sec": None,
                              "event_type_known": False, "boundaries_known": False, "legal_ground_truth": False,
                              "outside_anchor_interval_label": "unknown", "source_provenance": source["provenance"],
                              "allowed_use": "train_only_context_prior_or_eventness_bag" if role == "train" else "evaluation_only_do_not_condition_inference" if role == "validation" else "held_out_do_not_train_or_condition_inference"})
    return outputs, {"ok": True, "supplied_rows": len(original_rows), "unique_contents": len(anchors_by_content),
                     "positive_rows": len(positive_map), "negative_rows": len(original_rows) - len(positive_map),
                     "duplicate_contents_with_distinct_coarse_anchors": duplicate_disagreements,
                     "workbook_binary_and_timestamp_agreement": True, "duration_checks_passed": True,
                     "split_unique_content_counts": {role: len(rows) for role, rows in outputs.items()},
                     "class_agnostic_only": True, "event_type_or_boundary_or_legal_truth_created": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/manifests/private/user_coarse_anchors.source.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    parser.add_argument("--workbook", type=Path, default=Path("open_labels.xlsx"))
    parser.add_argument("--audited-rows", type=Path, default=Path("data/manifests/workbook_rows.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/automated_provisional_split.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--pins-output", type=Path, default=Path("data/splits/user_coarse_anchors.pins.json"))
    args = parser.parse_args()
    try:
        source_digest, workbook_digest = sha256_file(args.input), sha256_file(args.workbook)
        split = json.loads(args.split.read_text(encoding="utf-8-sig"))
        if split.get("source_workbook_sha256") != workbook_digest or split.get("source_manifest_sha256") != sha256_file(args.manifest):
            raise ValueError("coarse_anchor_frozen_source_binding_mismatch")
        outputs, report = build_anchor_exports(json.loads(args.input.read_text(encoding="utf-8-sig")), read_jsonl(args.manifest),
                                               load_workbook_rows(args.workbook, "labels"), read_jsonl(args.audited_rows), split,
                                               source_sha256=source_digest, workbook_sha256=workbook_digest)
        paths = {role: args.output_root / ("user_coarse_anchors." + role + ".jsonl") for role in SPLITS}
        for path in [*paths.values(), args.pins_output]:
            if path.exists():
                raise ValueError("coarse_anchor_export_already_exists")
            if not path.resolve().is_relative_to(Path("data").resolve()):
                raise ValueError("coarse_anchor_export_must_remain_private_data")
        for role in SPLITS:
            write_jsonl(paths[role], outputs[role])
        pins = {"schema_version": "forte.user-coarse-anchor-pins.v1", "anchor_origin": "user_provided_coarse_table",
                "anchor_source_sha256": source_digest, "provided_label_source_sha256": workbook_digest,
                "assignments_sha256": split["assignments_sha256"], "split_file_sha256": sha256_file(args.split),
                "target_scope": "class_agnostic_approximate_event_anchor", "uncertainty_sec": None,
                "role_manifests": {role: {"path": str(paths[role].resolve()), "sha256": label_records_digest(outputs[role]), "records": len(outputs[role])} for role in SPLITS},
                "role_policy": {"train": "may_focus_train_context_and_class_agnostic_eventness_bags", "validation": "post_prediction_coarse_diagnostic_only", "final_check": "held_out_no_training_or_inference_conditioning"}}
        write_json(args.pins_output, pins)
        report.update(anchor_source_sha256=source_digest, assignments_sha256=split["assignments_sha256"],
                      pins_sha256=sha256_file(args.pins_output), role_manifest_sha256={role: pins["role_manifests"][role]["sha256"] for role in SPLITS})
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)) else "coarse_anchor_ingest_failed"
        print(json.dumps({"ok": False, "error_code": code, "event_type_or_boundary_or_legal_truth_created": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
