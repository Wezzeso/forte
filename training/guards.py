"""Content-safe training checks. Exceptions contain codes, never private targets."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "forte.training.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TrainingGateError(ValueError):
    """The code is safe to print in a cloud-backed development log."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise TrainingGateError(code)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise TrainingGateError("invalid_or_missing_json") from None


def load_annotations(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, ValueError):
        raise TrainingGateError("invalid_or_missing_annotations") from None


def local_file(path: str, root: Path) -> Path:
    """Resolve links too; an exported path cannot leave the approved media root."""
    require(isinstance(path, str) and bool(path) and "://" not in path, "media_path_invalid")
    candidate = Path(path)
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    require(candidate.is_relative_to(root.resolve()), "media_outside_approved_root")
    require(candidate.is_file(), "media_missing")
    return candidate


def locked_records(split: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(split.get("state") == "locked", "split_not_locked")
    require(split.get("locked") is True and split.get("training_authorized") is True, "split_training_not_authorized")
    require(split.get("scenario_groups_reviewed") is True, "scenario_groups_unreviewed")
    require(bool(split.get("approved_by")) and bool(split.get("locked_at")), "split_approval_missing")
    require(bool(HEX64.fullmatch(str(split.get("source_manifest_sha256", "")))), "source_manifest_hash_missing")
    records = split.get("records")
    require(isinstance(records, list) and bool(records), "split_records_empty")
    require(split.get("records_sha256") == canonical_sha256(sorted(
        records, key=lambda row: row.get("sample_id", row.get("recording_id", "")))), "split_records_hash_mismatch")
    groups: dict[str, str] = {}
    contents: dict[str, str] = {}
    indexed: dict[str, dict[str, Any]] = {}
    allowed = {"train", "validation", "final_check"}
    for row in records:
        require(isinstance(row, dict), "split_record_invalid")
        content, group, partition = row.get("content_sha256"), row.get("group_id"), row.get("split")
        require(isinstance(content, str) and bool(HEX64.fullmatch(content)), "split_content_hash_invalid")
        require(isinstance(group, str) and bool(group), "split_group_missing")
        require(partition in allowed, "split_partition_invalid")
        require(groups.get(group, partition) == partition, "scenario_group_leakage")
        require(contents.get(content, partition) == partition, "duplicate_content_leakage")
        if content in indexed:
            require(indexed[content].get("group_id") == group, "duplicate_content_group_mismatch")
        groups[group], contents[content], indexed[content] = partition, partition, row
    require(set(groups.values()) == allowed, "split_partition_coverage_missing")
    return indexed


def _interval(start: Any, end: Any) -> bool:
    return (isinstance(start, (float, int)) and not isinstance(start, bool)
            and isinstance(end, (float, int)) and not isinstance(end, bool)
            and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end)


def validate_target_scope(records: list[dict[str, Any]], expected_count: int = 19) -> None:
    """E1/E2 use full observation targets; optional screening is not training data."""
    require(expected_count == 19 and bool(records)
            and all(len(record["requested_fact_ids"]) == 19 for record in records),
            "training_requires_full_19_fact_targets")


def validate_target(target: Any, frame_ids: set[str], requested: list[str], event_ids: list[str]) -> None:
    """Mirror the A08 raw decision contract, without ever printing a target."""
    require(isinstance(target, dict) and set(target) == {"facts", "candidates"}, "target_root_invalid")
    require(isinstance(target["facts"], list) and isinstance(target["candidates"], list), "target_lists_invalid")
    require(bool(requested) and len(set(requested)) == len(requested), "requested_fact_ids_invalid")
    seen: set[str] = set()
    for fact in target["facts"]:
        require(isinstance(fact, dict), "target_fact_invalid")
        require(set(fact) <= {"fact_id", "state", "supporting_frame_ids", "notes"}, "target_fact_fields_invalid")
        fact_id, state, support = fact.get("fact_id"), fact.get("state"), fact.get("supporting_frame_ids")
        require(fact_id in requested and fact_id not in seen, "target_fact_id_invalid")
        require(state in {"present", "absent", "unclear"}, "target_fact_state_invalid")
        require(isinstance(support, list) and set(support) <= frame_ids, "target_fact_support_invalid")
        require(state == "unclear" or bool(support), "target_fact_support_missing")
        require(fact.get("notes") is None or isinstance(fact["notes"], str), "target_notes_invalid")
        # Use the shared contract too; do not maintain a divergent FactDecision representation.
        from forte.shared.schemas import FactDecision
        try:
            FactDecision.model_validate(fact)
        except Exception:
            raise TrainingGateError("target_shared_fact_contract_invalid") from None
        seen.add(fact_id)
    require(seen == set(requested), "target_fact_coverage_missing")
    candidate_ids: set[str] = set()
    fields = {"candidate_id", "event_type_id", "start_frame_id", "end_frame_id",
              "supporting_fact_ids", "supporting_frame_ids", "visibility_notes"}
    for event in target["candidates"]:
        require(isinstance(event, dict) and set(event) == fields, "target_candidate_fields_invalid")
        require(isinstance(event["candidate_id"], str) and bool(event["candidate_id"])
                and event["candidate_id"] not in candidate_ids, "target_candidate_id_invalid")
        candidate_ids.add(event["candidate_id"])
        require(event["event_type_id"] in event_ids, "target_event_type_unreviewed")
        require(event["start_frame_id"] in frame_ids and event["end_frame_id"] in frame_ids,
                "target_candidate_bounds_invalid")
        require(isinstance(event["supporting_frame_ids"], list) and bool(event["supporting_frame_ids"])
                and set(event["supporting_frame_ids"]) <= frame_ids, "target_candidate_frames_invalid")
        require(isinstance(event["supporting_fact_ids"], list) and bool(event["supporting_fact_ids"])
                and set(event["supporting_fact_ids"]) <= seen, "target_candidate_facts_invalid")
        require(isinstance(event["visibility_notes"], list)
                and all(isinstance(note, str) for note in event["visibility_notes"]), "target_visibility_invalid")


def validate_training_inputs(manifest: dict[str, Any], split: dict[str, Any],
                             annotations: list[dict[str, Any]], *, split_hash: str,
                             annotations_hash: str, media_root: Path,
                             verify_frames: bool = True) -> list[dict[str, Any]]:
    indexed = locked_records(split)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "training_schema_invalid")
    require(manifest.get("split_manifest_sha256") == split_hash, "split_file_hash_mismatch")
    require(manifest.get("annotations_sha256") == annotations_hash, "annotations_file_hash_mismatch")
    require(bool(manifest.get("approved_by")), "training_approval_missing")
    reviews = {row.get("content_sha256"): row for row in annotations}
    require(len(reviews) == len(annotations), "duplicate_review_record")
    records = manifest.get("records")
    require(isinstance(records, list) and bool(records), "training_records_empty")
    sample_ids: set[str] = set()
    for row in records:
        require(isinstance(row, dict), "training_record_invalid")
        sample_id = row.get("sample_id")
        require(isinstance(sample_id, str) and bool(sample_id) and sample_id not in sample_ids,
                "training_sample_id_invalid")
        sample_ids.add(sample_id)
        content = row.get("content_sha256")
        source = indexed.get(content)
        require(source is not None and source["split"] == "train" and row.get("split") == "train",
                "training_partition_forbidden")
        require(row.get("group_id") == source["group_id"], "training_group_mismatch")
        require(any(parent.get("content_sha256") == content
                    and row.get("source_recording_id") == parent.get("recording_id", parent.get("sample_id"))
                    for parent in split["records"]),
                "training_source_recording_mismatch")
        require(row.get("data_origin") == "development", "jury_or_unknown_origin_forbidden")
        review = reviews.get(content)
        require(review is not None, "source_review_missing")
        require(review.get("review_status") == "human_reviewed" and review.get("review_state") == "reviewed",
                "source_review_incomplete")
        require(review.get("full_review_complete") is True, "source_full_review_incomplete")
        require(review.get("split_lock_verified") is True and review.get("split_manifest_sha256") == split_hash,
                "source_review_split_lock_unverified")
        require(review.get("source_role") in {"organizer_development", "team_training"},
                "source_jury_or_unknown_origin_forbidden")
        require(review.get("group_id") == source["group_id"] and review.get("split") == "train",
                "review_group_or_split_mismatch")
        require(row.get("annotation_record_sha256") == canonical_sha256(review), "review_record_hash_mismatch")
        require(row.get("review_status") == "human_reviewed" and row.get("reviewer_id")
                and row.get("reviewed_at") and review.get("reviewer_id") and review.get("reviewed_at"),
                "training_target_unreviewed")
        require(not review.get("unresolved_items"), "source_review_unresolved")
        start, end = row.get("start_sec"), row.get("end_sec")
        require(_interval(start, end), "training_interval_invalid")
        frames = row.get("frames")
        require(isinstance(frames, list) and 2 <= len(frames) <= 16, "training_frame_budget_invalid")
        frame_ids, previous = set(), -1.0
        for frame in frames:
            require(isinstance(frame, dict), "frame_invalid")
            fid, pts = frame.get("frame_id"), frame.get("pts_sec")
            require(isinstance(fid, str) and bool(fid) and fid not in frame_ids, "frame_id_invalid")
            require(_interval(pts, pts) and start <= pts <= end and pts > previous, "frame_timestamp_invalid")
            frame_ids.add(fid)
            previous = pts
            if verify_frames:
                resolved = local_file(frame.get("path"), media_root)
                require(HEX64.fullmatch(str(frame.get("sha256", ""))) is not None
                        and file_sha256(resolved) == frame["sha256"], "frame_checksum_mismatch")
        reviewed_frames = {frame.get("frame_id"): frame for frame in review.get("frames", [])}
        require(all(frame["frame_id"] in reviewed_frames
                    and reviewed_frames[frame["frame_id"]].get("pts_sec") == frame["pts_sec"]
                    for frame in frames), "training_frames_not_in_review")
        requested, event_ids = row.get("requested_fact_ids"), row.get("reviewed_event_ids")
        require(isinstance(requested, list) and isinstance(event_ids, list), "target_definitions_missing")
        validate_target(row.get("target"), frame_ids, requested, event_ids)
        # Reuse A08's actual parser/binding policy, including candidate/fact support relationships.
        from forte.decision import validate_decisions
        from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef
        try:
            request = ClipRequest(
                job_id="training", clip_id=sample_id, start_sec=start, end_sec=end,
                frame_refs=[FrameRef(frame_id=frame["frame_id"], pts_sec=frame["pts_sec"],
                                     path=str((media_root / frame["path"]).resolve()), width=1, height=1)
                            for frame in frames],
                fact_definitions=[FactDefinition(fact_id=fact_id, description="Approved definition in frozen prompt")
                                  for fact_id in requested],
            )
            validate_decisions(row["target"], request, reviewed_event_ids=event_ids)
        except Exception:
            raise TrainingGateError("target_a08_contract_invalid") from None
        for event in row["target"]["candidates"]:
            times = {frame["frame_id"]: frame["pts_sec"] for frame in frames}
            require(times[event["start_frame_id"]] <= times[event["end_frame_id"]], "candidate_time_reversed")
        is_negative = row.get("negative_verified") is True
        if is_negative:
            require(not row["target"]["candidates"], "negative_target_has_event")
            require(any(window.get("negative_verified") is True and window.get("context_complete") is True
                        and window.get("visibility") == "clear"
                        and window.get("start_sec", math.inf) <= start and end <= window.get("end_sec", -1)
                        for window in review.get("negative_windows", [])), "negative_window_unverified")
        else:
            require(bool(row["target"]["candidates"]), "normal_window_requires_explicit_review")
            require(any(event.get("start_sec", math.inf) <= end and start <= event.get("end_sec", -1)
                        for event in review.get("events", [])), "positive_window_outside_review")
            require(all(any(event.get("event_type_id") == candidate["event_type_id"]
                            and event.get("start_sec", math.inf) <= end and start <= event.get("end_sec", -1)
                            for event in review.get("events", [])) for candidate in row["target"]["candidates"]),
                    "target_event_not_in_source_review")
        prompt = row.get("prompt")
        require(isinstance(prompt, str) and bool(prompt) and "<|" not in prompt, "prompt_invalid")
        require(content not in prompt and row["source_recording_id"] not in prompt
                and not any(frame["path"] in prompt for frame in frames), "source_identity_in_prompt_forbidden")
        require(bool(row.get("prompt_version")) and bool(row.get("config_version")), "prompt_provenance_missing")
        from forte.decision import PROMPT_VERSION, CONFIG_VERSION
        require(row["prompt_version"] == PROMPT_VERSION and row["config_version"] == CONFIG_VERSION,
                "training_prompt_version_not_current")
        serialized = json.dumps(row["target"], ensure_ascii=False)
        require("<|" not in serialized, "target_control_token_forbidden")
    return records
