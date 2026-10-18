"""Separate provisional supervision; never create human-reviewed annotations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
import math
import re
from typing import Any

from training.guards import HEX64, canonical_sha256, require
from .targets import FeatureSequence, SparseTargets, _build_targets


@dataclass(frozen=True)
class TeacherWindowTargets:
    sequence_frame_indices: tuple[int, ...]
    targets: SparseTargets
    provenance_sha256: str
    label_origin: str = "local_teacher_silver"
    human_verified: bool = False


@dataclass(frozen=True)
class WeakSupervision:
    provided_video_label: int
    provided_label_source_sha256: str
    provided_label_record_sha256: str
    teacher_windows: tuple[TeacherWindowTargets, ...]
    masked_teacher_event_conflicts: int = 0
    track: str = "weak_candidate_unselected"
    human_verified: bool = False


def teacher_windows_from_machine_record(sequence: FeatureSequence, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize A03's exact separate machine export without inventing observations."""
    require(record.get("schema_version") == "forte.machine-annotations.v1"
            and record.get("annotation_origin") == "local_model"
            and record.get("review_status") == "machine_consistency_checked"
            and record.get("human_reviewed") is False, "weak_machine_export_origin_invalid")
    require(record.get("content_sha256") == sequence.content_sha256, "weak_machine_export_source_mismatch")
    source = record.get("provenance", {})
    provenance = {"model_id": source.get("model_id"), "model_revision": source.get("model_revision"),
                  "artifacts_sha256": source.get("model_manifest_sha256"),
                  "prompt_sha256": source.get("prompt_sha256"),
                  "source_proposal_sha256": source.get("source_proposal_sha256"),
                  "inference_location": source.get("inference_location")}
    frame_ids = {frame.frame_id for frame in sequence.frames}
    windows = []
    for original in record.get("windows", []):
        window_provenance = {**provenance, **original.get("provenance", {})}
        # Current audits bind prompts per window, never to a whole recording.
        if original.get("provenance", {}).get("prompt_sha256") is not None:
            window_provenance["prompt_sha256"] = original["provenance"]["prompt_sha256"]
        require(set(original.get("frame_ids", [])) <= frame_ids, "weak_machine_window_frame_binding_invalid")
        target = deepcopy(original.get("target", {"facts": [], "candidates": []}))
        supervised = original.get("fact_supervision_mask", {})
        target["facts"] = [fact for fact in target["facts"] if supervised.get(fact["fact_id"]) is True]
        conflicts = {(item.get("frame_id"), item.get("fact_id"))
                     for item in original.get("support_reviews", [])
                     if item.get("supervision_mask") is False
                     and item.get("mask_reason") == "direct_anchor_window_support_conflict"}
        if conflicts:
            retained = []
            for fact in target["facts"]:
                fact["supporting_frame_ids"] = [frame for frame in fact["supporting_frame_ids"]
                                                if (frame, fact["fact_id"]) not in conflicts]
                if fact["state"] in {"present", "absent"} and not fact["supporting_frame_ids"]:
                    continue
                retained.append(fact)
            target["facts"] = retained
        supervised_ids = {fact["fact_id"] for fact in target["facts"]}
        target["candidates"] = [event for event in target["candidates"]
                                if set(event["supporting_fact_ids"]) <= supervised_ids
                                and not any(frame in event["supporting_frame_ids"] and fact in event["supporting_fact_ids"]
                                            for frame, fact in conflicts)]
        context = original.get("context_masks", {})
        complete = context.get("procedure_complete") is True and context.get("continuous_visibility") is True
        normalized = {"annotation_origin": "local_model", "review_status": "machine_consistency_checked",
                        "human_reviewed": False, "machine_consistency_passed": True,
                        "teacher_provenance": window_provenance, "content_sha256": sequence.content_sha256,
                        "group_id": sequence.group_id, "split": sequence.split,
                        "start_sec": original.get("start_sec"), "end_sec": original.get("end_sec"),
                        "target": target, "fact_context": {fact_id: {
                            "review_status": "machine_consistency_checked", "context_complete": complete,
                            "visibility_adequate": context.get("continuous_visibility") is True}
                            for fact_id in supervised_ids}, "masked_support_conflicts": len(conflicts)}
        # A03/A06 may supply explicit native-frame physical observations. Their
        # per-item review/provenance is checked by _build_targets; window facts
        # are never automatically broadcast to frames.
        for name in ("frame_fact_targets", "support_reviews", "boundary_reviews"):
            if name in original:
                items = original[name]
                if name == "frame_fact_targets":
                    items = [item for item in items if item.get("supervision_mask") is True]
                elif name == "support_reviews":
                    items = [item for item in items if item.get("supervision_mask") is not False]
                normalized[name] = deepcopy(items)
        windows.append(normalized)
    return windows


def mil_logit_reference(instance_logits: list[float], temperature: float = .25) -> float:
    """Stable log-mean-exp MIL reference. Positive bags do not label every frame."""
    require(bool(instance_logits) and 0 < temperature <= 1
            and all(math.isfinite(value) for value in instance_logits), "weak_mil_input_invalid")
    maximum = max(instance_logits)
    return maximum + temperature * math.log(sum(math.exp((value - maximum) / temperature)
                                               for value in instance_logits) / len(instance_logits))


def build_weak_supervision(sequence: FeatureSequence, provided_label: dict[str, Any], *,
                           teacher_windows: list[dict[str, Any]], fact_ids: tuple[str, ...],
                           event_type_ids: tuple[str, ...], expected_label_source_sha256: str) -> WeakSupervision:
    """Call only after A02's distinct automated experiment authorization gate.

    This builder consumes a train-only organizer label and normalized A03 teacher
    windows. Labels remain loss targets outside model inputs and never become
    factual states. Empty teacher data permits binary MIL training only.
    """
    sequence.validate()
    require(sequence.split == "train" and provided_label.get("split") == "train", "weak_training_partition_forbidden")
    require(provided_label.get("content_sha256", provided_label.get("content_hash")) == sequence.content_sha256
            and provided_label.get("group_id") == sequence.group_id, "weak_provided_label_identity_mismatch")
    require(type(provided_label.get("provided_video_label")) is bool, "weak_provided_video_label_missing")
    require(HEX64.fullmatch(expected_label_source_sha256) is not None
            and provided_label.get("provided_label_source_sha256") == expected_label_source_sha256,
            "weak_provided_label_source_mismatch")
    sparse = []
    event_conflicts = 0
    for original_window in teacher_windows:
        window = deepcopy(original_window)
        require(window.get("annotation_origin") == "local_model"
                and window.get("review_status") == "machine_consistency_checked"
                and window.get("human_reviewed") is False
                and window.get("machine_consistency_passed") is True, "weak_teacher_status_invalid")
        provenance = window.get("teacher_provenance", {})
        require(isinstance(provenance.get("model_id"), str) and bool(provenance["model_id"])
                and re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("model_revision", ""))) is not None
                and provenance.get("inference_location") == "team_local_service", "weak_teacher_identity_missing")
        require(all(isinstance(provenance.get(key), str) and HEX64.fullmatch(provenance[key]) for key in
                    ("artifacts_sha256", "prompt_sha256", "source_proposal_sha256")), "weak_teacher_provenance_missing")
        if provided_label["provided_video_label"] is False:
            # A provided negative bag contradicts a positive violation proposal.
            # Mask that teacher event; preserve its factual observations and do
            # not manufacture boundary negatives or absent factual states.
            event_conflicts += len(window["target"]["candidates"])
            window["target"]["candidates"] = []
        start, end = window.get("start_sec"), window.get("end_sec")
        require(isinstance(start, (int, float)) and isinstance(end, (int, float)) and 0 <= start <= end,
                "weak_teacher_window_bounds_invalid")
        indices = tuple(index for index, frame in enumerate(sequence.frames) if start <= frame.pts_sec <= end)
        require(bool(indices), "weak_teacher_window_frames_missing")
        # Sparse teacher windows stay at their actual times; they never label
        # the entire recording's 19 facts just because the bag is positive.
        cropped = replace(sequence, frames=tuple(sequence.frames[index] for index in indices))
        target = _build_targets(cropped, window, fact_ids=fact_ids, event_type_ids=event_type_ids,
                                required_review_status="machine_consistency_checked")
        sparse.append(TeacherWindowTargets(indices, target, canonical_sha256(provenance)))
    return WeakSupervision(int(provided_label["provided_video_label"]), expected_label_source_sha256,
                           canonical_sha256(provided_label), tuple(sparse), event_conflicts)
