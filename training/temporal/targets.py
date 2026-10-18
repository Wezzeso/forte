"""Architecture-independent reviewed supervision. Unmarked pairs remain masked."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from training.guards import HEX64, canonical_sha256, require, validate_target
from forte.decision.head_contract import FULL_FACT_COUNT, STATE_ORDER

IGNORE = -100
STATE_IDS = {state: index for index, state in enumerate(STATE_ORDER)}


def canonical_source_frame_id(content_sha256: str, source_pts: int, numerator: int, denominator: int) -> str:
    """Use A05's pure source identity; never fabricate VideoAsset/JPEG references."""
    from forte.video.batches import source_frame_id
    try:
        return source_frame_id(content_sha256, source_pts, numerator, denominator)
    except ValueError:
        require(False, "feature_source_identity_invalid")
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class FeatureFrame:
    frame_id: str
    source_pts: int
    pts_sec: float
    next_pts_sec: float | None
    view_mask: tuple[bool, ...]
    view_roles: tuple[str, ...]
    normalized_boxes: tuple[tuple[float, float, float, float], ...]
    next_source_pts: int | None = None
    source_end_bound_sec: float | None = None

    @classmethod
    def from_source_identity(cls, identity: Any, *, view_mask: tuple[bool, ...], view_roles: tuple[str, ...],
                             normalized_boxes: tuple[tuple[float, float, float, float], ...]) -> FeatureFrame:
        """Consume A05's metadata directly without fake JPEG references."""
        return cls(identity.frame_id, identity.source_pts, identity.pts_sec, identity.next_pts_sec,
                   view_mask, view_roles, normalized_boxes, identity.next_source_pts, identity.source_end_bound_sec)


@dataclass(frozen=True)
class FeatureSequence:
    """Provenance stays outside head tensors. No source name/binary label input."""
    content_sha256: str
    group_id: str
    split: str
    time_base_num: int
    time_base_den: int
    origin_pts: int
    embedding_dim: int
    encoder_manifest_sha256: str
    decoder_config_sha256: str
    sampling_config_sha256: str
    crop_config_sha256: str
    frames: tuple[FeatureFrame, ...]

    def validate(self) -> None:
        require(self.split in {"train", "validation", "final_check"}, "feature_split_invalid")
        require(bool(self.group_id) and 1 <= self.embedding_dim <= 8192, "feature_metadata_invalid")
        require(all(HEX64.fullmatch(value) for value in (self.content_sha256, self.encoder_manifest_sha256,
                    self.decoder_config_sha256, self.sampling_config_sha256, self.crop_config_sha256)),
                "feature_provenance_hash_invalid")
        require(type(self.time_base_num) is int and type(self.time_base_den) is int
                and type(self.origin_pts) is int and self.time_base_num > 0 and self.time_base_den > 0
                and bool(self.frames), "feature_timeline_invalid")
        require(len(self.frames) <= 10000, "feature_frame_budget_exceeded")
        require(len({frame.frame_id for frame in self.frames}) == len(self.frames), "feature_frame_id_duplicate")
        views = len(self.frames[0].view_mask)
        previous = -1.0
        for frame in self.frames:
            require(type(frame.source_pts) is int, "feature_integer_source_pts_required")
            require(frame.frame_id == canonical_source_frame_id(self.content_sha256, frame.source_pts,
                                                               self.time_base_num, self.time_base_den),
                    "feature_source_frame_id_mismatch")
            expected = (frame.source_pts - self.origin_pts) * self.time_base_num / self.time_base_den
            require(math.isfinite(frame.pts_sec) and frame.pts_sec >= 0 and frame.pts_sec > previous
                    and abs(expected - frame.pts_sec) <= 1e-6, "feature_source_pts_mismatch")
            if frame.next_pts_sec is not None:
                require(type(frame.next_source_pts) is int and frame.next_source_pts > frame.source_pts,
                        "feature_native_next_pts_missing")
                native_next = (frame.next_source_pts - self.origin_pts) * self.time_base_num / self.time_base_den
                require(math.isfinite(frame.next_pts_sec) and frame.next_pts_sec > frame.pts_sec
                        and abs(native_next - frame.next_pts_sec) <= 1e-6 and frame.source_end_bound_sec is None,
                        "feature_native_frame_interval_invalid")
            else:
                require(frame.next_source_pts is None and frame.source_end_bound_sec is not None
                        and math.isfinite(frame.source_end_bound_sec) and frame.source_end_bound_sec > frame.pts_sec,
                        "feature_eof_bound_missing")
            require(0 < views <= 8 and len(frame.view_mask) == views == len(frame.view_roles)
                    == len(frame.normalized_boxes), "feature_view_shape_invalid")
            require(any(valid and role == "global" for valid, role in zip(frame.view_mask, frame.view_roles)),
                    "feature_global_view_missing")
            for valid, role, box in zip(frame.view_mask, frame.view_roles, frame.normalized_boxes):
                require(type(valid) is bool and role in {"global", "roi"}, "feature_view_metadata_invalid")
                require(len(box) == 4 and all(math.isfinite(value) and 0 <= value <= 1 for value in box)
                        and box[0] < box[2] and box[1] < box[3], "feature_box_invalid")
                require(not valid or role != "global" or box == (0, 0, 1, 1), "feature_global_view_cropped")
            previous = frame.pts_sec


@dataclass(frozen=True)
class SparseTargets:
    """Lists are private training tensors; report only aggregate coverage counts."""
    fact_ids: tuple[str, ...]
    event_type_ids: tuple[str, ...]
    window_states: list[int]
    frame_states: list[list[int]]
    frame_support: list[list[int]]
    event_boundaries: list[list[list[int]]]
    event_brackets: tuple[dict[str, Any], ...]
    target_sha256: str

    def coverage(self) -> dict[str, int]:
        return {"window_fact_labels": sum(value != IGNORE for value in self.window_states),
                "window_unclear_labels": self.window_states.count(STATE_IDS["unclear"]),
                "frame_fact_labels": sum(value != IGNORE for row in self.frame_states for value in row),
                "support_positive_labels": sum(value == 1 for row in self.frame_support for value in row),
                "support_negative_labels": sum(value == 0 for row in self.frame_support for value in row),
                "boundary_positive_labels": sum(value == 1 for frame in self.event_boundaries for event in frame for value in event),
                "boundary_negative_labels": sum(value == 0 for frame in self.event_boundaries for event in frame for value in event),
                "reviewed_event_brackets": len(self.event_brackets)}


def _build_targets(sequence: FeatureSequence, window: dict[str, Any], *, fact_ids: tuple[str, ...],
                   event_type_ids: tuple[str, ...], required_review_status: str) -> SparseTargets:
    """No weak video labels, pseudo facts, interval filling, or absent-by-default."""
    sequence.validate()
    require(sequence.split == "train" and window.get("split") == "train", "temporal_training_partition_forbidden")
    require(window.get("content_sha256") == sequence.content_sha256 and window.get("group_id") == sequence.group_id,
            "temporal_training_identity_mismatch")
    require(window.get("review_status") == required_review_status, "temporal_target_origin_mismatch")
    if required_review_status == "human_reviewed":
        require(window.get("reviewer_id") and window.get("reviewed_at"), "temporal_target_not_human_reviewed")
    start, end = window.get("start_sec"), window.get("end_sec")
    require(isinstance(start, (int, float)) and isinstance(end, (int, float)) and 0 <= start <= end
            and all(start <= frame.pts_sec <= end for frame in sequence.frames), "temporal_window_bounds_invalid")
    require(len(fact_ids) == FULL_FACT_COUNT and len(set(fact_ids)) == FULL_FACT_COUNT, "temporal_full_19_facts_required")
    require(len(set(event_type_ids)) == len(event_type_ids), "temporal_event_ids_duplicate")
    frame_index = {frame.frame_id: index for index, frame in enumerate(sequence.frames)}
    fact_index = {fact_id: index for index, fact_id in enumerate(fact_ids)}
    event_index = {event_id: index for index, event_id in enumerate(event_type_ids)}
    validation_target = window.get("target")
    if required_review_status == "machine_consistency_checked":
        require(isinstance(validation_target, dict) and isinstance(validation_target.get("facts"), list),
                "temporal_machine_target_invalid")
        actual_ids = [fact.get("fact_id") for fact in validation_target["facts"]]
        require(len(set(actual_ids)) == len(actual_ids) and set(actual_ids) <= set(fact_ids),
                "temporal_machine_fact_axis_invalid")
        require(all(set(event.get("supporting_fact_ids", [])) <= set(actual_ids)
                    for event in validation_target.get("candidates", [])), "temporal_machine_candidate_fact_missing")
        # Fill only a validator copy, never training labels. The assembler below
        # iterates original sparse facts, so missing pairs remain IGNORE.
        validation_target = deepcopy(validation_target)
        validation_target["facts"].extend({"fact_id": fact_id, "state": "unclear", "supporting_frame_ids": []}
                                           for fact_id in fact_ids if fact_id not in actual_ids)
    validate_target(validation_target, set(frame_index), list(fact_ids), list(event_type_ids))
    count, facts, events = len(sequence.frames), len(fact_ids), len(event_type_ids)
    window_states = [IGNORE] * facts
    frame_states = [[IGNORE] * facts for _ in range(count)]
    frame_support = [[IGNORE] * facts for _ in range(count)]
    boundaries = [[[IGNORE, IGNORE] for _ in range(events)] for _ in range(count)]

    def assign(matrix, frame_id, fact_id, value):
        require(frame_id in frame_index and fact_id in fact_index, "temporal_annotation_reference_invalid")
        target = matrix[frame_index[frame_id]]
        index = fact_index[fact_id]
        require(target[index] in {IGNORE, value}, "temporal_annotation_conflict")
        target[index] = value

    context = window.get("fact_context", {})
    def focal_question_bound(item):
        definition, provenance = item.get("question_definition"), item.get("provenance", {})
        expected_id = "ANCHOR__" + str(item.get("fact_id", ""))
        return (isinstance(definition, dict) and definition.get("fact_id") == expected_id
                and isinstance(definition.get("description"), str) and bool(definition["description"])
                and item.get("question_fact_id") == expected_id
                and provenance.get("focal_question_sha256") == canonical_sha256(definition)
                and provenance.get("prompt_sha256") == window.get("teacher_provenance", {}).get("prompt_sha256")
                and isinstance(provenance.get("prompt_sha256"), str)
                and HEX64.fullmatch(provenance["prompt_sha256"]) is not None)
    for decision in window["target"]["facts"]:
        fact_id, state = decision["fact_id"], decision["state"]
        index = fact_index[fact_id]
        if state == "absent":
            item = context.get(fact_id, {})
            # Coverage is not visibility. Unsupported absence stays unlabelled.
            if not (item.get("review_status") == required_review_status and item.get("context_complete") is True
                    and item.get("visibility_adequate") is True):
                continue
        window_states[index] = STATE_IDS[state]
        for frame_id in decision["supporting_frame_ids"]:
            assign(frame_support, frame_id, fact_id, 1)
    for item in window.get("frame_fact_targets", []):
        require(item.get("review_status") == required_review_status and item.get("state") in STATE_IDS,
                "temporal_frame_state_unreviewed")
        if required_review_status == "machine_consistency_checked":
            physical = {"CASH_VISIBLE", "CASH_ON_WORK_SURFACE", "CASH_HANDLER_PRESENT",
                        "CASH_HANDLER_LEAVES_WORKSTATION", "CASH_HANDOVER_VISIBLE", "CASH_COUNT_VISIBLE",
                        "CASH_PLACED_IN_CLOTHING_POCKET", "CASH_PLACED_IN_PERSONAL_CONTAINER"}
            require(item.get("fact_id") in physical and item.get("scope") == "exact_anchor_frame"
                    and item.get("derivation") == "direct_anchor_teacher_question"
                    and item.get("localizable_category") == "physical_anchor_atom"
                    and item.get("human_verified") is False and item.get("supervision_mask") is True
                    and (item.get("state") == "unclear" or item.get("frame_id") in set(item.get("supporting_frame_ids", [])))
                    and set(item.get("supporting_frame_ids", [])) <= set(frame_index)
                    and set(item.get("context_frame_ids", [])) <= set(frame_index)
                    and focal_question_bound(item),
                    "temporal_silver_frame_state_not_directly_localized")
        assign(frame_states, item["frame_id"], item["fact_id"], STATE_IDS[item["state"]])
    for item in window.get("support_reviews", []):
        require(item.get("review_status") == required_review_status and type(item.get("supports_state")) is bool,
                "temporal_support_target_unreviewed")
        if required_review_status == "machine_consistency_checked" and item["supports_state"] is False:
            reference = item.get("negative_reference_kind", item.get("reference_kind"))
            require(reference == "focal_frame_explicit_unclear" and item.get("teacher_state") == "unclear"
                    and item.get("focal_frame_id") == item.get("frame_id")
                    and item.get("supervision_mask") is True and item.get("human_verified") is False
                    and focal_question_bound(item), "temporal_silver_support_negative_not_explicit")
        assign(frame_support, item["frame_id"], item["fact_id"], int(item["supports_state"]))
    brackets = []
    for candidate in window["target"]["candidates"]:
        event = event_index[candidate["event_type_id"]]
        start, end = frame_index[candidate["start_frame_id"]], frame_index[candidate["end_frame_id"]]
        require(start <= end, "temporal_event_boundary_reversed")
        boundaries[start][event][0] = boundaries[end][event][1] = 1
        brackets.append({key: candidate[key] for key in ("candidate_id", "event_type_id", "start_frame_id",
                                                        "end_frame_id", "supporting_fact_ids", "supporting_frame_ids")})
    for item in window.get("boundary_reviews", []):
        require(item.get("review_status") == required_review_status and type(item.get("is_boundary")) is bool,
                "temporal_boundary_target_unreviewed")
        require(item.get("frame_id") in frame_index and item.get("event_type_id") in event_index
                and item.get("boundary") in {"start", "end"}, "temporal_boundary_reference_invalid")
        target = boundaries[frame_index[item["frame_id"]]][event_index[item["event_type_id"]]]
        index = 0 if item["boundary"] == "start" else 1
        value = int(item["is_boundary"])
        require(target[index] in {IGNORE, value}, "temporal_annotation_conflict")
        target[index] = value
    digest = canonical_sha256({"window_states": window_states, "frame_states": frame_states,
                               "frame_support": frame_support, "event_boundaries": boundaries,
                               "event_brackets": brackets, "fact_ids": fact_ids, "event_type_ids": event_type_ids})
    return SparseTargets(fact_ids, event_type_ids, window_states, frame_states, frame_support,
                         boundaries, tuple(brackets), digest)


def build_targets(sequence: FeatureSequence, window: dict[str, Any], *, fact_ids: tuple[str, ...],
                  event_type_ids: tuple[str, ...]) -> SparseTargets:
    """Human-verified E1 entry point remains strict; teacher records are rejected."""
    require(window.get("review_status") == "human_reviewed", "temporal_target_not_human_reviewed")
    return _build_targets(sequence, window, fact_ids=fact_ids, event_type_ids=event_type_ids,
                          required_review_status="human_reviewed")
