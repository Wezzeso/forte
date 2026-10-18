"""Provided approximate points supervise existential eventness, not boundaries."""
from __future__ import annotations

import math
from pathlib import Path

from training.guards import file_sha256, load_annotations, require


def load_coarse_anchors(config: dict, *, split: str, labels: list[dict]) -> dict[str, tuple[float, ...]]:
    key = "provided_" + split + "_anchors"
    if not config.get(key):
        return {}
    path = Path(config[key]).resolve()
    require(file_sha256(path) == config.get("expected_provided_" + split + "_anchors_sha256"),
            "weak_coarse_anchor_role_pin_mismatch")
    expected = {row.get("content_sha256", row.get("content_hash")): row for row in labels}
    rows = load_annotations(path)
    require(len(rows) == len(expected), "weak_coarse_anchor_population_mismatch")
    result = {}
    for row in rows:
        content = row.get("content_sha256", row.get("content_hash"))
        require(content in expected and content not in result and row.get("split") == split
                and row.get("group_id") == expected[content]["group_id"], "weak_coarse_anchor_partition_mismatch")
        require(row.get("anchor_origin") == "user_provided_coarse_table"
                and row.get("anchor_source_sha256") == config.get("expected_coarse_anchor_source_sha256")
                and row.get("provided_label_source_sha256") == config["expected_provided_label_source_sha256"]
                and row.get("assignments_sha256") == config["experiment_authorization"]["expected_assignments_sha256"]
                and row.get("target_scope") == "class_agnostic_approximate_event_anchor"
                and row.get("timing_precision") == "coarse_seconds"
                and row.get("boundaries_known") is False and row.get("event_type_known") is False,
                "weak_coarse_anchor_provenance_invalid")
        points = row.get("coarse_anchor_sec")
        require(isinstance(points, list) and all(type(point) in {int, float} and math.isfinite(point) and point >= 0 for point in points)
                and row.get("provided_video_label") is expected[content]["provided_video_label"]
                and (expected[content]["provided_video_label"] is True or not points), "weak_coarse_anchor_targets_invalid")
        result[content] = tuple(float(point) for point in points)
    return result


def anchor_frame_indices(sequence, point_sec: float, tolerance_sec: float) -> tuple[int, ...]:
    require(type(tolerance_sec) in {float, int} and math.isfinite(tolerance_sec) and 0 < tolerance_sec <= 5,
            "weak_coarse_anchor_tolerance_invalid")
    source_end = sequence.frames[-1].source_end_bound_sec
    require(source_end is not None and 0 <= point_sec <= source_end, "weak_coarse_anchor_outside_source")
    indices = tuple(index for index, frame in enumerate(sequence.frames)
                    if abs(frame.pts_sec - point_sec) <= tolerance_sec)
    require(bool(indices), "weak_coarse_anchor_has_no_observed_frames")
    return indices


def coarse_positive_anchor_loss(outputs, anchor_indices: tuple[tuple[int, ...], ...], *, temperature: float):
    """Only observed frames in each declared tolerance band form a positive bag."""
    import torch
    from torch.nn import functional as F
    logits = outputs["video_instance_logits"]
    require(logits.shape[0] == 1 and 0 < temperature <= 1, "weak_coarse_anchor_batch_invalid")
    if not anchor_indices:
        return logits.sum() * 0
    scores = []
    for indices in anchor_indices:
        values = logits[0, list(indices)]
        scores.append(temperature * (torch.logsumexp(values / temperature, dim=0) - math.log(len(indices))))
    scores = torch.stack(scores)
    return F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
