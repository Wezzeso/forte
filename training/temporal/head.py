"""Small random-initialized temporal head over frozen features; no encoder loader.

Import only inside an explicitly prepared ML environment. It does not allocate a
GPU, download weights, decode media, select a checkpoint or emit legal verdicts.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from forte.decision.head_contract import FULL_FACT_COUNT, STATE_ORDER


class FrozenFeatureFactHead(nn.Module):
    def __init__(self, embedding_dim: int, *, hidden_dim: int = 128, fact_count: int = FULL_FACT_COUNT, event_count: int = 0):
        super().__init__()
        if fact_count != FULL_FACT_COUNT or len(STATE_ORDER) != 3 or min(embedding_dim, hidden_dim) < 1 or event_count < 0:
            raise ValueError("invalid_temporal_head_shape")
        self.fact_count, self.event_count = fact_count, event_count
        self.project = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.view_role = nn.Embedding(2, hidden_dim)
        self.view_box = nn.Linear(4, hidden_dim)
        self.time = nn.Linear(2, hidden_dim)
        self.temporal = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
                                      nn.GELU(), nn.Conv1d(hidden_dim, hidden_dim, 1), nn.GELU(),
                                      nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=2, padding=2, groups=hidden_dim), nn.GELU())
        self.window_states = nn.Linear(hidden_dim, fact_count * 3)
        self.frame_states = nn.Linear(hidden_dim, fact_count * 3)
        self.support = nn.Linear(hidden_dim, fact_count)
        self.boundary = nn.Linear(hidden_dim, event_count * 2) if event_count else None

    def _encode(self, features, *, view_mask, frame_mask, pts_sec, view_roles, normalized_boxes):
        # Only features/timestamps/masks enter the head. Provenance and labels do not.
        if features.ndim != 4 or view_mask.shape != features.shape[:3] or frame_mask.shape != features.shape[:2]:
            raise ValueError("invalid_temporal_feature_shape")
        if pts_sec.shape != frame_mask.shape or not bool(frame_mask.any(dim=1).all()):
            raise ValueError("invalid_temporal_time_mask")
        # Collation is explicitly right-padded; the first frame is always real.
        lengths = frame_mask.sum(dim=1)
        expected_mask = torch.arange(frame_mask.shape[1], device=frame_mask.device)[None, :] < lengths[:, None]
        if not bool((frame_mask == expected_mask).all()):
            raise ValueError("temporal_padding_must_be_right_contiguous")
        valid_pts = pts_sec[frame_mask]
        if not bool(torch.isfinite(valid_pts).all()) or not bool((valid_pts >= 0).all()):
            raise ValueError("invalid_temporal_valid_timestamp")
        # Sanitize padding BEFORE the time projection; NaN*0 would stay NaN.
        pts_sec = pts_sec.masked_fill(~frame_mask, 0)
        adjacent_valid = frame_mask[:, 1:] & frame_mask[:, :-1]
        if not bool(((pts_sec[:, 1:] - pts_sec[:, :-1])[adjacent_valid] > 0).all()):
            raise ValueError("nonchronological_temporal_timestamps")
        if view_roles.shape != view_mask.shape or normalized_boxes.shape != (*view_mask.shape, 4):
            raise ValueError("invalid_temporal_view_geometry")
        if not bool(((view_roles == 0) | (view_roles == 1)).all()):
            raise ValueError("invalid_temporal_view_role")
        if not bool((view_mask.any(dim=2) | ~frame_mask).all()):
            raise ValueError("valid_temporal_frame_has_no_view")
        if not bool(((view_mask & (view_roles == 0)).any(dim=2) | ~frame_mask).all()):
            raise ValueError("valid_temporal_frame_has_no_global_view")
        effective_views = view_mask & frame_mask.unsqueeze(-1)
        features = features.detach().to(self.project[0].weight.dtype).masked_fill(~effective_views.unsqueeze(-1), 0)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("nonfinite_temporal_features")
        boxes = normalized_boxes.masked_fill(~effective_views.unsqueeze(-1), 0)
        if not bool(torch.isfinite(boxes).all()) or not bool(((boxes >= 0) & (boxes <= 1)).all()):
            raise ValueError("invalid_temporal_view_boxes")
        values = self.project(features) + self.view_role(view_roles) + self.view_box(boxes.to(features.dtype))
        views = effective_views.to(values.dtype).unsqueeze(-1)
        values = (values * views).sum(dim=2) / views.sum(dim=2).clamp_min(1)
        origin = pts_sec[:, :1]
        delta = torch.cat([torch.zeros_like(origin), pts_sec[:, 1:] - pts_sec[:, :-1]], dim=1).clamp_min(0)
        time = torch.stack([(pts_sec - origin).clamp_min(0) / 143.0, delta.clamp_max(143.0) / 143.0], dim=-1)
        mask = frame_mask.to(values.dtype).unsqueeze(-1)
        values = (values + self.time(time.to(values.dtype))) * mask
        temporal = values.transpose(1, 2)
        for layer in self.temporal:
            temporal = layer(temporal) * mask.transpose(1, 2)
        values = (values + temporal.transpose(1, 2)) * mask
        pooled = values.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return values, pooled

    def _outputs(self, values, pooled):
        batch, frames = values.shape[:2]
        # Names/axis order match A08 TemporalHeadOutput. Identity/source metadata
        # are bound by its adapter rather than fabricated by this tensor module.
        return {"window_logits": self.window_states(pooled).reshape(batch, self.fact_count, 3),
                "frame_logits": self.frame_states(values).reshape(batch, frames, self.fact_count, 3),
                "support_logits": self.support(values),
                "boundary_logits": (self.boundary(values).reshape(batch, frames, self.event_count, 2)
                                    if self.boundary is not None else None)}

    def forward(self, features, **kwargs):
        return self._outputs(*self._encode(features, **kwargs))


class FrozenFeatureWeakMILHead(FrozenFeatureFactHead):
    """Separate candidate track: organizer labels train only binary eventness."""

    def __init__(self, embedding_dim: int, *, hidden_dim: int = 128, event_count: int = 0,
                 mil_temperature: float = .25):
        super().__init__(embedding_dim, hidden_dim=hidden_dim, event_count=event_count)
        if not 0 < mil_temperature <= 1:
            raise ValueError("invalid_weak_mil_temperature")
        self.mil_temperature = mil_temperature
        self.video_instances = nn.Linear(hidden_dim, 1)

    def forward(self, features, **kwargs):
        values, pooled = self._encode(features, **kwargs)
        outputs = self._outputs(values, pooled)
        logits = self.video_instances(values).squeeze(-1)
        mask = kwargs["frame_mask"]
        # Log-mean-exp approximates an existential bag score without forcing all
        # instances in a positive recording to be positive or leaking bag labels.
        count = mask.sum(dim=1).to(logits.dtype)
        bag = self.mil_temperature * (torch.logsumexp(
            (logits / self.mil_temperature).masked_fill(~mask, float("-inf")), dim=1) - count.log())
        outputs.update(video_instance_logits=logits, video_logit=bag)
        return outputs


def weak_video_mil_loss(outputs, provided_video_labels, frame_mask, *, negative_instance_weight: float = 1.0):
    """Provided negative bags can supervise eventness, never nineteen fact states."""
    labels = provided_video_labels.to(outputs["video_logit"].dtype)
    if labels.shape != outputs["video_logit"].shape or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("invalid_provided_video_label_tensor")
    bag_loss = F.binary_cross_entropy_with_logits(outputs["video_logit"], labels)
    negative = frame_mask & (labels == 0).unsqueeze(-1)
    negative_loss = (F.binary_cross_entropy_with_logits(outputs["video_instance_logits"][negative],
                                                      torch.zeros_like(outputs["video_instance_logits"][negative]))
                     if bool(negative.any()) else outputs["video_instance_logits"].sum() * 0)
    return bag_loss + negative_instance_weight * negative_loss, {"provided_binary_bag": bag_loss,
                                                               "provided_negative_eventness": negative_loss}


def sparse_loss(outputs, targets, *, component_weights=None):
    """Ignore every unreviewed pair. Explicit unclear has class ID2, not a negative."""
    def ce(logits, labels):
        known = labels != -100
        return F.cross_entropy(logits[known], labels[known]) if bool(known.any()) else logits.sum() * 0

    def bce(logits, labels):
        known = labels != -100
        return F.binary_cross_entropy_with_logits(logits[known], labels[known].to(logits.dtype)) if bool(known.any()) else logits.sum() * 0

    weights = {"window": 1.0, "frame": 1.0, "support": 1.0, "boundary": 1.0}
    if component_weights is not None:
        if set(component_weights) != set(weights) or any(type(value) not in {int, float}
                or not 0 <= value <= 4 for value in component_weights.values()):
            raise ValueError("invalid_sparse_component_weights")
        weights.update(component_weights)
    # Disabled branches are not evaluated into the loss graph. In particular,
    # boundary logits receive no optimization from coarse point targets.
    components = {}
    for name, output_name, target_name, loss_fn in (
            ("window", "window_logits", "window_states", ce),
            ("frame", "frame_logits", "frame_states", ce),
            ("support", "support_logits", "frame_support", bce)):
        if weights[name]:
            components[name] = weights[name] * loss_fn(outputs[output_name], targets[target_name])
    if outputs["boundary_logits"] is not None and weights["boundary"]:
        components["boundary"] = weights["boundary"] * bce(outputs["boundary_logits"], targets["event_boundaries"])
    return sum(components.values(), outputs["video_logit"].sum() * 0 if "video_logit" in outputs
               else outputs["window_logits"].sum() * 0), components
