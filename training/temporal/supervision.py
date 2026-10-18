"""Actual target coverage and whole-branch freezing, independent of labels at inference."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace

from training.guards import canonical_sha256, require
from forte.decision.head_contract import STATE_ORDER


def component_weights(config: dict) -> dict:
    weights = config.get("component_loss_weights", {"window": 1.0, "frame": 1.0, "support": 1.0, "boundary": 1.0})
    require(isinstance(weights, dict) and set(weights) == {"window", "frame", "support", "boundary"}
            and all(type(value) in {int, float} and 0 <= value <= 4 for value in weights.values()),
            "weak_component_loss_weights_invalid")
    return dict(weights)


def freeze_unsupervised_branches(model, coverage: dict, *, weights: dict, teacher_weight: float) -> dict:
    enabled = {"window_states": teacher_weight * weights["window"] > 0 and coverage.get("silver_window_fact_labels", 0) > 0,
               "frame_states": teacher_weight * weights["frame"] > 0 and coverage.get("silver_frame_fact_labels", 0) > 0,
               "support": teacher_weight * weights["support"] > 0 and
                    coverage.get("silver_support_positive_labels", 0) + coverage.get("silver_support_negative_labels", 0) > 0,
               "boundary": teacher_weight * weights["boundary"] > 0 and
                    coverage.get("silver_boundary_positive_labels", 0) + coverage.get("silver_boundary_negative_labels", 0) > 0}
    result = {}
    for name, active in enabled.items():
        branch = getattr(model, name)
        if branch is not None:
            branch.requires_grad_(active)
            for parameter in branch.parameters():
                parameter.grad = None
            result[name] = {"trainable": active, "reason": "supervised_enabled_loss" if active else "no_targets_or_disabled_loss"}
    return result


def class_coverage(data, *, fact_ids: tuple, event_ids: tuple) -> dict:
    facts = {fact: {"window": Counter({state: 0 for state in STATE_ORDER}),
                    "frame": Counter({state: 0 for state in STATE_ORDER}),
                    "support_positive": 0, "support_negative": 0} for fact in fact_ids}
    events = {event: {"start_positive": 0, "end_positive": 0, "negative": 0} for event in event_ids}
    unique_frames, unique_support, conflicts = {}, {}, 0
    for sequence, _, _, _, windows, *_ in data:
        for window in windows:
            target = window.targets
            for index, fact in enumerate(fact_ids):
                state = target.window_states[index]
                if state != -100:
                    facts[fact]["window"][STATE_ORDER[state]] += 1
            for local, native in enumerate(window.sequence_frame_indices):
                frame_id = sequence.frames[native].frame_id
                for index, fact in enumerate(fact_ids):
                    state, support = target.frame_states[local][index], target.frame_support[local][index]
                    key = (sequence.content_sha256, frame_id, fact)
                    if state != -100:
                        facts[fact]["frame"][STATE_ORDER[state]] += 1
                        if key in unique_frames and unique_frames[key] != state:
                            conflicts += 1
                        unique_frames[key] = state
                    if support != -100:
                        facts[fact]["support_positive" if support else "support_negative"] += 1
                        if key in unique_support and unique_support[key] != support:
                            conflicts += 1
                        unique_support[key] = support
                for index, event in enumerate(event_ids):
                    start, end = target.event_boundaries[local][index]
                    events[event]["start_positive"] += start == 1
                    events[event]["end_positive"] += end == 1
                    events[event]["negative"] += (start == 0) + (end == 0)
    return {"per_fact": facts, "per_event": events, "unique_direct_frame_fact_pairs": len(unique_frames),
            "unique_frame_support_pairs": len(unique_support), "cross_window_conflicting_pair_observations": conflicts,
            "counts_are_target_observations_not_independent_human_ground_truth": True}


def mask_cross_window_conflicts(data):
    """Mask contradictory exact source pairs; keep every raw teacher receipt."""
    states, supports = {}, {}
    for sequence, _, _, _, windows, *_ in data:
        for window in windows:
            for local, native in enumerate(window.sequence_frame_indices):
                for index, fact in enumerate(window.targets.fact_ids):
                    key = (sequence.content_sha256, sequence.frames[native].frame_id, fact)
                    for rows, mapping in ((window.targets.frame_states, states), (window.targets.frame_support, supports)):
                        value = rows[local][index]
                        if value != -100:
                            mapping.setdefault(key, set()).add(value)
    state_conflicts = {key for key, values in states.items() if len(values) > 1}
    support_conflicts = {key for key, values in supports.items() if len(values) > 1}
    output, masked, dropped = [], 0, 0
    for record in data:
        sequence, features, manifest, label, windows, *tail = record
        copied = []
        for window in windows:
            target = deepcopy(window.targets)
            for local, native in enumerate(window.sequence_frame_indices):
                for index, fact in enumerate(target.fact_ids):
                    key = (sequence.content_sha256, sequence.frames[native].frame_id, fact)
                    for rows, conflicts in ((target.frame_states, state_conflicts), (target.frame_support, support_conflicts)):
                        if key in conflicts and rows[local][index] != -100:
                            rows[local][index] = -100
                            masked += 1
            target = replace(target, target_sha256=canonical_sha256({"original_target_sha256": target.target_sha256,
                "mask_policy": "exact_source_pair_disagreement_to_unknown.v1",
                "frame_states": target.frame_states, "frame_support": target.frame_support}))
            if any(target.coverage().values()):
                copied.append(replace(window, targets=target))
            else:
                dropped += 1
        output.append((sequence, features, manifest, label, copied, *tail))
    return output, {"policy": "exact_source_pair_disagreement_to_unknown.v1",
        "conflicting_frame_state_pairs": len(state_conflicts), "conflicting_support_pairs": len(support_conflicts),
        "masked_target_observations": masked, "fully_masked_teacher_windows": dropped,
        "original_teacher_receipts_modified": False}
