"""Optional next-experiment balancing; does not alter the frozen SILVER runner.

These weights balance supported exact-frame physical SILVER labels. They do
not turn teacher agreement into human truth or video-level legal supervision.
Per-fact state/content balance prevents easy handler-presence positives from
masking a lack of positive action examples. A new fit needs a new plan/release.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from .data import SILVER_PHYSICAL_FACT_IDS
from .guards import HEX64, canonical_sha256, require


def plan_balanced_silver(rows: list[dict[str, Any]], *, minimum_class_contents: int = 2) -> dict[str, Any]:
    """Assign equal mass per supported fact, state, content and unique anchor.

    Absent-only axes and states from too few independent contents get no loss.
    Duplicate windows for the same source anchor share one anchor's mass.
    Content independence is provisional; these checks establish content count,
    not scenario independence. Returned sample weights are private metadata.
    """
    require(type(minimum_class_contents) is int and minimum_class_contents >= 1,
            "silver_balance_minimum_contents_invalid")
    require(isinstance(rows, list) and bool(rows), "silver_balance_rows_empty")
    groups: dict[tuple[str, str], dict[str, dict[str, list[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen_samples, raw_counts, masked_counts = set(), Counter(), Counter()
    anchor_states: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    row_by_id = {}
    for row in rows:
        require(row.get("split") == "train" and row.get("annotation_origin") == "local_model"
                and row.get("human_reviewed") is False and row.get("target_scope") == "exact_anchor_frame",
                "silver_balance_scope_forbidden")
        sample, content, focal = row.get("sample_id"), row.get("content_sha256"), row.get("focal_frame_id")
        require(all(isinstance(value, str) and value for value in (sample, content, focal))
                and HEX64.fullmatch(content) is not None and sample not in seen_samples,
                "silver_balance_sample_identity_invalid")
        seen_samples.add(sample)
        row_by_id[sample] = row
        frames = {frame["frame_id"] for frame in row.get("frames", [])}
        require(focal in frames, "silver_balance_focal_frame_missing")
        target, masks = row.get("target", {}), row.get("fact_supervision_mask", {})
        require(target.get("candidates") == [] and isinstance(target.get("facts"), list)
                and isinstance(masks, dict) and all(type(value) is bool for value in masks.values()),
                "silver_balance_partial_targets_invalid")
        seen_facts = set()
        for fact in target["facts"]:
            fact_id, state = fact.get("fact_id"), fact.get("state")
            require(fact_id in SILVER_PHYSICAL_FACT_IDS and fact_id not in seen_facts and fact_id in masks
                    and state in {"present", "absent", "unclear"}, "silver_balance_fact_invalid")
            seen_facts.add(fact_id)
            if not masks[fact_id]:
                masked_counts[fact_id] += 1
                continue
            require(state in {"present", "absent"} and focal in fact.get("supporting_frame_ids", [])
                    and set(fact["supporting_frame_ids"]) <= frames, "silver_balance_fact_unsupported")
            raw_counts[(fact_id, state)] += 1
            anchor_states[(content, focal, fact_id)].add(state)
            groups[(fact_id, state)][content][focal].append(sample)
        require(seen_facts == set(masks), "silver_balance_mask_mismatch")
    require(all(len(states) == 1 for states in anchor_states.values()), "silver_balance_conflicting_anchor_unmasked")
    facts = sorted({fact for fact, _ in raw_counts})
    active, deferred = [], {}
    for fact in facts:
        counts = {state: len(groups[(fact, state)]) for state in ("present", "absent")}
        if min(counts.values()) >= minimum_class_contents:
            active.append(fact)
        else:
            deferred[fact] = "missing_state" if min(counts.values()) == 0 else "insufficient_state_contents"
    require(bool(active), "silver_balance_no_supported_two_class_facts")
    weights = {sample: {fact["fact_id"]: 0.0 for fact in row["target"]["facts"]}
               for sample, row in row_by_id.items()}
    mass_per_state = 1.0 / (2 * len(active))
    for fact in active:
        for state in ("present", "absent"):
            by_content = groups[(fact, state)]
            for by_anchor in by_content.values():
                for samples in by_anchor.values():
                    weight = mass_per_state / (len(by_content) * len(by_anchor) * len(samples))
                    for sample in samples:
                        weights[sample][fact] = weight
    active_samples = [sample for sample, values in weights.items() if sum(values.values()) > 0]
    fact_report = {}
    for fact in facts:
        states = {}
        for state in ("present", "absent"):
            by_content = groups[(fact, state)]
            states[state] = {"targets": raw_counts[(fact, state)], "contents": len(by_content),
                             "unique_anchors": sum(len(anchors) for anchors in by_content.values()),
                             "supervision_mass": mass_per_state if fact in active else 0.0}
        fact_report[fact] = {"states": states, "masked_targets": masked_counts[fact],
                             "eligible_for_fit": fact in active, "deferred_reason": deferred.get(fact)}
    total = sum(sum(values.values()) for values in weights.values())
    require(math.isclose(total, 1.0, abs_tol=1e-12), "silver_balance_mass_invalid")
    plan = {"schema_version": "forte.silver-balance-plan.v1", "input_rows_sha256": canonical_sha256(rows),
            "minimum_class_contents": minimum_class_contents,
            "method": "equal_fact_state_content_anchor_mass", "active_fact_ids": active,
            "deferred_fact_ids": deferred, "sample_fact_weights": weights,
            "active_sample_ids": active_samples,
            "aggregate": {"split": "train", "scope": "exact_anchor_frame", "rows": len(rows),
                "active_rows": len(active_samples), "contents": len({row["content_sha256"] for row in rows}),
                "present_targets": sum(count for (fact, state), count in raw_counts.items() if state == "present"),
                "absent_targets": sum(count for (fact, state), count in raw_counts.items() if state == "absent"),
                "present_supervision_mass": 0.5, "absent_supervision_mass": 0.5,
                "facts": fact_report, "human_verified": False, "scenario_independence_verified": False,
                "legal_targets": 0, "boundary_targets": 0, "whole_video_targets": 0}}
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def balanced_rows(rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Clone partial targets; defer unsupported axes without changing originals."""
    require(plan.get("schema_version") == "forte.silver-balance-plan.v1"
            and plan.get("input_rows_sha256") == canonical_sha256(rows), "silver_balance_rows_binding_mismatch")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    require(plan.get("plan_sha256") == canonical_sha256(payload), "silver_balance_plan_digest_mismatch")
    result = []
    for original in rows:
        weights = plan["sample_fact_weights"][original["sample_id"]]
        if not any(value > 0 for value in weights.values()):
            continue
        row = deepcopy(original)
        for fact in row["target"]["facts"]:
            enabled = weights[fact["fact_id"]] > 0
            row["fact_supervision_mask"][fact["fact_id"]] = enabled
            if not enabled:
                fact["state"], fact["supporting_frame_ids"] = "unclear", []
        row["fact_loss_weights"] = dict(weights)
        row["balance_plan_sha256"] = plan["plan_sha256"]
        result.append(row)
    return result


def fact_token_loss_weights(labels: Sequence[int], positions_by_fact: Mapping[str, Sequence[int]],
                            fact_weights: Mapping[str, float]) -> list[float]:
    """Split each fact's mass across its supervised tokens, ignoring text length.

    Positions come from the real collator's verified per-fact character/token
    alignment. Prompt/padding/unknown/candidate/visual tokens must stay -100.
    Token positions cannot overlap and missing positive-weight spans fail closed.
    """
    require(all(type(value) is int for value in labels), "silver_balance_labels_invalid")
    require(set(positions_by_fact) <= set(fact_weights), "silver_balance_fact_positions_unexpected")
    result, seen = [0.0] * len(labels), set()
    for fact, weight in fact_weights.items():
        require(type(weight) in {int, float} and math.isfinite(weight) and weight >= 0,
                "silver_balance_weight_invalid")
        positions = positions_by_fact.get(fact, [])
        if weight == 0:
            require(not positions, "silver_balance_disabled_fact_has_tokens")
            continue
        require(bool(positions) and len(positions) == len(set(positions)), "silver_balance_fact_tokens_missing_or_duplicate")
        for position in positions:
            require(type(position) is int and 0 < position < len(labels) and labels[position] != -100
                    and position not in seen, "silver_balance_token_position_invalid")
            seen.add(position)
            result[position] = float(weight) / len(positions)
    require(bool(seen) and seen == {index for index, value in enumerate(labels) if value != -100},
            "silver_balance_supervised_tokens_unaccounted")
    return result


def weighted_causal_fact_loss(logits: Any, labels: Any, weights: Any, *, example_count: int) -> Any:
    """Use global mass without cancelling it by normalizing each row separately.

    Mean accumulation over a uniformly shuffled single epoch of example_count
    rows gives the declared fact/state/content objective. Future executors must
    pass labels=None to model.forward and use this differentiable loss instead.
    This does not itself allocate a model, issue inference, or run an optimizer.
    """
    import torch
    import torch.nn.functional as functional
    require(type(example_count) is int and example_count > 0, "silver_balance_example_count_invalid")
    require(logits.ndim == 3 and labels.ndim == 2 and weights.shape == labels.shape
            and logits.shape[:2] == labels.shape, "silver_balance_loss_shape_invalid")
    require(bool(torch.isfinite(weights).all().item()) and bool((weights >= 0).all().item())
            and not bool((weights[:, 0] != 0).any().item()), "silver_balance_loss_weight_invalid")
    require(not bool(((labels == -100) & (weights != 0)).any().item()), "silver_balance_mask_has_weight")
    active = weights[:, 1:] > 0
    require(bool(active.any().item()), "silver_balance_loss_all_masked")
    # Gather supervised tokens before FP32 cross entropy to avoid materializing
    # an additional full sequence by 248k-vocabulary FP32 logits tensor.
    losses = functional.cross_entropy(logits[:, :-1, :][active].float(), labels[:, 1:][active], reduction="none")
    result = (losses * weights[:, 1:][active].float()).sum() * example_count
    require(bool(torch.isfinite(result).item()), "silver_balance_loss_nonfinite")
    return result
