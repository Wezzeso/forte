"""Synthetic next-fit balancing checks; no footage, model calls or GPU."""
from copy import deepcopy
import math

import pytest

from training.guards import TrainingGateError, canonical_sha256
from training.silver_balance import (balanced_rows, fact_token_loss_weights,
                                     plan_balanced_silver, weighted_causal_fact_loss)


def rows_fixture():
    result = []
    for index in range(4):
        focal = "focal-" + str(index)
        states = {"CASH_VISIBLE": "present" if index < 2 else "absent",
                  "CASH_HANDLER_PRESENT": "present" if index % 2 == 0 else "absent",
                  "CASH_PLACED_IN_CLOTHING_POCKET": "absent",
                  "CASH_COUNT_VISIBLE": "present" if index == 0 else "absent"}
        result.append({"sample_id": "sample-" + str(index), "content_sha256": str(index + 1) * 64,
            "focal_frame_id": focal, "frames": [{"frame_id": focal}], "split": "train",
            "annotation_origin": "local_model", "human_reviewed": False, "target_scope": "exact_anchor_frame",
            "fact_supervision_mask": dict.fromkeys(states, True),
            "target": {"facts": [{"fact_id": fact, "state": state, "supporting_frame_ids": [focal]}
                                  for fact, state in states.items()], "candidates": []}})
    return result


def test_each_eligible_fact_state_gets_equal_mass_and_sparse_axes_are_deferred():
    plan = plan_balanced_silver(rows_fixture())
    assert plan["active_fact_ids"] == ["CASH_HANDLER_PRESENT", "CASH_VISIBLE"]
    assert plan["deferred_fact_ids"] == {"CASH_COUNT_VISIBLE": "insufficient_state_contents",
                                          "CASH_PLACED_IN_CLOTHING_POCKET": "missing_state"}
    assert math.isclose(sum(sum(weights.values()) for weights in plan["sample_fact_weights"].values()), 1)
    for fact in plan["active_fact_ids"]:
        for state in ("present", "absent"):
            assert plan["aggregate"]["facts"][fact]["states"][state]["supervision_mass"] == .25
    assert plan["aggregate"]["present_supervision_mass"] == .5
    assert plan["aggregate"]["human_verified"] is False
    assert plan["aggregate"]["whole_video_targets"] == 0


def test_repeated_windows_do_not_add_supervision_mass():
    original = rows_fixture()
    expected = plan_balanced_silver(original)["sample_fact_weights"]["sample-0"]["CASH_VISIBLE"]
    duplicate = deepcopy(original[0])
    duplicate["sample_id"] = "repeated-window"
    plan = plan_balanced_silver(original + [duplicate])
    assert plan["sample_fact_weights"]["sample-0"]["CASH_VISIBLE"] == expected / 2
    assert plan["sample_fact_weights"]["repeated-window"]["CASH_VISIBLE"] == expected / 2


def test_each_content_keeps_equal_mass_when_it_has_more_anchors():
    rows = rows_fixture()
    extra = deepcopy(rows[0])
    extra.update(sample_id="extra-anchor", focal_frame_id="new-focal", frames=[{"frame_id": "new-focal"}])
    for fact in extra["target"]["facts"]:
        fact["supporting_frame_ids"] = ["new-focal"]
    plan = plan_balanced_silver(rows + [extra])
    first_content_mass = sum(plan["sample_fact_weights"][sample]["CASH_VISIBLE"] for sample in ("sample-0", "extra-anchor"))
    assert first_content_mass == plan["sample_fact_weights"]["sample-1"]["CASH_VISIBLE"]


def test_any_constant_state_per_fact_scores_only_half_balanced_mass():
    rows = rows_fixture()
    plan = plan_balanced_silver(rows)
    predicted = {"CASH_VISIBLE": "absent", "CASH_HANDLER_PRESENT": "present"}
    agreement_mass = sum(plan["sample_fact_weights"][row["sample_id"]][fact["fact_id"]]
        for row in rows for fact in row["target"]["facts"]
        if predicted.get(fact["fact_id"]) == fact["state"])
    assert math.isclose(agreement_mass, .5)


def test_no_evidence_or_synthetic_positives_are_added():
    rows = rows_fixture()
    before = canonical_sha256(rows)
    plan = plan_balanced_silver(rows)
    balanced = balanced_rows(rows, plan)
    assert canonical_sha256(rows) == before
    for row in balanced:
        for fact in row["target"]["facts"]:
            if fact["fact_id"] in plan["deferred_fact_ids"]:
                assert fact["state"] == "unclear"
                assert fact["supporting_frame_ids"] == []
                assert row["fact_supervision_mask"][fact["fact_id"]] is False


@pytest.mark.parametrize("key,value", [("split", "validation"), ("split", "final_check"),
    ("human_reviewed", True), ("annotation_origin", "workbook"), ("target_scope", "whole_video")])
def test_scope_and_roles_cannot_be_promoted(key, value):
    rows = rows_fixture()
    rows[0][key] = value
    with pytest.raises(TrainingGateError, match="scope_forbidden"):
        plan_balanced_silver(rows)


def test_unknown_fact_never_enters_balanced_mass():
    rows = rows_fixture()
    for row in rows:
        row["fact_supervision_mask"]["CASH_COUNT_VISIBLE"] = False
        row["target"]["facts"][-1].update(state="unclear", supporting_frame_ids=[])
    plan = plan_balanced_silver(rows)
    assert all(weights["CASH_COUNT_VISIBLE"] == 0 for weights in plan["sample_fact_weights"].values())


def test_coarse_positive_video_label_does_not_relabel_physical_absence():
    rows = rows_fixture()
    before = plan_balanced_silver(rows)["sample_fact_weights"]
    for row in rows:
        row["provided_video_label"] = True
    after = plan_balanced_silver(rows)
    assert after["sample_fact_weights"] == before
    assert after["aggregate"]["whole_video_targets"] == 0
    assert rows[2]["target"]["facts"][0]["state"] == "absent"


def test_supported_targets_must_have_original_focal_support():
    rows = rows_fixture()
    rows[0]["target"]["facts"][0]["supporting_frame_ids"] = []
    with pytest.raises(TrainingGateError, match="fact_unsupported"):
        plan_balanced_silver(rows)


def test_no_two_class_supported_axis_fails_closed():
    rows = rows_fixture()
    with pytest.raises(TrainingGateError, match="no_supported_two_class_facts"):
        plan_balanced_silver(rows, minimum_class_contents=3)


def test_unmasked_conflicting_anchor_is_rejected():
    rows = rows_fixture()
    conflict = deepcopy(rows[0])
    conflict["sample_id"] = "conflict"
    conflict["target"]["facts"][0]["state"] = "absent"
    with pytest.raises(TrainingGateError, match="conflicting_anchor_unmasked"):
        plan_balanced_silver(rows + [conflict])


def test_balancing_plan_is_bound_to_exact_rows_and_digest():
    rows = rows_fixture()
    plan = plan_balanced_silver(rows)
    changed = deepcopy(rows)
    changed[0]["sample_id"] = "changed"
    with pytest.raises(TrainingGateError, match="rows_binding_mismatch"):
        balanced_rows(changed, plan)
    plan["sample_fact_weights"]["sample-0"]["CASH_VISIBLE"] = 999
    with pytest.raises(TrainingGateError, match="plan_digest_mismatch"):
        balanced_rows(rows, plan)


def test_fact_mass_does_not_scale_with_json_or_support_token_length():
    labels = [-100, 7, 8, 9, 10, -100]
    weights = fact_token_loss_weights(labels, {"short": [1], "long": [2, 3, 4]}, {"short": .4, "long": .6})
    assert weights[0] == weights[-1] == 0
    assert weights[1] == .4
    assert math.isclose(sum(weights[2:5]), .6)


@pytest.mark.parametrize("positions,weights", [({"fact": []}, {"fact": 1.0}),
    ({"fact": [0]}, {"fact": 1.0}), ({"fact": [1, 1]}, {"fact": 1.0}),
    ({"fact": [1]}, {"fact": 0.0}), ({"fact": [1]}, {"fact": float("nan")}),
    ({"fact": [1]}, {"fact": -1.0})])
def test_unsafe_token_weights_are_rejected(positions, weights):
    with pytest.raises(TrainingGateError):
        fact_token_loss_weights([-100, 7], positions, weights)


def test_every_supervised_token_must_be_attributed_to_a_supported_fact():
    with pytest.raises(TrainingGateError, match="supervised_tokens_unaccounted"):
        fact_token_loss_weights([-100, 7, 8], {"fact": [1]}, {"fact": 1.0})


def test_weighted_loss_preserves_global_row_mass_and_gradients_when_torch_available():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 3, 2), requires_grad=True)
    labels = torch.tensor([[-100, 0, 1]])
    weights = torch.tensor([[0.0, .1, .3]])
    loss = weighted_causal_fact_loss(logits, labels, weights, example_count=4)
    assert torch.allclose(loss, torch.tensor(math.log(2) * .4 * 4))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0
    # Per-row normalization would cancel the .4 mass; halving global weights
    # must halve this row's contribution to the full fixed-epoch objective.
    half = weighted_causal_fact_loss(logits.detach(), labels, weights / 2, example_count=4)
    assert torch.allclose(half, loss.detach() / 2)
