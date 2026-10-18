"""Synthetic dictionaries only: fixed public metrics never open data or Torch."""
from __future__ import annotations

import copy

import pytest

from training.guards import TrainingGateError
from training.specialists.public_metrics import score_public_detections


def _object(name="single_banknote", box=None):
    return {"class_name": name, "bbox": [0, 0, 10, 10] if box is None else box}


def _detection(name="single_banknote", confidence=.8, box=None):
    return {**_object(name, box), "confidence": confidence}


def _row(sample_id, objects, **extra):
    return {"sample_id": sample_id, "split": "sealed_test", "included": True,
            "origin": "public", "objects": objects, **extra}


def _prediction(sample_id, detections):
    return {"sample_id": sample_id, "complete": True, "detections": detections}


@pytest.fixture
def inputs():
    lock = {"schema_version": "forte.specialists.public-dataset-lock.v1", "state": "frozen",
            "origin": "public", "classes": ["single_banknote", "id_card"],
            "records": [_row("cash", [_object()]), _row("id", [_object("id_card")]),
                        _row("coin-a", [], kind="coin"),
                        _row("coin-b", [], source_record={"kind": "coin", "origin": "public"}),
                        _row("other-empty", [], kind="background"),
                        _row("train", [_object()], split="train"),
                        _row("excluded", [_object()], included=False)]}
    predictions = [_prediction("cash", [_detection(), _detection(confidence=.7)]),
                   _prediction("id", []),
                   _prediction("coin-a", [_detection(confidence=.5), _detection("id_card", .6)]),
                   _prediction("coin-b", [_detection(confidence=.49999)]),
                   _prediction("other-empty", [_detection("id_card")])]
    return lock, predictions


def test_metrics_count_duplicates_misses_and_coin_false_alarm_images(inputs):
    result = score_public_detections(*inputs)

    assert result["scope"] == "public_object_detection_not_bank_violation"
    assert result["confidence_threshold"] == result["iou_threshold"] == .5
    assert result["images"] == 5
    assert result["per_class"] == {
        "single_banknote": {"tp": 1, "fp": 2, "fn": 0, "positive_instances": 1,
                            "precision": 1 / 3, "recall": 1},
        "id_card": {"tp": 0, "fp": 2, "fn": 1, "positive_instances": 1,
                    "precision": 0, "recall": 0}}
    assert result["negative_images"] == 2
    assert result["false_positive_negative_images"] == 1
    assert result["negative_image_FPR"] == .5
    assert result["total_false_positives"] == 4


def test_iou_and_confidence_boundaries_are_inclusive():
    lock = {"schema_version": "forte.specialists.public-dataset-lock.v1", "state": "frozen",
            "origin": "public", "classes": ["single_banknote"],
            "records": [_row("a", [_object()])]}
    predictions = [_prediction("a", [_detection(confidence=.5, box=[0, 0, 20, 10])])]

    result = score_public_detections(lock, predictions)

    assert result["per_class"]["single_banknote"]["tp"] == 1
    assert result["negative_image_FPR"] is None


def test_same_class_required_for_match(inputs):
    lock, _ = inputs
    predictions = [_prediction(row["sample_id"], []) for row in lock["records"]
                   if row["split"] == "sealed_test" and row["included"]]
    predictions[0]["detections"] = [_detection("id_card")]

    result = score_public_detections(lock, predictions)

    assert result["per_class"]["single_banknote"]["fn"] == 1
    assert result["per_class"]["id_card"]["fp"] == 1
    assert result["per_class"]["single_banknote"]["precision"] is None


def test_greedy_matching_uses_confidence_and_each_truth_once():
    lock = {"schema_version": "forte.specialists.public-dataset-lock.v1", "state": "frozen",
            "origin": "public", "classes": ["single_banknote"],
            "records": [_row("a", [_object(box=[0, 0, 10, 10]), _object(box=[5, 0, 10, 10])])]}
    # The lower-confidence box overlaps both truths at IoU .6. If input order
    # were used its tie would consume truth zero, leaving the high-confidence
    # box (which overlaps only truth zero) unmatched.
    predictions = [_prediction("a", [_detection(confidence=.6, box=[2.5, 0, 10, 10]),
                                       _detection(confidence=.9, box=[0, 0, 10, 10])])]

    metrics = score_public_detections(lock, predictions)["per_class"]["single_banknote"]

    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 0, 0)


def test_missing_class_support_is_undefined_not_perfect(inputs):
    lock, predictions = inputs
    lock["records"][1]["objects"] = []
    result = score_public_detections(lock, predictions)
    assert result["per_class"]["id_card"]["recall"] is None


@pytest.mark.parametrize("change,code", [
    (lambda lock, predictions: predictions.pop(), "prediction_coverage"),
    (lambda lock, predictions: predictions.append(copy.deepcopy(predictions[0])), "prediction_coverage"),
    (lambda lock, predictions: predictions[1].update(sample_id="cash"), "prediction_identity"),
    (lambda lock, predictions: predictions[1].update(sample_id="train"), "prediction_identity"),
    (lambda lock, predictions: predictions[1].update(complete=False), "prediction_incomplete"),
    (lambda lock, predictions: predictions[1].update(error="failed"), "prediction_incomplete"),
    (lambda lock, predictions: predictions[1].update(error_reasons=["failed"]), "prediction_incomplete"),
    (lambda lock, predictions: predictions[1].update(processed=False), "prediction_incomplete"),
    (lambda lock, predictions: predictions[1].update(detections=None), "prediction_incomplete"),
    (lambda lock, predictions: lock.update(origin="own"), "public_lock_required"),
    (lambda lock, predictions: lock["records"][0].update(origin="own"), "public_record_required"),
    (lambda lock, predictions: lock["records"][0].update(source_record={"origin": "own"}), "public_record_required"),
    (lambda lock, predictions: lock.update(classes=["cup"]), "classes_invalid"),
    (lambda lock, predictions: lock.update(classes=["single_banknote", "single_banknote"]), "classes_invalid"),
    (lambda lock, predictions: lock["records"][0]["objects"][0].update(class_name="cup"), "truth_class_invalid"),
    (lambda lock, predictions: lock["records"][1].update(sample_id="cash"), "sample_identity_invalid"),
])
def test_incomplete_forged_or_nonpublic_inputs_fail_closed(inputs, change, code):
    lock, predictions = inputs
    change(lock, predictions)
    with pytest.raises(TrainingGateError, match="public_metrics_" + code):
        score_public_detections(lock, predictions)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -.1, 1.1, True, "0.9"])
def test_invalid_confidence_is_rejected_before_thresholding(inputs, confidence):
    lock, predictions = inputs
    predictions[0]["detections"][0]["confidence"] = confidence
    with pytest.raises(TrainingGateError, match="public_metrics_detection_invalid"):
        score_public_detections(lock, predictions)


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 1, -1], [0, 0, float("nan"), 1],
                                  [0, float("inf"), 1, 1], [0, 0, True, 1], [0, 0, 1], "box"])
def test_invalid_detection_geometry_is_rejected_even_below_threshold(inputs, box):
    lock, predictions = inputs
    predictions[0]["detections"] = [_detection(confidence=.1, box=box)]
    with pytest.raises(TrainingGateError, match="public_metrics_prediction_bbox_invalid"):
        score_public_detections(lock, predictions)


def test_invalid_truth_geometry_is_rejected(inputs):
    lock, predictions = inputs
    lock["records"][0]["objects"][0]["bbox"] = [0, 0, -1, 1]
    with pytest.raises(TrainingGateError, match="public_metrics_truth_bbox_invalid"):
        score_public_detections(lock, predictions)


def test_unknown_detection_class_is_rejected_even_below_threshold(inputs):
    lock, predictions = inputs
    predictions[0]["detections"] = [_detection("cup", confidence=.1)]
    with pytest.raises(TrainingGateError, match="public_metrics_detection_invalid"):
        score_public_detections(lock, predictions)


def test_inputs_remain_unchanged(inputs):
    before = copy.deepcopy(inputs)
    score_public_detections(*inputs)
    assert inputs == before
