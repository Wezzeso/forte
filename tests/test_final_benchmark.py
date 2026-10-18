from copy import deepcopy

import pytest

from evaluation.final_benchmark import AggregateError, receipt_reference, validate_api_load, validate_binary_verdicts, validate_crop_stress, validate_fixed_detections


def test_binary_failures_never_receive_clean_negative_credit():
    report = {"n": 4, "positive": 2, "negative": 2, "tp": 1, "fn": 1, "fp": 0, "tn": 1,
              "failed": 1, "failed_positive": 0, "failed_negative": 1, "processed": 3, "correct": 2,
              "accuracy": .5, "precision": 1, "recall": .5}
    assert validate_binary_verdicts(report)["accuracy"] == .5
    report["tn"] = 2
    report["correct"] = 3
    report["accuracy"] = .75
    with pytest.raises(AggregateError):
        validate_binary_verdicts(report)


def test_load_failure_and_deadline_gate_cannot_be_overridden_by_pass_flag():
    report = {"parallel": 6, "requests": 6, "passed": 6, "failed": 0, "deadline_seconds": 240,
              "elapsed_seconds": 162, "max_request_seconds": 161, "pass": True}
    assert validate_api_load(report)["accuracy_measured"] is False
    report["max_request_seconds"] = 241
    with pytest.raises(AggregateError):
        validate_api_load(report)
    report.update({"passed": 5, "failed": 1, "max_request_seconds": 161})
    with pytest.raises(AggregateError):
        validate_api_load(report)


def detection():
    return {"images": 5, "confidence_threshold": .5, "iou_threshold": .5, "per_class": {
        "single_banknote": {"tp": 1, "fp": 1, "fn": 1, "positive_instances": 2, "precision": .5, "recall": .5},
        "id_card": {"tp": 1, "fp": 0, "fn": 0, "positive_instances": 1, "precision": 1, "recall": 1}},
        "negative_images": 2, "false_positive_negative_images": 1, "negative_image_FPR": .5,
        "total_false_positives": 1}


def check_detection(value):
    return validate_fixed_detections(value, expected_images=5, expected_positive_instances={"single_banknote": 2, "id_card": 1}, expected_negatives=2)


def test_detector_aggregate_recomputes_counts_and_keeps_denominators_distinct():
    result = check_detection(detection())
    assert result["annotated_object_recall"] == 2 / 3
    assert result["negative_image_FPR"] == .5
    assert result["images"] == 5


@pytest.mark.parametrize("field,value", [("images", 4), ("negative_images", 1), ("total_false_positives", 0), ("false_positive_negative_images", 2), ("negative_image_FPR", 0), ("confidence_threshold", .1), ("iou_threshold", .25)])
def test_detector_rejects_inconsistent_support_or_rates(field, value):
    report = detection()
    report[field] = value
    with pytest.raises(AggregateError):
        check_detection(report)


def test_detector_rejects_hidden_missing_objects_and_nan():
    report = detection()
    report["per_class"]["single_banknote"]["fn"] = 0
    with pytest.raises(AggregateError):
        check_detection(report)
    report = detection()
    report["per_class"]["single_banknote"]["precision"] = float("nan")
    with pytest.raises(AggregateError):
        check_detection(report)


def crop():
    condition = {"counts": {"total": 3, "correct": 2, "true:id_card:pred:id_card": 1,
                             "true:single_banknote:pred:single_banknote": 1,
                             "true:single_banknote:pred:invalid_output": 1},
                 "per_class": {"id_card": {"examples": 1, "correct": 1, "accuracy": 1},
                               "single_banknote": {"examples": 2, "correct": 1, "accuracy": .5}},
                 "macro_class_accuracy": .75}
    return {"conditions": {key: deepcopy(condition) for key in ("0", "64", "32", "16")}, "macro_accuracy": .75}


def check_crop(value):
    return validate_crop_stress(value, expected_support={"id_card": 1, "single_banknote": 2})


def test_crop_invalid_outputs_are_errors_and_views_do_not_inflate_originals():
    result = check_crop(crop())
    assert result["unique_originals"] == 3
    assert result["macro_accuracy"] == .75
    assert result["conditions"]["16"]["pooled_accuracy"] == 2 / 3
    assert result["conditions"]["16"]["invalid_outputs"] == 1
    assert result["conditions"]["16"]["per_class"]["single_banknote"]["fn"] == 1


def test_crop_timing_uses_recorded_microbatch_scope_and_exact_support():
    report = crop()
    report["conditions"]["16"]["timing"] = {"examples": 3, "total_seconds": 2, "generation_seconds": 1.5,
        "crops_per_second": 1.5, "mean_generation_seconds": .5,
        "scope": "single_crop_microbatches_includes_preprocessing_not_whole_video_api_latency"}
    assert check_crop(report)["conditions"]["16"]["timing"]["crops_per_second"] == 1.5
    report["conditions"]["16"]["timing"]["crops_per_second"] = 99
    with pytest.raises(AggregateError):
        check_crop(report)


@pytest.mark.parametrize("mutation", ["missing_condition", "support", "correct", "macro", "unknown_prediction"])
def test_crop_rejects_inconsistent_or_incomplete_aggregates(mutation):
    report = crop()
    if mutation == "missing_condition":
        del report["conditions"]["16"]
    elif mutation == "support":
        report["conditions"]["16"]["per_class"]["id_card"]["examples"] = 2
    elif mutation == "correct":
        report["conditions"]["16"]["counts"]["correct"] = 3
    elif mutation == "macro":
        report["macro_accuracy"] = .99
    else:
        report["conditions"]["16"]["counts"]["true:id_card:pred:background"] = 1
    with pytest.raises(AggregateError):
        check_crop(report)


def test_receipt_reference_binds_exact_bytes_and_requires_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "receipt.json"
    source.write_bytes(b"{}\n")
    assert receipt_reference(source, root) == {"path": "receipt.json", "sha256": "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356", "bytes": 3}
    with pytest.raises(AggregateError):
        receipt_reference(tmp_path / "outside.json", root)
