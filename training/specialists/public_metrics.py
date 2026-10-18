"""Fixed-threshold public object metrics; no model, image, or filesystem I/O."""
from __future__ import annotations

from .common import number, require


CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
ALLOWED_CLASSES = frozenset({"single_banknote", "id_card"})


def _box(value: object, *, prediction: bool) -> list[float]:
    # Detectors may emit coordinates just outside an image. Keep those boxes
    # unmodified for IoU; reject nonfinite/extreme values and nonpositive areas.
    minimum = -1e9 if prediction else 0
    require(isinstance(value, list) and len(value) == 4
            and all(number(v, minimum, 1e9) for v in value[:2])
            and all(number(v, 0, 1e9) and v > 0 for v in value[2:]),
            "public_metrics_prediction_bbox_invalid" if prediction else "public_metrics_truth_bbox_invalid")
    return value


def _iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def score_public_detections(lock: dict, predictions: list[dict]) -> dict:
    """Score the included sealed test exactly once per supplied prediction row.

    The caller verifies dataset/checkpoint hashes and controls sealed-test
    exposure. This pure scorer validates the report's contents and computes
    fixed confidence >= .5, same-class IoU >= .5 one-to-one matches. It does
    not select a checkpoint or measure bank-violation accuracy.
    """
    require(isinstance(lock, dict)
            and lock.get("schema_version") == "forte.specialists.public-dataset-lock.v1"
            and lock.get("state") == "frozen" and lock.get("origin") == "public",
            "public_metrics_public_lock_required")
    classes = lock.get("classes")
    require(isinstance(classes, list) and bool(classes)
            and all(isinstance(name, str) and name in ALLOWED_CLASSES for name in classes)
            and len(set(classes)) == len(classes), "public_metrics_classes_invalid")
    records = lock.get("records")
    require(isinstance(records, list) and all(isinstance(row, dict) for row in records),
            "public_metrics_records_required")
    rows = [row for row in records if row.get("split") == "sealed_test" and row.get("included") is True]
    require(bool(rows), "public_metrics_sealed_test_missing")
    expected = {}
    for row in rows:
        source = row.get("source_record", {})
        require(isinstance(source, dict)
                and row.get("origin", "public") == "public"
                and source.get("origin", "public") == "public", "public_metrics_public_record_required")
        sample_id = row.get("sample_id")
        require(isinstance(sample_id, str) and bool(sample_id) and sample_id not in expected,
                "public_metrics_sample_identity_invalid")
        require(isinstance(row.get("objects"), list), "public_metrics_truth_objects_required")
        for obj in row["objects"]:
            require(isinstance(obj, dict) and obj.get("class_name") in classes,
                    "public_metrics_truth_class_invalid")
            _box(obj.get("bbox"), prediction=False)
        expected[sample_id] = row
    require(isinstance(predictions, list) and len(predictions) == len(rows),
            "public_metrics_prediction_coverage")
    indexed = {}
    for prediction in predictions:
        require(isinstance(prediction, dict), "public_metrics_prediction_invalid")
        sample_id = prediction.get("sample_id")
        require(isinstance(sample_id, str) and sample_id in expected and sample_id not in indexed,
                "public_metrics_prediction_identity")
        require(prediction.get("complete") is True and not prediction.get("error")
                and not prediction.get("error_reasons") and prediction.get("processed", True) is True
                and isinstance(prediction.get("detections"), list), "public_metrics_prediction_incomplete")
        for detection in prediction["detections"]:
            require(isinstance(detection, dict) and detection.get("class_name") in classes
                    and number(detection.get("confidence"), 0, 1), "public_metrics_detection_invalid")
            _box(detection.get("bbox"), prediction=True)
        indexed[sample_id] = prediction
    require(set(indexed) == set(expected), "public_metrics_prediction_coverage")

    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in classes}
    negative_images = false_positive_negative_images = 0
    for sample_id, row in expected.items():
        accepted = [d for d in indexed[sample_id]["detections"] if d["confidence"] >= CONFIDENCE_THRESHOLD]
        if not row["objects"] and (row.get("kind") == "coin" or row.get("source_record", {}).get("kind") == "coin"):
            negative_images += 1
            false_positive_negative_images += bool(accepted)
        for name in classes:
            truths = [obj["bbox"] for obj in row["objects"] if obj["class_name"] == name]
            unmatched = set(range(len(truths)))
            detections = sorted((d for d in accepted if d["class_name"] == name), key=lambda d: -d["confidence"])
            for detection in detections:
                # Ascending indices make ties reproducible rather than relying
                # on set traversal order. A matched truth cannot match twice.
                best = max(sorted(unmatched), key=lambda index: _iou(detection["bbox"], truths[index]), default=None)
                if best is not None and _iou(detection["bbox"], truths[best]) >= IOU_THRESHOLD:
                    unmatched.remove(best)
                    counts[name]["tp"] += 1
                else:
                    counts[name]["fp"] += 1
            counts[name]["fn"] += len(unmatched)
    per_class = {}
    for name, count in counts.items():
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        per_class[name] = {**count, "positive_instances": tp + fn,
                           "precision": tp / (tp + fp) if tp + fp else None,
                           "recall": tp / (tp + fn) if tp + fn else None}
    return {"scope": "public_object_detection_not_bank_violation", "split": "sealed_test",
            "confidence_threshold": CONFIDENCE_THRESHOLD, "iou_threshold": IOU_THRESHOLD,
            "images": len(rows), "per_class": per_class, "negative_images": negative_images,
            "false_positive_negative_images": false_positive_negative_images,
            "negative_image_FPR": false_positive_negative_images / negative_images if negative_images else None,
            "total_false_positives": sum(count["fp"] for count in counts.values())}
