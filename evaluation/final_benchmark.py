"""Validate aggregate-only benchmark receipts without loading media or predictions.

These checks recompute reported ratios and support, not model inference.  They do
not establish the independence of source images or the completeness of publisher
annotations; those limitations belong in the final benchmark's scope fields.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping


class AggregateError(ValueError):
    """An aggregate receipt contradicts its own counts or declared support."""


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AggregateError(f"{name} must be a nonnegative integer")
    return value


def _ratio(actual: Any, expected: float | None, name: str) -> None:
    if expected is None:
        if actual is not None:
            raise AggregateError(f"{name} must be null for an empty denominator")
    elif (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(actual)
        or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
    ):
        raise AggregateError(f"{name} disagrees with counts")


def receipt_reference(path: str | Path, root: str | Path) -> dict[str, Any]:
    """Bind an existing source file without exposing absolute operator paths."""
    base = Path(root).resolve()
    source = Path(path).resolve()
    try:
        relative = source.relative_to(base).as_posix()
    except ValueError as exc:
        raise AggregateError("receipt is outside the workspace") from exc
    contents = source.read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}


def validate_binary_verdicts(report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep analyzer failures in whole-video accuracy's denominator."""
    names = ("n", "positive", "negative", "tp", "fp", "tn", "fn", "failed", "failed_positive", "failed_negative", "processed", "correct")
    values = {name: _count(report.get(name), name) for name in names}
    if (values["positive"] + values["negative"] != values["n"]
        or values["tp"] + values["fn"] + values["failed_positive"] != values["positive"]
        or values["tn"] + values["fp"] + values["failed_negative"] != values["negative"]
        or values["failed_positive"] + values["failed_negative"] != values["failed"]
        or values["processed"] + values["failed"] != values["n"]
        or values["tp"] + values["tn"] != values["correct"]):
        raise AggregateError("binary verdict counts disagree or credit failures")
    for name, numerator, denominator in (
        ("accuracy", values["correct"], values["n"]),
        ("precision", values["tp"], values["tp"] + values["fp"]),
        ("recall", values["tp"], values["positive"]),
    ):
        values[name] = numerator / denominator if denominator else None
        _ratio(report.get(name), values[name], name)
    values["negative_video_false_alarm_rate"] = values["fp"] / values["negative"] if values["negative"] else None
    return values


def validate_api_load(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an aggregate request cohort without treating it as accuracy."""
    values = {name: _count(report.get(name), name) for name in ("parallel", "requests", "passed", "failed")}
    if not values["parallel"] or not values["requests"] or values["passed"] + values["failed"] != values["requests"]:
        raise AggregateError("load request counts disagree")
    for name in ("deadline_seconds", "elapsed_seconds", "max_request_seconds"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise AggregateError("load timing must be finite and nonnegative")
        values[name] = value
    if not values["deadline_seconds"]:
        raise AggregateError("load deadline must be positive")
    passed = values["failed"] == 0 and values["max_request_seconds"] <= values["deadline_seconds"]
    if report.get("pass") is not passed:
        raise AggregateError("load pass flag disagrees with failures or deadline")
    values["pass"] = passed
    values["timing_scope"] = "client_start_through_upload_queue_processing_poll_and_terminal_schema_validation"
    values["accuracy_measured"] = False
    return values


def validate_fixed_detections(
    report: Mapping[str, Any], *, expected_images: int,
    expected_positive_instances: Mapping[str, int], expected_negatives: int,
    expected_confidence: float = .5, expected_iou: float = .5,
) -> dict[str, Any]:
    """Recompute the fixed operating point from its sanitized aggregate only."""
    _ratio(report.get("confidence_threshold"), expected_confidence, "confidence_threshold")
    _ratio(report.get("iou_threshold"), expected_iou, "iou_threshold")
    images = _count(report.get("images"), "images")
    negatives = _count(report.get("negative_images"), "negative_images")
    if images != expected_images or negatives != expected_negatives:
        raise AggregateError("image support differs from frozen split")
    classes = report.get("per_class")
    if not isinstance(classes, Mapping) or set(classes) != set(expected_positive_instances):
        raise AggregateError("class coverage differs from frozen protocol")
    output: dict[str, Any] = {}
    total_fp = total_tp = total_fn = 0
    for name, expected in expected_positive_instances.items():
        values = classes[name]
        tp, fp, fn = (_count(values.get(key), f"{name}.{key}") for key in ("tp", "fp", "fn"))
        support = _count(values.get("positive_instances"), f"{name}.positive_instances")
        if tp + fn != support or support != expected:
            raise AggregateError(f"{name} positive support disagrees")
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / support if support else None
        _ratio(values.get("precision"), precision, f"{name}.precision")
        _ratio(values.get("recall"), recall, f"{name}.recall")
        output[name] = {"tp": tp, "fp": fp, "fn": fn, "positive_instances": support,
                        "precision": precision, "recall": recall}
        total_fp += fp
        total_tp += tp
        total_fn += fn
    if _count(report.get("total_false_positives"), "total_false_positives") != total_fp:
        raise AggregateError("total false positives disagree with classes")
    false_images = _count(report.get("false_positive_negative_images"), "false_positive_negative_images")
    if false_images > negatives or false_images > total_fp:
        raise AggregateError("negative images with false positives exceed support")
    negative_fpr = false_images / negatives if negatives else None
    _ratio(report.get("negative_image_FPR"), negative_fpr, "negative_image_FPR")
    return {"images": images, "confidence_threshold": expected_confidence, "iou_threshold": expected_iou,
            "per_class": output, "negative_images": negatives,
            "false_positive_negative_images": false_images, "negative_image_FPR": negative_fpr,
            "total_false_positives": total_fp, "annotated_object_tp": total_tp,
            "annotated_object_fn": total_fn,
            "annotated_object_recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else None}


_CONFUSION = re.compile(r"^true:([^:]+):pred:([^:]+)$")


def validate_crop_stress(
    report: Mapping[str, Any], *, expected_support: Mapping[str, int],
    expected_conditions: tuple[str, ...] = ("0", "64", "32", "16"),
) -> dict[str, Any]:
    """Recompute confusion-derived accuracy, retaining invalid output as error.

    Each condition reuses the same originals.  The returned unique support is
    deliberately not multiplied by the number of resolution conditions.
    """
    conditions = report.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(expected_conditions):
        raise AggregateError("crop condition coverage differs from frozen protocol")
    labels = set(expected_support)
    summarized: dict[str, Any] = {}
    for condition in expected_conditions:
        item = conditions[condition]
        counts = item.get("counts", {})
        matrix = {(truth, prediction): 0 for truth in labels for prediction in labels | {"invalid_output"}}
        for key, value in counts.items():
            if key in {"total", "correct"}:
                continue
            match = _CONFUSION.fullmatch(key)
            if not match or match[1] not in labels or match[2] not in labels | {"invalid_output"}:
                raise AggregateError("unexpected crop confusion key")
            matrix[match[1], match[2]] = _count(value, key)
        total = sum(matrix.values())
        correct = sum(matrix[name, name] for name in labels)
        if _count(counts.get("total"), "total") != total or _count(counts.get("correct"), "correct") != correct:
            raise AggregateError("crop total or correct count disagrees with confusion")
        per_class: dict[str, Any] = {}
        declared = item.get("per_class", {})
        if set(declared) != labels:
            raise AggregateError("crop class coverage differs from frozen protocol")
        for label in sorted(labels):
            support = sum(value for (truth, _), value in matrix.items() if truth == label)
            if support != expected_support[label]:
                raise AggregateError("crop class support differs from frozen protocol")
            tp = matrix[label, label]
            fp = sum(value for (truth, prediction), value in matrix.items() if truth != label and prediction == label)
            fn = support - tp
            values = declared[label]
            if _count(values.get("examples"), "examples") != support or _count(values.get("correct"), "correct") != tp:
                raise AggregateError("crop per-class support disagrees with confusion")
            accuracy = tp / support if support else None
            _ratio(values.get("accuracy"), accuracy, "crop class accuracy")
            per_class[label] = {"tp": tp, "fp": fp, "fn": fn, "examples": support,
                                "precision": tp / (tp + fp) if tp + fp else None, "recall": accuracy}
        macro = sum(values["recall"] for values in per_class.values()) / len(labels)
        _ratio(item.get("macro_class_accuracy"), macro, "macro_class_accuracy")
        summarized[condition] = {"correct": correct, "total": total,
                                 "pooled_accuracy": correct / total if total else None,
                                 "macro_class_accuracy": macro, "per_class": per_class,
                                 "invalid_outputs": sum(matrix[truth, "invalid_output"] for truth in labels)}
        if "timing" in item:
            timing = item["timing"]
            if _count(timing.get("examples"), "timing.examples") != total:
                raise AggregateError("crop timing support disagrees with scores")
            seconds = timing.get("total_seconds")
            generation = timing.get("generation_seconds")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0
                or isinstance(generation, bool) or not isinstance(generation, (int, float))
                or not math.isfinite(generation) or not 0 <= generation <= seconds):
                raise AggregateError("crop timing must be finite and ordered")
            _ratio(timing.get("crops_per_second"), total / seconds, "crops_per_second")
            _ratio(timing.get("mean_generation_seconds"), generation / total, "mean_generation_seconds")
            if timing.get("scope") != "single_crop_microbatches_includes_preprocessing_not_whole_video_api_latency":
                raise AggregateError("crop timing scope is missing or unexpected")
            summarized[condition]["timing"] = dict(timing)
    macro = sum(item["macro_class_accuracy"] for item in summarized.values()) / len(summarized)
    _ratio(report.get("macro_accuracy"), macro, "macro_accuracy")
    return {"unique_originals": sum(expected_support.values()), "unique_class_support": dict(expected_support),
            "conditions": summarized, "macro_accuracy": macro,
            "views_reuse_originals": True, "task": "oracle_box_two_class_crop_classification"}
