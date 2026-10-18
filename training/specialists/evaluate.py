"""Evaluate a pinned candidate on untouched own-recording camera holdout."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .common import (canonical_sha256, cli_error, file_sha256, load_json, number,
                     require, sha, write_json)
from .prepare import verify_dataset
from .train import load_model, verify_run


def iou(a: list[float], b: list[float]) -> float:
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - x) * max(0, bottom - y)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def score_predictions(lock: dict, predictions: dict, thresholds: dict) -> dict:
    rows = [r for r in lock["records"] if r["split"] == "holdout"]
    require(bool(rows) and all(r["origin"] == "own" for r in rows), "specialist_holdout_missing")
    classes = lock["classes"]
    require(set(thresholds) == set(classes) and all(number(v, 0, 1) for v in thresholds.values()),
            "specialist_thresholds_invalid")
    items = predictions.get("images")
    require(isinstance(items, list) and len(items) == len(rows), "specialist_holdout_prediction_coverage")
    indexed = {p.get("sample_id"): p for p in items if isinstance(p, dict)}
    require(len(indexed) == len(items) and set(indexed) == {r["sample_id"] for r in rows},
            "specialist_holdout_prediction_identity")
    counts = {name: Counter(tp=0, fp=0, fn=0) for name in classes}
    clean_images, clean_fp = 0, 0
    for row in rows:
        prediction = indexed[row["sample_id"]]
        require(prediction.get("complete") is True and isinstance(prediction.get("detections"), list),
                "specialist_prediction_incomplete")
        detections = prediction["detections"]
        for detection in detections:
            require(isinstance(detection, dict) and detection.get("class_name") in classes
                    and number(detection.get("confidence"), 0, 1), "specialist_detection_invalid")
            box = detection.get("bbox")
            require(isinstance(box, list) and len(box) == 4 and all(number(v, -1e9, 1e9) for v in box)
                    and box[2] > 0 and box[3] > 0, "specialist_prediction_bbox_invalid")
        if not row["objects"]:
            clean_images += 1
            clean_fp += sum(d["confidence"] >= thresholds[d["class_name"]] for d in detections)
        for name in classes:
            truths = [o["bbox"] for o in row["objects"] if o["class_name"] == name]
            unmatched = set(range(len(truths)))
            found = sorted((d for d in detections if d["class_name"] == name
                            and d["confidence"] >= thresholds[name]), key=lambda d: -d["confidence"])
            for detection in found:
                best = max(unmatched, key=lambda index: iou(detection["bbox"], truths[index]), default=None)
                if best is not None and iou(detection["bbox"], truths[best]) >= .5:
                    unmatched.remove(best)
                    counts[name]["tp"] += 1
                else:
                    counts[name]["fp"] += 1
            counts[name]["fn"] += len(unmatched)
    per_class = {}
    for name, count in counts.items():
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        per_class[name] = dict(count) | {"positive_instances": tp + fn,
                                        "recall": tp / (tp + fn) if tp + fn else None,
                                        "precision": tp / (tp + fp) if tp + fp else None}
    return {"split": "holdout", "iou_threshold": .5, "thresholds": thresholds,
            "images": len(rows), "clean_images": clean_images, "clean_false_positives": clean_fp,
            "total_false_positives": sum(c["fp"] for c in counts.values()), "per_class": per_class,
            "heldout_data_sha256": canonical_sha256(rows),
            "evaluation_protocol_sha256": canonical_sha256({"split": "holdout", "iou": .5,
                                                               "thresholds": thresholds,
                                                               "class_names": lock["class_names"]})}


def evaluate(config: dict) -> dict:
    require(config.get("schema_version") == "forte.specialists.evaluate.v1", "specialist_evaluate_schema")
    root = Path(config["dataset_dir"]).resolve()
    lock = verify_dataset(root, config.get("expected_dataset_lock_sha256", ""))
    run_path = Path(config["run"]).resolve()
    run, checkpoint = verify_run(run_path, config.get("expected_run_sha256", ""))
    require(run["dataset_lock_sha256"] == config["expected_dataset_lock_sha256"]
            and run["class_names"] == lock["class_names"], "specialist_evaluation_dataset_mismatch")
    output = Path(config["output_dir"]).resolve()
    require(not output.exists(), "specialist_evaluation_output_already_exists")
    output.mkdir(parents=True)
    model = load_model(run["model_class"], checkpoint, resolution=run["resolution"], device=config.get("device", "cuda"))
    inverse = {value: name for name, value in lock["prediction_class_ids"].items()}
    predictions = {"schema_version": "forte.specialists.predictions.v1", "images": []}
    rows = [r for r in lock["records"] if r["split"] == "holdout"]
    require(bool(rows), "specialist_holdout_missing")
    for row in rows:
        # Match serving's low proposal threshold. Per-class inclusive thresholds
        # are applied by the scorer, including a confidence exactly at the gate.
        raw = model.predict(str(root / row["export_path"]), threshold=min(.1, *run["thresholds"].values()))
        require(raw.class_id is not None and raw.confidence is not None, "specialist_model_prediction_invalid")
        require(len(raw.xyxy) == len(raw.class_id) == len(raw.confidence), "specialist_model_prediction_length")
        detections = []
        for box, class_id, confidence in zip(raw.xyxy, raw.class_id, raw.confidence):
            require(int(class_id) in inverse, "specialist_model_class_id_mismatch")
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append({"class_name": inverse[int(class_id)], "confidence": float(confidence),
                               "bbox": [x1, y1, x2 - x1, y2 - y1]})
        predictions["images"].append({"sample_id": row["sample_id"], "complete": True, "detections": detections})
    # Detect data/checkpoint mutation during evaluation before issuing proof.
    verify_dataset(root, run["dataset_lock_sha256"])
    verify_run(run_path, config["expected_run_sha256"])
    write_json(output / "predictions.json", predictions)
    metrics = {"schema_version": "forte.specialists.metrics.v1", "state": "complete",
               "run_sha256": file_sha256(run_path), "checkpoint_sha256": run["checkpoint_sha256"],
               "dataset_lock_sha256": run["dataset_lock_sha256"], "config_sha256": run["config_sha256"],
               "predictions": "predictions.json", "predictions_sha256": file_sha256(output / "predictions.json"),
               "class_names": lock["class_names"], "model_class": run["model_class"],
               **score_predictions(lock, predictions, run["thresholds"])}
    write_json(output / "metrics.json", metrics)
    return {"ok": True, "state": "complete", "metrics_sha256": file_sha256(output / "metrics.json"),
            "total_false_positives": metrics["total_false_positives"], "per_class": metrics["per_class"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(load_json(args.config))))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
