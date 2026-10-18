"""Synthetic provenance/leakage/metric proof tests; never read supplied media."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from training.guards import TrainingGateError, file_sha256
from training.specialists.common import write_json
from training.specialists.evaluate import score_predictions
from training.specialists.gate import gate
from training.specialists.prepare import prepare, validate_manifest, verify_dataset
from training.specialists.train import preflight


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    inventory = tmp_path / "organizer.json"
    write_json(inventory, [{"content_sha256": "a" * 64}])
    rows = []
    for role, count in (("train", 1), ("validation", 1), ("holdout", 40)):
        recording = source / f"{role}.mov"
        recording.write_bytes(f"synthetic-source-{role}".encode())
        for n in range(count):
            index = len(rows)
            path = source / f"{index}.png"
            Image.new("RGB", (16, 16), color=(index + 1, 2, 3)).save(path)
            positive = role != "holdout" or n < 20
            rows.append({"sample_id": f"s-{index}", "path": path.name,
                         "content_sha256": file_sha256(path), "source_recording_path": recording.name,
                         "source_recording_sha256": file_sha256(recording), "origin": "own", "split": role,
                         "scenario_group": f"g-{role}", "camera_id": f"camera-{role}",
                         "width": 16, "height": 16, "labels_reviewed": True,
                         "exhaustive_labels": True, "label_reviewer": "synthetic-test",
                         "label_reviewed_at": "2026-10-18T00:00:00Z",
                         "provenance": {"reference": "synthetic-fixture-not-real-consent", "participants_consented": True,
                                        "recording_authorized": True, "commercial_use_allowed": True},
                         "objects": [{"class_name": "id_card", "bbox": [1, 1, 4, 3]},
                                     {"class_name": "key_in_lock", "bbox": [8, 8, 2, 3]}] if positive else []})
    manifest = {"schema_version": "forte.specialists.data.v1", "frozen": True,
                "reviewed_by": "synthetic-test", "reviewed_at": "2026-10-18", "media_root": str(source),
                "organizer_inventory": {"path": str(inventory), "sha256": file_sha256(inventory)},
                "classes": ["id_card", "key_in_lock"], "records": rows}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    output = tmp_path / "dataset"
    summary = prepare(manifest_path, output)
    lock = verify_dataset(output, summary["dataset_lock_sha256"])
    weights = tmp_path / "local-pretrain.pth"
    weights.write_bytes(b"synthetic-weights-not-a-model")
    config = {"schema_version": "forte.specialists.train.v1", "rfdetr_version": "1.11.2",
              "model_class": "RFDETRSmall", "dataset_dir": str(output),
              "expected_dataset_lock_sha256": summary["dataset_lock_sha256"],
              "pretrain_weights": str(weights), "expected_pretrain_weights_sha256": file_sha256(weights),
              "output_dir": str(tmp_path / "run"), "device": "cuda", "epochs": 1,
              "batch_size": 1, "grad_accum_steps": 1, "resolution": 768, "lr": .0001, "seed": 1,
              "thresholds": {"id_card": .5, "key_in_lock": .5},
              "gate_policy": {"minimum_recall": .95, "minimum_positive_instances": 20, "minimum_clean_images": 20},
              "compute_release_authorization": {"released_by_A18_or_root": True, "release_reference": "synthetic-test"}}
    return {"manifest": manifest, "manifest_path": manifest_path, "root": output,
            "lock": lock, "config": config, "base": tmp_path}


def test_deterministic_export_and_physically_excluded_holdout(dataset):
    second = prepare(dataset["manifest_path"], dataset["base"] / "second")
    assert second["dataset_sha256"] == dataset["lock"]["dataset_sha256"]
    test = json.loads((dataset["root"] / "test/_annotations.coco.json").read_text())
    assert test["images"] == [] and test["annotations"] == []
    assert dataset["lock"]["class_names"] == {"0": "id_card", "1": "key_in_lock"}
    assert dataset["lock"]["coco_category_ids"] == {"id_card": 1, "key_in_lock": 2}


@pytest.mark.parametrize("change,code", [
    (lambda m: m["records"][0].update(origin="organizer"), "organizer_training_forbidden"),
    (lambda m: m["records"][0]["provenance"].update(participants_consented=False), "recording_consent_required"),
    (lambda m: m["records"][0].update(source_recording_sha256="a" * 64), "organizer_source_rebranded"),
    (lambda m: m["records"][0].update(source_recording_sha256="b" * 64), "source_recording_hash_mismatch"),
    (lambda m: m["records"][1].update(scenario_group="g-train"), "partition_leakage"),
    (lambda m: m["records"][2].update(camera_id="camera-train"), "holdout_camera_leakage"),
    (lambda m: m["records"][2].update(derived_from_sample_id="s-0"), "derivation_leakage"),
    (lambda m: m["records"][0].update(exhaustive_labels=False), "exhaustive_human_labels_required"),
    (lambda m: m["records"][0]["objects"][0].update(bbox=[15, 15, 4, 4]), "bbox_outside_image"),
    (lambda m: m.update(frozen=False), "data_review_required"),
])
def test_manifest_fails_closed_on_leakage_consent_and_bad_boxes(dataset, change, code):
    manifest = copy.deepcopy(dataset["manifest"])
    change(manifest)
    with pytest.raises(TrainingGateError, match=code):
        validate_manifest(manifest, dataset["base"])


def test_dataset_tampering_is_rejected(dataset):
    path = dataset["root"] / "train/_annotations.coco.json"
    path.write_text("{}")
    with pytest.raises(TrainingGateError, match="dataset_file_changed"):
        verify_dataset(dataset["root"], dataset["config"]["expected_dataset_lock_sha256"])


def test_preflight_checks_hashes_without_loading_detector(dataset):
    _, lock = preflight(dataset["config"])
    assert lock["state"] == "frozen"
    config = copy.deepcopy(dataset["config"])
    config["rfdetr_version"] = "latest"
    with pytest.raises(TrainingGateError, match="package_pin_required"):
        preflight(config)
    config = copy.deepcopy(dataset["config"])
    config["gate_policy"]["minimum_recall"] = .9
    with pytest.raises(TrainingGateError, match="gate_policy_invalid"):
        preflight(config)


def proof(dataset):
    root = dataset["base"] / "run"
    root.mkdir()
    write_json(root / "config.json", dataset["config"])
    checkpoint = root / "checkpoint_best_total.pth"
    checkpoint.write_bytes(b"synthetic-boundary-test-not-trained-weights")
    config = dataset["config"]
    run = {"schema_version": "forte.specialists.run.v1", "state": "trained_unselected",
           "rfdetr_version": "1.11.2", "model_class": "RFDETRSmall", "class_names": dataset["lock"]["class_names"],
           "checkpoint": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint), "resolution": 768,
           "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
           "config_sha256": file_sha256(root / "config.json"), "thresholds": config["thresholds"],
           "gate_policy": config["gate_policy"]}
    write_json(root / "run.json", run)
    predictions = {"schema_version": "forte.specialists.predictions.v1", "images": [
        {"sample_id": r["sample_id"], "complete": True,
         "detections": [dict(o, confidence=.9) for o in r["objects"]]}
        for r in dataset["lock"]["records"] if r["split"] == "holdout"]}
    metrics_root = dataset["base"] / "metrics"
    metrics_root.mkdir()
    write_json(metrics_root / "predictions.json", predictions)
    metrics = {"schema_version": "forte.specialists.metrics.v1", "state": "complete",
               **{key: run[key] for key in ("checkpoint_sha256", "dataset_lock_sha256", "config_sha256", "class_names", "model_class")},
               "run_sha256": file_sha256(root / "run.json"), "predictions": "predictions.json",
               "predictions_sha256": file_sha256(metrics_root / "predictions.json"),
               **score_predictions(dataset["lock"], predictions, config["thresholds"])}
    write_json(metrics_root / "metrics.json", metrics)
    return root / "run.json", metrics_root / "metrics.json", predictions


def test_promotion_recomputes_per_class_recall_and_retains_type_lock(dataset):
    run, metrics, _ = proof(dataset)
    result = gate(run, metrics)
    assert result["eligible"] is True
    assert result["approved_classes"] == ["id_card", "key_in_lock"]
    assert result["approved_types"] == []
    assert result["class_names"] == {"0": "id_card", "1": "key_in_lock"}


def test_promotion_rejects_forged_metric_and_checkpoint(dataset):
    run, metrics, _ = proof(dataset)
    metric = json.loads(metrics.read_text())
    metric["per_class"]["id_card"]["recall"] = .99
    write_json(metrics, metric)
    with pytest.raises(TrainingGateError, match="metrics_recompute_mismatch"):
        gate(run, metrics)
    (run.parent / "checkpoint_best_total.pth").write_bytes(b"changed")
    with pytest.raises(TrainingGateError, match="checkpoint_hash_mismatch"):
        gate(run, metrics)


def test_single_false_positive_blocks_gate(dataset):
    run, metrics_path, predictions = proof(dataset)
    clean_id = next(r["sample_id"] for r in dataset["lock"]["records"]
                    if r["split"] == "holdout" and not r["objects"])
    next(p for p in predictions["images"] if p["sample_id"] == clean_id)["detections"] = [
        {"class_name": "id_card", "confidence": .99, "bbox": [1, 1, 4, 3]}]
    write_json(metrics_path.parent / "predictions.json", predictions)
    metrics = json.loads(metrics_path.read_text())
    metrics.update(score_predictions(dataset["lock"], predictions, dataset["config"]["thresholds"]))
    metrics["predictions_sha256"] = file_sha256(metrics_path.parent / "predictions.json")
    write_json(metrics_path, metrics)
    result = gate(run, metrics_path)
    assert not result["eligible"] and "heldout_false_positives" in result["reasons"]
    assert result["approved_classes"] == []


def test_missing_failed_or_duplicate_holdout_predictions_cannot_pass(dataset):
    _, _, predictions = proof(dataset)
    incomplete = copy.deepcopy(predictions)
    incomplete["images"].pop()
    with pytest.raises(TrainingGateError, match="prediction_coverage"):
        score_predictions(dataset["lock"], incomplete, dataset["config"]["thresholds"])
    predictions["images"][0]["complete"] = False
    with pytest.raises(TrainingGateError, match="prediction_incomplete"):
        score_predictions(dataset["lock"], predictions, dataset["config"]["thresholds"])


def test_mlflow_exports_only_allowlisted_local_provenance_and_scalars(dataset, monkeypatch):
    import sys
    from types import SimpleNamespace
    from training.specialists.track import track
    run_path, metrics_path, _ = proof(dataset)
    captured = {}
    class Context:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id="synthetic-mlflow-run"))
        def __exit__(self, *args):
            return False
    fake = SimpleNamespace(
        set_tracking_uri=lambda uri: captured.update(uri=uri),
        get_experiment_by_name=lambda _: None,
        create_experiment=lambda name, artifact_location: captured.update(artifact=artifact_location) or "1",
        start_run=lambda experiment_id: Context(),
        log_params=lambda values: captured.update(params=values),
        log_metrics=lambda values: captured.update(metrics=values),
        log_dict=lambda values, filename: captured.update(provenance=values, filename=filename),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    result = track(run_path, dataset["base"] / "mlflow", metrics_path)
    assert result["local_only"] and captured["uri"].startswith("sqlite:///")
    assert captured["artifact"].startswith("file:")
    assert captured["metrics"]["class.id_card.recall"] == 1
    assert "records" not in captured["provenance"] and "dataset_dir" not in captured["provenance"]


def test_mlflow_existing_remote_artifact_store_is_rejected(dataset, monkeypatch):
    import sys
    from types import SimpleNamespace
    from training.specialists.track import track
    run_path, _, _ = proof(dataset)
    fake = SimpleNamespace(set_tracking_uri=lambda _: None,
                           get_experiment_by_name=lambda _: SimpleNamespace(artifact_location="https://example.invalid"))
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    with pytest.raises(TrainingGateError, match="remote_artifact_store_forbidden"):
        track(run_path, dataset["base"] / "mlflow")
