"""Public pretraining contracts; no organizer/public payload or GPU is read."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from training.guards import TrainingGateError, file_sha256
from training.specialists.common import write_json
from training.specialists.public_train import (model_and_train_kwargs, preflight,
                                                claim_sealed_test,
                                                initialize_target_heads, parameter_sha256,
                                                fixed_detection_evaluation,
                                                remaining_seconds, scalar_metrics,
                                                sealed_test_view)


def prediction_fixture(tmp_path, monkeypatch, class_ids):
    image = tmp_path / "sealed_test" / "example.jpg"
    image.parent.mkdir()
    image.write_bytes(b"synthetic-predictor-does-not-decode")
    lock = {"class_names": {"0": "single_banknote", "1": "id_card"},
            "records": [{"sample_id": "example", "split": "sealed_test", "included": True,
                         "export_path": "sealed_test/example.jpg"}]}
    raw = SimpleNamespace(xyxy=[[1, 2, 11, 12] for _ in class_ids], class_id=class_ids,
                          confidence=[.9 for _ in class_ids])
    selected = SimpleNamespace(predict=lambda path, threshold: raw)
    monkeypatch.setattr("training.specialists.public_metrics.score_public_detections",
                        lambda lock, predictions: {"synthetic_metric": True})
    return selected, lock


def test_fixed_predictions_drop_only_documented_no_object_slot(tmp_path, monkeypatch):
    selected, lock = prediction_fixture(tmp_path, monkeypatch, [0, 2, 1, 2])
    metrics, predictions = fixed_detection_evaluation(selected, tmp_path, lock)
    assert metrics["background_proposals_excluded"] == 2
    assert [row["class_name"] for row in predictions[0]["detections"]] == ["single_banknote", "id_card"]


@pytest.mark.parametrize("invalid_id", [-1, 3, 99, 2.0, 2.5, "2", True])
def test_fixed_predictions_reject_other_or_noninteger_ids(tmp_path, monkeypatch, invalid_id):
    selected, lock = prediction_fixture(tmp_path, monkeypatch, [invalid_id])
    with pytest.raises(TrainingGateError, match="public_training_final_prediction_class_id"):
        fixed_detection_evaluation(selected, tmp_path, lock)


def config_fixture(tmp_path):
    weights = tmp_path / "pretrained.pth"
    weights.write_bytes(b"synthetic-test-not-weights")
    return {"schema_version": "forte.specialists.public-train.v1", "rfdetr_version": "1.11.2",
            "recipe_predeclared": True, "recipe_reference": "synthetic-predeclared-test",
            "model_class": "RFDETRNano", "freeze_encoder": True, "gradient_checkpointing": False,
            "amp_dtype": "bf16", "batch_size": 1, "eval_batch_size": 1, "grad_accum_steps": 8,
            "multi_scale": False, "use_ema": False, "epochs": 30, "resolution": 384,
            "seed": 1, "lr": .0001, "gpu_memory_budget_mib": 5632,
            "minimum_free_margin_mib": 1024, "max_fit_seconds": 14400, "evaluation_reserve_seconds": 1200,
            "test_exposure_registry_dir": str(tmp_path / "test-exposures"),
            "stop_at_utc": "2099-01-01T00:00:00Z", "dataset_dir": str(tmp_path / "dataset"),
            "expected_classes": ["single_banknote"],
            "train_data_policy": {"schema_version": "forte.specialists.public-augmentation.v1", "seed": 1,
                "class_balanced_sampling": True, "small_object_zoomout": {"probability": .8,
                    "reference_resolution": 512, "longest_box_min_px": 12, "longest_box_max_px": 96}},
            "expected_dataset_lock_sha256": "a" * 64, "pretrain_weights": str(weights),
            "expected_pretrain_weights_sha256": file_sha256(weights), "output_dir": str(tmp_path / "run"),
            "compute_release_authorization": {"released_by_A18_or_root": True, "release_reference": "synthetic-test"}}


@pytest.fixture
def fake_public_data(monkeypatch):
    lock = {"schema_version": "forte.specialists.public-dataset-lock.v1", "origin": "public",
            "classes": ["single_banknote"], "class_names": {"0": "single_banknote"}, "dataset_sha256": "b" * 64,
            "records": [{"split": "train", "included": True, "image_id": 1, "sample_id": "note",
                         "content_sha256": "c" * 64, "source_group": "note-source", "template_group": "note-template",
                         "objects": [{"class_name": "single_banknote"}]},
                        {"split": "train", "included": True, "image_id": 2, "sample_id": "coin",
                         "content_sha256": "d" * 64, "source_group": "coin-source", "template_group": "coin-template",
                         "kind": "coin", "objects": []}]}
    monkeypatch.setitem(sys.modules, "training.specialists.public_data",
                        SimpleNamespace(verify_public_dataset=lambda root, expected: lock))
    monkeypatch.setattr("training.specialists.public_train.load_public_dataset", lambda root, expected: lock)
    return lock


def test_preflight_accepts_only_separate_public_schema(tmp_path, fake_public_data):
    config = config_fixture(tmp_path)
    _, lock = preflight(config)
    assert lock["classes"] == ["single_banknote"]
    fake_public_data["schema_version"] = "forte.specialists.dataset-lock.v1"
    with pytest.raises(TrainingGateError, match="requires_public_data"):
        preflight(config)


@pytest.mark.parametrize("field,value,code", [
    ("recipe_predeclared", False, "recipe_not_predeclared"),
    ("freeze_encoder", False, "bounded_recipe_required"),
    ("amp_dtype", "fp8", "bounded_recipe_required"),
    ("batch_size", 2, "bounded_recipe_required"),
    ("multi_scale", True, "bounded_recipe_required"),
    ("gpu_memory_budget_mib", 7000, "memory_budget"),
    ("minimum_free_margin_mib", 0, "memory_margin"),
    ("resolution", 768, "resolution"),
    ("stop_at_utc", "2000-01-01T00:00:00Z", "deadline_elapsed"),
    ("device", "cuda:1", "device_must_be_local_zero"),
])
def test_unbounded_recipe_cannot_run(tmp_path, fake_public_data, field, value, code):
    config = config_fixture(tmp_path)
    config[field] = value
    with pytest.raises(TrainingGateError, match=code):
        preflight(config)


def test_pretrained_weights_hash_cannot_be_substituted(tmp_path, fake_public_data):
    config = config_fixture(tmp_path)
    Path(config["pretrain_weights"]).write_bytes(b"changed")
    with pytest.raises(TrainingGateError, match="pretrain_hash_mismatch"):
        preflight(config)


def test_stop_uses_explicit_utc_not_host_timezone():
    now = datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc)
    assert remaining_seconds({"stop_at_utc": "2026-10-18T06:01:00Z"}, now=now) == 60
    with pytest.raises(TrainingGateError, match="deadline_must_be_utc"):
        remaining_seconds({"stop_at_utc": "2026-10-18T11:01:00+05:00"}, now=now)


def test_upstream_constructor_and_train_kwargs_are_separate(tmp_path):
    config = config_fixture(tmp_path)
    model, training = model_and_train_kwargs(config, tmp_path / "data", tmp_path / "run", ["single_banknote"])
    assert model["freeze_encoder"] and model["num_classes"] == 1
    assert "freeze_encoder" not in training and "max_time" not in training
    assert training["grad_accum_steps"] == 8 and not training["run_test"]
    assert not any(training[k] for k in ("wandb", "tensorboard", "mlflow", "clearml"))


def test_scalar_metrics_do_not_export_private_strings_or_nonfinite_values():
    assert scalar_metrics({"loss": .3, "secret_path": "private", "nan": float("nan"), "inf": float("inf")}) == {"loss": .3}


def test_sealed_view_is_hash_bound_and_separate(tmp_path):
    root, output = tmp_path / "dataset", tmp_path / "output"
    (root / "sealed_test").mkdir(parents=True)
    output.mkdir()
    annotation = root / "sealed_test/_annotations.coco.json"
    write_json(annotation, {"categories": [{"id": 1, "name": "single_banknote"}], "images": [], "annotations": []})
    image = root / "sealed_test/synthetic.png"
    image.write_bytes(b"synthetic-not-opened")
    lock = {"classes": ["single_banknote"], "files": {
        "sealed_test/_annotations.coco.json": file_sha256(annotation),
        "sealed_test/synthetic.png": file_sha256(image)}}
    view = sealed_test_view(root, output, lock)
    assert (view / "test/synthetic.png").read_bytes() == image.read_bytes()
    assert (root / "sealed_test").is_dir() and not (root / "test").exists()
    with pytest.raises(TrainingGateError, match="sealed_view_already_exists"):
        sealed_test_view(root, output, lock)


def test_test_attempt_is_dataset_scoped_even_when_output_directory_changes(tmp_path):
    config = config_fixture(tmp_path)
    lock = {"files": {"sealed_test/synthetic.png": "b" * 64}}
    first = claim_sealed_test(config, lock, "c" * 64)
    assert first.is_file()
    config["output_dir"] = str(tmp_path / "different-output")
    with pytest.raises(TrainingGateError, match="sealed_test_already_exposed"):
        claim_sealed_test(config, lock, "d" * 64)


def test_target_classifier_semantics_are_reset_before_baseline():
    torch = pytest.importorskip("torch")
    heads = torch.nn.ModuleList([torch.nn.Linear(4, 3) for _ in range(2)])
    model = torch.nn.Module()
    model.class_embed = torch.nn.Linear(4, 3)
    model.transformer = torch.nn.Module()
    model.transformer.enc_out_class_embed = heads
    model.two_stage = True
    wrapper = SimpleNamespace(model=model)
    with torch.no_grad():
        model.class_embed.weight.fill_(99)
        for head in heads:
            head.weight.fill_(99)
    receipt = initialize_target_heads(wrapper, ["single_banknote", "id_card"])
    assert receipt["source_classifier_semantics"] == "discarded"
    assert not torch.equal(model.class_embed.weight, torch.full_like(model.class_embed.weight, 99))
    assert all(torch.equal(head.weight, model.class_embed.weight) for head in heads)
    assert model.class_names == ["single_banknote", "id_card"]


def test_nonfinite_parameter_cannot_satisfy_real_update_proof():
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(2, 2)
    with torch.no_grad():
        module.weight[0, 0] = float("nan")
    with pytest.raises(TrainingGateError, match="nonfinite_parameter"):
        parameter_sha256(module, trainable=True)


def test_parameter_proof_handles_scalar_and_bf16_parameter_bytes():
    torch = pytest.importorskip("torch")
    module = torch.nn.Module()
    module.scalar = torch.nn.Parameter(torch.tensor(1.0))
    module.vector = torch.nn.Parameter(torch.tensor([1, 2], dtype=torch.bfloat16))
    before = parameter_sha256(module, trainable=True)
    with torch.no_grad():
        module.scalar.add_(1)
    assert before != parameter_sha256(module, trainable=True)
