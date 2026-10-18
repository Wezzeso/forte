"""Fake-model API rehearsal tests; no real model, GPU, or media payload is used."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from training.guards import TrainingGateError
from training.specialists.common import canonical_sha256, file_sha256, write_json
from training.specialists import public_dev_validate as dev
from training.specialists.public_bundle import verify_fit_bundle


@pytest.fixture
def rehearsal(tmp_path, monkeypatch):
    root, run_dir = tmp_path / "bundle", tmp_path / "run"
    root.mkdir()
    run_dir.mkdir()
    records, files = [], {}
    for role, part in (("train", "train"), ("validation", "valid"), ("sealed_test", "sealed_test")):
        (root / part).mkdir()
        relative = part + "/synthetic.jpg"
        if role != "sealed_test":
            (root / relative).write_bytes(b"synthetic-not-a-media-payload")
            files[relative] = file_sha256(root / relative)
        else:
            # Deliberately absent: even stat/open of a sealed pixel would fail.
            files[relative] = "a" * 64
        records.append({"sample_id": role, "split": role, "included": True, "origin": "public",
                        "export_path": relative, "content_sha256": files[relative],
                        "source_group": role, "template_group": role, "image_id": 1,
                        "width": 20, "height": 10,
                        "objects": [{"class_name": "single_banknote", "bbox": [0, 0, 10, 5]}]})
        annotation = part + "/_annotations.coco.json"
        write_json(root / annotation, {"images": [{"id": 1, "file_name": "synthetic.jpg", "width": 20, "height": 10}],
                                      "annotations": [], "categories": [{"id": 1, "name": "single_banknote"}]})
        files[annotation] = file_sha256(root / annotation)
    # Unique fixture bytes avoid exact-content leakage across train and valid.
    (root / "valid/synthetic.jpg").write_bytes(b"different-synthetic-not-media")
    files["valid/synthetic.jpg"] = file_sha256(root / "valid/synthetic.jpg")
    records[1]["content_sha256"] = files["valid/synthetic.jpg"]
    lock = {"schema_version": "forte.specialists.public-dataset-lock.v1", "state": "frozen", "origin": "public",
            "classes": ["single_banknote"], "class_names": {"0": "single_banknote"},
            "prediction_class_ids": {"single_banknote": 0}, "coco_category_ids": {"single_banknote": 1},
            "files": files, "records": records}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(root / "dataset-lock.json", lock)
    config = {"dataset_dir": str(root), "expected_dataset_lock_sha256": file_sha256(root / "dataset-lock.json"),
              "stop_at_utc": "2099-01-01T00:00:00Z", "gpu_memory_budget_mib": 5632,
              "minimum_free_margin_mib": 1024}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "evaluation-protocol.json", {"scope": "synthetic-protocol"})
    checkpoint = run_dir / "checkpoint_best_total.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint-not-a-model")
    run = {"schema_version": "forte.specialists.public-run.v1", "state": "trained_unselected_pending_test",
           "defer_sealed_test": True, "test_evaluation_count": 0, "serving_eligible": False,
           "rfdetr_version": "1.11.2", "class_names": lock["class_names"],
           "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
           "config_sha256": file_sha256(run_dir / "config.json"), "checkpoint": checkpoint.name,
           "checkpoint_sha256": file_sha256(checkpoint),
           "evaluation_protocol_sha256": file_sha256(run_dir / "evaluation-protocol.json")}
    run_path = run_dir / "run.json"
    write_json(run_path, run)
    calls = {"images": [], "from_checkpoint": [], "evaluate": [], "predict": [], "cap": []}

    class FakeImage:
        size = (20, 10)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def verify(self):
            return None

    def open_image(path):
        path = Path(path)
        assert path.parent.name in {"train", "valid"}, "sealed pixels must never be inspected"
        calls["images"].append(path)
        return FakeImage()

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace(open=open_image)))

    def guard(config_value):
        dev.remaining_seconds(config_value)
        bundle = Path(config_value["dataset_dir"])
        return bundle, verify_fit_bundle(bundle, config_value["expected_dataset_lock_sha256"])

    monkeypatch.setattr(dev, "preflight", guard)
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                           mem_get_info=lambda index: (8 * 1024**3, 80 * 1024**3),
                           set_per_process_memory_fraction=lambda fraction, index: calls["cap"].append((fraction, index)),
                           reset_peak_memory_stats=lambda index: None, synchronize=lambda: None,
                           max_memory_allocated=lambda index: 1024**3, max_memory_reserved=lambda index: 2 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(dev.importlib.metadata, "version", lambda name: "1.11.2")

    class FakeSelected:
        class_names = ["single_banknote"]
        native_result = {"val/mAP_50_95": .4, "val/mAP_50": .6}
        move_on_predict = True
        def __init__(self):
            self.tensor = SimpleNamespace(device="cpu")
            self.buffer = SimpleNamespace(device="cpu")
            self.model = SimpleNamespace(device="cuda:0", model=SimpleNamespace(
                parameters=lambda: [self.tensor], buffers=lambda: [self.buffer]))
        def evaluate(self, **kwargs):
            calls["evaluate"].append(kwargs)
            assert kwargs["split"] == "val"
            return self.native_result
        def predict(self, path, *, threshold):
            assert Path(path).parent.name == "valid", "fixed rehearsal must use development pixels only"
            calls["predict"].append((path, threshold))
            if self.move_on_predict:
                self.tensor.device = self.buffer.device = "cuda:0"
            return SimpleNamespace(class_id=[0], confidence=[.8], xyxy=[[0, 0, 10, 5]])

    model = FakeSelected()
    def from_checkpoint(path, **kwargs):
        calls["from_checkpoint"].append((path, kwargs))
        return model
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETR=SimpleNamespace(from_checkpoint=from_checkpoint)))
    return SimpleNamespace(root=root, run=run, run_dir=run_dir, run_path=run_path,
                           run_sha=file_sha256(run_path), output=tmp_path / "development-rehearsal",
                           lock=lock, config=config, checkpoint=checkpoint, calls=calls, model=model)


def test_native_and_fixed_rehearsal_only_access_development_and_keep_run_immutable(rehearsal):
    fixture = rehearsal
    original_run = fixture.run_path.read_bytes()
    result = dev.validate_development(fixture.run_path, fixture.run_sha, fixture.output)
    receipt = json.loads((fixture.output / "receipt.json").read_text())

    assert result["ok"] and result["scope"] == "development_only" and not result["serving_eligible"]
    assert fixture.run_path.read_bytes() == original_run
    assert result["test_evaluation_count"] == receipt["test_evaluation_count"] == 0
    assert receipt["sealed_pixels_accessed"] is False and receipt["api_shape_errors"] == []
    assert receipt["native_selected_dev_metrics"]["val/mAP_50_95"] == .4
    assert receipt["fixed_development_metrics"]["split"] == "validation"
    assert receipt["fixed_development_metrics"]["scope"] == "development_only"
    assert receipt["fixed_development_metrics"]["per_class"]["single_banknote"]["tp"] == 1
    assert fixture.calls["from_checkpoint"] == [(str(fixture.checkpoint), {"device": "cuda:0"})]
    assert fixture.calls["predict"] == [(str(fixture.root / "valid/synthetic.jpg"), .1)]
    native = fixture.calls["evaluate"][0]
    assert native == {"dataset_dir": str(fixture.root), "output_dir": str(fixture.output / "native-dev"),
                      "split": "val", "device": "cuda:0", "batch_size": 1, "eval_batch_size": 1,
                      "amp_dtype": "bf16", "num_workers": 0, "multi_scale": False,
                      "eval_max_dets": 500, "log_per_class_metrics": True}
    assert fixture.calls["cap"] == [(5632 * 1024**2 / (80 * 1024**3), 0)]
    assert receipt["intended_prediction_device"] == "cuda:0"
    snapshots = receipt["gpu_device_snapshots"]
    assert [row["phase"] for row in snapshots] == ["before_native_evaluate", "after_native_evaluate", "after_first_fixed_predict"]
    assert snapshots[0]["parameter_devices"] == snapshots[1]["parameter_devices"] == ["cpu"]
    assert snapshots[2]["parameter_devices"] == snapshots[2]["buffer_devices"] == ["cuda:0"]
    assert not (fixture.root / "sealed_test/synthetic.jpg").exists()


def test_metadata_view_filters_to_included_validation_and_preserves_paths(rehearsal):
    before = copy.deepcopy(rehearsal.lock)
    rehearsal.lock["records"].append({"split": "validation", "included": False, "export_path": "valid/excluded.jpg"})
    view = dev.development_view(rehearsal.lock)
    assert len(view["records"]) == 1
    assert view["records"][0]["sample_id"] == "validation"
    assert view["records"][0]["split"] == "sealed_test"
    assert view["records"][0]["export_path"] == "valid/synthetic.jpg"
    assert all(path.startswith("valid/") for path in view["files"])
    assert rehearsal.lock["records"][:3] == before["records"]


@pytest.mark.parametrize("path", ["sealed_test/synthetic.jpg", "../sealed_test/synthetic.jpg", "/valid/a.jpg", "valid\\a.jpg"])
def test_validation_metadata_cannot_redirect_to_sealed_or_external_pixels(rehearsal, path):
    rehearsal.lock["records"][1]["export_path"] = path
    with pytest.raises(TrainingGateError, match="public_dev_image_path_not_validation"):
        dev.development_view(rehearsal.lock)


@pytest.mark.parametrize("artifact,code", [("run.json", "specialist_input_hash_mismatch"),
                                           ("config.json", "specialist_input_hash_mismatch"),
                                           ("checkpoint_best_total.pth", "public_dev_checkpoint_changed"),
                                           ("evaluation-protocol.json", "public_dev_protocol_changed")])
def test_changed_immutable_artifacts_fail_before_model_loading(rehearsal, artifact, code):
    (rehearsal.run_dir / artifact).write_bytes(b"changed")
    with pytest.raises(TrainingGateError, match=code):
        dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    assert not rehearsal.calls["from_checkpoint"]


@pytest.mark.parametrize("native,code", [([], "native_metrics_shape_invalid"), ({}, "native_metrics_shape_invalid"),
                                         ({"val/mAP": float("nan")}, "native_metrics_values_invalid"),
                                         ({"test/mAP": .2}, "native_metrics_values_invalid")])
def test_native_api_shape_errors_are_recorded_without_fixed_or_test_predictions(rehearsal, native, code):
    rehearsal.model.native_result = native
    with pytest.raises(TrainingGateError, match="public_dev_" + code):
        dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    receipt = json.loads((rehearsal.output / "receipt.json").read_text())
    assert receipt["state"] == "failed" and receipt["scope"] == "development_only"
    assert receipt["failure_stage"] == "native_evaluate_val"
    assert receipt["api_shape_errors"][0]["code"] == "public_dev_" + code
    assert not rehearsal.calls["predict"] and receipt["test_evaluation_count"] == 0


def test_low_gpu_headroom_stops_before_checkpoint_load(rehearsal, monkeypatch):
    sys.modules["torch"].cuda.mem_get_info = lambda index: (6 * 1024**3, 80 * 1024**3)
    with pytest.raises(TrainingGateError, match="public_dev_shared_gpu_headroom_insufficient"):
        dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    assert not rehearsal.calls["from_checkpoint"]


def test_native_failure_does_not_write_private_exception_text(rehearsal, monkeypatch):
    def fail(**kwargs):
        raise TypeError("synthetic secret /private/example.mov")
    monkeypatch.setattr(rehearsal.model, "evaluate", fail)
    with pytest.raises(TypeError):
        dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    receipt = (rehearsal.output / "receipt.json").read_text()
    assert "/private/example.mov" not in receipt and "public_dev_api_execution_failed" in receipt


def test_output_must_be_fresh(rehearsal):
    rehearsal.output.mkdir()
    with pytest.raises(TrainingGateError, match="public_dev_output_not_new_or_inside_dataset"):
        dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    assert not rehearsal.calls["from_checkpoint"]


def test_cpu_parameters_after_first_prediction_are_explicit_failure(rehearsal):
    rehearsal.model.move_on_predict = False
    with pytest.raises(TrainingGateError, match="public_dev_first_prediction_not_cuda_zero"):
        dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    receipt = json.loads((rehearsal.output / "receipt.json").read_text())
    assert receipt["failure_stage"] == "fixed_detection_evaluation"
    assert receipt["gpu_device_snapshots"][-1]["parameter_devices"] == ["cpu"]
    assert receipt["test_evaluation_count"] == 0


def test_explicit_cpu_wrapper_target_fails_before_evaluation(rehearsal):
    rehearsal.model.model.device = "cpu"
    with pytest.raises(TrainingGateError, match="public_dev_wrapper_target_not_cuda_zero"):
        dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    assert not rehearsal.calls["evaluate"] and not rehearsal.calls["predict"]


def test_existing_native_and_fixed_calls_record_synchronized_phase_wall_time(rehearsal, monkeypatch):
    clock = iter([100.0, 110.0, 112.0, 120.0, 124.0, 130.0])
    monkeypatch.setattr(dev.time, "monotonic", lambda: next(clock))
    synchronizations = []
    sys.modules["torch"].cuda.synchronize = lambda: synchronizations.append(True)

    dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    receipt = json.loads((rehearsal.output / "receipt.json").read_text())

    assert receipt["phase_timings"] == {
        "native_evaluate_val": {"wall_seconds": 2.0, "development_images": 1, "images_per_second": .5},
        "fixed_detection_evaluation": {"wall_seconds": 4.0, "development_images": 1, "images_per_second": .25}}
    assert len(synchronizations) == 4
    assert len(rehearsal.calls["evaluate"]) == len(rehearsal.calls["predict"]) == 1
    assert len(rehearsal.calls["from_checkpoint"]) == 1
    assert receipt["elapsed_s"] == 30.0


def test_zero_duration_phase_records_zero_finite_throughput(rehearsal, monkeypatch):
    clock = iter([100.0, 110.0, 110.0, 120.0, 120.0, 130.0])
    monkeypatch.setattr(dev.time, "monotonic", lambda: next(clock))

    dev.validate_development(rehearsal.run_path, rehearsal.run_sha, rehearsal.output)
    receipt = json.loads((rehearsal.output / "receipt.json").read_text())

    assert all(phase == {"wall_seconds": 0.0, "development_images": 1, "images_per_second": 0.0}
               for phase in receipt["phase_timings"].values())
    assert len(rehearsal.calls["evaluate"]) == len(rehearsal.calls["predict"]) == 1


@pytest.mark.parametrize("duration", [-1.0, float("inf"), float("nan"), True])
def test_invalid_phase_durations_cannot_enter_receipt(duration):
    with pytest.raises(TrainingGateError, match="public_dev_phase_timing_invalid"):
        dev.phase_timing(duration, 1)
