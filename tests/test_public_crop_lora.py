"""Synthetic safeguards only; these tests do not train or score a model."""
import builtins
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from training.guards import TrainingGateError, canonical_sha256
from training import public_crop_lora as crop
from training.specialists.common import file_sha256, write_json


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    config = {"model_type": "qwen3_vl", "architectures": [crop.MODEL_CLASS],
        "text_config": {"model_type": "qwen3_vl_text", "num_hidden_layers": 28, "hidden_size": 2048,
                        "num_attention_heads": 16, "num_key_value_heads": 8},
        "vision_config": {"patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}}
    write_json(model / "config.json", config)
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json"):
        write_json(model / name, {})
    (model / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"synthetic fixture, never load as weights")
    monkeypatch.setattr(crop, "WEIGHTS_SHA256", file_sha256(model / "model.safetensors"))
    manifest = tmp_path / "model-manifest.json"
    crop.snapshot(model, manifest)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows, files = [], {}
    for split in ("train", "validation", "sealed_test"):
        for n, cls in enumerate(crop.CLASSES):
            image = dataset / split / f"{cls}.png"
            image.parent.mkdir(exist_ok=True)
            Image.new("RGB", (80, 40), (10 + n, 20, len(split))).save(image)
            path, digest = image.relative_to(dataset).as_posix(), file_sha256(image)
            files[path] = digest
            row = {"sample_id": f"{split}-{cls}", "split": split, "included": True,
                "dataset_id": "dny3dgvvw8" if n == 0 else "MIDV-Holo", "export_path": path,
                "content_sha256": digest, "source_group": f"source-{split}-{n}",
                "template_group": f"template-{split}-{n}", "width": 80, "height": 40,
                "objects": [{"class_name": cls, "bbox": [1.2, 2.5, 70.1, 30.3]}]}
            rows.append(row)
            # Sealed pixels deliberately absent. Training preflight must not read them.
            if split == "sealed_test":
                image.unlink()
    lock = {"schema_version": "forte.specialists.public-dataset-lock.v1", "state": "frozen",
        "origin": "public", "dataset_kind": "combined_public", "classes": crop.CLASSES,
        "class_names": {"0": crop.CLASSES[0], "1": crop.CLASSES[1]}, "records": rows, "files": files}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(dataset / "dataset-lock.json", lock)
    cfg = {"schema_version": crop.SCHEMA, "model_id": crop.MODEL_ID, "revision": crop.REVISION,
        "runtime_image_digest": "sha256:" + "a" * 64,
        "model_dir": str(model), "model_manifest": str(manifest),
        "expected_model_manifest_sha256": file_sha256(manifest), "dataset_dir": str(dataset),
        "expected_dataset_lock_sha256": file_sha256(dataset / "dataset-lock.json"),
        "runtime_versions": crop.PINS, "recipe_predeclared": True, "recipe_reference": "synthetic-test-recipe",
        "verified_dataset_import_reference": "synthetic-fixture-not-publisher-data",
        "optimizer_steps": 80, "grad_accum_steps": 4, "seed": 1337, "lora_rank": 4, "lora_alpha": 8,
        "max_length": 256, "max_pixels": 65536, "train_per_class": 256, "validation_per_class": 16,
        "test_per_class": 32, "gpu_memory_budget_mib": 5120, "learning_rate": 1e-4,
        "max_fit_seconds": 3600, "evaluation_reserve_seconds": 900, "minimum_free_margin_mib": 1024,
        "train_short_side_curriculum": crop.SHORT_SIDES, "evaluation_short_sides": crop.SHORT_SIDES,
        "test_exposure_registry_dir": str(tmp_path / "registry"), "output_dir": str(tmp_path / "run"),
        "stop_at_utc": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()}
    return cfg, lock


def rewrite_lock(cfg, lock):
    lock.pop("dataset_sha256", None)
    lock["dataset_sha256"] = canonical_sha256(lock)
    path = Path(cfg["dataset_dir"]) / "dataset-lock.json"
    write_json(path, lock)
    cfg["expected_dataset_lock_sha256"] = file_sha256(path)


def test_preflight_never_imports_gpu_or_reads_sealed_pixels(inputs, monkeypatch):
    cfg, _ = inputs
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "bitsandbytes"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    checked = crop.preflight(cfg)
    assert set(checked["rows"]) == {"train", "validation"}
    assert {r["answer"] for r in checked["rows"]["train"]} == set(crop.CLASSES)
    assert checked["rows"]["train"][0]["crop_xyxy"] == [1, 2, 72, 33]


@pytest.mark.parametrize("field", ["source_group", "template_group", "scene_group", "combined_group", "physical_document_group"])
def test_split_leakage_rejected(inputs, field):
    cfg, lock = inputs
    lock["records"][0][field] = "leaked-group"
    lock["records"][2][field] = "leaked-group"
    rewrite_lock(cfg, lock)
    with pytest.raises(TrainingGateError, match="partition_leakage"):
        crop.preflight(cfg)


def test_organizer_or_own_dataset_cannot_enter(inputs):
    cfg, lock = inputs
    lock["origin"] = "own"
    rewrite_lock(cfg, lock)
    with pytest.raises(TrainingGateError, match="combined_public"):
        crop.preflight(cfg)


def test_modified_training_bytes_fail(inputs):
    cfg, lock = inputs
    (Path(cfg["dataset_dir"]) / lock["records"][0]["export_path"]).write_bytes(b"changed")
    with pytest.raises(TrainingGateError, match="image_hash"):
        crop.preflight(cfg)


@pytest.mark.parametrize("box", [[0, 0, 0, 5], [-1, 0, 5, 5], [0, 0, 81, 5], [0, 0, float("nan"), 1]])
def test_invalid_boxes_never_become_qa(inputs, box):
    _, lock = inputs
    lock["records"][0]["objects"][0]["bbox"] = box
    with pytest.raises(TrainingGateError, match="published_box"):
        crop.crop_rows(lock, "train", 16, 1337)


def test_no_procedural_or_unknown_class_targets(inputs):
    _, lock = inputs
    lock["records"][0]["objects"][0]["class_name"] = "no_authenticity_check"
    with pytest.raises(TrainingGateError, match="published_box"):
        crop.crop_rows(lock, "train", 16, 1337)


def test_fixed_selection_ignores_record_order(inputs):
    _, lock = inputs
    first = crop.crop_rows(lock, "train", 16, 1337)
    lock["records"].reverse()
    assert first == crop.crop_rows(lock, "train", 16, 1337)


@pytest.mark.parametrize("field,value", [("gpu_memory_budget_mib", 6000), ("max_pixels", 262144),
    ("lora_rank", 16), ("max_fit_seconds", 3601), ("train_short_side_curriculum", [0])])
def test_bounded_frozen_recipe(inputs, field, value):
    cfg, _ = inputs
    cfg[field] = value
    with pytest.raises(TrainingGateError):
        crop.preflight(cfg)


def test_compute_release_required_before_importing_gpu(inputs):
    cfg, _ = inputs
    checked = crop.preflight(cfg)
    with pytest.raises(TrainingGateError, match="compute_not_released"):
        crop._runtime(cfg, checked)


def test_full_fit_requires_memory_smoke_before_creating_output(inputs):
    cfg, _ = inputs
    with pytest.raises(TrainingGateError):
        crop.train(cfg)
    assert not Path(cfg["output_dir"]).exists()


def test_stress_downsamples_never_enlarges_or_reconstructs():
    class Processor:
        def __call__(self, **kwargs):
            return kwargs["images"][0].size
    wrapper = crop._ResizeProcessor(Processor())
    wrapper.short_side = 16
    assert wrapper(images=[Image.new("RGB", (160, 80))]) == (32, 16)
    assert wrapper(images=[Image.new("RGB", (8, 4))]) == (8, 4)


def test_non_square_processor_min_area_does_not_equal_max_budget():
    class Processor:
        def __call__(self, **kwargs):
            return kwargs["images_kwargs"]["size"]
    wrapper = crop._ResizeProcessor(Processor())
    original = {"size": {"shortest_edge": 65536, "longest_edge": 65536}}
    assert wrapper(images=[], images_kwargs=original) == {"shortest_edge": 1024, "longest_edge": 65536}
    assert original["size"]["shortest_edge"] == 65536


def test_final_eval_shared_registry_blocks_new_run_dirs(inputs, monkeypatch):
    cfg, lock = inputs
    from training.specialists.public_train import claim_sealed_test
    claim_sealed_test(cfg, lock, "a" * 64)
    output = Path(cfg["output_dir"])
    (output / "adapter").mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        (output / "adapter" / name).write_text("fixture", encoding="utf-8")
    write_json(output / "config.json", cfg)
    write_json(output / "evaluation-protocol.json", crop.evaluation_protocol(cfg))
    run = {"state": "trained_unselected", "schema_version": crop.RUN_SCHEMA,
        "config_sha256": file_sha256(output / "config.json"),
        "dataset_lock_sha256": cfg["expected_dataset_lock_sha256"],
        "evaluation_protocol_sha256": file_sha256(output / "evaluation-protocol.json"),
        "adapter_files": {p.name: file_sha256(p) for p in (output / "adapter").iterdir()}}
    write_json(output / "run.json", run)
    with pytest.raises(TrainingGateError, match="already_exposed"):
        crop.final_evaluate(cfg, file_sha256(output / "run.json"))
    assert not (output / "sealed-test-access.json").exists()


def test_changed_snapshot_cannot_pass_inventory(inputs):
    cfg, _ = inputs
    (Path(cfg["model_dir"]) / "tokenizer.json").write_text("modified", encoding="utf-8")
    with pytest.raises(TrainingGateError, match="snapshot_hash"):
        crop.preflight(cfg)


def test_exhausted_deadline_rejected(inputs):
    cfg, _ = inputs
    cfg["stop_at_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with pytest.raises(TrainingGateError, match="deadline_reserve"):
        crop.preflight(cfg)


def test_cohort_eval_claims_before_test_pixels_preserves_marker_and_scores_only_adapter(inputs, monkeypatch):
    cfg, _ = inputs
    output = Path(cfg["output_dir"])
    (output / "adapter").mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        (output / "adapter" / name).write_text("fixture", encoding="utf-8")
    write_json(output / "config.json", cfg)
    write_json(output / "evaluation-protocol.json", crop.evaluation_protocol(cfg))
    run = {"state": "trained_unselected", "schema_version": crop.RUN_SCHEMA,
        "config_sha256": file_sha256(output / "config.json"),
        "dataset_lock_sha256": cfg["expected_dataset_lock_sha256"],
        "evaluation_protocol_sha256": file_sha256(output / "evaluation-protocol.json"),
        "adapter_files": {p.name: file_sha256(p) for p in (output / "adapter").iterdir()}}
    write_json(output / "run.json", run)
    from training.specialists import public_cohort
    events = []
    marker = output.parent / "immutable-attempt.json"
    marker.write_text('{"state":"attempted"}', encoding="utf-8")
    marker_before = marker.read_bytes()
    def claim(*args, **kwargs):
        events.append("claim")
        return marker
    monkeypatch.setattr(public_cohort, "claim_cohort_member", claim)
    monkeypatch.setattr(public_cohort, "verify_cohort", lambda *args: events.append("reverify"))
    real_verify = crop.verify_crop_files
    def verify(root, rows):
        if rows[0]["split"] == "sealed_test":
            assert "claim" in events
            events.append("test_pixels")
        else:
            real_verify(root, rows)
    monkeypatch.setattr(crop, "verify_crop_files", verify)
    monkeypatch.setattr(crop, "_runtime", lambda *args: ("base", "processor"))
    monkeypatch.setattr(crop, "PublicCropCollator", lambda *args, **kwargs: "collator")
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: "adapter")))
    def evaluate(model, *args):
        assert model == "adapter"
        events.append("adapter_evaluation")
        return {"macro_accuracy": 0.5, "synthetic_fixture": True}
    monkeypatch.setattr(crop, "stress_evaluate", evaluate)
    result = crop.final_evaluate(cfg, file_sha256(output / "run.json"),
        cohort_receipt=output.parent / "cohort.json", expected_cohort_sha256="a" * 64)
    assert result["adapter_accuracy"] == 0.5
    assert events == ["claim", "test_pixels", "reverify", "reverify", "adapter_evaluation"]
    assert marker.read_bytes() == marker_before


def test_adapter_config_tamper_blocks_descriptor_before_cohort_reservation(inputs):
    cfg, _ = inputs
    output = Path(cfg["output_dir"])
    (output / "adapter").mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        (output / "adapter" / name).write_text("fixture", encoding="utf-8")
    run = {"state": "trained_unselected", "adapter_files": {
        p.name: file_sha256(p) for p in (output / "adapter").iterdir()}}
    write_json(output / "run.json", run)
    (output / "adapter" / "adapter_config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(TrainingGateError, match="adapter_changed"):
        crop.cohort_descriptor(cfg, file_sha256(output / "run.json"))


def test_balanced_training_draws_do_not_hide_small_id_support():
    rows = [{"example_id": f"cash-{i}", "answer": "single_banknote"} for i in range(256)]
    rows += [{"example_id": f"id-{i}", "answer": "id_card"} for i in range(36)]
    draws = crop.balanced_draws(rows, 320, 1337)
    assert draws == crop.balanced_draws(rows, 320, 1337)
    assert [row["answer"] for row in draws] == crop.CLASSES * 160
    assert len({row["example_id"] for row in draws if row["answer"] == "id_card"}) == 36
    for side in crop.SHORT_SIDES:
        labels = [r["answer"] for i, r in enumerate(draws) if crop.SHORT_SIDES[(i // 2) % 4] == side]
        assert labels.count("id_card") == labels.count("single_banknote") == 40


def test_failure_diagnostic_bounds_chain_and_omits_exception_values():
    secret = "private-prompt-or-token-should-never-be-recorded"
    def fail(depth):
        if depth:
            return fail(depth - 1)
        raise RuntimeError(secret)
    try:
        try:
            fail(20)
        except RuntimeError as inner:
            raise AttributeError(secret, name="missing_runtime_symbol") from inner
    except AttributeError as exc:
        diagnostic = crop.failure_diagnostic(exc, "runtime_quantized_model_load")
    assert diagnostic["phase"] == "runtime_quantized_model_load"
    assert [entry["error_class"] for entry in diagnostic["exceptions"]] == ["AttributeError", "RuntimeError"]
    assert diagnostic["exceptions"][0]["symbol"] == "missing_runtime_symbol"
    assert len(diagnostic["exceptions"][1]["frames"]) == 8
    assert secret not in json.dumps(diagnostic)
    for entry in diagnostic["exceptions"]:
        for frame in entry["frames"]:
            assert set(frame) == {"file", "function", "line"}
            assert "/" not in frame["file"] and "\\" not in frame["file"]


def test_runtime_failure_persists_phase_without_claiming_optimizer_proof(inputs, monkeypatch):
    cfg, _ = inputs
    def fail_runtime(config, checked, *, progress):
        progress("runtime_quantized_model_load")
        raise RuntimeError("private-exception-value")
    monkeypatch.setattr(crop, "_runtime", fail_runtime)
    with pytest.raises(RuntimeError, match="private-exception-value"):
        crop.train(cfg, memory_smoke=True)
    output = Path(cfg["output_dir"])
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["state"] == "failed" and run["optimizer_steps"] == 0
    assert run["phase"] == run["failure_diagnostic"]["phase"] == "runtime_quantized_model_load"
    assert run["failure_diagnostic"]["exceptions"][0]["error_class"] == "RuntimeError"
    assert "private-exception-value" not in json.dumps(run)
    assert not (output / "step-proof.json").exists()
