"""Synthetic boundary tests only: no model downloads or GPU-training claims."""
from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from training.guards import TrainingGateError
from training.specialist_lora import (CropQACollator, LINK_FIELDS, MODEL_CLASS, MODEL_ID,
    MODEL_TYPE, PINS, inventory_targets, preflight, resolve_config, train,
    validate_crop_manifest, verify_model)
from training.specialists.common import file_sha256, write_json
from training.specialists.prepare import prepare


@pytest.fixture
def crop_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    inventory = tmp_path / "organizer.json"
    write_json(inventory, [{"content_sha256": "a" * 64}])
    sources = []
    for n, split in enumerate(("train", "validation", "holdout")):
        recording = source / f"{split}.mov"
        recording.write_bytes(("synthetic-recording-fixture-" + split).encode())
        image = source / f"{split}.png"
        Image.new("RGB", (16, 16), (n + 1, 3, 4)).save(image)
        sources.append({"sample_id": split, "path": image.name,
            "content_sha256": file_sha256(image), "source_recording_path": recording.name,
            "source_recording_sha256": file_sha256(recording), "origin": "own", "split": split,
            "scenario_group": "group-" + split, "camera_id": "camera-" + split,
            "width": 16, "height": 16, "labels_reviewed": True, "exhaustive_labels": True,
            "label_reviewer": "synthetic-test", "label_reviewed_at": "2026-10-18",
            "provenance": {"reference": "synthetic-test-not-real-consent", "commercial_use_allowed": True,
                           "participants_consented": True, "recording_authorized": True},
            "objects": [{"class_name": "id_card", "bbox": [1, 1, 5, 4]},
                        {"class_name": "key_in_lock", "bbox": [9, 9, 2, 3]}]})
    manifest = {"schema_version": "forte.specialists.data.v1", "frozen": True,
        "reviewed_by": "synthetic-test", "reviewed_at": "2026-10-18", "media_root": str(source),
        "organizer_inventory": {"path": str(inventory), "sha256": file_sha256(inventory)},
        "classes": ["id_card", "key_in_lock"], "records": sources}
    source_manifest = tmp_path / "source-manifest.json"
    write_json(source_manifest, manifest)
    dataset = tmp_path / "dataset"
    prepared = prepare(source_manifest, dataset)
    lock = json.loads((dataset / "dataset-lock.json").read_text())
    qa = {"schema_version": "forte.crop-lora.data.v1", "frozen": True,
          "reviewed_by": "synthetic-test", "reviewed_at": "2026-10-18",
          "dataset_lock_sha256": prepared["dataset_lock_sha256"], "records": []}
    for row in lock["records"]:
        if row["split"] == "holdout":
            continue
        qa["records"].append({"example_id": "qa-" + row["sample_id"], "source_sample_id": row["sample_id"],
            **{k: row[k] for k in LINK_FIELDS}, "crop_xyxy": [1, 1, 8, 7],
            "question": "Удостоверение или банкноты?", "answer": "Удостоверение.",
            "human_reviewed": True, "reviewer": "synthetic-test", "reviewed_at": "2026-10-18"})
    qa_path = tmp_path / "crop-qa.json"
    write_json(qa_path, qa)
    model = tmp_path / "model"
    model.mkdir()
    model_config = {"architectures": [MODEL_CLASS], "model_type": MODEL_TYPE,
        "text_config": {"model_type": "qwen3_vl_moe_text", "hidden_size": 2048,
                        "num_hidden_layers": 48, "num_experts": 128, "num_experts_per_tok": 8},
        "image_token_id": 101, "video_token_id": 102, "vision_start_token_id": 103,
        "vision_end_token_id": 104,
        "vision_config": {"patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}}
    write_json(model / "config.json", model_config)
    for name in ("tokenizer_config.json", "tokenizer.json", "preprocessor_config.json", "chat_template.json"):
        write_json(model / name, {})
    shard = "model-00001-of-00001.safetensors"
    (model / shard).write_bytes(b"synthetic-boundary-fixture-not-real-model-weights")
    write_json(model / "model.safetensors.index.json", {"weight_map": {"test_weight": shard}})
    snapshot = {"schema_version": "forte.crop-lora.snapshot.v1", "model_id": MODEL_ID,
        "revision": "b" * 40, "processor_revision": "b" * 40,
        "license": "apache-2.0", "license_verified": True, "reviewed_by": "synthetic-test",
        "reviewed_at": "2026-10-18", "files": {p.name: file_sha256(p) for p in model.iterdir()}}
    snapshot_path = tmp_path / "snapshot.json"
    write_json(snapshot_path, snapshot)
    config = {"schema_version": "forte.crop-lora.train.v1", "model_id": MODEL_ID, "model_type": MODEL_TYPE,
        "model_dir": str(model), "model_manifest": str(snapshot_path),
        "expected_model_manifest_sha256": file_sha256(snapshot_path), "base_revision": "b" * 40,
        "processor_revision": "b" * 40, "dataset_dir": str(dataset),
        "expected_dataset_lock_sha256": prepared["dataset_lock_sha256"], "crop_manifest": str(qa_path),
        "expected_crop_manifest_sha256": file_sha256(qa_path), "output_dir": str(tmp_path / "run"),
        "runtime_versions": dict(PINS), "optimizer_steps": 1, "grad_accum_steps": 1,
        "learning_rate": .0001, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .05,
        "max_length": 4096, "max_pixels": 262144, "minimum_free_cuda_gib": 72,
        "seed": 17, "merge_adapter": False}
    return SimpleNamespace(config=config, qa=qa, lock=lock, model=model, model_config=model_config,
                           base=tmp_path, dataset=dataset, snapshot=snapshot)


def test_preflight_imports_no_cuda_libraries_and_writes_nothing(crop_inputs, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "accelerate"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    checked = preflight(crop_inputs.config)
    assert {r["split"] for r in checked["rows"]} == {"train", "validation"}
    assert not (crop_inputs.base / "run").exists()


@pytest.mark.parametrize("change,code", [
    (lambda c: c.update(model_id="Qwen/Qwen3.6-27B"), "only_qwen3_vl_30b_moe"),
    (lambda c: c.update(model_type="qwen3_vl"), "only_qwen3_vl_30b_moe"),
    (lambda c: c.update(base_revision="main"), "immutable_revision"),
    (lambda c: c.update(processor_revision="c" * 40), "immutable_revision"),
    (lambda c: c["runtime_versions"].update(transformers="latest"), "runtime_pins"),
    (lambda c: c.update(optimizer_steps=0), "training_parameter"),
    (lambda c: c.update(learning_rate=float("nan")), "training_parameter"),
    (lambda c: c.update(merge_adapter="yes"), "merge_choice"),
    (lambda c: c.update(max_pixels=512), "pixel_budget_below_one_visual_token"),
])
def test_config_guards(crop_inputs, change, code):
    config = copy.deepcopy(crop_inputs.config)
    change(config)
    with pytest.raises(TrainingGateError, match=code):
        preflight(config)


def test_snapshot_hashes_and_uninventoried_files_are_rejected(crop_inputs):
    extra = crop_inputs.model / "chat_template.jinja"
    extra.write_text("unverified template")
    with pytest.raises(TrainingGateError, match="uninventoried"):
        verify_model(crop_inputs.config)
    extra.unlink()
    (crop_inputs.model / "tokenizer.json").write_text("changed")
    with pytest.raises(TrainingGateError, match="snapshot_hash"):
        verify_model(crop_inputs.config)


def test_declared_dense_architecture_cannot_be_substituted(crop_inputs):
    cfg = copy.deepcopy(crop_inputs.model_config)
    cfg["model_type"] = "qwen3_vl"
    write_json(crop_inputs.model / "config.json", cfg)
    crop_inputs.snapshot["files"]["config.json"] = file_sha256(crop_inputs.model / "config.json")
    write_json(Path(crop_inputs.config["model_manifest"]), crop_inputs.snapshot)
    crop_inputs.config["expected_model_manifest_sha256"] = file_sha256(Path(crop_inputs.config["model_manifest"]))
    with pytest.raises(TrainingGateError, match="architecture_mismatch"):
        verify_model(crop_inputs.config)


@pytest.mark.parametrize("field,value,code", [
    ("human_reviewed", False, "human_qa_review"),
    ("camera_id", "wrong-camera", "source_lineage"),
    ("scenario_group", "wrong-group", "source_lineage"),
    ("split", "validation", "source_lineage"),
    ("origin", "organizer", "source_lineage"),
    ("crop_xyxy", [0, 0, 17, 16], "outside_image"),
    ("crop_xyxy", [True, 1, 3, 3], "crop_box_invalid"),
    ("answer", "<|image_pad|>", "question_or_answer"),
    ("answer", " ", "question_or_answer"),
])
def test_crop_review_lineage_and_bounds(crop_inputs, field, value, code):
    manifest = copy.deepcopy(crop_inputs.qa)
    manifest["records"][0][field] = value
    with pytest.raises(TrainingGateError, match=code):
        validate_crop_manifest(manifest, crop_inputs.lock, crop_inputs.config["expected_dataset_lock_sha256"])


@pytest.mark.parametrize("origin,split", [("own", "holdout"), ("organizer", "organizer_eval")])
def test_eval_samples_never_enter_lora(crop_inputs, origin, split):
    lock = copy.deepcopy(crop_inputs.lock)
    lock["records"][0].update(origin=origin, split=split)
    qa = copy.deepcopy(crop_inputs.qa)
    qa["records"][0].update(origin=origin, split=split)
    with pytest.raises(TrainingGateError, match="evaluation_source_forbidden"):
        validate_crop_manifest(qa, lock, crop_inputs.config["expected_dataset_lock_sha256"])


def test_exact_language_attention_target_inventory_excludes_vision_and_experts():
    names = [f"model.language_model.layers.{i}.self_attn.{p}"
             for i in range(2) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    model = SimpleNamespace(named_modules=lambda: [(n, None) for n in names + [
        "model.visual.blocks.0.attn.qkv", "model.language_model.layers.0.mlp.experts.0.up_proj",
        "model.language_model.layers.0.mlp.gate"]])
    assert set(inventory_targets(model, 2)) == set(names)
    with pytest.raises(TrainingGateError, match="inventory_mismatch"):
        inventory_targets(model, 3)


class FakeVector:
    def __init__(self, values):
        self.values = values
    def tolist(self):
        return self.values


class FakeVisualTensor:
    def __init__(self, values, shape):
        self.values, self.shape, self.ndim = values, shape, len(shape)
    def tolist(self):
        return self.values
    def equal(self, other):
        return self.values == other.values and self.shape == other.shape


class FakeProcessor:
    def __init__(self, *, mismatch=False, visual_in_answer=False, visual_fault=None):
        self.mismatch, self.visual_in_answer = mismatch, visual_in_answer
        self.visual_fault = visual_fault
        self.seen_sizes = []
        self.image_processor = SimpleNamespace(size={"shortest_edge": 65536})
    def apply_chat_template(self, messages, **kwargs):
        return "prompt" if len(messages) == 1 else "promptanswer"
    def __call__(self, *, text, images, padding, truncation, return_tensors, images_kwargs):
        assert not padding and not truncation and return_tensors == "pt"
        assert images_kwargs["size"] == {"longest_edge": 262144, "shortest_edge": 65536}
        self.seen_sizes.append(images[0].size)
        full = text[0] == "promptanswer"
        ids = [1, 103, 101, 104, 4]
        if full:
            ids += [101 if self.visual_in_answer else 9, 10]
            if self.mismatch:
                ids[0] = 99
        grid, shape, pixel_values = [[1, 2, 2]], (4, 1536), [1, 2, 3]
        if self.visual_fault == "changed_pixels" and full:
            pixel_values = [7, 8, 9]
        elif self.visual_fault == "oversize":
            grid = [[1, 64, 64]]
        elif self.visual_fault == "bad_pixel_rows":
            shape = (3, 1536)
        elif self.visual_fault == "bad_pixel_width":
            shape = (4, 768)
        elif self.visual_fault == "token_count":
            grid, shape = [[1, 4, 4]], (16, 1536)
        return {"input_ids": [FakeVector(ids)], "attention_mask": [FakeVector([1] * len(ids))],
                "pixel_values": FakeVisualTensor(pixel_values, shape),
                "image_grid_thw": FakeVisualTensor(grid, (1, 3))}


def test_collator_only_supervises_answer_and_reads_exact_locked_crop(crop_inputs):
    checked = preflight(crop_inputs.config)
    processor = FakeProcessor()
    collator = CropQACollator(processor, crop_inputs.dataset, crop_inputs.model_config,
                             max_length=10, tensor_factory=lambda x: x)
    batch = collator([checked["rows"][0]])
    assert batch["labels"] == [[-100] * 5 + [9, 10]]
    assert processor.seen_sizes == [(7, 6), (7, 6)]


@pytest.mark.parametrize("options,length,code", [
    ({"mismatch": True}, 10, "prefix_alignment"),
    ({"visual_in_answer": True}, 10, "completion_contains_visual"),
    ({}, 5, "sequence_too_long"),
])
def test_collator_fails_on_ambiguous_loss_and_truncation(crop_inputs, options, length, code):
    checked = preflight(crop_inputs.config)
    collator = CropQACollator(FakeProcessor(**options), crop_inputs.dataset, crop_inputs.model_config,
                             max_length=length, tensor_factory=lambda x: x)
    with pytest.raises(TrainingGateError, match=code):
        collator([checked["rows"][0]])


@pytest.mark.parametrize("fault,code", [
    ("changed_pixels", "prompt_full_pixels_mismatch"),
    ("oversize", "visual_pixel_budget_exceeded"),
    ("bad_pixel_rows", "pixel_grid_mismatch"),
    ("bad_pixel_width", "pixel_grid_mismatch"),
    ("token_count", "visual_token_grid_mismatch"),
])
def test_collator_proves_visual_geometry_and_equal_prompt_pixels(crop_inputs, fault, code):
    checked = preflight(crop_inputs.config)
    collator = CropQACollator(FakeProcessor(visual_fault=fault), crop_inputs.dataset,
        crop_inputs.model_config, max_length=10, tensor_factory=lambda x: x)
    with pytest.raises(TrainingGateError, match=code):
        collator([checked["rows"][0]])


def test_loader_failure_never_becomes_training_proof(crop_inputs, monkeypatch):
    def fail(*args):
        raise TrainingGateError("crop_lora_cuda_unavailable")
    monkeypatch.setattr("training.specialist_lora._runtime_model", fail)
    with pytest.raises(TrainingGateError, match="cuda_unavailable"):
        train(crop_inputs.config)
    receipt = json.loads((crop_inputs.base / "run/run.json").read_text())
    assert receipt["state"] == "failed" and receipt["optimizer_steps"] == 0
    assert receipt["promotion_eligible"] is False and receipt["approved_types"] == []
    assert not (crop_inputs.base / "run/adapter").exists()


def test_config_paths_resolve_relative_to_config_file(crop_inputs):
    config = copy.deepcopy(crop_inputs.config)
    config["output_dir"] = "relative-output"
    path = crop_inputs.base / "run-config.json"
    write_json(path, config)
    assert resolve_config(path)["output_dir"] == str(crop_inputs.base / "relative-output")
