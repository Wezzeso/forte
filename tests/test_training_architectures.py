"""Synthetic architecture/scope checks; no private media, weights, or CUDA.

The torch shim exercises guards only. A real Transformers visual forward and
backward remains a separate mandatory candidate-image acceptance check.
"""
from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from training import architectures as subject
from training.guards import TrainingGateError, canonical_sha256


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def all(self):
        return self


class Tensor:
    def __init__(self, values=(0.0, 1.0), *, trainable=False):
        self.values = list(values)
        self.requires_grad = trainable
        self.grad = None
        self.shape = (len(self.values),)
        self.dtype = "float32"

    def detach(self):
        return self

    def contiguous(self):
        return self

    def reshape(self, *_):
        return self

    def view(self, *_):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return array("B", struct.pack("<" + "f" * len(self.values), *self.values))


class Linear:
    in_features = 4
    out_features = 8


class NonLinear:
    in_features = 4
    out_features = 8


class Model:
    def __init__(self, *, model_type="qwen3_5", layer_types=None):
        layer_types = layer_types or ["linear_attention", "full_attention"]
        self.config = SimpleNamespace(model_type=model_type, text_config=SimpleNamespace(
            num_hidden_layers=len(layer_types), layer_types=layer_types))
        self.modules = []
        for index, layer_type in enumerate(layer_types):
            if model_type == "qwen3_vl":
                policy = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj")}
            elif layer_type == "linear_attention":
                policy = {"linear_attn": ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")}
            else:
                policy = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj")}
            policy["mlp"] = ("gate_proj", "up_proj", "down_proj")
            for group, leaves in policy.items():
                self.modules.extend((f"model.language_model.layers.{index}.{group}.{leaf}", Linear()) for leaf in leaves)
        self.modules.extend((name, Linear()) for name in (
            "model.visual.blocks.0.attn.qkv", "model.visual.merger.mlp.0", "model.aligner.q_proj", "lm_head"))
        self.parameters = []

    def named_modules(self):
        return iter(self.modules)

    def named_parameters(self):
        return iter(self.parameters)


@pytest.fixture(autouse=True)
def synthetic_torch(monkeypatch):
    shim = SimpleNamespace(nn=SimpleNamespace(Linear=Linear), uint8="uint8",
        isfinite=lambda value: Scalar(all(math.isfinite(item) for item in value.values)),
        count_nonzero=lambda value: Scalar(sum(item != 0 for item in value.values)))
    monkeypatch.setitem(sys.modules, "torch", shim)


def adapter_model(model_type="qwen3_5"):
    model = Model(model_type=model_type)
    inventory = subject.inventory_language_lora_targets(model, model_type)
    for name in inventory["target_modules"]:
        model.parameters.append(("base_model.model." + name + ".base_layer.weight", Tensor()))
        for side in ("lora_A", "lora_B"):
            model.parameters.append(("base_model.model." + name + "." + side + ".default.weight",
                                     Tensor(trainable=True)))
    model.parameters.append(("base_model.model.model.visual.patch_embed.proj.weight", Tensor()))
    model.parameters.append(("base_model.model.model.visual.merger.mlp.0.weight", Tensor()))
    return model, inventory


def make_gradients(model):
    for name, parameter in model.parameters:
        if parameter.requires_grad:
            parameter.grad = Tensor((0.0, 1.0))


def change_adapters(model):
    for _, parameter in model.parameters:
        if parameter.requires_grad:
            parameter.values[0] = 0.25


def test_dispatch_is_explicit_and_shared_processor_is_verified():
    profile = subject.resolve_architecture("qwen3_5")
    assert profile.model_class_name == "Qwen3_5ForConditionalGeneration"
    assert profile.processor_class_names == ("Qwen3VLProcessor",)
    model_class, processor_class = object(), object()
    module = SimpleNamespace(Qwen3_5ForConditionalGeneration=model_class, AutoProcessor=processor_class)
    assert subject.resolve_training_classes(profile, module) == (model_class, processor_class)


@pytest.mark.parametrize("model_type", ["qwen3_8", "qwen3_5_text", "qwen3_5_moe", "unknown", None])
def test_unknown_or_text_only_architecture_is_rejected(model_type):
    with pytest.raises(TrainingGateError, match="unsupported_model_architecture"):
        subject.resolve_architecture(model_type)


def test_missing_transformers_class_is_rejected():
    with pytest.raises(TrainingGateError, match="training_architecture_class_unavailable"):
        subject.resolve_training_classes(subject.resolve_architecture("qwen3_5"), SimpleNamespace(AutoProcessor=object()))


def test_metadata_class_mismatch_is_rejected():
    model = Model()
    config = {"model_type": "qwen3_5", "architectures": ["Qwen3VLForConditionalGeneration"],
              "text_config": vars(model.config.text_config)}
    with pytest.raises(TrainingGateError, match="model_architecture_class_mismatch"):
        subject.validate_architecture_config(config)


def test_hybrid_all_layers_and_groups_are_covered():
    inventory = subject.inventory_language_lora_targets(Model(), "qwen3_5")
    assert inventory["groups"] == {"linear_attn": 5, "self_attn": 4, "mlp": 6}
    assert inventory["target_count"] == 15
    assert inventory["language_layer_count"] == 2
    assert inventory["vision_and_aligner_target_count"] == 0
    assert all("visual" not in name and "aligner" not in name for name in inventory["target_modules"])
    receipt = dict(inventory)
    digest = receipt.pop("inventory_sha256")
    assert digest == canonical_sha256(receipt)


def test_baseline_qv_recipe_is_preserved():
    inventory = subject.inventory_language_lora_targets(Model(model_type="qwen3_vl"), "qwen3_vl")
    assert inventory["target_count"] == 4
    assert inventory["groups"] == {"self_attn": 4}
    assert all(name.endswith((".q_proj", ".v_proj")) for name in inventory["target_modules"])


def test_attention_only_accidental_coverage_is_rejected():
    model = Model()
    model.modules = [(name, module) for name, module in model.modules if ".self_attn." in name]
    with pytest.raises(TrainingGateError, match="lora_layer_targets_missing"):
        subject.inventory_language_lora_targets(model, "qwen3_5")


def test_non_linear_target_is_rejected():
    model = Model()
    name, _ = model.modules[0]
    model.modules[0] = (name, NonLinear())
    with pytest.raises(TrainingGateError, match="lora_target_is_not_linear"):
        subject.inventory_language_lora_targets(model, "qwen3_5")


def test_missing_targets_rejected_even_if_visual_has_same_suffix():
    model = Model()
    model.modules = [(name, module) for name, module in model.modules if ".in_proj_qkv" not in name]
    model.modules.append(("model.visual.language_model.layers.0.linear_attn.in_proj_qkv", Linear()))
    with pytest.raises(TrainingGateError, match="lora_layer_targets_missing"):
        subject.inventory_language_lora_targets(model, "qwen3_5")


@pytest.mark.parametrize("types", [["full_attention"], ["linear_attention"], ["full_attention", "future_attention"]])
def test_hybrid_config_must_exercise_both_actual_families(types):
    with pytest.raises(TrainingGateError, match="lora_hybrid_"):
        subject.inventory_language_lora_targets(Model(layer_types=types), "qwen3_5")


def test_trainable_adapter_pairs_match_complete_inventory():
    model, inventory = adapter_model()
    assert len(subject.validate_trainable_adapter(model, inventory=inventory)) == 30
    assert len(subject.validate_trainable_adapter(model)) == 30


@pytest.mark.parametrize("name", ["model.visual.q_proj.weight", "model.language_model.embed_tokens.weight",
                                  "lm_head.weight", "model.aligner.lora_A.default.weight"])
def test_any_base_or_vision_or_aligner_trainable_is_rejected(name):
    model, inventory = adapter_model()
    model.parameters.append((name, Tensor(trainable=True)))
    with pytest.raises(TrainingGateError, match="non_language_adapter_parameter_trainable"):
        subject.validate_trainable_adapter(model, inventory=inventory)


def test_missing_adapter_pair_is_rejected():
    model, inventory = adapter_model()
    model.parameters.pop(2)
    with pytest.raises(TrainingGateError, match="lora_trainable_pair_incomplete"):
        subject.validate_trainable_adapter(model, inventory=inventory)


def test_attention_only_trainable_subset_is_rejected():
    model, inventory = adapter_model()
    model.parameters = [(name, parameter) for name, parameter in model.parameters
                        if not parameter.requires_grad or ".self_attn." in name]
    with pytest.raises(TrainingGateError, match="lora_trainable_layer_coverage_incomplete"):
        subject.validate_trainable_adapter(model, inventory=inventory)
    with pytest.raises(TrainingGateError, match="lora_trainable_layer_coverage_incomplete"):
        subject.validate_trainable_adapter(model)


def test_finite_active_gradients_and_frozen_absence_are_proved():
    model, inventory = adapter_model()
    make_gradients(model)
    receipt = subject.validate_adapter_gradients(model, inventory=inventory)
    assert receipt["nonzero_gradient_groups"] == ["linear_attn", "mlp", "self_attn"]
    assert receipt["frozen_gradients_absent"] is True


def test_frozen_gradient_is_rejected():
    model, inventory = adapter_model()
    make_gradients(model)
    model.parameters[-1][1].grad = Tensor()
    with pytest.raises(TrainingGateError, match="frozen_parameter_received_gradient"):
        subject.validate_adapter_gradients(model, inventory=inventory)


@pytest.mark.parametrize("bad_gradient", [None, Tensor((float("nan"),))])
def test_missing_or_nonfinite_gradient_is_rejected(bad_gradient):
    model, inventory = adapter_model()
    make_gradients(model)
    model.parameters[1][1].grad = bad_gradient
    with pytest.raises(TrainingGateError, match="training_gradient_missing_or_nonfinite"):
        subject.validate_adapter_gradients(model, inventory=inventory)


def test_inactive_linear_attention_gradient_is_rejected():
    model, inventory = adapter_model()
    make_gradients(model)
    for name, parameter in model.parameters:
        if parameter.requires_grad and ".linear_attn." in name:
            parameter.grad = Tensor((0.0, 0.0))
    with pytest.raises(TrainingGateError, match="training_gradient_group_inactive"):
        subject.validate_adapter_gradients(model, inventory=inventory)


def test_two_step_proof_hashes_all_frozen_parameters_and_changed_groups():
    model, inventory = adapter_model()
    before = subject.capture_parameter_proof(model, inventory=inventory)
    change_adapters(model)
    receipt = subject.verify_parameter_update(model, before, optimizer_steps=2, inventory=inventory)
    assert receipt["frozen_parameters_checked"] == 17
    assert receipt["frozen_parameters_byte_identical"] is True
    assert len(receipt["changed_adapter_tensors"]) == 30
    assert receipt["changed_adapter_groups"] == ["linear_attn", "mlp", "self_attn"]


def test_any_frozen_change_is_rejected():
    model, inventory = adapter_model()
    before = subject.capture_parameter_proof(model, inventory=inventory)
    change_adapters(model)
    model.parameters[-1][1].values[0] = 0.125
    with pytest.raises(TrainingGateError, match="frozen_parameter_changed"):
        subject.verify_parameter_update(model, before, optimizer_steps=2, inventory=inventory)


def test_unchanged_adapter_or_one_step_cannot_claim_smoke_completion():
    model, inventory = adapter_model()
    before = subject.capture_parameter_proof(model, inventory=inventory)
    with pytest.raises(TrainingGateError, match="adapter_parameter_group_unchanged"):
        subject.verify_parameter_update(model, before, optimizer_steps=2, inventory=inventory)
    with pytest.raises(TrainingGateError, match="optimizer_step_proof_insufficient"):
        subject.verify_parameter_update(model, before, optimizer_steps=1, inventory=inventory)


def test_parameter_receipt_tampering_is_rejected():
    model, inventory = adapter_model()
    before = subject.capture_parameter_proof(model, inventory=inventory)
    before["parameters"][next(iter(before["parameters"]))]["sha256"] = "0" * 64
    with pytest.raises(TrainingGateError, match="parameter_proof_digest_mismatch"):
        subject.verify_parameter_update(model, before, optimizer_steps=2, inventory=inventory)


def test_nonfinite_adapter_weight_cannot_be_hashed_as_success():
    model, inventory = adapter_model()
    model.parameters[1][1].values[0] = float("nan")
    with pytest.raises(TrainingGateError, match="adapter_parameter_nonfinite"):
        subject.capture_parameter_proof(model, inventory=inventory)


def test_inventory_matches_pinned_public_27b_weight_names():
    snapshot = Path(__file__).resolve().parents[1] / "configs/serving/qwen38/public-snapshot"
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip("Pinned public metadata has not been staged; no model weights required")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen3_5"
    subject.validate_architecture_config(config)
    model = Model()
    model.config = SimpleNamespace(**config)
    model.modules = [(name.removesuffix(".weight"), Linear()) for name in index["weight_map"]
                     if name.endswith(".weight") and subject._language_target(name.removesuffix(".weight"))]
    inventory = subject.inventory_language_lora_targets(model, "qwen3_5")
    assert inventory["language_layer_count"] == 64
    assert inventory["full_attention_layer_count"] == 16
    assert inventory["linear_attention_layer_count"] == 48
    assert inventory["groups"] == {"linear_attn": 240, "self_attn": 64, "mlp": 192}
    assert inventory["target_count"] == 496
    assert all(name + ".weight" in index["weight_map"] for name in inventory["target_modules"])
    image = json.loads((snapshot / "preprocessor_config.json").read_text(encoding="utf-8"))
    video = json.loads((snapshot / "video_preprocessor_config.json").read_text(encoding="utf-8"))
    assert image["processor_class"] == video["processor_class"] == "Qwen3VLProcessor"
