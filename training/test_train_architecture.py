"""Synthetic architecture/loader integration checks; no weights, GPU or media."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from training.guards import TrainingGateError
from training.train import (check_local_model, configure_adapter,
                            load_model_and_processor, validate_model_processor)


TOKENS = {"<|image_pad|>": 9, "<|video_pad|>": 10,
          "<|vision_start|>": 3, "<|vision_end|>": 4}


class Tokenizer:
    def __len__(self):
        return 100

    def convert_tokens_to_ids(self, token):
        return TOKENS[token]


def pair(model_type="qwen3_5"):
    name = "Qwen3_5ForConditionalGeneration" if model_type == "qwen3_5" else "Qwen3VLForConditionalGeneration"
    model = type(name, (), {})()
    model.config = SimpleNamespace(
        model_type=model_type, image_token_id=9, video_token_id=10,
        vision_start_token_id=3, vision_end_token_id=4,
        vision_config=SimpleNamespace(patch_size=16, temporal_patch_size=2, spatial_merge_size=2),
        text_config=SimpleNamespace(use_cache=True), use_cache=True)
    model.get_input_embeddings = lambda: SimpleNamespace(num_embeddings=128)
    model.to = Mock(return_value=model)
    model.named_modules = lambda: []
    processor = type("Qwen3VLProcessor", (), {})()
    processor.tokenizer = Tokenizer()
    processor.image_token_id = 9
    processor.video_token_id = 10
    processor.image_processor = SimpleNamespace(patch_size=16, temporal_patch_size=2, merge_size=2)
    processor.video_processor = SimpleNamespace(patch_size=16, temporal_patch_size=2, merge_size=2)
    return model, processor


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_5"])
def test_correct_model_and_shared_qwen3vl_processor_pass(model_type):
    model, processor = pair(model_type)
    receipt = validate_model_processor(model, processor, model_type)
    assert receipt["processor_model_compatibility_verified"] is True
    assert receipt["model_type"] == model_type
    assert receipt["processor_class"] == "Qwen3VLProcessor"


@pytest.mark.parametrize("mutation,reason", [
    (lambda m, p: setattr(m.config, "model_type", "qwen3_vl"), "loaded_model_architecture_mismatch"),
    (lambda m, p: setattr(p, "image_token_id", 8), "model_processor_token_mismatch"),
    (lambda m, p: setattr(m.config, "vision_start_token_id", 8), "model_processor_token_mismatch"),
    (lambda m, p: setattr(p.image_processor, "patch_size", 14), "model_processor_patch_geometry_mismatch"),
    (lambda m, p: setattr(p.video_processor, "merge_size", 1), "model_processor_patch_geometry_mismatch"),
    (lambda m, p: setattr(p, "video_processor", None), "visual_processor_missing"),
    (lambda m, p: setattr(m, "get_input_embeddings", lambda: SimpleNamespace(num_embeddings=99)),
     "model_tokenizer_vocabulary_mismatch"),
])
def test_instantiated_pair_mismatches_fail_closed(mutation, reason):
    model, processor = pair()
    mutation(model, processor)
    with pytest.raises(TrainingGateError, match=reason):
        validate_model_processor(model, processor, "qwen3_5")


def test_text_only_class_cannot_be_used_for_multimodal_training():
    model, processor = pair()
    model.__class__ = type("Qwen3_5ForCausalLM", (), {})
    with pytest.raises(TrainingGateError, match="loaded_model_class_mismatch"):
        validate_model_processor(model, processor, "qwen3_5")


def local_config(root: Path, model_type="qwen3_5"):
    model = root / "model"
    model.mkdir()
    name = "Qwen3_5ForConditionalGeneration" if model_type == "qwen3_5" else "Qwen3VLForConditionalGeneration"
    (model / "config.json").write_text(json.dumps({"model_type": model_type, "architectures": [name]}))
    (model / "tokenizer_config.json").write_text("{}")
    (model / "synthetic.safetensors").write_bytes(b"synthetic; not model weights")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"revision": "a" * 40, "processor_revision": "a" * 40}))
    return {"model_dir": str(model), "processor_dir": str(model), "model_manifest": str(manifest),
            "base_revision": "a" * 40, "processor_revision": "a" * 40, "model_type": model_type}


def test_qwen38_snapshot_uses_separate_candidate_verifier(tmp_path):
    config = local_config(tmp_path)
    verify = Mock()
    with patch.dict(sys.modules, {"serving.qwen38_identity": SimpleNamespace(verify_candidate_base=verify)}), \
         patch("forte.model.manifest.verify_local_snapshot") as baseline:
        model, processor = check_local_model(config)
    assert model == processor
    verify.assert_called_once_with(model, config["model_manifest"])
    baseline.assert_not_called()


def test_candidate_revision_mismatch_never_verifies_weights(tmp_path):
    config = local_config(tmp_path)
    config["base_revision"] = config["processor_revision"] = "b" * 40
    verify = Mock()
    with patch.dict(sys.modules, {"serving.qwen38_identity": SimpleNamespace(verify_candidate_base=verify)}):
        with pytest.raises(TrainingGateError, match="candidate_manifest_revision_mismatch"):
            check_local_model(config)
    verify.assert_not_called()


def test_baseline_verifier_path_stays_separate(tmp_path):
    config = local_config(tmp_path, "qwen3_vl")
    manifest = {"revision": "a" * 40, "processor_revision": "a" * 40,
                "model_id": "Qwen/Qwen3-VL-8B-Instruct"}
    with patch("forte.model.manifest.load_manifest", return_value=manifest), \
         patch("forte.model.manifest.verify_local_snapshot") as verify:
        check_local_model(config)
    verify.assert_called_once()


def loader_environment(model, processor):
    model_class = type(model)
    model_class.from_pretrained = Mock(return_value=model)
    processor_class = SimpleNamespace(from_pretrained=Mock(return_value=processor))
    torch = SimpleNamespace(bfloat16="bf16", version=SimpleNamespace(cuda="12.9"),
                            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                                                 get_device_capability=lambda _: (8, 0)))
    transformers = SimpleNamespace(AutoProcessor=processor_class, **{model_class.__name__: model_class})
    return torch, transformers, processor_class


def test_offline_loader_dispatches_actual_qwen35_class(tmp_path):
    config = local_config(tmp_path)
    model, processor = pair()
    torch, transformers, processor_class = loader_environment(model, processor)
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        loaded, processed = load_model_and_processor(config, Path(config["model_dir"]), Path(config["processor_dir"]))
    assert loaded is model and processed is processor
    kwargs = type(model).from_pretrained.call_args.kwargs
    assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
    assert kwargs["dtype"] == "bf16" and kwargs["attn_implementation"] == "sdpa"
    assert "quantization_config" not in kwargs
    assert model._forte_runtime_load_receipt["nf4_runtime_verified"] is False
    model.to.assert_called_once_with("cuda")
    processor_class.from_pretrained.assert_called_once_with(Path(config["processor_dir"]),
                                                           local_files_only=True, trust_remote_code=False)


@pytest.mark.parametrize("override,reason", [
    ({"required_compute_capability": [9, 0]}, "training_gpu_compute_capability_mismatch"),
    ({"quantization": "nvfp4"}, "training_quantization_unsupported"),
])
def test_unverified_gpu_or_quantization_profile_cannot_load_model(tmp_path, override, reason):
    config = {**local_config(tmp_path), **override}
    model, processor = pair()
    torch, transformers, processor_class = loader_environment(model, processor)
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        with pytest.raises(TrainingGateError, match=reason):
            load_model_and_processor(config, Path(config["model_dir"]), Path(config["processor_dir"]))
    type(model).from_pretrained.assert_not_called()
    processor_class.from_pretrained.assert_not_called()


@pytest.mark.parametrize("quantized,reason", [(False, "training_nf4_not_loaded"),
                                             (True, "training_nf4_modules_missing")])
def test_nf4_claim_requires_actual_quantized_modules(tmp_path, quantized, reason):
    config = local_config(tmp_path)
    config["quantization"] = "nf4"
    model, processor = pair()
    model.is_loaded_in_4bit = quantized
    torch, transformers, _ = loader_environment(model, processor)
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=type("Linear4bit", (), {})), __version__="synthetic")
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "bitsandbytes": bnb}):
        with pytest.raises(TrainingGateError, match=reason):
            load_model_and_processor(config, Path(config["model_dir"]), Path(config["processor_dir"]))
    model.to.assert_not_called()


def test_nf4_runtime_verification_records_actual_loaded_path(tmp_path):
    config = local_config(tmp_path)
    config["quantization"] = "nf4"
    model, processor = pair()
    model.is_loaded_in_4bit = True
    torch, transformers, _ = loader_environment(model, processor)
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    linear = type("Linear4bit", (), {})()
    linear.weight = SimpleNamespace(quant_type="nf4", quant_state=object())
    linear.compute_dtype = "bf16"
    model.named_modules = lambda: [("model.language_model.synthetic", linear)]
    bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=type(linear)), __version__="synthetic")
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "bitsandbytes": bnb}):
        loaded, _ = load_model_and_processor(config, Path(config["model_dir"]), Path(config["processor_dir"]))
    quant = type(model).from_pretrained.call_args.kwargs["quantization_config"]
    assert quant == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bf16",
                     "bnb_4bit_use_double_quant": True}
    assert loaded._forte_runtime_load_receipt["nf4_runtime_verified"] is True


def test_configure_adapter_consumes_and_records_actual_inventory():
    model, _ = pair()
    model.requires_grad_ = Mock()
    model.enable_input_require_grads = Mock()
    model.gradient_checkpointing_enable = Mock()
    model.train = Mock()
    inventory = {"target_modules": ["model.language_model.layers.0.linear_attn.in_proj_qkv"],
                 "target_count": 1, "synthetic": True}
    trainable = [("synthetic_language_lora", object())]
    lora_config = Mock()
    peft = SimpleNamespace(LoraConfig=lora_config, TaskType=SimpleNamespace(CAUSAL_LM="causal"),
                           get_peft_model=Mock(return_value=model))
    with patch.dict(sys.modules, {"peft": peft}), \
         patch("training.train.inventory_language_lora_targets", return_value=inventory) as inspect, \
         patch("training.train.validate_trainable_adapter", return_value=trainable):
        adapted, parameters = configure_adapter(model, {"model_type": "qwen3_5"})
    assert adapted is model and parameters == trainable
    inspect.assert_called_once_with(model, "qwen3_5")
    assert lora_config.call_args.kwargs["target_modules"] == inventory["target_modules"]
    assert model._forte_lora_target_inventory == inventory
    assert model.config.use_cache is False and model.config.text_config.use_cache is False
    model.requires_grad_.assert_called_once_with(False)
