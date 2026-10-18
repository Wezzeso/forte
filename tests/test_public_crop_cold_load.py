"""Synthetic CPU cold-load contracts; no model weights, torch, pixels or GPU."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import public_crop_cold_load as cold
from training import public_crop_lora as crop
from training.guards import TrainingGateError
from training.specialists.common import file_sha256, load_json, write_json


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", {"model_type": "qwen3_vl", "architectures": [crop.MODEL_CLASS],
        "text_config": {"model_type": "qwen3_vl_text", "num_hidden_layers": 28, "hidden_size": 2048,
                        "num_attention_heads": 16, "num_key_value_heads": 8},
        "vision_config": {"patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}})
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json"):
        write_json(model / name, {})
    (model / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"synthetic model bytes, never load")
    monkeypatch.setattr(crop, "WEIGHTS_SHA256", file_sha256(model / "model.safetensors"))
    manifest = tmp_path / "model-manifest.json"
    crop.snapshot(model, manifest)
    output = tmp_path / "fit"
    output.mkdir()
    cfg = {"schema_version": crop.SCHEMA, "model_id": crop.MODEL_ID, "revision": crop.REVISION,
        "runtime_versions": crop.PINS, "runtime_image_digest": "sha256:" + "a" * 64,
        "model_dir": str(model), "model_manifest": str(manifest),
        "expected_model_manifest_sha256": file_sha256(manifest),
        "dataset_dir": str(tmp_path / "absent-dataset-must-not-be-read"),
        "expected_dataset_lock_sha256": "d" * 64, "output_dir": str(output),
        "test_exposure_registry_dir": str(tmp_path / "absent-registry"),
        "optimizer_steps": 80, "lora_rank": 4, "lora_alpha": 8,
        "max_pixels": 65536, "max_length": 256, "gpu_memory_budget_mib": 5120, "grad_accum_steps": 4}
    write_json(output / "config.json", cfg)
    write_json(output / "evaluation-protocol.json", {"source_sha256": cold.TRAINING_SOURCE_SHA256,
               "dataset_lock_sha256": cfg["expected_dataset_lock_sha256"]})
    write_json(output / "crop-manifest.json", {"synthetic": True})
    write_json(output / "step-proof.json", [{"synthetic": True}])
    adapter = output / "adapter"
    adapter.mkdir()
    write_json(adapter / "adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none", "inference_mode": True,
        "target_modules": cold.expected_targets()})
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter bytes, never load")
    run = {"schema_version": crop.RUN_SCHEMA, "state": "trained_unselected", "training_completed": True,
        "memory_smoke_only": False, "optimizer_steps": 80, "model_id": crop.MODEL_ID,
        "revision": crop.REVISION, "config_sha256": file_sha256(output / "config.json"),
        "model_manifest_sha256": cfg["expected_model_manifest_sha256"],
        "dataset_lock_sha256": cfg["expected_dataset_lock_sha256"], "runtime_versions": crop.PINS,
        "runtime_image_digest": cfg["runtime_image_digest"], "memory_recipe": crop.memory_recipe(cfg),
        "adapter_before_sha256": "b" * 64, "adapter_after_sha256": "c" * 64,
        "frozen_before_sha256": "f" * 64, "frozen_after_sha256": "f" * 64,
        "sealed_test_used_for_training_or_selection": False, "promotion_eligible": False,
        "target_modules": cold.expected_targets(), "trainable_parameters": 224 * 4,
        "adapter_files": {p.name: file_sha256(p) for p in adapter.iterdir()}}
    for name, field in (("evaluation-protocol.json", "evaluation_protocol_sha256"),
                        ("crop-manifest.json", "crop_manifest_sha256"), ("step-proof.json", "step_proof_sha256")):
        run[field] = file_sha256(output / name)
    write_json(output / "run.json", run)
    return SimpleNamespace(config=cfg, config_path=output / "config.json", run=run,
        run_path=output / "run.json", sha=file_sha256(output / "run.json"), receipt=tmp_path / "receipt.json")


class Tensor:
    def __init__(self, values=(1.0, 2.0, 3.0, 4.0), dtype="float32", device="cpu", requires_grad=False):
        self.values, self.dtype, self.device = values, dtype, SimpleNamespace(type=device)
        self.shape, self.requires_grad = (2, 2), requires_grad

    def detach(self):
        return self

    def numel(self):
        return len(self.values)


class Torch:
    bfloat16 = "bfloat16"

    def __init__(self):
        self.initialized = False
        self.cuda = SimpleNamespace(is_initialized=lambda: self.initialized)

    @staticmethod
    def equal(a, b):
        return a.values == b.values

    @staticmethod
    def isfinite(tensor):
        return SimpleNamespace(all=lambda: all(math.isfinite(value) for value in tensor.values))


@pytest.fixture
def fake_runtime(artifacts, monkeypatch):
    torch, calls = Torch(), []
    keys = [f"base_model.model.{target}.lora_{side}.weight"
            for target in cold.expected_targets() for side in ("A", "B")]
    saved = {key: Tensor() for key in keys}
    actual = copy.deepcopy(saved)
    live = {name.replace(".weight", ".default.weight"): value for name, value in actual.items()}
    layers = {"base_model.model." + target: SimpleNamespace(lora_A={"default": object()},
        lora_B={"default": object()}, active_adapters=["default"], disable_adapters=False,
        merged_adapters=[]) for target in cold.expected_targets()}
    base_parameter = Tensor(dtype="bfloat16")
    buffers = [Tensor(dtype="float32")]
    class Adapted:
        active_adapters = ["default"]
        peft_config = {"default": SimpleNamespace(inference_mode=True)}
        def eval(self):
            calls.append("eval")
            return self
        def named_parameters(self):
            return list(live.items()) + [("base_model.model.lm_head.weight", base_parameter)]
        def parameters(self):
            return [value for _, value in self.named_parameters()]
        def buffers(self):
            return buffers
        def named_modules(self):
            return list(layers.items())
    adapted = Adapted()
    class Qwen3VLForConditionalGeneration:
        config = SimpleNamespace()
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("base", path, kwargs))
            return cls(), {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        def parameters(self):
            return [base_parameter]
    class PeftModel:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            calls.append(("peft", path, kwargs))
            return adapted
    def state_dict(model, **kwargs):
        assert model is adapted
        calls.append(("state", kwargs))
        return actual
    def load_file(path, **kwargs):
        calls.append(("saved", path, kwargs))
        return saved
    monkeypatch.setattr(cold, "runtime", lambda: (torch, Qwen3VLForConditionalGeneration, PeftModel, state_dict, load_file))
    return SimpleNamespace(torch=torch, saved=saved, actual=actual, live=live, layers=layers,
        adapted=adapted, base_parameter=base_parameter, buffers=buffers, calls=calls)


def test_completed_fit_artifacts_do_not_require_dataset_or_training_deadline(artifacts, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("training preflight or dataset access is forbidden")
    monkeypatch.setattr(crop, "preflight", forbidden)
    monkeypatch.setattr(crop, "dataset_metadata", forbidden)
    checked = cold.verify_inputs(artifacts.config_path, artifacts.sha)
    assert checked["run"]["optimizer_steps"] == 80
    assert "snapshot:model.safetensors" in checked["hashes"]
    assert not Path(artifacts.config["dataset_dir"]).exists()


def test_exact_cold_load_uses_explicit_cpu_local_frozen_contract(artifacts, fake_runtime):
    result = cold.cold_load(artifacts.config_path, artifacts.sha, artifacts.receipt)
    assert result["state"] == "passed" and result["adapter_tensor_count"] == 224
    assert result["keys_shapes_dtypes_values_exact"] and result["inputs_unchanged"]
    assert not result["inference_executed"] and not result["nf4_equivalence_verified"]
    assert not result["dataset_accessed"] and not result["gpu_inference_verified"]
    base = next(call for call in fake_runtime.calls if call[0] == "base")
    assert base[2] == {"local_files_only": True, "trust_remote_code": False, "dtype": "bfloat16",
        "device_map": {"": "cpu"}, "attn_implementation": "sdpa", "output_loading_info": True}
    peft = next(call for call in fake_runtime.calls if call[0] == "peft")
    assert peft[2] == {"adapter_name": "default", "is_trainable": False, "local_files_only": True,
        "torch_device": "cpu", "ephemeral_gpu_offload": False, "autocast_adapter_dtype": True}


def replace_serialized_targets(artifacts, targets):
    path = Path(artifacts.config["output_dir"]) / "adapter" / "adapter_config.json"
    config = load_json(path)
    config["target_modules"] = targets
    write_json(path, config)
    artifacts.run["adapter_files"][path.name] = file_sha256(path)
    write_json(artifacts.run_path, artifacts.run)
    artifacts.sha = file_sha256(artifacts.run_path)


def test_native_peft_four_suffix_serialization_keeps_exact_live_layer_proof(artifacts, fake_runtime):
    replace_serialized_targets(artifacts, ["o_proj", "v_proj", "q_proj", "k_proj"])
    result = cold.cold_load(artifacts.config_path, artifacts.sha, artifacts.receipt)
    assert result["state"] == "passed" and result["adapter_tensor_count"] == 224
    assert result["keys_shapes_dtypes_values_exact"]


@pytest.mark.parametrize("targets", [
    ["k_proj", "o_proj", "q_proj", "v_proj", "q_proj"],
    ["k_proj", "o_proj", "q_proj", "v_proj", "gate_proj"],
    ["k_proj", "o_proj", "q_proj"], ["all-linear"], "q_proj|k_proj|v_proj|o_proj",
    cold.expected_targets() + [cold.expected_targets()[0]],
    cold.expected_targets() + ["model.visual.blocks.0.attn.qkv"],
])
def test_minimized_targets_reject_duplicates_extras_missing_and_broader_targets(artifacts, targets):
    replace_serialized_targets(artifacts, targets)
    with pytest.raises(TrainingGateError, match="adapter_config_mismatch"):
        cold.verify_inputs(artifacts.config_path, artifacts.sha)


@pytest.mark.parametrize("cuda,nvidia", [(None, "none"), ("0", "none"), ("", None), ("", "all")])
def test_visible_or_unasserted_cuda_rejected_before_artifact_or_runtime_access(artifacts, monkeypatch, cuda, nvidia):
    for key, value in (("CUDA_VISIBLE_DEVICES", cuda), ("NVIDIA_VISIBLE_DEVICES", nvidia)):
        monkeypatch.delenv(key, raising=False) if value is None else monkeypatch.setenv(key, value)
    monkeypatch.setattr(cold, "verify_inputs", lambda *a: pytest.fail("artifact access before visibility guard"))
    with pytest.raises(TrainingGateError, match="gpu_must_be_hidden"):
        cold.cold_load(artifacts.config_path, artifacts.sha, artifacts.receipt)
    assert not artifacts.receipt.exists()


@pytest.mark.parametrize("field,value", [("state", "failed"), ("memory_smoke_only", True),
    ("training_completed", False), ("optimizer_steps", 79), ("frozen_after_sha256", "e" * 64),
    ("adapter_after_sha256", "b" * 64)])
def test_incomplete_or_unproven_training_rejected(artifacts, field, value):
    artifacts.run[field] = value
    write_json(artifacts.run_path, artifacts.run)
    with pytest.raises(TrainingGateError):
        cold.verify_inputs(artifacts.config_path, file_sha256(artifacts.run_path))


@pytest.mark.parametrize("mutation,error", [
    ("missing", "adapter_keys"), ("extra", "adapter_keys"), ("value", "adapter_values"),
    ("dtype", "tensor_metadata"), ("shape", "tensor_metadata"), ("nan", "adapter_values"),
    ("live_extra", "live_adapter_keys"), ("trainable", "trainable_parameter"),
    ("buffer_cuda", "non_cpu_tensor"), ("disabled", "adapter_layer_state"),
    ("merged", "adapter_layer_state"), ("cuda_initialized", "base_initialized_cuda")])
def test_silently_ignored_or_changed_adapter_and_device_failures_are_rejected(artifacts, fake_runtime, mutation, error):
    first = next(iter(fake_runtime.actual))
    tensor = fake_runtime.actual[first]
    if mutation == "missing": fake_runtime.actual.pop(first)
    elif mutation == "extra": fake_runtime.actual["extra.weight"] = Tensor()
    elif mutation == "value": tensor.values = (0.0, 2.0, 3.0, 4.0)
    elif mutation == "dtype": tensor.dtype = "bfloat16"
    elif mutation == "shape": tensor.shape = (4,)
    elif mutation == "nan": tensor.values = (float("nan"), 2.0, 3.0, 4.0)
    elif mutation == "live_extra": fake_runtime.live["other.lora_A.default.weight"] = Tensor()
    elif mutation == "trainable": tensor.requires_grad = True
    elif mutation == "buffer_cuda": fake_runtime.buffers[0].device.type = "cuda"
    elif mutation == "disabled": next(iter(fake_runtime.layers.values())).disable_adapters = True
    elif mutation == "merged": next(iter(fake_runtime.layers.values())).merged_adapters = ["default"]
    elif mutation == "cuda_initialized": fake_runtime.torch.initialized = True
    with pytest.raises(TrainingGateError, match=error):
        cold.cold_load(artifacts.config_path, artifacts.sha, artifacts.receipt)
    assert load_json(artifacts.receipt)["state"] == "failed"


def test_output_is_never_overwritten(artifacts, fake_runtime):
    artifacts.receipt.write_text("preserve", encoding="utf-8")
    with pytest.raises(TrainingGateError, match="output_exists"):
        cold.cold_load(artifacts.config_path, artifacts.sha, artifacts.receipt)
    assert artifacts.receipt.read_text() == "preserve" and not fake_runtime.calls


def test_adapter_changed_during_load_fails_postload_reverification(artifacts, fake_runtime):
    original_eval = fake_runtime.adapted.eval
    def changed_eval():
        (Path(artifacts.config["output_dir"]) / "adapter" / "adapter_model.safetensors").write_bytes(b"modified")
        return original_eval()
    fake_runtime.adapted.eval = changed_eval
    with pytest.raises(TrainingGateError, match="adapter_changed"):
        cold.cold_load(artifacts.config_path, artifacts.sha, artifacts.receipt)
    assert load_json(artifacts.receipt)["phase"] == "reverify_inputs"


def test_frozen_trainer_source_pin_is_enforced(artifacts, monkeypatch):
    monkeypatch.setattr(cold, "TRAINING_SOURCE_SHA256", "0" * 64)
    with pytest.raises(TrainingGateError, match="training_source_changed"):
        cold.verify_inputs(artifacts.config_path, artifacts.sha)


def test_package_pin_check_runs_before_torch_import(artifacts, monkeypatch):
    monkeypatch.setattr(cold.importlib.metadata, "version", lambda name: "wrong")
    with pytest.raises(TrainingGateError, match="runtime_version_mismatch:torch"):
        cold.runtime()
