"""CPU regression for the real PEFT inference adapter activation contract.

Uses only a generated Linear module and a temporary generated adapter. It does
not load Qwen, use CUDA, read a private checkpoint, or run model generation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def run_cpu_regression() -> dict:
    import peft
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, load_peft_weights

    class TinyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(7, 5, bias=False, device="cpu")

        def forward(self, values):
            return self.projection(values)

    def digest_tensors(model):
        return {
            name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in model.named_parameters()
        }

    torch.manual_seed(20261017)
    base = TinyBase()
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    source = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["projection"]))
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if ".lora_" in name:
                parameter.copy_(torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter) / 100)
    values = torch.arange(14, dtype=torch.float32).reshape(2, 7) / 10

    with tempfile.TemporaryDirectory(prefix="forte-peft-cpu-") as temporary:
        source.save_pretrained(temporary, safe_serialization=True)
        expected = load_peft_weights(temporary, device="cpu", local_files_only=True)

        def reload_adapter():
            fresh_base = TinyBase()
            fresh_base.load_state_dict(base_state)
            model = PeftModel.from_pretrained(
                fresh_base, temporary, adapter_name="candidate", is_trainable=False, local_files_only=True
            )
            assert all(not parameter.requires_grad for parameter in model.parameters())
            loaded = {
                name: value for name, value in model.named_parameters()
                if ".lora_A.candidate." in name or ".lora_B.candidate." in name
            }
            assert len(loaded) == len(expected) == 2
            assert all(torch.equal(value.detach(), expected[name.replace(".candidate.", ".")])
                       for name, value in loaded.items())
            return model

        old = reload_adapter()
        old_before = digest_tensors(old)
        with torch.inference_mode():
            old_output_before = old(values).clone()
        old.set_adapter("candidate")
        old.eval()
        old_trainable = [name for name, parameter in old.named_parameters() if parameter.requires_grad]
        assert len(old_trainable) == 2 and all(".lora_" in name for name in old_trainable)
        assert digest_tensors(old) == old_before
        with torch.inference_mode():
            assert torch.equal(old(values), old_output_before)

        fixed = reload_adapter()
        fixed_before = digest_tensors(fixed)
        fixed.set_adapter("candidate", inference_mode=True)
        fixed.eval()
        assert fixed.active_adapter == "candidate"
        assert all(not parameter.requires_grad for parameter in fixed.parameters())
        assert digest_tensors(fixed) == fixed_before == old_before
        with torch.inference_mode():
            assert torch.equal(fixed(values), old_output_before)

        # inference_mode=True freezes the adapter; it must not silently conceal
        # a separate unexpected trainable base parameter from the serving guard.
        base_parameter = next(parameter for name, parameter in fixed.named_parameters() if ".lora_" not in name)
        base_parameter.requires_grad_(True)
        fixed.set_adapter("candidate", inference_mode=True)
        assert base_parameter.requires_grad is True
        assert not all(not parameter.requires_grad for parameter in fixed.parameters())
        assert digest_tensors(fixed) == fixed_before

    return {
        "schema_version": "forte.qwen38.peft-inference-cpu-regression.v1",
        "status": "passed",
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
        "generated_linear_shape": [7, 5],
        "generated_lora_rank": 2,
        "adapter_tensor_count": 2,
        "old_default_unfreezes_active_adapter": True,
        "eval_does_not_freeze_adapter": True,
        "explicit_inference_mode_freezes_adapter": True,
        "all_parameter_bytes_unchanged": True,
        "cpu_forward_output_exactly_equal": True,
        "unexpected_trainable_base_still_detected": True,
        "private_model_or_adapter_read": False,
        "qwen_loaded": False,
        "cuda_used": False,
        "model_generation_calls": 0,
    }


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("peft"),
                     "real torch/peft CPU dependencies are unavailable")
class PeftInferenceActivationTests(unittest.TestCase):
    def test_real_peft_activation_preserves_values_and_freezes_only_inference_adapter(self):
        result = run_cpu_regression()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["unexpected_trainable_base_still_detected"])


if __name__ == "__main__":
    print(json.dumps(run_cpu_regression(), sort_keys=True))
