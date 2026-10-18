"""Synthetic proof/quarantine checks; no private model operation is performed."""
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.qwen38_proof import compare_frozen_snapshots, save_quarantined_adapter


def snapshot(records):
    return {"schema_version": "forte.qwen38.frozen-snapshot.v1",
            "scope": "all_frozen_named_parameters_and_quant_state", "parameters": records,
            "parameters_sha256": canonical_sha256(records)}


def test_full_frozen_diff_keeps_all_changed_names_and_values():
    before = snapshot({"bias": {"raw_sha256": "a", "dtype": "float32"},
                       "weight": {"raw_sha256": "b", "dtype": "uint8"}})
    after = snapshot({"bias": {"raw_sha256": "c", "dtype": "bfloat16"},
                      "weight": {"raw_sha256": "d", "dtype": "uint8"}})
    result = compare_frozen_snapshots(before, after)
    assert result["unchanged"] is False
    assert result["changed_parameter_count"] == 2
    assert [item["name"] for item in result["changes"]] == ["bias", "weight"]
    assert result["changes"][0]["before"]["dtype"] == "float32"
    assert result["changes"][0]["after"]["dtype"] == "bfloat16"


def test_frozen_added_missing_and_tampered_snapshots_fail_closed():
    before, after = snapshot({"weight": {"raw_sha256": "a"}}), snapshot({"bias": {"raw_sha256": "b"}})
    result = compare_frozen_snapshots(before, after)
    assert result["missing_names"] == ["weight"] and result["added_names"] == ["bias"]
    after["parameters"]["bias"]["raw_sha256"] = "tampered"
    with pytest.raises(TrainingGateError, match="integrity"):
        compare_frozen_snapshots(before, after)


def test_identical_frozen_proof_is_exact():
    values = snapshot({"weight": {"raw_sha256": "a", "quant_state_sha256": "b"}})
    assert compare_frozen_snapshots(values, deepcopy(values))["unchanged"] is True
    changed = snapshot({"weight": {"raw_sha256": "a", "quant_state_sha256": "different"}})
    assert compare_frozen_snapshots(values, changed)["unchanged"] is False


def test_quarantine_preserves_weights_but_forbids_selection_and_zero_steps():
    class Model:
        peft_config = {"default": SimpleNamespace()}
        def save_pretrained(self, directory, *, safe_serialization):
            assert safe_serialization
            directory.mkdir(parents=True)
            (directory / "adapter_model.safetensors").write_bytes(b"synthetic-stepped-weights")
    config = {"base_revision": "a" * 40, "model_id": "Qwen/Qwen3.8-27B", "runtime_image_sha256": "b" * 64,
              "code_archive_sha256": "c" * 64}
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        result = save_quarantined_adapter(Model(), output, config, reason_code="silver_frozen_tensors_changed",
                                         failure_stage="frozen_after", optimizer_steps=2)
        assert result["status"] == "quarantined" and result["selection"] == "forbidden"
        assert result["serving_eligible"] is False and result["optimizer_steps"] == 2
        assert result["frozen_base_verified_by_this_quarantine"] is False
        assert (output / "quarantine/adapter/adapter_model.safetensors").is_file()
        with pytest.raises(TrainingGateError, match="already_exists"):
            save_quarantined_adapter(Model(), output, config, reason_code="failed", failure_stage="failed", optimizer_steps=2)
        with pytest.raises(TrainingGateError, match="zero_step"):
            save_quarantined_adapter(Model(), output, config, reason_code="failed", failure_stage="failed", optimizer_steps=0)


def test_exact_nf4_bias_initialization_and_postbaseline_byte_guard():
    torch = pytest.importorskip("torch")
    from training.qwen38_proof import canonicalize_nf4_bias_storage, frozen_snapshot
    class Linear4bit(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32), requires_grad=False)
            self.compute_dtype = torch.bfloat16
    model = torch.nn.Sequential(Linear4bit([0.5, 2.0, -8.0]))
    bnb = SimpleNamespace(__version__="synthetic", nn=SimpleNamespace(Linear4bit=Linear4bit))
    with patch.dict("sys.modules", {"bitsandbytes": bnb}):
        receipt = canonicalize_nf4_bias_storage(model)
        assert receipt["biases_storage_changed"] == 1 and receipt["all_values_exactly_preserved"]
        before = frozen_snapshot(model)
        assert compare_frozen_snapshots(before, frozen_snapshot(model))["unchanged"]
        model[0].bias.data.add_(1)
        assert compare_frozen_snapshots(before, frozen_snapshot(model))["unchanged"] is False
        with pytest.raises(TrainingGateError, match="not_exactly_reversible"):
            canonicalize_nf4_bias_storage(torch.nn.Sequential(Linear4bit([0.1234567])))
        model[0].bias.requires_grad_(True)
        with pytest.raises(TrainingGateError, match="scope_invalid"):
            canonicalize_nf4_bias_storage(model)


def test_nf4_generated_proof_requires_exact_full_base_image_and_helper():
    from training import qwen38_proof
    from training.qwen38_silver import verify_nf4_initialization_proof
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "proof.json"
        config = {"quantization": "nf4", "model_id": "Qwen/Qwen3.8-27B", "base_revision": "a" * 40,
                  "runtime_image_sha256": "b" * 64, "nf4_bias_initialization_proof": str(path)}
        proof = {"schema_version": "forte.qwen38.nf4-initialization-proof.v1", "status": "passed",
            "private_input_used": False, "model_id": config["model_id"], "base_revision": config["base_revision"],
            "base_manifest_sha256": "c" * 64, "runtime_image_sha256": config["runtime_image_sha256"],
            "helper_source_sha256": file_sha256(Path(qwen38_proof.__file__)), "optimizer_steps": 2,
            "all_initialization_values_exactly_preserved": True, "frozen_post_initialization_bytes_unchanged": True,
            "quantization_state_unchanged": True, "changed_adapter_groups": ["self_attn", "linear_attn", "mlp"]}
        path.write_text(json.dumps(proof))
        config["expected_nf4_bias_initialization_proof_sha256"] = file_sha256(path)
        assert verify_nf4_initialization_proof(config, base_manifest_sha256="c" * 64)["optimizer_steps"] == 2
        proof["helper_source_sha256"] = "d" * 64
        path.write_text(json.dumps(proof))
        config["expected_nf4_bias_initialization_proof_sha256"] = file_sha256(path)
        with pytest.raises(TrainingGateError, match="proof_invalid"):
            verify_nf4_initialization_proof(config, base_manifest_sha256="c" * 64)


def test_failure_codes_never_copy_raw_exception_text():
    from training.qwen38_silver import safe_failure_code
    assert safe_failure_code(TrainingGateError("silver_frozen_tensors_changed")) == "silver_frozen_tensors_changed"
    assert safe_failure_code(TrainingGateError("private filename and target text")) == "silver_local_training_error"
    assert safe_failure_code(RuntimeError("private observation")) == "silver_local_training_error"


def test_two_real_cpu_steps_frozen_failure_persists_diff_metrics_and_quarantine():
    """Exercise the actual phase2 failure path with generated CPU tensors."""
    torch = pytest.importorskip("torch")
    from training.qwen38_silver import train
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            for group in ("self_attn", "linear_attn", "mlp"):
                child = torch.nn.Module()
                child.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2)))
                setattr(self.language_model, group, child)
            self.base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
            self.peft_config = {"default": SimpleNamespace()}
            self._forte_runtime_load_receipt = {"synthetic": True}
            self._forte_lora_target_inventory = {"synthetic": True}
            self.calls = 0
        def forward(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                with torch.no_grad():
                    self.base.add_(1)
            return SimpleNamespace(loss=sum((parameter ** 2).sum()
                for parameter in self.parameters() if parameter.requires_grad))
        def save_pretrained(self, directory, *, safe_serialization):
            directory.mkdir(parents=True)
            (directory / "adapter_model.safetensors").write_bytes(b"actual-cpu-stepped-fixture")
    class Collator:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, rows):
            return {"input_ids": SimpleNamespace(to=lambda device: torch.ones(1, dtype=torch.long))}
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary)
        model_dir, media = root / "model", root / "media"
        model_dir.mkdir(); media.mkdir()
        manifest, runtime, reservation = root / "manifest.json", root / "runtime.json", root / "reservation.json"
        for path in (manifest, runtime, reservation): path.write_text("{}")
        config = {"model_manifest": str(manifest), "runtime_lock": str(runtime),
            "expected_runtime_lock_sha256": file_sha256(runtime), "quantization": "none",
            "output_dir": str(root / "run"), "media_root": str(media), "base_revision": "a" * 40,
            "processor_revision": "a" * 40, "runtime_image_sha256": "b" * 64, "code_archive_sha256": "c" * 64,
            "expected_attempt_plan_sha256": "d" * 64, "attempt_reservation": str(reservation),
            "max_frames": 4, "max_pixels_per_image": 65536, "max_length": 4096, "seed": 20261017,
            "max_wall_seconds": 1800, "stop_utc": "2026-10-18T06:01:25Z", "max_steps": 2,
            "learning_rate": 1e-5, "gradient_accumulation_steps": 1, "model_id": "Qwen/Qwen3.8-27B"}
        model = Model()
        patches = {
            "training.qwen38_silver.load_inputs": ([{"synthetic": True}] * 2, {},
                {"coverage": {"target_rows_sha256": "e" * 64}, "supervision": {
                    "supervision_objective_sha256": "f" * 64, "effective_rows_sha256": "1" * 64,
                    "effective_fact_weights_sha256": "2" * 64, "balance_plan": None, "alias_receipt": None}}),
            "training.train.check_local_model": (model_dir, model_dir),
            "training.qwen38_silver.validate_compute_release": {"expires_at": config["stop_utc"]},
            "training.qwen38_silver.validate_attempt_accounting": {"requested_attempts": 4,
                "attempts": [{"attempt_id": str(index), "split": "train"} for index in range(4)]},
            "training.qwen38_silver._verify_smoke_proof": None,
            "training.runtime_check.verify_runtime": {"status": "passed"},
            "training.train.load_model_and_processor": (model, None),
            "training.train.configure_adapter": (model, [(name, parameter) for name, parameter
                                                          in model.named_parameters() if parameter.requires_grad]),
            "training.architectures.validate_adapter_gradients": {"synthetic": True},
            "torch.cuda.manual_seed_all": None, "torch.cuda.reset_peak_memory_stats": None,
            "torch.cuda.synchronize": None, "torch.cuda.get_device_name": "Synthetic CPU test",
            "torch.cuda.is_initialized": True, "torch.cuda.max_memory_allocated": 123,
            "torch.cuda.max_memory_reserved": 456}
        for name, value in patches.items(): stack.enter_context(patch(name, return_value=value))
        stack.enter_context(patch("training.data.SilverCompletionCollator", Collator))
        stack.enter_context(patch("training.qwen38_silver.datetime", Clock))
        stack.enter_context(patch("training.qwen38_silver.sys.platform", "linux"))
        stack.enter_context(patch("training.qwen38_silver.os.statvfs", return_value=SimpleNamespace(f_flag=1), create=True))
        stack.enter_context(patch("training.qwen38_silver.os.ST_RDONLY", 1, create=True))
        with pytest.raises(TrainingGateError, match="silver_frozen_tensors_changed"):
            train(config, smoke=True)
        run = json.loads((root / "run/run.json").read_text())
        assert run["optimizer_steps"] == 2 and run["attempted_forwards"] == 2
        assert run["status"] == "failed" and run["reason_code"] == "silver_frozen_tensors_changed"
        assert run["failure_stage"] == "frozen_after_compare"
        assert run["peak_vram_bytes"] == 123 and run["peak_reserved_vram_bytes"] == 456
        assert len(run["losses"]) == 2 and len(run["optimizer_step_metrics"]) == 2
        assert run["frozen_tensors_unchanged"] is False
        assert [item["status"] for item in run["attempt_statuses"]] == ["completed", "completed",
            "cancelled_before_attempt", "cancelled_before_attempt"]
        difference = json.loads((root / "run/proof/frozen.difference.json").read_text())
        assert difference["changes"][0]["name"] == "base"
        quarantine = json.loads((root / "run/quarantine/receipt.json").read_text())
        assert quarantine["optimizer_steps"] == 2 and quarantine["selection"] == "forbidden"
        assert quarantine["serving_eligible"] is False
