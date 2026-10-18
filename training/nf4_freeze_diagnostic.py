"""Generated-only full-base NF4 freeze diagnosis; no private-data mount or calls.

The output records raw storage, quantization-state and dequantized fingerprints.
It must not be treated as a private training run, quality result or selected model.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any

from .guards import canonical_sha256, file_sha256, require


def raw_tensor_hash(value: Any) -> str:
    """Drop Tensor subclasses before moving bytes, to avoid subclass .cpu()."""
    import torch
    tensor = value.detach().as_subclass(torch.Tensor).contiguous().reshape(-1).view(torch.uint8).cpu()
    return hashlib.sha256(memoryview(tensor.numpy()).cast("B")).hexdigest()


def legacy_tensor_hash(value: Any) -> str:
    import torch
    tensor = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(tensor).cast("B")).hexdigest()


def quant_state_fingerprint(state: Any) -> Any:
    if state is None:
        return None
    import torch
    if isinstance(state, torch.Tensor):
        return {"shape": list(state.shape), "dtype": str(state.dtype), "sha256": raw_tensor_hash(state)}
    if isinstance(state, dict):
        return {str(key): quant_state_fingerprint(value) for key, value in sorted(state.items())}
    if isinstance(state, (tuple, list)):
        return [quant_state_fingerprint(value) for value in state]
    if isinstance(state, (str, int, float, bool)) or state is None:
        return state
    if hasattr(state, "as_dict"):
        return quant_state_fingerprint(state.as_dict(packed=False))
    return str(state)


def logical_tensor_hash(parameter: Any) -> str:
    """Canonical FP32 values, dequantized using this parameter's exact state."""
    import torch
    state = getattr(parameter, "quant_state", None)
    if state is not None:
        from bitsandbytes.functional import dequantize_4bit
        value = dequantize_4bit(parameter.detach().as_subclass(torch.Tensor), state)
    else:
        value = parameter.detach().as_subclass(torch.Tensor)
    return raw_tensor_hash(value.to(torch.float32)) if value.is_floating_point() else raw_tensor_hash(value)


def snapshot(model: Any, *, reference: dict | None = None, include_trainable: bool = False) -> dict:
    """Full raw/state coverage; recompute logical values for every storage change."""
    records, legacy_mutations = {}, []
    started = time.monotonic()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not include_trainable:
            continue
        state_before = quant_state_fingerprint(getattr(parameter, "quant_state", None))
        raw_before = raw_tensor_hash(parameter)
        previous = reference.get("parameters", {}).get(name) if reference else None
        unchanged_before = previous is not None and previous["raw_sha256"] == raw_before \
            and previous["dtype"] == str(parameter.dtype) and previous["shape"] == list(parameter.shape) \
            and previous["quant_state_sha256"] == canonical_sha256(state_before)
        logical_before = previous["logical_sha256"] if unchanged_before else logical_tensor_hash(parameter)
        legacy = legacy_tensor_hash(parameter)
        raw_after = raw_tensor_hash(parameter)
        state_after = quant_state_fingerprint(getattr(parameter, "quant_state", None))
        if raw_before != raw_after or state_before != state_after:
            legacy_mutations.append(name)
        record = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                  "class": type(parameter).__module__ + "." + type(parameter).__name__,
                  "requires_grad": bool(parameter.requires_grad), "raw_sha256": raw_after,
                  "legacy_raw_sha256": legacy, "quant_state": state_after,
                  "quant_state_sha256": canonical_sha256(state_after),
                  "bnb_quantized": getattr(parameter, "bnb_quantized", None),
                  "quant_type": getattr(parameter, "quant_type", None)}
        identical = previous is not None and all(record.get(key) == previous.get(key)
            for key in ("shape", "dtype", "raw_sha256", "quant_state_sha256"))
        record["logical_sha256"] = logical_before if raw_before == raw_after and state_before == state_after \
            else logical_tensor_hash(parameter)
        record["legacy_hash_preserved_logical_values"] = record["logical_sha256"] == logical_before
        record["logical_proof"] = "identical_raw_dtype_shape_and_quant_state" if identical else "dequantized_fp32_hash"
        records[name] = record
    return {"schema_version": "forte.nf4.parameter-snapshot.v1", "parameters": records,
            "parameters_sha256": canonical_sha256(records), "legacy_hash_mutated_parameters": legacy_mutations,
            "elapsed_seconds": time.monotonic() - started}


def compare(before: dict, after: dict) -> dict:
    prior, current = before["parameters"], after["parameters"]
    require(set(prior) == set(current), "nf4_diagnostic_parameter_names_changed")
    changes = []
    for name, previous in prior.items():
        record = current[name]
        fields = [key for key in ("shape", "dtype", "raw_sha256", "legacy_raw_sha256", "quant_state_sha256")
                  if previous.get(key) != record.get(key)]
        if fields or previous["logical_sha256"] != record["logical_sha256"]:
            changes.append({"name": name, "changed_fields": fields,
                "logical_equal": previous["logical_sha256"] == record["logical_sha256"],
                "before": previous, "after": record})
    return {"changed_parameter_count": len(changes), "logical_changed_parameter_count":
            sum(not item["logical_equal"] for item in changes), "changes": changes}


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def diagnose(model_dir: Path, output: Path, image: str, manifest: Path) -> dict:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    import torch
    import bitsandbytes
    from PIL import Image
    from .train import check_local_model, configure_adapter, load_model_and_processor
    from .data import SilverCompletionCollator
    from .architectures import validate_adapter_gradients

    require(not output.exists(), "nf4_diagnostic_output_already_exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    result = {"schema_version": "forte.nf4.generated-freeze-diagnostic.v1", "private_input_used": False,
              "generated_forward_recipe": "two solid-color JPEGs and supported/unclear synthetic physical facts",
              "runtime_image_sha256": image, "status": "running", "optimizer_steps": 0,
              "source_sha256": file_sha256(Path(__file__))}
    write(output / "diagnostic.json", result)
    config = {"model_dir": str(model_dir), "processor_dir": str(model_dir), "model_type": "qwen3_5",
              "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
              "processor_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
              "model_manifest": str(manifest), "quantization": "nf4", "required_compute_capability": [8, 0],
              "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05}
    torch.manual_seed(20261017)
    torch.cuda.manual_seed_all(20261017)
    torch.cuda.reset_peak_memory_stats()
    # Verify the complete separate public inventory before model allocation.
    model_path, processor_path = check_local_model(config)
    model, processor = load_model_and_processor(config, model_path, processor_path)
    result["runtime_load"] = model._forte_runtime_load_receipt
    model, trainable = configure_adapter(model, config)
    result["trainable_parameters"] = sum(parameter.numel() for _, parameter in trainable)
    source = inspect.getsourcefile(bitsandbytes.nn.Linear4bit)
    result["bitsandbytes_source_sha256"] = file_sha256(Path(source))
    write(output / "Linear4bit.forward.public-source.txt", {"source": inspect.getsource(bitsandbytes.nn.Linear4bit.forward)})
    frames = []
    for index, pts in enumerate((0.25, 1.25)):
        path = output / f"generated-{index}.jpg"
        Image.new("RGB", (64, 64), (index * 100, 50, 150)).save(path)
        frames.append({"path": path.name, "frame_id": f"generated-{index}", "pts_sec": pts})
    row = {"frames": frames, "prompt": "Return requested synthetic physical fact JSON.", "split": "train",
           "annotation_origin": "local_model", "human_reviewed": False, "target_scope": "exact_anchor_frame",
           "focal_frame_id": "generated-0", "fact_supervision_mask": {"CASH_VISIBLE": True,
            "CASH_COUNT_VISIBLE": False}, "target": {"facts": [{"fact_id": "CASH_VISIBLE", "state": "present",
            "supporting_frame_ids": ["generated-0"]}, {"fact_id": "CASH_COUNT_VISIBLE", "state": "unclear",
            "supporting_frame_ids": []}], "candidates": []}}
    batch = SilverCompletionCollator(processor, output, model_type="qwen3_5", max_pixels=65536)([row])
    batch = {name: tensor.to("cuda") for name, tensor in batch.items()}
    result["input_tokens"] = int(batch["input_ids"].shape[-1])
    result["supervised_tokens"] = int((batch["labels"] != -100).sum().item())
    snapshots = {}
    previous = None
    for stage in ("before_forward", "after_first_readonly_forward", "after_second_readonly_forward"):
        if stage != "before_forward":
            model.eval()
            with torch.no_grad():
                require(bool(torch.isfinite(model(**batch).loss).item()), "nf4_generated_readonly_loss_nonfinite")
        print(json.dumps({"stage": stage, "status": "hashing_all_parameters"}), flush=True)
        current = snapshot(model, reference=previous)
        write(output / (stage + ".json"), current)
        snapshots[stage] = current
        if previous is not None:
            diff = compare(previous, current)
            write(output / (stage + ".diff.json"), diff)
            result[stage + "_diff"] = {key: value for key, value in diff.items() if key != "changes"}
        previous = current
        write(output / "diagnostic.json", result)
    result["hash_function_mutations"] = {stage: value["legacy_hash_mutated_parameters"]
                                         for stage, value in snapshots.items()}
    readonly_diff = compare(snapshots["before_forward"], snapshots["after_first_readonly_forward"])
    stabilized_diff = compare(snapshots["after_first_readonly_forward"], snapshots["after_second_readonly_forward"])
    require(readonly_diff["logical_changed_parameter_count"] == 0, "nf4_generated_readonly_logical_weights_changed")
    require(stabilized_diff["changed_parameter_count"] == 0, "nf4_generated_storage_not_stable_after_warmup")
    adapters_before = {name: raw_tensor_hash(parameter) for name, parameter in trainable}
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1e-5, weight_decay=0.01)
    model.train()
    gradient_proofs, losses = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        require(bool(torch.isfinite(loss).item()), "nf4_generated_training_loss_nonfinite")
        loss.backward()
        gradient_proofs.append(validate_adapter_gradients(model, inventory=model._forte_lora_target_inventory))
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
        optimizer.step()
        result["optimizer_steps"] += 1
        losses.append(float(loss.item()))
        write(output / "diagnostic.json", result)
    after = snapshot(model, reference=previous)
    write(output / "after_two_generated_optimizer_steps.json", after)
    final_diff = compare(previous, after)
    write(output / "after_two_generated_optimizer_steps.diff.json", final_diff)
    result.update(after_optimizer_diff={key: value for key, value in final_diff.items() if key != "changes"},
                  gradients=gradient_proofs, losses=losses,
                  changed_adapter_tensors=sum(raw_tensor_hash(parameter) != adapters_before[name]
                                               for name, parameter in trainable),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  elapsed_seconds=time.monotonic() - started)
    require(final_diff["changed_parameter_count"] == 0, "nf4_frozen_storage_changed_after_generated_steps")
    result["status"] = "passed"
    write(output / "diagnostic.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-image-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = diagnose(args.model_dir, args.output, args.runtime_image_sha256, args.manifest)
        print(json.dumps({key: value for key, value in result.items() if key not in {"runtime_load", "gradients"}}))
        return 0
    except Exception as exc:
        # Generated-only traceback is retained by the operator in protected logs.
        import traceback
        traceback.print_exc()
        print(json.dumps({"status": "failed", "reason": str(exc), "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
