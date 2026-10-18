"""Generated-only actual 27B test of the exact NF4 initialization helper."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def prove(model_dir: Path, output: Path, image: str, manifest: Path, helper_hash: str) -> dict:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    import torch
    from PIL import Image
    from . import qwen38_proof
    from .guards import file_sha256, require
    from .train import check_local_model, configure_adapter, load_model_and_processor
    from .data import SilverCompletionCollator
    from .architectures import validate_adapter_gradients
    from .nf4_freeze_diagnostic import raw_tensor_hash, write

    require(file_sha256(Path(qwen38_proof.__file__)) == helper_hash, "initialization_helper_pin_mismatch")
    require(not output.exists(), "initialization_proof_output_already_exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    config = {"model_dir": str(model_dir), "processor_dir": str(model_dir), "model_type": "qwen3_5",
              "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
              "processor_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
              "model_manifest": str(manifest), "quantization": "nf4", "required_compute_capability": [8, 0],
              "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05}
    torch.manual_seed(20261017)
    torch.cuda.manual_seed_all(20261017)
    torch.cuda.reset_peak_memory_stats()
    model_path, processor_path = check_local_model(config)
    model, processor = load_model_and_processor(config, model_path, processor_path)
    model, trainable = configure_adapter(model, config)
    initialization = qwen38_proof.canonicalize_nf4_bias_storage(model)
    write(output / "canonicalization.json", initialization)
    print(json.dumps({"stage": "canonicalization_completed", "biases": initialization["biases_storage_changed"]}), flush=True)
    before = qwen38_proof.frozen_snapshot(model)
    write(output / "before_generated_forward.frozen.json", before)
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
    model.eval()
    with torch.no_grad():
        require(bool(torch.isfinite(model(**batch).loss).item()), "initialization_generated_readonly_loss_nonfinite")
    after_readonly = qwen38_proof.frozen_snapshot(model)
    write(output / "after_readonly.frozen.json", after_readonly)
    readonly_diff = qwen38_proof.compare_frozen_snapshots(before, after_readonly)
    write(output / "after_readonly.diff.json", readonly_diff)
    require(readonly_diff["unchanged"], "initialization_frozen_changed_on_readonly_forward")
    adapter_before = {name: raw_tensor_hash(parameter) for name, parameter in trainable}
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1e-5, weight_decay=0.01)
    model.train()
    losses, gradients = [], []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        require(bool(torch.isfinite(loss).item()), "initialization_generated_loss_nonfinite")
        loss.backward()
        gradients.append(validate_adapter_gradients(model, inventory=model._forte_lora_target_inventory))
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        print(json.dumps({"stage": "generated_optimizer_step_completed", "step": step + 1}), flush=True)
    after = qwen38_proof.frozen_snapshot(model)
    write(output / "after_two_steps.frozen.json", after)
    difference = qwen38_proof.compare_frozen_snapshots(before, after)
    write(output / "after_two_steps.diff.json", difference)
    require(difference["unchanged"], "initialization_frozen_changed_after_generated_steps")
    changed = [name for name, parameter in trainable if raw_tensor_hash(parameter) != adapter_before[name]]
    groups = sorted({group for name in changed for group in ("self_attn", "linear_attn", "mlp")
                     if "." + group + "." in name})
    require(groups == ["linear_attn", "mlp", "self_attn"], "initialization_adapter_group_unchanged")
    result = {"schema_version": "forte.qwen38.nf4-initialization-proof.v1", "status": "passed",
              "private_input_used": False, "full_model_id": "Qwen/Qwen3.8-27B", "model_id": "Qwen/Qwen3.8-27B",
              "base_revision": config["base_revision"],
              "base_manifest_sha256": file_sha256(manifest), "runtime_image_sha256": image,
              "helper_source_sha256": helper_hash, "proof_source_sha256": file_sha256(Path(__file__)),
              "optimizer_steps": 2, "readonly_generated_forwards": 1,
              "frozen_postcanonical_bytes_unchanged": True, "frozen_tensors_unchanged": True,
              "all_initialization_values_exactly_preserved": initialization["all_values_exactly_preserved"],
              "frozen_post_initialization_bytes_unchanged": True, "quantization_state_unchanged": True,
              "frozen_parameters_checked": difference["parameters_checked"],
              "canonicalization": initialization, "initialization": initialization,
              "gradients": gradients, "losses": losses,
              "changed_adapter_tensors": len(changed), "changed_adapter_groups": groups,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "elapsed_seconds": time.monotonic() - started}
    write(output / "initialization-proof.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-image-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initialization-helper-sha256", required=True)
    args = parser.parse_args()
    result = prove(args.model_dir, args.output, args.runtime_image_sha256, args.manifest,
                   args.initialization_helper_sha256)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"canonicalization", "initialization", "gradients"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
