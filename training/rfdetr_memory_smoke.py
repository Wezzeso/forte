"""One synthetic RF-DETR optimizer step under a bounded CUDA allocator.

This is a memory/runtime smoke, not a detector fit or an accuracy result. It uses
two generated boxes and the official local COCO checkpoint. Run each variant in
a fresh process. The host launcher owns admission control and its GPU lock.

The PyTorch allocator cap is the first CUDA runtime entry, before model/tensor
placement on CUDA. PyTorch's setter initializes the CUDA context internally; it
cannot cap CUDA context, library, or another process's allocations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator
from unittest.mock import patch


MIB = 1024**2
MAX_ALLOCATOR_MIB = 5120
MIN_GLOBAL_FREE_MIB = 6656
CLASS_NAMES = ["single_banknote", "id_card"]
OFFICIAL_MD5 = {
    "nano": "fb6504cce7fbdc783f7a46991f07639f",
    "small": "fb37061c1af7bace359c91b723a8d5c1",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(OFFICIAL_MD5), default="nano")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--allocator-limit-mib", type=int, default=MAX_ALLOCATOR_MIB)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_sha256):
        parser.error("--expected-sha256 must contain exactly 64 hexadecimal digits")
    if not 0 < args.allocator_limit_mib <= MAX_ALLOCATOR_MIB:
        parser.error(f"--allocator-limit-mib must be between 1 and {MAX_ALLOCATOR_MIB}")
    args.expected_sha256 = args.expected_sha256.lower()
    args.weights = args.weights.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    if args.weights == args.report:
        parser.error("--report must differ from --weights")
    return args


def file_digests(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            sha256.update(block)
            md5.update(block)
    return sha256.hexdigest(), md5.hexdigest()


def gpu_snapshot(selected_uuid: str | None = None) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    if selected_uuid is not None:
        command.append(f"--id={selected_uuid}")
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    rows = list(csv.reader(io.StringIO(result.stdout.strip())))
    require(bool(rows), "nvidia-smi returned no GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if len(rows) > 1:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        require(len(tokens) == 1, "Expose one GPU or set a single CUDA_VISIBLE_DEVICES entry")
        token = tokens[0]
        rows = [row for row in rows if row[0].strip() == token or row[1].strip().startswith(token)]
    require(len(rows) == 1, "Cannot unambiguously match the single CUDA GPU")
    row = [value.strip() for value in rows[0]]
    require(len(row) == 5, "Unexpected nvidia-smi GPU query output")
    total_mib, free_mib = int(row[3]), int(row[4])
    require(total_mib > 0 and free_mib >= 0, "Invalid nvidia-smi memory values")
    return {
        "index": int(row[0]),
        "uuid": row[1],
        "name": row[2],
        "total_mib": total_mib,
        "free_mib": free_mib,
    }


@contextmanager
def no_network() -> Iterator[None]:
    """Fail closed on Python network connections, including accidental telemetry."""
    def denied(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access is disabled during the synthetic memory smoke")

    with (
        patch.object(socket.socket, "connect", denied),
        patch.object(socket.socket, "connect_ex", denied),
        patch.object(socket, "create_connection", denied),
    ):
        yield


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as out:
            temporary_path = Path(out.name)
            json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_smoke(args: argparse.Namespace, report: dict[str, Any]) -> None:
    # These are set before importing Torch/Transformers/RF-DETR or logger packages.
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled",
        "OTEL_SDK_DISABLED": "true",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
    })
    # Keep the allocator limit meaningful; CUDA-async and expandable allocations
    # are outside the narrow native caching-allocator smoke tested here.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ["PYTORCH_ALLOC_CONF"] = "backend:native"

    require(args.weights.is_file(), "Local pretrained weights do not exist")
    actual_sha256, actual_md5 = file_digests(args.weights)
    report["weights_sha256"] = actual_sha256
    report["weights_md5"] = actual_md5
    require(actual_sha256 == args.expected_sha256, "Pretrained weights SHA-256 mismatch")
    require(actual_md5 == OFFICIAL_MD5[args.variant], "Weights do not match the official variant MD5")
    require(version("rfdetr") == "1.11.2", "This smoke requires the inspected rfdetr==1.11.2 API")

    with no_network():
        import torch
        from pytorch_lightning import Callback
        from torch.utils.data import DataLoader

        from rfdetr.config import RFDETRNanoConfig, RFDETRSmallConfig, TrainConfig
        from rfdetr.training import RFDETRModelModule, build_trainer
        from rfdetr.utilities.tensors import collate_fn

        report["runtime_versions"] = {
            "rfdetr": version("rfdetr"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "pytorch_lightning": version("pytorch-lightning"),
            "transformers": version("transformers"),
        }
        require(not torch.cuda.is_initialized(), "CUDA was initialized before the allocator cap")
        torch.manual_seed(20261018)
        config_class = RFDETRNanoConfig if args.variant == "nano" else RFDETRSmallConfig
        mc = config_class(
            pretrain_weights=str(args.weights),
            num_classes=len(CLASS_NAMES),
            device="cpu",
            freeze_encoder=True,
            gradient_checkpointing=False,
            compile=False,
            cuda_graphs=False,
        )
        tc = TrainConfig(
            dataset_dir=None,
            output_dir=str(args.report.parent / "unused-smoke-output"),
            class_names=CLASS_NAMES,
            epochs=1,
            batch_size=1,
            eval_batch_size=1,
            grad_accum_steps=1,
            amp_dtype="bf16",
            optimizer="adamw",
            use_ema=False,
            multi_scale=False,
            num_workers=0,
            accelerator="gpu",
            devices=1,
            tensorboard=False,
            wandb=False,
            mlflow=False,
            compute_train_metrics=False,
            compute_val_loss=False,
            save_dataset_grids=False,
            num_sanity_val_steps=0,
        )

        def existing_weights_only(path: str, redownload: bool = False, **kwargs: Any) -> None:
            require(not redownload, "Checkpoint load requested a forbidden redownload")
            require(Path(path).resolve() == args.weights, "Loader requested different pretrained weights")
            require(Path(path).is_file(), "Local pretrained weights disappeared")

        with patch("rfdetr.models.weights.download_pretrain_weights", existing_weights_only):
            module = RFDETRModelModule(mc, tc)
        require(all(p.device.type == "cpu" for p in module.parameters()), "Module construction placed parameters on GPU")
        require(not torch.cuda.is_initialized(), "CPU module construction unexpectedly initialized CUDA")
        require(mc.num_classes == 2, "Pretrained loader changed the explicit two-class configuration")
        require(module.model.class_embed.out_features == 3, "Expected two classes plus the background output")
        encoder_parameters = list(module.model.backbone[0].encoder.parameters())
        require(all(not p.requires_grad for p in encoder_parameters), "Encoder is not fully frozen")
        head_before = module.model.class_embed.weight.detach().cpu().clone()
        report["model"] = {
            "resolution": mc.resolution,
            "class_names": CLASS_NAMES,
            "head_outputs": module.model.class_embed.out_features,
            "encoder_frozen": True,
            "total_parameters": sum(p.numel() for p in module.parameters()),
            "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "gradient_checkpointing": False,
            "compile": False,
            "use_ema": False,
            "batch_size": 1,
            "target_boxes": 2,
        }

        before = gpu_snapshot()
        report["gpu"] = {key: value for key, value in before.items() if key != "free_mib"}
        report["global_free_before_mib"] = before["free_mib"]
        require(before["free_mib"] >= MIN_GLOBAL_FREE_MIB, f"GPU free memory is below {MIN_GLOBAL_FREE_MIB} MiB")
        require(args.allocator_limit_mib < before["total_mib"], "Allocator limit exceeds total GPU memory")
        limit_bytes = args.allocator_limit_mib * MIB
        # The setter initializes CUDA internally. It is intentionally called
        # before availability/property queries, Trainer construction or CUDA data.
        # The first cap is conservative about nvidia-smi's rounded MiB total.
        first_fraction = limit_bytes / ((before["total_mib"] + 1) * MIB)
        torch.cuda.set_per_process_memory_fraction(first_fraction, device=0)
        props = torch.cuda.get_device_properties(0)
        fraction = limit_bytes / props.total_memory
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        report["allocator_fraction"] = fraction
        report["cuda_total_memory_bytes"] = props.total_memory
        report["allocator_cap_set_before_cuda_model_allocation"] = True
        report["allocator_cap_before_context_creation"] = False
        require(torch.cuda.is_bf16_supported(), "The selected GPU does not support bf16")
        actual_free, _ = torch.cuda.mem_get_info(device=0)
        report["cuda_actual_free_before_fit_mib"] = actual_free / MIB
        require(actual_free >= MIN_GLOBAL_FREE_MIB * MIB, f"CUDA actual free memory is below {MIN_GLOBAL_FREE_MIB} MiB")

        class Probe(Callback):
            saw_backward = False
            loss: float | None = None
            head_gradient_finite = False
            head_gradient_l2: float | None = None

            def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
                grad = pl_module.model.class_embed.weight.grad
                require(grad is not None, "The classification head received no gradient")
                require(bool(torch.isfinite(grad).all()), "The classification head gradient is not finite")
                self.head_gradient_l2 = float(grad.detach().float().norm().cpu())
                require(self.head_gradient_l2 > 0, "The classification head gradient is zero")
                self.head_gradient_finite = True
                self.saw_backward = True

            def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                self.loss = float(loss.detach().float().cpu())
                require(math.isfinite(self.loss), "The genuine detection loss is not finite")

        side = mc.resolution
        image = torch.rand(3, side, side)
        target = {
            "boxes": torch.tensor([[0.33, 0.45, 0.25, 0.15], [0.68, 0.55, 0.22, 0.20]]),
            "labels": torch.tensor([0, 1], dtype=torch.int64),
            "image_id": torch.tensor([1]),
            "orig_size": torch.tensor([side, side]),
            "size": torch.tensor([side, side]),
            "area": torch.tensor([side**2 * 0.25 * 0.15, side**2 * 0.22 * 0.20]),
            "iscrowd": torch.tensor([0, 0]),
        }
        loader = DataLoader([(image, target)], batch_size=1, num_workers=0, collate_fn=collate_fn)
        probe = Probe()
        trainer = build_trainer(
            tc,
            mc,
            accelerator="gpu",
            include_training_callbacks=False,
            callbacks=[probe],
            max_steps=1,
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        require(str(trainer.precision) == "bf16-mixed", "Trainer did not select bf16-mixed precision")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(module, train_dataloaders=loader)
        torch.cuda.synchronize()
        require(trainer.global_step == 1, "Expected exactly one native optimizer step")
        require(probe.saw_backward and probe.head_gradient_finite, "A finite real backward pass was not observed")
        require(probe.loss is not None and math.isfinite(probe.loss), "A finite real detection loss was not observed")
        optimizer = trainer.optimizers[0]
        require(isinstance(optimizer, torch.optim.AdamW), "Native optimizer is not AdamW")
        stepped_states = sum(
            1 for state in optimizer.state.values()
            if "exp_avg" in state and "exp_avg_sq" in state and int(state["step"]) == 1
        )
        require(stepped_states > 0, "AdamW moment tensors and step counter were not initialized")
        head_updated = not torch.equal(head_before, module.model.class_embed.weight.detach().cpu())
        require(head_updated, "The optimizer step did not change the classification head")
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        require(peak_reserved <= limit_bytes, "Observed CUDA reservation exceeded the allocator limit")
        report.update({
            "precision": str(trainer.precision),
            "loss": probe.loss,
            "loss_finite": True,
            "backward_observed": probe.saw_backward,
            "head_gradient_finite": probe.head_gradient_finite,
            "head_gradient_l2": probe.head_gradient_l2,
            "head_updated": head_updated,
            "optimizer": type(optimizer).__name__,
            "optimizer_steps": trainer.global_step,
            "optimizer_states_stepped": stepped_states,
            "peak_allocated_mib": peak_allocated / MIB,
            "peak_reserved_mib": peak_reserved / MIB,
            "status": "passed",
        })


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": "forte.rfdetr.synthetic-memory-smoke.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "scope": "one synthetic optimizer step; no real-data fit or accuracy claim",
        "variant": args.variant,
        "weights": str(args.weights),
        "expected_sha256": args.expected_sha256,
        "expected_md5": OFFICIAL_MD5[args.variant],
        "allocator_limit_mib": args.allocator_limit_mib,
        "minimum_global_free_mib": MIN_GLOBAL_FREE_MIB,
        "allocator_limit_scope": "PyTorch native caching allocator only; CUDA contexts/libraries/other processes excluded",
        "network_policy": "Python socket connections denied; local weights only; download helper fails closed",
        "host_admission_control": "external launcher holds the GPU lock for the entire container lifetime",
        "checkpointing": False,
        "telemetry": False,
    }
    try:
        run_smoke(args, report)
    except Exception as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]})
    finally:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            try:
                report.setdefault("peak_allocated_mib", torch.cuda.max_memory_allocated() / MIB)
                report.setdefault("peak_reserved_mib", torch.cuda.max_memory_reserved() / MIB)
            except Exception:
                pass
        if "gpu" in report:
            try:
                after = gpu_snapshot(report["gpu"]["uuid"])
                report["global_free_after_mib"] = after["free_mib"]
            except Exception as exc:
                report["global_free_after_error"] = f"{type(exc).__name__}: {exc}"
                if report["status"] == "passed":
                    report.update({"status": "failed", "error_type": type(exc).__name__, "error": "Could not record final global GPU free memory"})
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(args.report, report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
