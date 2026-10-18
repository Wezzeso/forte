"""Bounded public-data RF-DETR pretraining; artifacts never authorize serving."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import operator
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import threading
import time

from .common import (canonical_sha256, cli_error, file_sha256, load_json, number,
                     pinned_json, relative_file, require, sha, write_json)
from .train import RFDETR_VERSION

SCHEMA = "forte.specialists.public-train.v1"
MAX_BUDGET_MIB = 5632


def load_public_dataset(root: Path, expected_sha256: str) -> dict:
    from .public_bundle import verify_fit_bundle
    return verify_fit_bundle(root, expected_sha256)


def remaining_seconds(config: dict, *, now: datetime | None = None) -> float:
    try:
        deadline = datetime.fromisoformat(config["stop_at_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        require(False, "public_training_utc_deadline_required")
    require(deadline.tzinfo is not None and deadline.utcoffset() == timedelta(0),
            "public_training_deadline_must_be_utc")
    remaining = (deadline - (now or datetime.now(timezone.utc))).total_seconds()
    require(remaining > 0, "public_training_deadline_elapsed")
    return remaining


def preflight(config: dict) -> tuple[Path, dict]:
    require(config.get("schema_version") == SCHEMA, "public_training_schema")
    require(config.get("rfdetr_version") == RFDETR_VERSION, "public_training_package_pin")
    require(config.get("model_class") in {"RFDETRNano", "RFDETRSmall"}, "public_training_model_class")
    require(config.get("recipe_predeclared") is True and config.get("recipe_reference"),
            "public_training_recipe_not_predeclared")
    require(type(config.get("defer_sealed_test", False)) is bool, "public_training_defer_flag_invalid")
    require(config.get("freeze_encoder") is True and config.get("amp_dtype") == "bf16"
            and config.get("batch_size") == 1 and config.get("eval_batch_size") == 1
            and config.get("multi_scale") is False and config.get("use_ema") is False,
            "public_training_bounded_recipe_required")
    for field in ("epochs", "grad_accum_steps", "seed"):
        require(type(config.get(field)) is int and config[field] >= (0 if field == "seed" else 1),
                "public_training_integer_parameter")
    require(type(config.get("resolution")) is int and 256 <= config["resolution"] <= 512
            and config["resolution"] % 32 == 0, "public_training_resolution")
    require(number(config.get("lr"), 1e-8, 1e-3), "public_training_learning_rate")
    require(type(config.get("gradient_checkpointing", False)) is bool
            and type(config.get("early_stopping_patience", 5)) is int
            and 2 <= config.get("early_stopping_patience", 5) <= 10, "public_training_optional_recipe_parameters")
    require(type(config.get("gpu_memory_budget_mib")) is int
            and 1024 <= config["gpu_memory_budget_mib"] <= MAX_BUDGET_MIB,
            "public_training_memory_budget")
    require(type(config.get("minimum_free_margin_mib")) is int and config["minimum_free_margin_mib"] >= 1024,
            "public_training_memory_margin")
    require(number(config.get("max_fit_seconds"), 1, 5 * 3600)
            and number(config.get("evaluation_reserve_seconds"), 300, 3600), "public_training_time_budget")
    remaining_seconds(config)
    require(isinstance(config.get("test_exposure_registry_dir"), str) and config["test_exposure_registry_dir"]
            and "://" not in config["test_exposure_registry_dir"], "public_training_test_exposure_registry_required")
    root = Path(config.get("dataset_dir", "")).resolve()
    lock = load_public_dataset(root, config.get("expected_dataset_lock_sha256", ""))
    require(lock.get("classes") and lock.get("class_names")
            == {str(i): name for i, name in enumerate(lock["classes"])}, "public_training_class_mapping")
    require(config.get("expected_classes") == lock["classes"], "public_training_expected_classes_mismatch")
    require(lock.get("schema_version") == "forte.specialists.public-dataset-lock.v1",
            "public_training_requires_public_data")
    from .public_augmentation import validate_policy
    validate_policy(config.get("train_data_policy"), lock, seed=config["seed"])
    weights = Path(config.get("pretrain_weights", "")).resolve()
    require(weights.is_file() and sha(config.get("expected_pretrain_weights_sha256"))
            and file_sha256(weights) == config["expected_pretrain_weights_sha256"],
            "public_training_pretrain_hash_mismatch")
    authorization = config.get("compute_release_authorization", {})
    require(authorization.get("released_by_A18_or_root") is True and authorization.get("release_reference"),
            "public_training_compute_not_released")
    require(config.get("device", "cuda:0") == "cuda:0", "public_training_device_must_be_local_zero")
    return root, lock


def parameter_sha256(module, *, trainable: bool | None = None) -> str:
    """Hash real tensor bytes, names, shapes and dtypes; supports bf16 tensors."""
    import torch
    digest = hashlib.sha256()
    matched = 0
    for name, parameter in sorted(module.named_parameters()):
        if trainable is not None and parameter.requires_grad is not trainable:
            continue
        require(bool(torch.isfinite(parameter.detach()).all()), "public_training_nonfinite_parameter")
        header = json.dumps([name, list(parameter.shape), str(parameter.dtype)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        data = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        matched += 1
    require(matched > 0, "public_training_parameter_group_empty")
    return digest.hexdigest()


def initialize_target_heads(module, classes: list[str]) -> dict:
    """Discard COCO semantic rows; retain pretrained localization/backbone.

    RF-DETR's upstream resize helper truncates/tiles source classifier weights.
    Resetting every target classifier explicitly is necessary before calling
    this a newly initialized target-class baseline.
    """
    import torch
    head = module.model.class_embed
    require(isinstance(head, torch.nn.Linear) and head.out_features == len(classes) + 1,
            "public_training_target_head_shape_mismatch")
    head.reset_parameters()
    with torch.no_grad():
        head.bias.fill_(-math.log(99.0))
    encoder_heads = getattr(module.model.transformer, "enc_out_class_embed", None)
    if getattr(module.model, "two_stage", False):
        require(encoder_heads is not None and len(encoder_heads) > 0, "public_training_encoder_class_heads_missing")
        for encoder_head in encoder_heads:
            require(isinstance(encoder_head, torch.nn.Linear) and encoder_head.weight.shape == head.weight.shape,
                    "public_training_encoder_class_head_shape_mismatch")
            encoder_head.load_state_dict(head.state_dict())
    module.model.class_names = list(classes)
    return {"method": "fresh_linear_reset_focal_prior_0.01", "target_head_rows": head.out_features,
            "encoder_classifier_count": len(encoder_heads) if encoder_heads is not None else 0,
            "localization_weights": "pretrained_preserved", "source_classifier_semantics": "discarded"}


def scalar_metrics(value: dict) -> dict[str, float]:
    result = {}
    for key, item in value.items():
        if hasattr(item, "detach"):
            item = item.detach().cpu().item()
        if number(item, -1e20, 1e20):
            result[str(key)] = float(item)
    return result


def model_and_train_kwargs(config: dict, root: Path, output: Path, classes: list[str]) -> tuple[dict, dict]:
    """Keep the pinned upstream constructor/training argument scopes explicit."""
    model = {"num_classes": len(classes), "model_name": config["model_class"],
             "pretrain_weights": str(Path(config["pretrain_weights"]).resolve()), "device": "cpu",
             "resolution": config["resolution"], "freeze_encoder": True,
             "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
             "compile": False, "cuda_graphs": False}
    training = {"dataset_dir": str(root), "output_dir": str(output), "epochs": config["epochs"],
                "batch_size": 1, "eval_batch_size": 1, "grad_accum_steps": config["grad_accum_steps"],
                "lr": config["lr"], "lr_encoder": 0.0, "seed": config["seed"], "amp_dtype": "bf16",
                "multi_scale": False, "num_workers": 0, "use_ema": False,
                "augmentation_backend": "torchvision", "scale_jitter": False, "expanded_scales": False,
                "wandb": False, "tensorboard": False, "mlflow": False, "clearml": False,
                "run_test": False, "save_dataset_grids": False, "class_names": classes,
                "eval_max_dets": 500,
                "log_per_class_metrics": True, "early_stopping": True,
                "early_stopping_patience": int(config.get("early_stopping_patience", 5)),
                "early_stopping_min_delta": 0.0001, "early_stopping_use_ema": False}
    return model, training


def evaluation_protocol(config: dict, lock: dict) -> dict:
    sealed_files = {path: digest for path, digest in lock["files"].items() if path.startswith("sealed_test/")}
    return {"schema_version": "forte.specialists.public-evaluation-protocol.v1", "participant": "detector",
            "member_id": "detector", "split": "sealed_test", "classes": lock["classes"],
            "selection_scope": "public_train_dev_only", "test_reuse_for_tuning_allowed": False,
            "sealed_inventory_sha256": canonical_sha256(sealed_files),
            "metric": "RF-DETR COCO bbox mAP50:95 and per-class AP/AR", "max_detections": 500,
            "fixed_detection_metrics": {"confidence": 0.5, "iou": 0.5,
                "matching": "one_to_one_same_class_descending_confidence",
                "background_policy": "discard_only_final_no_object_slot",
                "negative_image_population": "empty_coin_images_only",
                "metrics": ["per_class_precision", "per_class_recall", "coin_negative_image_FPR"],
                "threshold_tuning_after_test": False},
            "fixed_metrics_source_sha256": file_sha256(Path(__file__).with_name("public_metrics.py")),
            "batch_size": 1, "eval_batch_size": 1, "amp_dtype": "bf16", "multi_scale": False,
            "rfdetr_version": RFDETR_VERSION, "resolution": config["resolution"],
            "runner_sha256": file_sha256(Path(__file__)),
            "dataset_lock_sha256": config["expected_dataset_lock_sha256"]}


def sealed_test_view(root: Path, output: Path, lock: dict) -> Path:
    """Materialize a separate test view only after the dev-selected fit ends."""
    view = output / "sealed-test-view"
    require(not view.exists(), "public_training_sealed_view_already_exists")
    for part in ("train", "valid", "test"):
        (view / part).mkdir(parents=True)
    categories = [{"id": i + 1, "name": name, "supercategory": "none"}
                  for i, name in enumerate(lock["classes"])]
    for part in ("train", "valid"):
        write_json(view / part / "_annotations.coco.json", {"images": [], "annotations": [], "categories": categories})
    names = [path for path in lock["files"] if path.startswith("sealed_test/")]
    require(bool(names), "public_training_sealed_test_files_missing")
    for relative in names:
        source = relative_file(root, relative)
        target = view / "test" / Path(relative).name
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        require(file_sha256(target) == lock["files"][relative], "public_training_sealed_copy_hash_mismatch")
    require((view / "test/_annotations.coco.json").is_file(), "public_training_sealed_annotations_missing")
    return view


def claim_sealed_test(config: dict, lock: dict, checkpoint_sha256: str) -> Path:
    files = {path: digest for path, digest in lock["files"].items() if path.startswith("sealed_test/")}
    require(bool(files), "public_training_test_inventory_empty")
    identity = canonical_sha256(files)
    registry = Path(config["test_exposure_registry_dir"]).resolve()
    registry.mkdir(parents=True, exist_ok=True)
    receipt = registry / f"public-sealed-{identity}.json"
    try:
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump({"state": "attempted", "sealed_inventory_sha256": identity,
                       "checkpoint_sha256": checkpoint_sha256,
                       "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
                       "attempted_at_utc": datetime.now(timezone.utc).isoformat()}, stream)
    except FileExistsError:
        require(False, "public_training_sealed_test_already_exposed")
    return receipt


def fixed_detection_evaluation(selected, root: Path, lock: dict) -> tuple[dict, list[dict]]:
    from .public_metrics import score_public_detections
    inverse = {int(key): name for key, name in lock["class_names"].items()}
    predictions = []
    background_proposals_excluded = 0
    for row in lock["records"]:
        if row.get("split") != "sealed_test" or row.get("included") is not True:
            continue
        raw = selected.predict(str(relative_file(root, row["export_path"])), threshold=0.1)
        require(not isinstance(raw, list) and raw.class_id is not None and raw.confidence is not None,
                "public_training_final_prediction_invalid")
        require(len(raw.xyxy) == len(raw.class_id) == len(raw.confidence), "public_training_final_prediction_length")
        detections = []
        for box, class_id, confidence in zip(raw.xyxy, raw.class_id, raw.confidence):
            try:
                require(not isinstance(class_id, bool), "public_training_final_prediction_class_id")
                prediction_id = operator.index(class_id)
            except (TypeError, ValueError):
                require(False, "public_training_final_prediction_class_id")
            # RF-DETR 1.11.2 ranks all sigmoid columns and predict() can expose
            # the reserved final no-object slot. It is not a foreground class.
            if prediction_id == len(inverse):
                background_proposals_excluded += 1
                continue
            require(prediction_id in inverse, "public_training_final_prediction_class_id")
            x1, y1, x2, y2 = [float(value) for value in box]
            detections.append({"class_name": inverse[prediction_id], "confidence": float(confidence),
                               "bbox": [x1, y1, x2 - x1, y2 - y1]})
        predictions.append({"sample_id": row["sample_id"], "complete": True, "detections": detections})
    metrics = score_public_detections(lock, predictions)
    metrics["background_proposals_excluded"] = background_proposals_excluded
    return metrics, predictions


def cli_deadline_watchdog(config: dict):
    """Cover setup, baseline, fitting and test even when native calls block.

    This is CLI-only: importing/calling helper functions never terminates their
    host process. The owning host should retain its external timeout as well.
    """
    remaining = remaining_seconds(config)
    def stop() -> None:
        output = Path(config["output_dir"]).resolve()
        if output.is_dir():
            try:
                write_json(output / "deadline-stop.json", {"state": "absolute_deadline_reached",
                           "stop_at_utc": config["stop_at_utc"], "exit_code": 124})
            except Exception:
                pass
        os._exit(124)
    timer = threading.Timer(remaining, stop)
    timer.daemon = True
    timer.start()
    return timer


def run(config: dict, *, smoke: bool = False, defer_test: bool = False) -> dict:
    root, lock = preflight(config)
    output = Path(config["output_dir"]).resolve()
    require(not output.exists(), "public_training_output_already_exists")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "dataset-lock.json", lock)
    write_json(output / "evaluation-protocol.json", evaluation_protocol(config, lock))
    deferred = defer_test or config.get("defer_sealed_test", False)
    run_manifest = {"schema_version": "forte.specialists.public-run.v1", "state": "preparing",
                    "scope": "public_object_pretraining_only", "serving_eligible": False, "approved_types": [],
                    "mode": "bounded_smoke" if smoke else "fixed_recipe_train",
                    "rfdetr_version": RFDETR_VERSION, "model_class": config["model_class"],
                    "class_names": lock["class_names"], "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
                    "dataset_sha256": lock["dataset_sha256"], "config_sha256": file_sha256(output / "config.json"),
                    "runner_sha256": file_sha256(Path(__file__)),
                    "pretrain_weights_sha256": config["expected_pretrain_weights_sha256"],
                    "baseline_scope": "initial_untrained_target_head_on_pretrained_backbone",
                    "test_evaluation_count": 0, "defer_sealed_test": deferred,
                    "evaluation_protocol_sha256": file_sha256(output / "evaluation-protocol.json"),
                    "recipe_reference": config["recipe_reference"]}
    write_json(output / "run.json", run_manifest)
    started = time.monotonic()
    try:
        require(importlib.metadata.version("rfdetr") == RFDETR_VERSION, "public_training_installed_package_mismatch")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        import pytorch_lightning as pl
        from rfdetr.config import RFDETRNanoConfig, RFDETRSmallConfig, TrainConfig
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "public_training_bf16_cuda_required")
        free, total = torch.cuda.mem_get_info(0)
        cap = config["gpu_memory_budget_mib"] * 1024 * 1024
        margin = config["minimum_free_margin_mib"] * 1024 * 1024
        require(free >= cap + margin, "public_training_shared_gpu_headroom_insufficient")
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        pl.seed_everything(config["seed"], workers=True)
        model_kwargs, train_kwargs = model_and_train_kwargs(config, root, output, lock["classes"])
        model_config = ({"RFDETRNano": RFDETRNanoConfig, "RFDETRSmall": RFDETRSmallConfig}[config["model_class"]])(**model_kwargs)
        train_config = TrainConfig(**train_kwargs)
        write_json(output / "effective-recipe.json", {"model": model_kwargs, "training": train_kwargs})
        module = RFDETRModelModule(model_config, train_config)
        from .public_augmentation import balanced_public_datamodule
        data, data_policy = balanced_public_datamodule(RFDETRDataModule, model_config, train_config, lock,
                                                      config["train_data_policy"])
        run_manifest["training_data_policy"] = data_policy
        run_manifest["augmentation_source_sha256"] = file_sha256(Path(__file__).with_name("public_augmentation.py"))
        write_json(output / "training-data-policy.json", data_policy)
        run_manifest["target_head_initialization"] = initialize_target_heads(module, lock["classes"])
        require(any(p.requires_grad for p in module.parameters()) and any(not p.requires_grad for p in module.parameters()),
                "public_training_frozen_encoder_or_trainable_head_missing")
        before_trainable = parameter_sha256(module, trainable=True)
        before_frozen = parameter_sha256(module, trainable=False)
        run_manifest.update(initial_trainable_parameters_sha256=before_trainable,
                            initial_frozen_parameters_sha256=before_frozen,
                            runtime={"torch": torch.__version__, "cuda": torch.version.cuda,
                                     "pytorch_lightning": pl.__version__, "gpu": torch.cuda.get_device_name(0),
                                     "free_before_mib": free / 1024**2, "allocator_budget_mib": config["gpu_memory_budget_mib"]})
        # This eval-only trainer cannot write/select a best training checkpoint.
        eval_trainer = build_trainer(train_config, model_config, include_training_callbacks=False,
                                     accelerator="gpu", devices=[0], logger=False, enable_progress_bar=False,
                                     limit_val_batches=2 if smoke else 1.0)
        run_manifest["state"] = "baseline_evaluating"
        write_json(output / "run.json", run_manifest)
        baseline = eval_trainer.validate(module, datamodule=data)
        write_json(output / "baseline-dev.json", {"scope": run_manifest["baseline_scope"],
                   "metrics": scalar_metrics(baseline[0] if baseline else {}),
                   "trainable_parameters_sha256": before_trainable,
                   "dataset_lock_sha256": run_manifest["dataset_lock_sha256"]})
        del eval_trainer
        gc.collect()
        require(parameter_sha256(module, trainable=True) == before_trainable
                and parameter_sha256(module, trainable=False) == before_frozen,
                "public_training_baseline_mutated_parameters")
        torch.cuda.empty_cache()
        budget = min(config["max_fit_seconds"], remaining_seconds(config) - config["evaluation_reserve_seconds"])
        require(budget > 0, "public_training_no_fit_time_remaining")
        class Audit(pl.Callback):
            def __init__(self):
                self.losses = []
                self.batch_count = 0
                self.fit_started = time.monotonic()
            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                gradients = [parameter.grad for parameter in pl_module.parameters()
                             if parameter.requires_grad and parameter.grad is not None]
                require(bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
                        "public_training_nonfinite_or_missing_gradient")
                require(any(bool(torch.count_nonzero(gradient)) for gradient in gradients),
                        "public_training_supervised_gradient_is_zero")
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.batch_count += 1
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                if hasattr(loss, "detach"):
                    loss = float(loss.detach().cpu())
                require(number(loss, -1e20, 1e20), "public_training_nonfinite_loss")
                self.losses.append(loss)
                if len(self.losses) > 200:
                    self.losses.pop(0)
                require(torch.cuda.memory_reserved(0) <= cap, "public_training_memory_budget_exceeded")
                try:
                    left = remaining_seconds(config)
                except Exception:
                    trainer.should_stop = True
                    return
                if left <= config["evaluation_reserve_seconds"]:
                    trainer.should_stop = True
            def on_train_epoch_end(self, trainer, pl_module):
                write_json(output / "progress.json", {"epoch": int(trainer.current_epoch),
                           "optimizer_steps": int(trainer.global_step), "physical_batches": self.batch_count,
                           "recent_losses": self.losses[-20:], "elapsed_s": time.monotonic() - started,
                           "max_memory_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
                           "max_memory_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
                           "metrics": scalar_metrics(dict(trainer.callback_metrics))})
        audit = Audit()
        trainer = build_trainer(train_config, model_config, accelerator="gpu", devices=[0],
                                max_time=timedelta(seconds=budget), min_epochs=0, min_steps=0,
                                max_steps=3 if smoke else -1, logger=False, enable_progress_bar=False,
                                limit_val_batches=2 if smoke else 1.0, num_sanity_val_steps=0)
        trainer.callbacks.append(audit)
        run_manifest["state"] = "training"
        write_json(output / "run.json", run_manifest)
        trainer.fit(module, datamodule=data)
        torch.cuda.synchronize()
        after_trainable = parameter_sha256(module, trainable=True)
        require(int(trainer.global_step) > 0 and after_trainable != before_trainable,
                "public_training_no_real_optimizer_update")
        require(parameter_sha256(module, trainable=False) == before_frozen, "public_training_frozen_parameters_changed")
        run_manifest.update(optimizer_steps=int(trainer.global_step), physical_batches=audit.batch_count,
                            final_trainable_parameters_sha256=after_trainable,
                            final_frozen_parameters_sha256=before_frozen,
                            recent_losses=audit.losses[-20:],
                            max_memory_allocated_mib=torch.cuda.max_memory_allocated(0) / 1024**2,
                            max_memory_reserved_mib=torch.cuda.max_memory_reserved(0) / 1024**2,
                            fit_wall_s=time.monotonic() - audit.fit_started)
        checkpoint = relative_file(output, "checkpoint_best_total.pth")
        run_manifest.update(checkpoint=checkpoint.name, checkpoint_sha256=file_sha256(checkpoint),
                            state=("smoke_completed_unselected" if smoke else
                                   "trained_unselected_pending_test" if deferred else "trained_unselected"))
        run_manifest["final_epoch_dev_metrics"] = scalar_metrics(dict(trainer.callback_metrics))
        selected_callback = next((callback for callback in trainer.callbacks
                                  if type(callback).__name__ == "BestModelCallback"), None)
        if selected_callback is not None:
            best_score = getattr(selected_callback, "best_model_score", None)
            if best_score is not None:
                run_manifest["selected_checkpoint_dev_metric"] = {
                    "metric": str(selected_callback.monitor), "value": float(best_score.detach().cpu())}
        write_json(output / "run.json", run_manifest)
        if not smoke and not deferred:
            # Recheck data and deadline; never open the sealed test if fitting
            # failed, no real update occurred, or time for honest evaluation ran out.
            load_public_dataset(root, config["expected_dataset_lock_sha256"])
            require(remaining_seconds(config) > 60, "public_training_no_test_time_remaining")
            exposure = claim_sealed_test(config, lock, run_manifest["checkpoint_sha256"])
            run_manifest["test_evaluation_count"] = 1
            run_manifest["test_exposure_receipt"] = str(exposure)
            run_manifest["test_state"] = "attempted"
            write_json(output / "run.json", run_manifest)
            test_receipt = output / "sealed-test-attempt.json"
            with test_receipt.open("x", encoding="utf-8") as stream:
                json.dump({"state": "started", "checkpoint_sha256": run_manifest["checkpoint_sha256"],
                           "dataset_lock_sha256": run_manifest["dataset_lock_sha256"]}, stream)
            view = sealed_test_view(root, output, lock)
            del trainer, module, data
            gc.collect()
            torch.cuda.empty_cache()
            # Load the selected checkpoint through RF-DETR's supported metadata-
            # aware API. Public test values cannot influence checkpoint selection.
            from rfdetr import RFDETR
            selected = RFDETR.from_checkpoint(str(checkpoint), device="cpu")
            require(list(selected.class_names) == lock["classes"], "public_training_selected_class_mapping_mismatch")
            test_metrics = selected.evaluate(dataset_dir=str(view), output_dir=str(output / "sealed-test"),
                                             split="test", batch_size=1, eval_batch_size=1,
                                             device="cuda:0",
                                             amp_dtype="bf16", num_workers=0, multi_scale=False,
                                             eval_max_dets=500,
                                             log_per_class_metrics=True)
            write_json(output / "sealed-test-metrics.json", {"state": "complete", "metrics": scalar_metrics(test_metrics),
                       "checkpoint_sha256": run_manifest["checkpoint_sha256"], "selection_scope": "public_dev_only"})
            write_json(test_receipt, {"state": "complete", "checkpoint_sha256": run_manifest["checkpoint_sha256"],
                                     "metrics_sha256": file_sha256(output / "sealed-test-metrics.json")})
            run_manifest["test_state"] = "complete"
            exposure_record = load_json(exposure)
            exposure_record.update(state="complete", metrics_sha256=file_sha256(output / "sealed-test-metrics.json"))
            write_json(exposure, exposure_record)
        run_manifest["elapsed_s"] = time.monotonic() - started
        write_json(output / "run.json", run_manifest)
        return {"ok": True, "state": run_manifest["state"], "run_sha256": file_sha256(output / "run.json"),
                "checkpoint_sha256": run_manifest["checkpoint_sha256"], "serving_eligible": False,
                "test_evaluation_count": run_manifest["test_evaluation_count"]}
    except Exception as exc:
        run_manifest["state"] = "failed"
        run_manifest["error"] = str(exc) if isinstance(exc, ValueError) and str(exc).startswith("public_training_") else "public_training_execution_failed"
        run_manifest["elapsed_s"] = time.monotonic() - started
        write_json(output / "run.json", run_manifest)
        raise


def final_evaluate(run_path: Path, expected_run_sha256: str, cohort_receipt: Path,
                   protocol_path: Path, *, expected_cohort_sha256: str) -> dict:
    """Score a previously frozen detector once as a registered cohort member."""
    run_path = run_path.resolve()
    selected_run = pinned_json(run_path, expected_run_sha256)
    require(selected_run.get("schema_version") == "forte.specialists.public-run.v1"
            and selected_run.get("state") == "trained_unselected_pending_test"
            and selected_run.get("defer_sealed_test") is True
            and selected_run.get("test_evaluation_count") == 0, "public_training_final_run_not_pending")
    config_path = relative_file(run_path.parent, "config.json")
    require(file_sha256(config_path) == selected_run.get("config_sha256"), "public_training_final_config_changed")
    config = load_json(config_path)
    root, lock = preflight(config)
    require(selected_run.get("dataset_lock_sha256") == config["expected_dataset_lock_sha256"]
            and selected_run.get("class_names") == lock["class_names"], "public_training_final_data_changed")
    checkpoint = relative_file(run_path.parent, selected_run.get("checkpoint"))
    require(file_sha256(checkpoint) == selected_run.get("checkpoint_sha256"), "public_training_final_checkpoint_changed")
    protocol = pinned_json(protocol_path, selected_run.get("evaluation_protocol_sha256", ""))
    require(protocol == evaluation_protocol(config, lock), "public_training_final_protocol_changed")
    require(remaining_seconds(config) > 60, "public_training_no_test_time_remaining")
    from .public_cohort import claim_cohort_member
    attempt = claim_cohort_member(config, lock, selected_run["checkpoint_sha256"],
                                  cohort_receipt=cohort_receipt, expected_cohort_sha256=expected_cohort_sha256,
                                  member_id="detector", run_path=run_path,
                                  config_path=config_path, protocol_path=protocol_path)
    output = run_path.parent / "cohort-detector-evaluation"
    require(not output.exists(), "public_training_final_output_exists")
    output.mkdir()
    write_json(output / "attempt.json", {"state": "attempted", "run_sha256": expected_run_sha256,
               "checkpoint_sha256": selected_run["checkpoint_sha256"], "test_evaluation_count": 1})
    started = time.monotonic()
    try:
        require(importlib.metadata.version("rfdetr") == RFDETR_VERSION, "public_training_installed_package_mismatch")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "public_training_bf16_cuda_required")
        free, total = torch.cuda.mem_get_info(0)
        cap = config["gpu_memory_budget_mib"] * 1024**2
        require(free >= cap + config["minimum_free_margin_mib"] * 1024**2,
                "public_training_shared_gpu_headroom_insufficient")
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        view = sealed_test_view(root, output, lock)
        from rfdetr import RFDETR
        selected = RFDETR.from_checkpoint(str(checkpoint), device="cuda:0")
        require(list(selected.class_names) == lock["classes"], "public_training_selected_class_mapping_mismatch")
        metrics = selected.evaluate(dataset_dir=str(view), output_dir=str(output / "native-metrics"),
                                    split="test", device="cuda:0", batch_size=1, eval_batch_size=1,
                                    amp_dtype="bf16", num_workers=0, multi_scale=False,
                                    eval_max_dets=500,
                                    log_per_class_metrics=True)
        fixed_metrics, predictions = fixed_detection_evaluation(selected, root, lock)
        write_json(output / "fixed-confidence-predictions.json", predictions)
        torch.cuda.synchronize()
        result = {"schema_version": "forte.specialists.public-final-evaluation.v1", "state": "complete",
                  "member_id": "detector", "run_sha256": expected_run_sha256,
                  "checkpoint_sha256": selected_run["checkpoint_sha256"],
                  "dataset_lock_sha256": selected_run["dataset_lock_sha256"],
                  "config_sha256": selected_run["config_sha256"],
                  "evaluation_protocol_sha256": selected_run["evaluation_protocol_sha256"],
                  "cohort_receipt_sha256": file_sha256(cohort_receipt), "metrics": scalar_metrics(metrics),
                  "fixed_detection_metrics": fixed_metrics,
                  "fixed_predictions_sha256": file_sha256(output / "fixed-confidence-predictions.json"),
                  "test_evaluation_count": 1, "elapsed_s": time.monotonic() - started,
                  "max_memory_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
                  "max_memory_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
                  "serving_eligible": False}
        write_json(output / "result.json", result)
        write_json(output / "attempt.json", {"state": "complete", "result_sha256": file_sha256(output / "result.json"),
                                             "test_evaluation_count": 1})
        return {"ok": True, "state": "complete", "result_sha256": file_sha256(output / "result.json"),
                "serving_eligible": False, "test_evaluation_count": 1}
    except Exception:
        write_json(output / "attempt.json", {"state": "failed", "test_evaluation_count": 1,
                                             "elapsed_s": time.monotonic() - started})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--defer-test", action="store_true")
    parser.add_argument("--final-eval-run", type=Path)
    parser.add_argument("--expected-run-sha256")
    parser.add_argument("--cohort-receipt", type=Path)
    parser.add_argument("--expected-cohort-sha256")
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        if args.final_eval_run:
            require(config == load_json(args.final_eval_run.resolve().parent / "config.json"),
                    "public_training_final_cli_config_mismatch")
        if args.preflight:
            _, lock = preflight(config)
            result = {"ok": True, "state": "public_metadata_preflight_passed", "training_executed": False,
                      "dataset_sha256": lock["dataset_sha256"], "serving_eligible": False}
        else:
            watchdog = cli_deadline_watchdog(config)
            try:
                if args.final_eval_run:
                    require(not args.smoke and not args.defer_test and args.expected_run_sha256
                            and args.cohort_receipt and sha(args.expected_cohort_sha256) and args.protocol,
                            "public_training_final_eval_arguments_required")
                    result = final_evaluate(args.final_eval_run, args.expected_run_sha256,
                                            args.cohort_receipt, args.protocol,
                                            expected_cohort_sha256=args.expected_cohort_sha256)
                else:
                    result = run(config, smoke=args.smoke, defer_test=args.defer_test)
            finally:
                watchdog.cancel()
        print(json.dumps(result))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
