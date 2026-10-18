"""Fine-tune a local RF-DETR checkpoint after a hash/provenance preflight."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random

from .common import (canonical_sha256, cli_error, file_sha256, load_json, number,
                     relative_file, require, sha, write_json)
from .prepare import verify_dataset

RFDETR_VERSION = "1.11.2"
MODEL_CLASSES = {"RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase"}


def preflight(config: dict) -> tuple[Path, dict]:
    require(config.get("schema_version") == "forte.specialists.train.v1", "specialist_train_schema")
    require(config.get("rfdetr_version") == RFDETR_VERSION, "specialist_package_pin_required")
    require(config.get("model_class") in MODEL_CLASSES, "specialist_model_class_not_approved")
    root = Path(config.get("dataset_dir", "")).resolve()
    lock = verify_dataset(root, config.get("expected_dataset_lock_sha256", ""))
    weights = Path(config.get("pretrain_weights", "")).resolve()
    require(weights.is_file() and sha(config.get("expected_pretrain_weights_sha256"))
            and file_sha256(weights) == config["expected_pretrain_weights_sha256"],
            "specialist_local_pretrain_weights_hash_mismatch")
    require(config.get("device") in {"cuda", "cuda:0"}, "specialist_cuda_device_required")
    for key in ("epochs", "batch_size", "grad_accum_steps"):
        require(type(config.get(key)) is int and config[key] >= 1, "specialist_training_parameter_invalid")
    require(type(config.get("seed")) is int and config["seed"] >= 0, "specialist_seed_required")
    require(type(config.get("resolution")) is int and config["resolution"] > 0
            and config["resolution"] % (56 if config["model_class"] == "RFDETRBase" else 32) == 0,
            "specialist_resolution_invalid")
    require(number(config.get("lr"), 1e-9, 1), "specialist_learning_rate_invalid")
    thresholds = config.get("thresholds")
    require(isinstance(thresholds, dict) and set(thresholds) == set(lock["classes"])
            and all(number(t, 0, 1) for t in thresholds.values()), "specialist_thresholds_required")
    policy = config.get("gate_policy", {})
    require(number(policy.get("minimum_recall"), .95, 1)
            and type(policy.get("minimum_positive_instances")) is int
            and policy["minimum_positive_instances"] >= 20
            and type(policy.get("minimum_clean_images")) is int
            and policy["minimum_clean_images"] >= 20,
            "specialist_gate_policy_invalid")
    for role in ("train", "validation"):
        subset = [r for r in lock["records"] if r["split"] == role]
        labels = {o["class_name"] for r in subset for o in r["objects"]}
        require(set(lock["classes"]) <= labels, "specialist_training_class_support_missing")
    require(config.get("compute_release_authorization", {}).get("released_by_A18_or_root") is True
            and config["compute_release_authorization"].get("release_reference"),
            "specialist_compute_window_not_released")
    return root, lock


def load_model(model_class: str, weights: Path, *, resolution: int, device: str):
    try:
        version = importlib.metadata.version("rfdetr")
    except importlib.metadata.PackageNotFoundError:
        require(False, "specialist_rfdetr_dependency_missing")
    require(version == RFDETR_VERSION, "specialist_installed_package_mismatch")
    # All weights are local and pinned. Backbones missing from local caches fail
    # under the Hugging Face offline flags rather than fetching during fitting.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import rfdetr
    return getattr(rfdetr, model_class)(pretrain_weights=str(weights), resolution=resolution, device=device)


def train(config: dict) -> dict:
    root, lock = preflight(config)
    output = Path(config["output_dir"]).resolve()
    require(not output.exists(), "specialist_run_output_already_exists")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "dataset-lock.json", lock)
    run = {"schema_version": "forte.specialists.run.v1", "state": "preparing",
           "model_class": config["model_class"], "class_names": lock["class_names"],
           "coco_category_ids": lock["coco_category_ids"], "prediction_class_ids": lock["prediction_class_ids"],
           "rfdetr_version": RFDETR_VERSION, "resolution": config["resolution"],
           "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
           "dataset_sha256": lock["dataset_sha256"], "config_sha256": file_sha256(output / "config.json"),
           "thresholds": config["thresholds"], "gate_policy": config["gate_policy"],
           "approved_types": [], "seed": config["seed"]}
    write_json(output / "run.json", run)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            import torch
        except ImportError:
            require(False, "specialist_torch_dependency_missing")
        require(torch.cuda.is_available(), "specialist_cuda_unavailable")
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model = load_model(config["model_class"], Path(config["pretrain_weights"]).resolve(),
                           resolution=config["resolution"], device=config["device"])
        # Version 1.11.2 uses contiguous 0-based labels. Check the pinned package
        # implementation too, so a packaging change cannot silently shift IDs.
        from rfdetr.datasets.coco import _train_split_cat2label
        require(_train_split_cat2label(root) == {i + 1: i for i in range(len(lock["classes"]))},
                "specialist_package_class_mapping_mismatch")
        run["state"] = "training"
        write_json(output / "run.json", run)
        model.train(dataset_dir=str(root), output_dir=str(output),
                    epochs=config["epochs"], batch_size=config["batch_size"],
                    grad_accum_steps=config["grad_accum_steps"], lr=config["lr"],
                    resolution=config["resolution"], device=config["device"], seed=config["seed"],
                    num_workers=int(config.get("num_workers", 2)),
                    wandb=False, tensorboard=False, mlflow=False, clearml=False, run_test=False,
                    save_dataset_grids=False)
        # The predeclared checkpoint is selected by RF-DETR validation mAP, never
        # by holdout/test results. No fallback to an arbitrary checkpoint file.
        checkpoint = relative_file(output, "checkpoint_best_total.pth")
        require(checkpoint.stat().st_size > 0, "specialist_checkpoint_empty")
        verify_dataset(root, config["expected_dataset_lock_sha256"])
        run.update(state="trained_unselected", checkpoint=checkpoint.name,
                   checkpoint_sha256=file_sha256(checkpoint),
                   runtime={"torch": torch.__version__, "cuda": torch.version.cuda,
                            "device": torch.cuda.get_device_name(0)})
        write_json(output / "run.json", run)
        return {"ok": True, "state": run["state"], "checkpoint_sha256": run["checkpoint_sha256"],
                "run_sha256": file_sha256(output / "run.json"), "promotion_eligible": False}
    except Exception:
        run["state"] = "failed"
        run["error"] = "specialist_training_failed"
        write_json(output / "run.json", run)
        raise


def verify_run(run_path: Path, expected: str | None = None) -> tuple[dict, Path]:
    require(run_path.is_file(), "specialist_run_missing")
    if expected is not None:
        require(sha(expected) and file_sha256(run_path) == expected, "specialist_run_hash_mismatch")
    run = load_json(run_path)
    require(run.get("schema_version") == "forte.specialists.run.v1"
            and run.get("state") == "trained_unselected", "specialist_run_not_trained")
    require(run.get("model_class") in MODEL_CLASSES and run.get("rfdetr_version") == RFDETR_VERSION,
            "specialist_run_model_identity_invalid")
    checkpoint = relative_file(run_path.parent, run.get("checkpoint"))
    require(file_sha256(checkpoint) == run.get("checkpoint_sha256"), "specialist_checkpoint_hash_mismatch")
    require(file_sha256(relative_file(run_path.parent, "config.json")) == run.get("config_sha256"),
            "specialist_run_config_changed")
    return run, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        if args.preflight:
            _, lock = preflight(config)
            result = {"ok": True, "state": "metadata_preflight_passed", "training_executed": False,
                      "dataset_sha256": lock["dataset_sha256"], "gpu_verified": False}
        else:
            result = train(config)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
