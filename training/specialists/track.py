"""Export selected run provenance/scalars to a local SQLite MLflow store."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import cli_error, file_sha256, load_json, number, relative_file, require
from .train import verify_run


def track(run_path: Path, tracking_dir: Path, metrics_path: Path | None = None) -> dict:
    run = load_json(run_path)
    require(run.get("schema_version") == "forte.specialists.run.v1"
            and run.get("state") in {"preparing", "training", "failed", "trained_unselected"},
            "specialist_tracking_run_schema")
    require(file_sha256(relative_file(run_path.parent, "config.json")) == run.get("config_sha256"),
            "specialist_tracking_config_mismatch")
    scalars = {}
    if run["state"] == "trained_unselected":
        verify_run(run_path)
    if metrics_path:
        require(run["state"] == "trained_unselected", "specialist_tracking_metrics_require_trained_run")
        # Recompute and verify before exporting any reported quality values.
        from .gate import gate
        proof = gate(run_path, metrics_path)
        scalars["class_gate_eligible"] = int(proof["eligible"])
        scalars["clean_images"] = proof["clean_images"]
        scalars["total_false_positives"] = proof["total_false_positives"]
        for name, values in proof["per_class"].items():
            require(name.isascii() and name.replace("_", "").isalnum(), "specialist_tracking_class_invalid")
            for field in ("tp", "fp", "fn", "positive_instances", "precision", "recall"):
                if number(values.get(field)):
                    scalars[f"class.{name}.{field}"] = values[field]
    try:
        import mlflow
    except ImportError:
        require(False, "specialist_mlflow_optional_dependency_missing")
    tracking_dir = tracking_dir.resolve()
    require(not str(tracking_dir).startswith(("\\\\", "//")), "specialist_mlflow_network_path_forbidden")
    tracking_dir.mkdir(parents=True, exist_ok=True)
    # No user-supplied tracking URL, environment URL, autolog, or media upload.
    mlflow.set_tracking_uri("sqlite:///" + (tracking_dir / "mlflow.db").as_posix())
    experiment_name = "forte-small-object-specialists"
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        experiment_id = mlflow.create_experiment(experiment_name,
                                                 artifact_location=(tracking_dir / "artifacts").as_uri())
    else:
        artifact = str(existing.artifact_location)
        require(artifact == (tracking_dir / "artifacts").as_uri(),
                "specialist_mlflow_remote_artifact_store_forbidden")
        experiment_id = existing.experiment_id
    safe = {key: run[key] for key in ("schema_version", "state", "model_class", "rfdetr_version",
                                      "dataset_lock_sha256", "config_sha256", "dataset_sha256",
                                      "checkpoint_sha256", "resolution", "seed") if key in run}
    safe["run_sha256"] = file_sha256(run_path)
    if metrics_path:
        safe["metrics_sha256"] = file_sha256(metrics_path)
    with mlflow.start_run(experiment_id=experiment_id) as logged:
        mlflow.log_params({key: safe[key] for key in ("state", "model_class", "rfdetr_version", "resolution", "seed")
                           if key in safe})
        mlflow.log_metrics(scalars)
        mlflow.log_dict(safe, "provenance.json")
        run_id = logged.info.run_id
    return {"ok": True, "mlflow_run_id": run_id, "local_only": True,
            "run_sha256": safe["run_sha256"], "scalar_count": len(scalars)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--tracking-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(track(args.run.resolve(), args.tracking_dir, args.metrics)))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
