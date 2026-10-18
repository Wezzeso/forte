"""Operational six-upload preparation. Metadata work never launches inference.

Select freezes two TRAIN fixtures before new outputs. Bind requires actual
runtime and shared recipe pins. Execute reads an existing private key file
internally, then delegates to the released, failure-inclusive HTTP harness.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import tarfile

from benchmarks.performance import qwen38_acceptance as driver
from benchmarks.performance.repeatability import digest, file_digest, require, write_new
from forte.video.sampling import coverage_windows

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://forte-review-wd88a7vvt.gobrev.dev"
INTENDED_RECIPE = {"window_seconds": 8., "stride_seconds": 4., "sample_fps": 2.,
    "max_frames_per_window": 16, "max_refinement_passes": 8, "staged_screening": False,
    "max_concurrent_jobs": 1, "max_concurrent_clips": 1, "max_inflight_per_job": 1, "provider_max_attempts": 2}
BUNDLE_CODE = ("benchmarks/performance/qwen38_acceptance_ops.py",
    "benchmarks/performance/qwen38_acceptance.py", "benchmarks/performance/repeatability.py",
    "deploy/qwen38_acceptance_host.py", "training/qwen38_preflight.py", "training/guards.py", "scripts/audit_data.py",
    "scripts/build_reviewed_split.py", "scripts/exposure_registry.py", "scripts/split_gate.py", "evaluation/audit.py", "evaluation/io.py",
    "evaluation/metrics.py", "evaluation/qwen38_comparison.py", "evaluation/qwen38_pipeline.py",
    "evaluation/qwen38_server_export.py", "evaluation/qwen38_worker_trace.py", "evaluation/weak_supervision.py")


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def model_budget(population, recipe):
    """Bound generation HTTP attempts, including every configured retry slot."""
    for key, value in INTENDED_RECIPE.items():
        if key == "provider_max_attempts":
            continue
        require(recipe.get(key) == value, "acceptance_ops_recipe_changed")
    require(type(recipe.get("provider_max_attempts")) is int and recipe["provider_max_attempts"] in {1, 2},
            "acceptance_ops_provider_retry_budget_invalid")
    cases = []
    for source in population["cases"]:
        windows = coverage_windows(source["duration_sec"], window_sec=recipe["window_seconds"],
            stride_sec=recipe["stride_seconds"], sample_fps=recipe["sample_fps"],
            max_frames_per_window=recipe["max_frames_per_window"], max_windows=512, max_total_frames=8192)
        # Staging is disabled by the released profile: one full-fact request per
        # initial window; at most max_refinement_passes extra logical requests.
        logical = len(windows) + recipe["max_refinement_passes"]
        cases.append({"case_id": source["case_id"], "duration_sec": source["duration_sec"],
            "coverage_windows_per_job": len(windows), "fresh_upload_jobs": 3,
            "minimum_initial_model_requests_per_job": len(windows),
            "maximum_logical_model_requests_per_job": logical,
            "maximum_generation_http_attempts_per_job": logical * recipe["provider_max_attempts"]})
    return {"schema_version": "forte.qwen38-acceptance-model-budget.v1", "cases": cases,
        "source_jobs": 6, "independent_contents": 2, "fresh_uploads_per_content": 3,
        "minimum_initial_model_requests": sum(row["coverage_windows_per_job"] * 3 for row in cases),
        "maximum_generation_http_attempts": sum(row["maximum_generation_http_attempts_per_job"] * 3 for row in cases),
        "provider_retry_slots_included": True, "warmup_or_readiness_model_work_included": False,
        "failed_or_uncertain_source_jobs_consume_reserved_slots": True,
        "budget_is_not_observed_model_call_count": True, "recipe_sha256": digest(recipe)}


def select_population(roles_path, labels_path, recordings_path, output):
    """Select shortest source per supplied class, before any new predictions."""
    roles = driver.read(roles_path)
    train = {row["content_sha256"]: row for row in roles["records"] if row["split"] == "train"}
    labels = records(labels_path)
    require(labels and all(row.get("split") == "train" and row.get("provided_label_origin") == "organizer_workbook"
            and type(row.get("provided_video_label")) is bool for row in labels), "acceptance_ops_train_labels_required")
    by_hash = {}
    for row in records(recordings_path):
        content = row["content_sha256"]
        if content in train and (content not in by_hash or row["relative_path"] < by_hash[content]["relative_path"]):
            by_hash[content] = row
    output = driver.private(output)
    require(not output.exists(), "acceptance_ops_population_exists")
    choices = []
    for expected in (True, False):
        contents = {row["content_sha256"] for row in labels if row["provided_video_label"] is expected
                    and row["content_sha256"] in train}
        available = [by_hash[content] for content in contents if content in by_hash]
        require(available, "acceptance_ops_both_classes_required")
        chosen = min(available, key=lambda row: (row["duration_sec"], row["content_sha256"]))
        choices.append((chosen, expected))
    require(choices[0][0]["content_sha256"] != choices[1][0]["content_sha256"], "acceptance_ops_conflicting_classes")
    output.mkdir(parents=True)
    cases, references = [], []
    for index, (row, expected) in enumerate(choices, 1):
        content = row["content_sha256"]
        source = (ROOT / row["relative_path"]).resolve()
        require(source.is_relative_to((ROOT / "videos").resolve()) and source.is_file()
                and not source.is_symlink() and file_digest(source) == content,
                "acceptance_ops_source_binding_invalid")
        destination = output / "inputs" / (content + ".mov")
        destination.parent.mkdir(exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        require(file_digest(destination) == content, "acceptance_ops_copied_source_changed")
        case_id = f"fixture_{index:03d}"
        cases.append({"case_id": case_id, "sample_id": train[content]["sample_id"], "split": "train",
            "content_sha256": content, "source_bytes": destination.stat().st_size,
            "duration_sec": row["duration_sec"], "relative_input": "inputs/" + destination.name})
        references.append({"case_id": case_id, "sample_id": train[content]["sample_id"],
                           "content_sha256": content, "provided_video_label": expected,
                           "human_reverified": False})
    population = {"schema_version": "forte.qwen38-acceptance-population.v1", "cases": cases,
        "selection_policy": "shortest metadata duration within each provided TRAIN binary class; hash tie-break",
        "selected_before_new_outputs": True, "source_roles_sha256": file_digest(roles_path),
        "provided_train_labels_sha256": file_digest(labels_path), "source_manifest_sha256": file_digest(recordings_path),
        "runtime_binding": "PENDING_ACTUAL_SELECTED_MODEL_AND_SHARED_RECIPE", "model_calls_made": 0}
    budget = model_budget(population, INTENDED_RECIPE)
    write_new(output / "population.json", population)
    write_new(output / "reference.private.json", {"schema_version": "forte.qwen38-acceptance-class-reference.v1",
        "records": references, "not_in_model_prompt": True, "not_human_event_or_legal_gold": True})
    write_new(output / "worker-recipe.intended.json", INTENDED_RECIPE)
    write_new(output / "model-budget.intended.json", budget)
    return {"status": "population_prepared_runtime_pending", "positive_train_contents": 1,
        "negative_train_contents": 1, "source_jobs": 6, "model_calls_made": 0,
        "maximum_generation_http_attempts": budget["maximum_generation_http_attempts"],
        "minimum_initial_model_requests": budget["minimum_initial_model_requests"]}


def bind(population_path, runtime_config_path, worker_recipe_path, comparison_plan_path, output, arm):
    """Create executable plan only from real model and matched recipe pins."""
    population = driver.read(driver.private(population_path))
    require(population.get("schema_version") == "forte.qwen38-acceptance-population.v1"
            and len(population["cases"]) == 2 and all(row["split"] == "train" for row in population["cases"]),
            "acceptance_ops_population_invalid")
    runtime = driver.read(runtime_config_path)
    if arm == "qwen38_tuned":
        from serving.qwen38_identity import load_expected_runtime_identity
        actual_config_identity = load_expected_runtime_identity(runtime_config_path)
    elif arm == "qwen38_untuned":
        from serving.qwen38_untuned import expected_untuned_identity
        actual_config_identity = expected_untuned_identity(runtime)
    else:
        require(False, "acceptance_ops_selected_qwen_arm_required")
    identity = {key: actual_config_identity.get(key) for key in ("model_id", "model_revision", "adapter_sha256")}
    comparison = driver.read(driver.private(comparison_plan_path))
    require({key: comparison.get("arms", {}).get(arm, {}).get(key) for key in identity} == identity
            and driver.valid_digest(comparison.get("recipe_sha256"))
            and comparison["recipe_sha256"] == digest(comparison["recipe"]),
            "acceptance_ops_shared_comparison_not_bound")
    worker_recipe = driver.read(worker_recipe_path)
    budget = model_budget(population, worker_recipe)
    shared_settings = comparison["recipe"].get("worker_settings", comparison["recipe"].get("sampling", {}))
    require(all(shared_settings.get(key) == value for key, value in worker_recipe.items()
                if key != "provider_max_attempts"), "acceptance_ops_shared_worker_recipe_mismatch")
    require(comparison["recipe"].get("provider_max_attempts") == worker_recipe["provider_max_attempts"],
            "acceptance_ops_shared_retry_recipe_mismatch")
    output = driver.private(output)
    require(not output.exists(), "acceptance_ops_bound_output_exists")
    output.mkdir(parents=True)
    cases = []
    for row in population["cases"]:
        source = driver.private(population_path).parent / row["relative_input"]
        cases.append({**{key: row[key] for key in ("case_id", "sample_id", "content_sha256", "source_bytes", "split")},
                      "owned_path": str(source.resolve())})
    spec = {"schema_version": "forte.qwen38-acceptance-spec.v1", "arm": arm, "cases": cases,
        "expected_model_identity": identity, "recipe_sha256": comparison["recipe_sha256"],
        "comparison_plan_sha256": digest(comparison),
        "settings": {"base_url": BASE_URL, "whole_result_cache": False, "concurrency": 1,
            "normalization": ["job_id"], "deadline_seconds": 300., "request_timeout_seconds": 60., "poll_seconds": .25}}
    write_new(output / "spec.json", spec)
    result = driver.prepare(spec, output / "plan.json", repeats=3)
    plan = driver.read(output / "plan.json")
    limits = {row["case_id"]: row["maximum_generation_http_attempts_per_job"] for row in budget["cases"]}
    slots = [{"attempt_id": "model_slot_" + secrets.token_hex(16), "phase": "acceptance",
        "content_sha256": attempt["content_sha256"], "split": "train",
        "source_attempt_id": attempt["attempt_id"], "case_id": attempt["case_id"], "slot_index": index}
        for attempt in plan["attempts"] for index in range(limits[attempt["case_id"]])]
    write_new(output / "model-attempt-plan.json", {"schema_version": "forte.qwen38-attempt-plan.v1",
        "phase": "acceptance", "attempts": slots, "source_attempt_plan_sha256": file_digest(output / "plan.json"),
        "source_attempts": len(plan["attempts"]), "maximum_model_post_attempts": len(slots),
        "recipe_sha256": plan["recipe_sha256"], "failed_cancelled_and_unneeded_slots_never_refunded": True})
    write_new(output / "model-budget.json", budget)
    write_new(output / "binding.private.json", {"runtime_config_file_sha256": file_digest(runtime_config_path),
        "worker_recipe_file_sha256": file_digest(worker_recipe_path), "population_file_sha256": file_digest(population_path),
        "comparison_plan_file_sha256": file_digest(comparison_plan_path),
        "runtime_identity": actual_config_identity, "selection_or_calls_authorized": False})
    return {**result, "source_jobs": 6, "maximum_generation_http_attempts": budget["maximum_generation_http_attempts"]}


def trace_binding(bound_root, worker_recipe_path, model_release_path, model_reservation_path,
                  source_archive_path, app_image_sha256, output):
    """Freeze actual CPU trace pins after separate model-slot reservation."""
    bound_root = driver.private(bound_root)
    plan = driver.validate_plan(driver.read(bound_root / "plan.json"))
    budget = driver.read(bound_root / "model-budget.json")
    recipe = driver.read(worker_recipe_path)
    worker = {key: value for key, value in recipe.items() if key != "provider_max_attempts"}
    require(all(worker.get(key) == value for key, value in INTENDED_RECIPE.items() if key != "provider_max_attempts")
            and recipe["provider_max_attempts"] == 2, "acceptance_ops_trace_worker_profile_invalid")
    require(driver.valid_digest(app_image_sha256), "acceptance_ops_cpu_image_pin_required")
    module = importlib.util.find_spec("forte.pipeline")
    require(module is not None and module.origin, "acceptance_ops_actual_pipeline_source_missing")
    wrapper = ROOT / "evaluation/qwen38_worker_trace.py"
    require(wrapper.is_file(), "acceptance_ops_trace_wrapper_missing")
    def pinned(path):
        path = Path(path).resolve()
        require(path.is_file() and not path.is_symlink(), "acceptance_ops_trace_pin_file_missing")
        return {"path": str(path), "sha256": file_digest(path)}
    source_attempts = [{key: row[key] for key in ("attempt_id", "content_sha256", "split", "repeat_index")}
                       for row in plan["attempts"]]
    binding = {"schema_version": "forte.qwen38-worker-trace-binding.v1",
        "source_attempts": source_attempts, "expected_model_identity": plan["expected_model_identity"],
        "provider_max_attempts": 2, "maximum_model_requests": budget["maximum_generation_http_attempts"],
        "worker_settings": worker, "model_attempt_plan": pinned(bound_root / "model-attempt-plan.json"),
        "model_release": pinned(model_release_path), "model_reservation": pinned(model_reservation_path),
        "wrapper_sha256": file_digest(wrapper), "pipeline_sha256": file_digest(module.origin),
        "trace_source_archive": pinned(source_archive_path), "trace_runtime_image_sha256": app_image_sha256,
        "conditions_sha256": plan["conditions_sha256"], "recipe_sha256": plan["recipe_sha256"]}
    write_new(driver.private(output), binding)
    return {"status": "trace_binding_prepared_not_activated", "source_jobs": 6,
            "maximum_generation_http_attempts": budget["maximum_generation_http_attempts"], "model_calls_made": 0}


def execute_key_file(plan_path, release_path, reservation_path, budget_path, output, key_file):
    plan = driver.validate_plan(driver.read(driver.private(plan_path)))
    budget = driver.read(driver.private(budget_path))
    release = driver.read(driver.private(release_path))
    require(len(plan["cases"]) == 2 and len(plan["attempts"]) == 6 and plan["repeats"] == 3
            and all(row["split"] == "train" for row in plan["cases"]), "acceptance_ops_exact_six_jobs_required")
    require(release.get("maximum_generation_http_attempts") == budget["maximum_generation_http_attempts"]
            and release.get("model_budget_file_sha256") == file_digest(budget_path),
            "acceptance_ops_exact_model_budget_release_required")
    trace_path = Path(budget_path).parent / "trace-binding.json"
    require(trace_path.is_file() and release.get("worker_trace_binding_file_sha256") == file_digest(trace_path),
            "acceptance_ops_actual_worker_trace_binding_required")
    trace = driver.read(trace_path)
    require(trace.get("conditions_sha256") == plan["conditions_sha256"]
            and trace.get("maximum_model_requests") == budget["maximum_generation_http_attempts"],
            "acceptance_ops_worker_trace_scope_mismatch")
    # Key material never enters argv, output, the frozen plan, or the bundle.
    source = Path(key_file)
    require(source.is_file() and not source.is_symlink() and source.stat().st_size <= 65536,
            "acceptance_ops_key_file_invalid")
    key = source.read_text(encoding="utf-8").strip().splitlines()[0].split(",")[0].strip()
    require(bool(key), "acceptance_ops_key_file_empty")
    env_name = "FORTE_QWEN38_OPERATIONAL_ACCEPTANCE_KEY"
    previous = os.environ.get(env_name)
    try:
        os.environ[env_name] = key
        return asyncio.run(driver.execute(plan_path, release_path, reservation_path, output, env_name))
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


def export_report(run, data_root, reference_path, private_output, public_output):
    """Join genuine worker witnesses to immutable published result files."""
    from evaluation.qwen38_server_export import export_artifacts
    run, data_root = driver.private(run), Path(data_root).resolve()
    plan = driver.validate_plan(driver.read(run / "plan.json"))
    exported, failed = 0, 0
    outcomes = []
    for expected in plan["attempts"]:
        directory = run / expected["attempt_id"]
        terminal_path = directory / "attempt.json"
        if not terminal_path.is_file():
            continue
        row = driver.read(terminal_path)
        outcomes.append(row)
        if row.get("status") != "done":
            continue
        try:
            require(driver.ID.fullmatch(row["job_id"]), "acceptance_ops_export_job_invalid")
            witnesses = [path for path in (data_root / "trace-witnesses").glob(row["job_id"] + ".attempt-*.json")
                         if driver.read(path).get("status") == "done"]
            require(len(witnesses) == 1, "acceptance_ops_actual_witness_missing_or_ambiguous")
            witness = driver.read(witnesses[0])
            require(witness.get("source_attempt_id") == expected["attempt_id"], "acceptance_ops_witness_source_mismatch")
            audit_path = Path(witness["audit_path"]).resolve()
            result_path = (data_root / "results" / f"{row['job_id']}.{witness['attempt']}.{witness['result_sha256']}.json").resolve()
            require(audit_path.is_relative_to(data_root / "work") and result_path.is_relative_to(data_root / "results"),
                    "acceptance_ops_export_artifact_outside_owned_root")
            binding = {**{key: expected[key] for key in ("attempt_id", "content_sha256")},
                "job_id": row["job_id"], "conditions_sha256": plan["conditions_sha256"],
                "recipe_sha256": plan["recipe_sha256"], "expected_model_identity": plan["expected_model_identity"]}
            write_new(directory / "export-binding.private.json", binding)
            export_artifacts(audit_path=audit_path, result_path=result_path, binding=binding, witness=witness,
                             output=directory / "server-receipt.json")
            exported += 1
        except Exception:
            failed += 1
    summary = driver.report(run, private_output, public_output)
    references = driver.read(driver.private(reference_path))["records"]
    by_content = {row["content_sha256"]: row["provided_video_label"] for row in references}
    require(set(by_content) == {case["content_sha256"] for case in plan["cases"]}
            and sorted(by_content.values()) == [False, True], "acceptance_ops_reference_population_changed")
    matches = sum(row.get("status") == "done" and row.get("verdict") == (
        "violation" if by_content[row["content_sha256"]] else "no_violation") for row in outcomes)
    diagnostic = {"expected_attempts": 6, "processed": sum(row.get("status") == "done" for row in outcomes),
        "provided_video_label_agreements": matches, "positive_train_contents": 1, "negative_train_contents": 1,
        "human_event_or_legal_accuracy_established": False, "provided_labels_used_as_inference_input": False}
    write_new(run / "provided-class-diagnostic.aggregate.json", diagnostic)
    return {**summary, "server_receipts_exported": exported, "server_receipts_export_failed": failed,
            "provided_binary_diagnostic": diagnostic}


def bundle(population_path, output):
    population_path = driver.private(population_path)
    population = driver.read(population_path)
    output = driver.private(output)
    require(not output.exists(), "acceptance_ops_bundle_exists")
    paths = [ROOT / name for name in BUNDLE_CODE]
    paths.extend(population_path.parent / name for name in (
        "population.json", "reference.private.json", "worker-recipe.intended.json", "model-budget.intended.json"))
    paths.extend(population_path.parent / row["relative_input"] for row in population["cases"])
    # No API keys, model weights, labels workbook, broad media tree or release.
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(output, "x:gz") as target:
        for path in paths:
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT),
                    "acceptance_ops_bundle_member_invalid")
            relative = path.relative_to(ROOT).as_posix()
            target.add(path, arcname=relative, recursive=False)
            entries.append({"path": relative, "sha256": file_digest(path), "bytes": path.stat().st_size})
    write_new(output.with_suffix(output.suffix + ".manifest.json"), {"schema_version": "forte.qwen38-acceptance-ops-bundle.v1",
        "archive_sha256": file_digest(output), "entries": entries, "api_key_included": False,
        "root_release_included": False, "model_calls_made": 0})
    return {"status": "operational_bundle_prepared", "files": len(entries), "bytes": output.stat().st_size,
            "api_key_included": False, "source_jobs": 6, "model_calls_made": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    for name in ("roles", "labels", "recordings", "output"):
        select.add_argument("--" + name, type=Path, required=True)
    binding = sub.add_parser("bind")
    for name in ("population", "runtime-config", "worker-recipe", "comparison-plan", "output"):
        binding.add_argument("--" + name, type=Path, required=True)
    binding.add_argument("--arm", choices=("qwen38_tuned", "qwen38_untuned"), required=True)
    trace = sub.add_parser("trace-bind")
    for name in ("bound-root", "worker-recipe", "model-release", "model-reservation", "source-archive", "output"):
        trace.add_argument("--" + name, type=Path, required=True)
    trace.add_argument("--app-image-sha256", required=True)
    execution = sub.add_parser("execute")
    for name in ("plan", "release", "reservation", "budget", "output", "api-key-file"):
        execution.add_argument("--" + name, type=Path, required=True)
    exporting = sub.add_parser("export-report")
    for name in ("run", "data-root", "reference", "private-output", "public-output"):
        exporting.add_argument("--" + name, type=Path, required=True)
    packing = sub.add_parser("bundle")
    packing.add_argument("--population", type=Path, required=True)
    packing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            result = select_population(args.roles, args.labels, args.recordings, args.output)
        elif args.command == "bind":
            result = bind(args.population, args.runtime_config, args.worker_recipe, args.comparison_plan, args.output, args.arm)
        elif args.command == "bundle":
            result = bundle(args.population, args.output)
        elif args.command == "trace-bind":
            result = trace_binding(args.bound_root, args.worker_recipe, args.model_release, args.model_reservation,
                                   args.source_archive, args.app_image_sha256, args.output)
        elif args.command == "export-report":
            result = export_report(args.run, args.data_root, args.reference, args.private_output, args.public_output)
        else:
            result = execute_key_file(args.plan, args.release, args.reservation, args.budget, args.output, args.api_key_file)
        print(json.dumps(result, ensure_ascii=True))
        if args.command == "export-report":
            return int(result["less_than_ten_seconds"] != "demonstrated" or result["fresh_repeatability"] != "demonstrated")
        return int(args.command == "execute" and result["completed"] != result["expected_attempts"])
    except Exception:
        print(json.dumps({"status": "blocked", "reason_code": "acceptance_operational_prerequisite_missing",
                          "private_values_exported": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
