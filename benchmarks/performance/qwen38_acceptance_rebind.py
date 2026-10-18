"""CPU-only rebind to an actual selected worker; historical plans stay intact.

Run in the intended worker image/source/config context, with network disabled
and no GPU. Only metadata, source archives and the two frozen TRAIN files are
read. This never constructs a provider, pipeline, worker, model or HTTP client.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from benchmarks.performance import qwen38_acceptance as driver
from benchmarks.performance import qwen38_acceptance_ops as ops
from benchmarks.performance.repeatability import digest, file_digest, require, write_new

WORKER_FIELDS = ("window_seconds", "stride_seconds", "sample_fps", "max_frames_per_window",
    "max_concurrent_jobs", "max_concurrent_clips", "max_inflight_per_job", "compact_frame_ids",
    "concise_observations", "staged_screening", "max_refinement_passes", "job_timeout_seconds",
    "trusted_scene_context")
WORKER_MODULES = {
    "forte.pipeline": "src/forte/pipeline.py", "forte.config": "src/forte/config.py",
    "forte.model.adapter": "src/forte/model/adapter.py",
    "forte.video.sampling": "src/forte/video/sampling.py",
    "forte.video.decode": "src/forte/video/decode.py",
    "forte.decision.engine": "src/forte/decision/engine.py",
    "forte.decision.structured": "src/forte/decision/structured.py",
    "forte.decision.aliasing": "src/forte/decision/aliasing.py",
    "forte.events.builder": "src/forte/events/builder.py",
    "forte.evidence.assembly": "src/forte/evidence/assembly.py",
    "serving.qwen38_identity": "serving/qwen38_identity.py",
    "serving.qwen38_grammar": "serving/qwen38_grammar.py",
}
TRACE_MODULES = {"evaluation.qwen38_worker_trace": "evaluation/qwen38_worker_trace.py",
    "evaluation.qwen38_pipeline": "evaluation/qwen38_pipeline.py",
    "evaluation.qwen38_server_export": "evaluation/qwen38_server_export.py"}


def archive_pins(path, expected):
    """Hash bounded archive members without extraction or model/media access."""
    path = Path(path)
    require(driver.valid_digest(expected) and path.is_file() and not path.is_symlink()
            and file_digest(path) == expected, "rebind_actual_source_archive_required")
    rows, total = {}, 0
    with tarfile.open(path, "r:*") as source:
        for item in source:
            relative = PurePosixPath(item.name)
            require(item.isfile() and not item.issym() and not item.islnk()
                    and not relative.is_absolute() and relative.as_posix() == item.name
                    and all(part not in {".", ".."} for part in relative.parts)
                    and "\\" not in item.name and ":" not in item.name and item.name not in rows
                    and 0 <= item.size <= 16 * 1024 * 1024,
                    "rebind_source_archive_member_invalid")
            total += item.size
            require(len(rows) < 5000 and total <= 128 * 1024 * 1024, "rebind_source_archive_limit")
            stream = source.extractfile(item)
            require(stream is not None, "rebind_source_archive_unreadable")
            rows[item.name] = hashlib.sha256(stream.read()).hexdigest()
    return rows


def actual_modules(mapping, pins):
    found = {}
    for module, logical in mapping.items():
        spec = importlib.util.find_spec(module)
        require(spec is not None and spec.origin and spec.origin.endswith(".py"), "rebind_actual_module_source_missing")
        path = Path(spec.origin)
        actual = file_digest(path)
        require(path.is_file() and not path.is_symlink() and pins.get(logical) == actual,
                "rebind_actual_worker_module_differs_from_archive")
        found[logical] = {"path": str(path.resolve()), "sha256": actual}
    return found


def require_family_equivalence(family, actual_pins, model_pins, *, rules_sha256, taxonomy_sha256, equivalence=None):
    recipe = family["recipe"]
    require(family.get("schema_version") == "forte.qwen38-quality-family.v1"
            and family.get("recipe_sha256") == digest(recipe), "rebind_frozen_quality_family_invalid")
    prompt = recipe.get("prompt_implementation_pins")
    require(isinstance(prompt, dict) and prompt, "rebind_family_prompt_pins_missing")
    allowed = set()
    if equivalence is not None:
        from evaluation.qwen38_quality_equivalence import validate_equivalence
        validate_equivalence(family, equivalence, equivalence["actual_serving_source_archive_sha256"])
        allowed = {row["path"] for row in equivalence.get("audited_common_source_deltas", [])}
        require(allowed <= {"src/forte/pipeline.py"}, "rebind_unexpected_audited_common_delta")
    require(all(actual_pins.get(name, {}).get("sha256") == model_pins.get(name)
                and (model_pins.get(name) == expected or (name in allowed
                    and model_pins.get(name) == equivalence["actual_serving_archive_files"].get(name)))
                for name, expected in prompt.items()), "rebind_family_common_source_changed")
    require(recipe.get("rules_sha256") == rules_sha256 == model_pins.get("rules/catalog.json")
            and recipe.get("taxonomy_sha256") == taxonomy_sha256 == model_pins.get("annotations/taxonomy.v1.json"),
            "rebind_family_rules_or_taxonomy_changed")


def capture(*, runtime_config, family_path, worker_archive, worker_archive_sha256,
            model_archive, model_archive_sha256, trace_archive, trace_archive_sha256,
            cpu_image_sha256, equivalence_path, arm):
    """Import configuration helpers only; never initialize a model/provider."""
    from forte.config import WorkerSettings
    from forte.pipeline import _provider_configuration
    from serving.qwen38_grammar import POLICY

    require(driver.valid_digest(cpu_image_sha256), "rebind_actual_cpu_image_required")
    require(os.environ.get("FORTE_ACCEPTANCE_CONTEXT_CPU_IMAGE_SHA256") == cpu_image_sha256,
            "rebind_actual_cpu_image_metadata_mismatch")
    config = driver.read(runtime_config)
    if arm == "qwen38_tuned":
        from serving.qwen38_identity import load_expected_runtime_identity
        expected = load_expected_runtime_identity(runtime_config)
    elif arm == "qwen38_untuned":
        from serving.qwen38_untuned import expected_untuned_identity
        expected = expected_untuned_identity(config)
    else:
        require(False, "rebind_selected_qwen_arm_required")
    require(config.get("structured_decoding") == POLICY and config.get("package_versions", {}).get("xgrammar") == POLICY["version"]
            and expected["source_sha256"] == model_archive_sha256,
            "rebind_actual_native_grammar_or_source_missing")
    worker_pins = archive_pins(worker_archive, worker_archive_sha256)
    model_pins = archive_pins(model_archive, model_archive_sha256)
    trace_pins = archive_pins(trace_archive, trace_archive_sha256)
    actual = actual_modules(WORKER_MODULES, worker_pins)
    trace = actual_modules(TRACE_MODULES, trace_pins)
    settings = WorkerSettings.from_env()
    require(settings.model_candidate_launch_config_path is not None
            and settings.model_candidate_launch_config_path.resolve() == Path(runtime_config).resolve(),
            "rebind_actual_worker_candidate_config_mismatch")
    provider = _provider_configuration(settings)  # Dataclass configuration only.
    worker = {name: getattr(settings, name) for name in WORKER_FIELDS}
    worker_recipe = {**worker, "provider_max_attempts": provider.max_attempts}
    for key, expected_value in ops.INTENDED_RECIPE.items():
        require(worker_recipe.get(key) == expected_value, "rebind_actual_worker_profile_mismatch")
    require(provider.require_readiness is True and provider.max_tokens == config["generation"]["max_new_tokens"] == 4096
            and provider.max_frames == config["limits"]["max_images"] == 16
            and provider.request_timeout_sec == config["limits"]["request_timeout_seconds"],
            "rebind_actual_provider_profile_mismatch")
    family = driver.read(driver.private(family_path))
    from evaluation.qwen38_quality_equivalence import validate_equivalence
    equivalence = driver.read(driver.private(equivalence_path))
    validate_equivalence(family, equivalence, model_archive_sha256)
    require(equivalence.get("actual_serving_archive_files") == model_pins,
            "rebind_equivalence_actual_archive_manifest_mismatch")
    require_family_equivalence(family, actual, model_pins, rules_sha256=file_digest(settings.rules_path),
                               taxonomy_sha256=file_digest(settings.taxonomy_path), equivalence=equivalence)
    require(family["recipe"]["worker_settings"] == worker and family["recipe"]["provider_max_attempts"] == provider.max_attempts,
            "rebind_actual_worker_differs_from_quality_family")
    require(all(config["generation"].get(key) == value for key, value in family["recipe"]["generation"].items())
            and config["limits"]["max_context_tokens"] == family["recipe"]["context_tokens"],
            "rebind_actual_generation_or_context_differs_from_family")
    return {"schema_version": "forte.qwen38-acceptance-actual-context.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "arm": arm,
        "cpu_image_sha256": cpu_image_sha256, "worker_archive_sha256": worker_archive_sha256,
        "model_archive_sha256": model_archive_sha256, "trace_archive_sha256": trace_archive_sha256,
        "worker_archive_path": str(Path(worker_archive).resolve()),
        "model_archive_path": str(Path(model_archive).resolve()), "trace_archive_path": str(Path(trace_archive).resolve()),
        "runtime_config_file_sha256": file_digest(runtime_config), "expected_runtime_identity": expected,
        "runtime_config_path": str(Path(runtime_config).resolve()),
        "family_path": str(Path(family_path).resolve()), "quality_equivalence_path": str(Path(equivalence_path).resolve()),
        "expected_model_identity": {key: expected.get(key) for key in ("model_id", "model_revision", "adapter_sha256")},
        "worker_recipe": worker_recipe, "actual_worker_modules": actual, "actual_trace_modules": trace,
        "family_file_sha256": file_digest(family_path), "family_recipe_sha256": family["recipe_sha256"],
        "quality_source_equivalence_file_sha256": file_digest(equivalence_path),
        "quality_source_equivalence_receipt_sha256": equivalence["receipt_sha256"],
        "provider_profile": {"max_attempts": provider.max_attempts, "max_frames": provider.max_frames,
            "max_tokens": provider.max_tokens, "seed": provider.seed, "require_readiness": provider.require_readiness,
            "request_timeout_seconds": provider.request_timeout_sec,
            "generation_sha256": expected["generation_sha256"], "structured_decoding": config["structured_decoding"]},
        "configured_source_equivalence_verified": True, "loaded_model_or_gpu_execution_verified": False,
        "provider_or_worker_constructed": False, "model_calls_made": 0}


def rebind(*, output, population_path, **kwargs):
    """Produce fresh IDs and source/trace bindings; grants no execution release."""
    output = driver.private(output)
    require(not output.exists(), "rebind_new_output_required")
    context = capture(**kwargs)
    family = driver.read(kwargs["family_path"])
    config = driver.read(kwargs["runtime_config"])
    recipe = {**family["recipe"], "worker_settings": {key: value for key, value in context["worker_recipe"].items()
                                                     if key != "provider_max_attempts"},
        "provider_max_attempts": context["worker_recipe"]["provider_max_attempts"],
        "actual_selected_runtime": {"runtime_config_file_sha256": context["runtime_config_file_sha256"],
            "expected_runtime_identity": context["expected_runtime_identity"], "structured_decoding": config["structured_decoding"]},
        "actual_worker_archive_sha256": context["worker_archive_sha256"],
        "actual_trace_archive_sha256": context["trace_archive_sha256"],
        "acceptance_timing": "client_upload_start_to_full_result_byte; usable_UI_boundary_separate"}
    envelope = {"schema_version": "forte.qwen38-acceptance-envelope.v1",
        "frozen_quality_family_file_sha256": context["family_file_sha256"],
        "quality_family_not_rewritten": True, "arms": {kwargs["arm"]: context["expected_model_identity"]},
        "recipe": recipe, "recipe_sha256": digest(recipe)}
    output.mkdir(parents=True)
    write_new(output / "actual-context.private.json", context)
    write_new(output / "worker-recipe.actual.json", context["worker_recipe"])
    write_new(output / "acceptance-envelope.actual.json", envelope)
    result = ops.bind(population_path, kwargs["runtime_config"], output / "worker-recipe.actual.json",
                      output / "acceptance-envelope.actual.json", output / "bound", kwargs["arm"])
    write_new(output / "trace-context.private.json", {"actual_worker_context_file_sha256": file_digest(output / "actual-context.private.json"),
        "worker_pipeline_sha256": context["actual_worker_modules"]["src/forte/pipeline.py"]["sha256"],
        "wrapper_sha256": context["actual_trace_modules"]["evaluation/qwen38_worker_trace.py"]["sha256"],
        "cpu_image_sha256": context["cpu_image_sha256"], "actual_trace_archive_sha256": context["trace_archive_sha256"],
        "trace_binding_requires_new_exact_model_release_and_reservation": True})
    return {**result, "status": "actual_context_rebound_not_authorized", "historical_attempts_reused": False,
            "configured_source_equivalence_verified": True, "loaded_model_verified": False, "model_calls_made": 0}


def trace_rebind(rebound_root, model_release_path, model_reservation_path):
    """Bind actual module origins, never an API image's assumed worker code."""
    from training.qwen38_preflight import validate_attempt_reservation
    rebound_root = driver.private(rebound_root)
    old = driver.read(rebound_root / "actual-context.private.json")
    current = capture(runtime_config=old["runtime_config_path"], family_path=old["family_path"],
        worker_archive=old["worker_archive_path"], worker_archive_sha256=old["worker_archive_sha256"],
        model_archive=old["model_archive_path"], model_archive_sha256=old["model_archive_sha256"],
        trace_archive=old["trace_archive_path"], trace_archive_sha256=old["trace_archive_sha256"],
        cpu_image_sha256=old["cpu_image_sha256"], equivalence_path=old["quality_equivalence_path"], arm=old["arm"])
    require({key: value for key, value in current.items() if key != "captured_at_utc"}
            == {key: value for key, value in old.items() if key != "captured_at_utc"}, "rebind_actual_context_changed")
    bound = rebound_root / "bound"
    source_plan = driver.validate_plan(driver.read(bound / "plan.json"))
    model_plan = driver.read(bound / "model-attempt-plan.json")
    release = driver.read(driver.private(model_release_path))
    reservation = driver.read(driver.private(model_reservation_path))
    validate_attempt_reservation(reservation, model_plan, plan_sha256=file_digest(bound / "model-attempt-plan.json"), phase="acceptance")
    require(release.get("schema_version") == "forte.qwen38-root-release.v1"
            and release.get("released_by_root") is True and release.get("phase") == "acceptance"
            and release.get("maximum_attempts") == len(model_plan["attempts"])
            and release.get("attempt_plan_sha256") == file_digest(bound / "model-attempt-plan.json")
            and release.get("expected_model_identity") == current["expected_model_identity"]
            and reservation.get("release_sha256") == file_digest(model_release_path)
            and all(release.get(key) == reservation.get(key) == value for key, value in {
                "source_archive_sha256": current["model_archive_sha256"],
                "runtime_image_sha256": current["expected_runtime_identity"]["image_sha256"],
                "base_manifest_sha256": current["expected_runtime_identity"]["model_manifest_sha256"]}.items()),
            "rebind_exact_actual_model_release_required")
    def pinned(path):
        return {"path": str(Path(path).resolve()), "sha256": file_digest(path)}
    binding = {"schema_version": "forte.qwen38-worker-trace-binding.v1",
        "source_attempts": [{key: row[key] for key in ("attempt_id", "content_sha256", "split", "repeat_index")}
                            for row in source_plan["attempts"]],
        "expected_model_identity": current["expected_model_identity"], "provider_max_attempts": 2,
        "maximum_model_requests": len(model_plan["attempts"]),
        "worker_settings": {key: value for key, value in current["worker_recipe"].items() if key != "provider_max_attempts"},
        "model_attempt_plan": pinned(bound / "model-attempt-plan.json"),
        "model_release": pinned(model_release_path), "model_reservation": pinned(model_reservation_path),
        "wrapper_sha256": current["actual_trace_modules"]["evaluation/qwen38_worker_trace.py"]["sha256"],
        "pipeline_sha256": current["actual_worker_modules"]["src/forte/pipeline.py"]["sha256"],
        "trace_source_archive": pinned(current["trace_archive_path"]),
        "trace_runtime_image_sha256": current["cpu_image_sha256"],
        "conditions_sha256": source_plan["conditions_sha256"], "recipe_sha256": source_plan["recipe_sha256"],
        "actual_worker_context_file_sha256": file_digest(rebound_root / "actual-context.private.json"),
        "quality_source_equivalence_receipt_sha256": current["quality_source_equivalence_receipt_sha256"]}
    write_new(bound / "trace-binding.json", binding)
    return {"status": "actual_worker_trace_bound_not_activated", "source_jobs": 6,
            "maximum_generation_http_attempts": len(model_plan["attempts"]), "model_calls_made": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--population", type=Path, required=True)
    prep.add_argument("--runtime-config", type=Path, required=True)
    prep.add_argument("--family", type=Path, required=True)
    prep.add_argument("--quality-equivalence", type=Path, required=True)
    for name in ("worker", "model", "trace"):
        prep.add_argument("--" + name + "-archive", type=Path, required=True)
        prep.add_argument("--" + name + "-archive-sha256", required=True)
    prep.add_argument("--cpu-image-sha256", required=True)
    prep.add_argument("--arm", choices=("qwen38_tuned", "qwen38_untuned"), required=True)
    prep.add_argument("--output", type=Path, required=True)
    tracing = sub.add_parser("trace")
    for name in ("rebound-root", "model-release", "model-reservation"):
        tracing.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "trace":
            result = trace_rebind(args.rebound_root, args.model_release, args.model_reservation)
        else:
            result = rebind(output=args.output, population_path=args.population, runtime_config=args.runtime_config,
                family_path=args.family, worker_archive=args.worker_archive, worker_archive_sha256=args.worker_archive_sha256,
                model_archive=args.model_archive, model_archive_sha256=args.model_archive_sha256,
                trace_archive=args.trace_archive, trace_archive_sha256=args.trace_archive_sha256,
                cpu_image_sha256=args.cpu_image_sha256, equivalence_path=args.quality_equivalence, arm=args.arm)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status": "blocked", "reason_code": "acceptance_actual_context_binding_missing_or_mismatched",
                          "model_calls_made": 0, "private_values_exported": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
