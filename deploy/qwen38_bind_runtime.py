"""Bind actual completed Qwen3.8 artifacts; CPU-only, no CUDA/model imports.

Run this separately mounted helper in the measured immutable runtime image,
with the frozen source, completed run and source archive mounted read-only.
It verifies archive-to-executed-source hashes before importing that source's
config validators. No model load, private media read or serving mutation occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile


PACKAGES = ("torch", "transformers", "peft", "fastapi", "uvicorn", "pillow", "jsonschema", "bitsandbytes", "xgrammar")
OPTIONAL_KERNEL_PACKAGES = ("causal-conv1d", "fla-core", "flash-linear-attention")
CORE_FILES = frozenset({
    "serving/qwen38_identity.py", "serving/qwen38_launch.py", "serving/qwen38_service.py",
    "serving/stage_qwen38.py", "serving/readiness.py", "serving/security.py",
    "serving/qwen38_comparison_profile.py", "serving/qwen38_untuned.py",
    "serving/qwen38_grammar.py",
    "serving/qwen38_adapter_contract.py", "serving/qwen38_source_lineage.py",
    "src/forte/model/adapter.py", "src/forte/model/smoke.py",
    "configs/serving/qwen38/public-candidate.manifest.json",
})


class BindError(RuntimeError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise BindError(code)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "binding_duplicate_json_key")
        result[key] = value
    return result


def _float(value):
    result = float(value)
    require(math.isfinite(result), "binding_nonfinite_json")
    return result


def read_json(path: Path) -> dict:
    data = path.read_bytes()
    require(len(data) <= 4 * 1024 * 1024, "binding_json_too_large")
    result = json.loads(data, object_pairs_hook=_pairs, parse_float=_float,
                        parse_constant=lambda _: (_ for _ in ()).throw(BindError("binding_nonfinite_json")))
    require(isinstance(result, dict), "binding_json_object_required")
    return result


def relative_path(name: str) -> str:
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            "binding_unsafe_artifact_path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name
            and all(part not in {".", ".."} for part in relative.parts), "binding_unsafe_artifact_path")
    return name


def verify_source_tree(code_root: Path, archive: Path, expected_sha256: str) -> dict:
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None
            and sha256(archive) == expected_sha256, "binding_source_archive_mismatch")
    code_root = code_root.resolve()
    files, total = {}, 0
    with tarfile.open(archive, "r:*") as package:
        for member in package:
            name = relative_path(member.name)
            require(member.isfile() and not member.issym() and not member.islnk()
                    and name not in files and 0 <= member.size <= 16 * 1024 * 1024,
                    "binding_source_member_invalid")
            total += member.size
            require(len(files) < 5000 and total <= 128 * 1024 * 1024, "binding_source_archive_too_large")
            target = code_root / name
            require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(code_root),
                    "binding_source_file_missing_or_linked")
            stream = package.extractfile(member)
            require(stream is not None, "binding_source_member_unreadable")
            expected = hashlib.sha256(stream.read()).hexdigest()
            require(sha256(target) == expected, "binding_executed_source_bytes_mismatch")
            files[name] = expected
    require(CORE_FILES <= files.keys(), "binding_executed_core_source_missing")
    actual = {path.relative_to(code_root).as_posix() for path in code_root.rglob("*.py")}
    require(actual <= files.keys(), "binding_unarchived_executable_source")
    require(not any(code_root.rglob("*.pyc")), "binding_unreviewed_source_bytecode")
    return {"source_archive_sha256": expected_sha256, "archive_files_verified": len(files),
            "archive_bytes_verified": total, "executed_core_files": {name: files[name] for name in sorted(CORE_FILES)},
            "all_executed_python_files_sha256": canonical({name: digest for name, digest in files.items()
                                                          if name.endswith(".py")}),
            "source_verified_before_import": True}


def collect_packages() -> dict:
    # importlib.metadata never imports torch, bitsandbytes or initializes CUDA.
    result = {name: version(name) for name in PACKAGES}
    for name in OPTIONAL_KERNEL_PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def bind_quality_family(path: Path, expected_sha256: str, config: dict, code_root: Path,
                        *, lineage_proof: dict | None = None) -> dict:
    """Bind a frozen population/recipe without editing it or reading media."""
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None and sha256(path) == expected_sha256,
            "binding_quality_family_file_mismatch")
    family = read_json(path)
    recipe = family.get("recipe", {})
    require(family.get("schema_version") == "forte.qwen38-quality-family.v1"
            and family.get("recipe_sha256") == canonical(recipe), "binding_quality_family_invalid")
    worker = recipe.get("worker_settings", {})
    require(recipe.get("rule_selection") == {"enabled_rules": 4, "optional_expanded_rules_selected": False,
                                             "visible_fact_axes": 19}
            and worker.get("window_seconds") == 8 and worker.get("stride_seconds") == 4
            and worker.get("sample_fps") == 2 and worker.get("max_frames_per_window") == config["limits"]["max_images"]
            and worker.get("staged_screening") is False
            and worker.get("max_concurrent_clips") == worker.get("max_concurrent_jobs") == worker.get("max_inflight_per_job") == 1
            and worker.get("max_refinement_passes") == 8 and worker.get("job_timeout_seconds") == 285
            and recipe.get("context_tokens") == config["limits"]["max_context_tokens"]
            and all(recipe.get("generation", {}).get(key) == config["generation"][key]
                    for key in ("max_new_tokens", "seed", "do_sample", "temperature", "num_beams")),
            "binding_quality_family_runtime_profile_mismatch")
    prompt_pins = recipe.get("prompt_implementation_pins", {})
    require(isinstance(prompt_pins, dict) and bool(prompt_pins), "binding_quality_family_prompt_pins_missing")
    audited = []
    for name, digest in prompt_pins.items():
        relative_path(name)
        actual = sha256(code_root / name)
        if actual == digest:
            continue
        lineage = lineage_proof or {}
        delta = lineage.get("candidate_http_timeout_delta") or {}
        require(name == "src/forte/pipeline.py" and lineage.get("verified") is True
                and lineage.get("schema_version") == "forte.qwen38.training-serving-lineage.v1"
                and lineage.get("runtime_source_archive_sha256") == config["source_archive"]["sha256"]
                and delta.get("schema_version") == "forte.qwen38.candidate-http-timeout-delta.v1"
                and all(delta.get(key) is True for key in ("verified", "candidate_branch_only",
                    "other_ast_identical", "other_source_bytes_identical", "baseline_default_unchanged"))
                and delta.get("added_keyword") == "request_timeout_sec"
                and delta.get("added_value") == 'config["limits"]["request_timeout_seconds"]'
                and delta.get("old_source_sha256") == digest and delta.get("new_source_sha256") == actual,
                "binding_quality_family_prompt_bytes_changed")
        audited.append({"path": name, "proof": delta})
    require(sha256(code_root / "rules/catalog.json") == recipe.get("rules_sha256")
            and sha256(code_root / "annotations/taxonomy.v1.json") == recipe.get("taxonomy_sha256"),
            "binding_quality_family_rule_or_taxonomy_changed")
    return {"schema_version": "forte.qwen38.frozen-family-runtime-binding.v1",
            "family_file_sha256": expected_sha256, "family_semantic_sha256": canonical(family),
            "recipe_sha256": family["recipe_sha256"], "family_source_archive_sha256": recipe["source_archive_sha256"],
            "actual_serving_source_archive_sha256": config["source_archive"]["sha256"],
            "same_family_prompt_and_pipeline_bytes_verified": not audited,
            "family_prompt_pipeline_verified_with_audited_timeout": bool(audited),
            "prompt_implementation_pin_count": len(prompt_pins),
            "prompt_implementation_byte_identical_count": len(prompt_pins) - len(audited),
            "audited_prompt_implementation_delta_count": len(audited),
            "audited_prompt_implementation_deltas": audited,
            "full_rules": 4, "full_fact_axes": 19, "worker_settings": worker,
            "arm_processor_profile": {"max_pixels_per_image": config["limits"]["max_pixels_per_image"],
                                      "disclosed_separately_from_baseline": True},
            "generation_schema_enforcement": "xgrammar_hf_and_strict_post_generation_validation",
            "decoder_constrained_json": True, "family_rewritten": False}


def _write_private_new(path: Path, value: dict) -> None:
    require(not path.exists(), "binding_output_already_exists")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bind_runtime(*, code_root: Path, source_archive: Path, source_sha256: str, run_root: Path,
                 image_sha256: str, package_versions: dict, max_pixels: int = 262144,
                 max_new_tokens: int = 4096, max_context_tokens: int = 32768, max_images: int = 16,
                 request_timeout: int = 240, port: int = 8000,
                 family_path: Path | None = None, family_sha256: str | None = None,
                 runtime_proofs: tuple[Path, ...] = (),
                 training_source_archive: Path | None = None,
                 training_source_sha256: str | None = None) -> tuple[dict, dict, dict]:
    source = verify_source_tree(code_root, source_archive, source_sha256)
    require(re.fullmatch(r"[0-9a-f]{64}", image_sha256) is not None, "binding_image_sha256_required")
    require(set(PACKAGES) <= package_versions.keys()
            and all(isinstance(value, str) and value for value in package_versions.values()),
            "binding_package_versions_missing")
    require(package_versions["xgrammar"] == "0.2.7", "binding_xgrammar_version_unverified")
    run_path = run_root / "run.json"
    run = read_json(run_path)
    require(run.get("schema_version") == "forte.qwen38.silver-run.v1"
            and run.get("status") == "candidate_unselected" and run.get("gpu_executed") is True
            and run.get("dataset_origin") == "local_teacher_silver" and run.get("human_verified") is False
            and type(run.get("optimizer_steps")) is int and run["optimizer_steps"] >= 2,
            "binding_completed_actual_training_run_required")
    require(run.get("runtime_image_sha256") == image_sha256, "binding_training_image_mismatch")
    # A new serving validator does not rewrite the completed fit's source pin.
    # Cross-release use requires the exact original archive plus audited deltas.
    require((training_source_archive is None) == (training_source_sha256 is None),
            "binding_training_source_arguments_incomplete")
    training_archive = training_source_archive or source_archive
    training_digest = training_source_sha256 or source_sha256
    sys.path[:0] = [str(code_root), str(code_root / "src")]
    lineage_module = importlib.import_module("serving.qwen38_source_lineage")
    require(Path(lineage_module.__file__).resolve() == (code_root / "serving/qwen38_source_lineage.py").resolve(),
            "binding_lineage_import_source_mismatch")
    try:
        lineage = lineage_module.verify_source_lineage(runtime_archive=source_archive, runtime_sha256=source_sha256,
            training_archive=training_archive, training_sha256=training_digest,
            recorded_training_sha256=run.get("code_archive_sha256"))
    except lineage_module.SourceLineageError as error:
        raise BindError(str(error)) from None
    for key in ("finite_loss_verified", "gradients_verified", "frozen_tensors_unchanged",
                "peft_load_verified", "reload_forward_verified"):
        require(run.get(key) is True, "binding_training_verification_incomplete")
    require(run.get("processor_limits", {}).get("max_pixels_per_image") == max_pixels
            and run.get("seed") == 20261017, "binding_training_profile_mismatch")
    require(type(run.get("changed_adapter_tensors")) is int and run["changed_adapter_tensors"] > 0
            and run.get("changed_adapter_groups") == {"self_attn": True, "linear_attn": True, "mlp": True},
            "binding_changed_hybrid_adapter_evidence_missing")
    for key, value in run.get("versions", {}).items():
        if key in package_versions:
            require(package_versions[key] == value, "binding_training_package_mismatch")
    checksums = run.get("adapter_checksums")
    require(isinstance(checksums, dict) and {"adapter_config.json", "adapter_model.safetensors"} <= checksums.keys(),
            "binding_adapter_inventory_missing")
    adapter_root = (run_root / "adapter").resolve()
    actual = {path.relative_to(adapter_root).as_posix() for path in adapter_root.rglob("*") if path.is_file()}
    require(actual == set(checksums), "binding_adapter_extra_or_missing_files")
    for name, expected in checksums.items():
        relative_path(name)
        target = adapter_root / name
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None and not target.is_symlink()
                and target.resolve().is_relative_to(adapter_root) and sha256(target) == expected,
                "binding_adapter_bytes_mismatch")
    require(canonical(checksums) == run.get("adapter_tree_sha256"), "binding_adapter_tree_mismatch")
    adapter_config = read_json(adapter_root / "adapter_config.json")
    require(adapter_config.get("peft_type") == "LORA" and not adapter_config.get("modules_to_save")
            and adapter_config.get("base_model_name_or_path") == "Qwen/Qwen3.8-27B"
            and run.get("changed_adapter_groups") == {"self_attn": True, "linear_attn": True, "mlp": True},
            "binding_complete_language_hybrid_adapter_required")
    contract_module = importlib.import_module("serving.qwen38_adapter_contract")
    require(Path(contract_module.__file__).resolve() == (code_root / "serving/qwen38_adapter_contract.py").resolve(),
            "binding_adapter_contract_import_source_mismatch")
    try:
        adapter_proof = contract_module.verify_adapter_contract(run.get("adapted_module_inventory"),
            adapter_config, adapter_root / "adapter_model.safetensors")
    except contract_module.AdapterContractError as error:
        raise BindError(str(error)) from None
    manifest_path = code_root / "configs/serving/qwen38/public-candidate.manifest.json"
    manifest = read_json(manifest_path)
    require(sha256(manifest_path) == run.get("base_manifest_sha256"), "binding_training_base_manifest_mismatch")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    config = {
        "schema_version": "forte.qwen38.runtime.v1", "backend": "transformers_peft", "production_selected": False,
        "structured_decoding": {"backend": "xgrammar_hf", "version": "0.2.7",
            "schema_policy": "forte.decision-json-subset.v1", "any_order": False,
            "fact_order": "canonical_requested_fact_ids.v1",
            "any_whitespace": False, "strict_mode": True, "max_cache_entries": 8, "max_cache_bytes": 67108864},
        "model_id": manifest["model_id"], "model_revision": manifest["revision"],
        "processor_revision": manifest["processor_revision"], "served_model_name": "Qwen3.8-27B-forte-silver",
        "model_path": "/models/Qwen3.8-27B", "adapter_path": "/candidate/adapter",
        "base_manifest": {"path": "/app/configs/serving/qwen38/public-candidate.manifest.json", "sha256": sha256(manifest_path)},
        "tokenizer_sha256": entries["tokenizer.json"]["sha256"],
        "chat_template_sha256": entries["chat_template.jinja"]["sha256"],
        "training_run": {"path": "/candidate/run.json", "sha256": sha256(run_path)},
        "adapter_sha256": run["adapter_tree_sha256"], "adapter_config_sha256": checksums["adapter_config.json"],
        "source_archive": {"path": "/release/source.tar", "sha256": source_sha256},
        "training_source_archive": {"path": "/release/source.tar" if training_digest == source_sha256
                                    else "/release/training-source.tar", "sha256": training_digest},
        "training_source_lineage": lineage,
        "runtime_image_sha256": image_sha256, "package_versions": package_versions,
        "quantization": run["quantization"],
        "generation": {"do_sample": False, "temperature": 0, "seed": 20261017,
                       "max_new_tokens": max_new_tokens, "enable_thinking": False,
                       "preserve_thinking": False, "num_beams": 1},
        "device": "cuda:0", "host": "0.0.0.0", "port": port,
        "limits": {"max_images": max_images, "max_pixels_per_image": max_pixels,
                   "max_source_pixels_per_image": 64000000, "max_context_tokens": max_context_tokens,
                   "max_body_bytes": 100663296, "request_timeout_seconds": request_timeout},
        "prewarm_shapes": [{"frames": 1, "width": 448, "height": 448},
                           {"frames": min(8, max_images), "width": 672, "height": 672},
                           {"frames": max_images, "width": 1024, "height": 1024}],
        "executed_source_binding": source,
    }
    if family_path is not None:
        require(family_sha256 is not None, "binding_quality_family_sha256_required")
        config["quality_family_binding"] = bind_quality_family(family_path, family_sha256, config, code_root,
                                                              lineage_proof=lineage)
    proof_pins = []
    for proof_path in runtime_proofs:
        proof = read_json(proof_path)
        proof_pins.append({"sha256": sha256(proof_path), "schema_version": proof.get("schema_version"),
                           "status": proof.get("status"), "identity_claims_verified_by_binder": False})
    if proof_pins:
        config["external_runtime_proof_pins"] = proof_pins
    # Verify the original immutable implementation, not a working-tree edit.
    sys.path[:0] = [str(code_root), str(code_root / "src")]
    identity = importlib.import_module("serving.qwen38_identity")
    require(Path(identity.__file__).resolve() == (code_root / "serving/qwen38_identity.py").resolve(),
            "binding_validator_import_source_mismatch")
    identity.validate_config(config)
    require(adapter_config.get("revision") == config["model_revision"]
            and run.get("base_revision") == config["model_revision"]
            and run.get("processor_revision") == config["processor_revision"], "binding_adapter_revision_mismatch")
    comparison = importlib.import_module("serving.qwen38_comparison_profile")
    require(Path(comparison.__file__).resolve() == (code_root / "serving/qwen38_comparison_profile.py").resolve(),
            "binding_comparison_import_source_mismatch")
    untuned = {key: value for key, value in config.items() if key not in {
        "adapter_path", "training_run", "training_source_archive", "training_source_lineage"}}
    untuned.update(schema_version="forte.qwen38.untuned-runtime.v1", backend="transformers_base_diagnostic",
                   arm="qwen38_untuned", diagnostic_only=True, served_model_name="Qwen3.8-27B-forte-untuned-diagnostic",
                   adapter_sha256=None, adapter_config_sha256=None, training_run_sha256=None)
    untuned["runtime_profile_sha256"] = comparison.runtime_profile_sha256(untuned)
    profile_sha = comparison.require_matching_profiles(untuned, config)
    untuned_module = importlib.import_module("serving.qwen38_untuned")
    require(Path(untuned_module.__file__).resolve() == (code_root / "serving/qwen38_untuned.py").resolve(),
            "binding_untuned_import_source_mismatch")
    untuned_module.validate_untuned_config(untuned)
    receipt = {"schema_version": "forte.qwen38.actual-runtime-binding.v1", "status": "bound_unloaded",
               "captured_at_utc": datetime.now(timezone.utc).isoformat(), "source_verification": source,
               "run_sha256": sha256(run_path), "adapter_sha256": run["adapter_tree_sha256"],
               "image_sha256": image_sha256, "package_versions": package_versions,
               "candidate_config_sha256": canonical(config), "untuned_config_sha256": canonical(untuned),
               "comparison_runtime_profile_sha256": profile_sha, "base_shards_reverified_by_binder": False,
               "model_loaded": False, "gpu_initialized": False, "production_selected": False,
               "quality_family_binding": config.get("quality_family_binding"),
               "external_runtime_proof_pins": proof_pins,
               "binder_sha256": sha256(Path(__file__))}
    receipt["training_source_lineage"] = lineage
    receipt["adapter_topology"] = adapter_proof
    return config, untuned, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-packages", action="store_true")
    parser.add_argument("--verify-source-only", action="store_true")
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--training-source-archive", type=Path,
                        help="Exact immutable archive used for the completed fit, mounted read-only")
    parser.add_argument("--training-source-sha256", help="Reviewed original fit source digest; never replace run bytes")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--image-sha256")
    parser.add_argument("--packages-json", type=Path,
                        help="Previously captured package map from the exact runtime image; otherwise inspect current metadata")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quality-family", type=Path)
    parser.add_argument("--quality-family-sha256")
    parser.add_argument("--runtime-proof", action="append", type=Path, default=[],
                        help="Pin actual independently reviewed image/GPU proof receipt bytes without manufacturing a pass")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.collect_packages:
        print(json.dumps(collect_packages()))
        return 0
    if args.verify_source_only:
        require(args.code_root is not None and args.source_archive is not None and args.source_sha256 is not None,
                "binding_source_arguments_missing")
        print(json.dumps({"source_verification": verify_source_tree(args.code_root, args.source_archive,
                                                                  args.source_sha256),
                          "model_loaded": False, "gpu_initialized": False}))
        return 0
    require(all(getattr(args, key) is not None for key in
                ("code_root", "source_archive", "source_sha256", "run_root", "image_sha256", "output_dir")),
            "binding_required_arguments_missing")
    packages = read_json(args.packages_json) if args.packages_json else collect_packages()
    config, untuned, receipt = bind_runtime(code_root=args.code_root, source_archive=args.source_archive,
        source_sha256=args.source_sha256, run_root=args.run_root, image_sha256=args.image_sha256,
        package_versions=packages, max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens, max_images=args.max_images,
        request_timeout=args.request_timeout, port=args.port, family_path=args.quality_family,
        family_sha256=args.quality_family_sha256, runtime_proofs=tuple(args.runtime_proof),
        training_source_archive=args.training_source_archive, training_source_sha256=args.training_source_sha256)
    for name, value in (("candidate.runtime.bound.json", config), ("untuned.runtime.bound.json", untuned),
                        ("runtime-binding.receipt.json", receipt)):
        _write_private_new(args.output_dir / name, value)
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in {"source_verification", "package_versions"}}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "blocked", "reason_code": str(error)
                          if isinstance(error, BindError) else "binding_runtime_error",
                          "error_type": type(error).__name__}))
        raise SystemExit(2)
