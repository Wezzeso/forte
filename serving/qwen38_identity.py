"""Fail-closed, offline identity checks for the separate Qwen3.8 candidate.

This module never loads a model. A successful verification proves artifact
identity only; it does not establish training quality, GPU compatibility or
readiness. The rollback baseline verifier is deliberately not reused/relaxed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


class CandidateError(RuntimeError):
    """A public, static reason code without private paths or model output."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CandidateError(code)


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "candidate_duplicate_json_key")
        result[key] = value
    return result


def _finite_float(value):
    result = float(value)
    require(math.isfinite(result), "candidate_json_nonfinite_number")
    return result


def load_json(path: str | Path, *, max_bytes: int = 2 * 1024 * 1024) -> dict:
    try:
        raw = Path(path).read_bytes()
        require(len(raw) <= max_bytes, "candidate_json_too_large")
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_pairs,
                           parse_float=_finite_float,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        require(isinstance(value, dict), "candidate_json_object_required")
        return value
    except (OSError, ValueError, TypeError, RecursionError):
        raise CandidateError("candidate_json_unavailable_or_invalid") from None


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _local(value) -> Path:
    require(isinstance(value, str) and (Path(value).is_absolute() or PurePosixPath(value).is_absolute()),
            "candidate_absolute_local_path_required")
    require(not value.startswith(("//", "\\\\")), "candidate_network_path_forbidden")
    return Path(value)


def _bound_file(entry: dict) -> Path:
    require(isinstance(entry, dict) and _hash(entry.get("sha256")), "candidate_file_pin_missing")
    path = _local(entry.get("path"))
    require(path.is_file() and file_sha256(path) == entry["sha256"], "candidate_bound_file_mismatch")
    return path


def verify_candidate_base(snapshot: str | Path, manifest_path: str | Path) -> dict:
    """Use A07's complete public inventory verifier without changing baseline."""
    from serving.stage_qwen38 import validate_manifest, verify_snapshot

    manifest = validate_manifest(load_json(manifest_path))
    return verify_snapshot(Path(snapshot), manifest)


def validate_config(config: dict) -> None:
    require(config.get("schema_version") == "forte.qwen38.runtime.v1", "candidate_runtime_schema_invalid")
    require(config.get("backend") == "transformers_peft", "candidate_backend_unsupported")
    from serving.qwen38_grammar import POLICY
    require(config.get("structured_decoding") == POLICY, "candidate_structured_decoding_policy_required")
    require(config.get("model_id") == MODEL_ID and config.get("model_revision") == REVISION,
            "candidate_model_identity_invalid")
    require(config.get("processor_revision") == REVISION, "candidate_processor_revision_invalid")
    require(config.get("production_selected") is False, "candidate_requires_separate_selection_gate")
    require(isinstance(config.get("served_model_name"), str)
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", config["served_model_name"]) is not None,
            "candidate_alias_invalid")
    _local(config.get("model_path"))
    _local(config.get("adapter_path"))
    require(_hash(config.get("runtime_image_sha256")), "candidate_image_pin_missing")
    require(_hash(config.get("tokenizer_sha256")) and _hash(config.get("chat_template_sha256"))
            and _hash(config.get("adapter_sha256")) and _hash(config.get("adapter_config_sha256")),
            "candidate_component_pins_missing")
    for name in ("base_manifest", "training_run", "source_archive"):
        entry = config.get(name)
        require(isinstance(entry, dict) and _hash(entry.get("sha256")), "candidate_file_pin_missing")
        _local(entry.get("path"))
    if "training_source_archive" in config:
        entry = config["training_source_archive"]
        require(isinstance(entry, dict) and _hash(entry.get("sha256")), "candidate_training_source_pin_missing")
        _local(entry.get("path"))
    require(config.get("quantization") in (
        {"format": "nf4", "compute_dtype": "bfloat16", "double_quant": True},
        {"format": "none", "compute_dtype": "bfloat16", "double_quant": False}),
        "candidate_quantization_unsupported")
    generation = config.get("generation", {})
    require(set(generation) == {"do_sample", "temperature", "seed", "max_new_tokens",
                               "enable_thinking", "preserve_thinking", "num_beams"}
            and generation.get("do_sample") is False and generation.get("temperature") == 0
            and generation.get("seed") == 20261017 and generation.get("num_beams") == 1
            and generation.get("enable_thinking") is False and generation.get("preserve_thinking") is False
            and type(generation.get("max_new_tokens")) is int and 1 <= generation["max_new_tokens"] <= 8192,
            "candidate_generation_identity_invalid")
    packages = config.get("package_versions", {})
    required = {"torch", "transformers", "peft", "fastapi", "uvicorn", "pillow", "jsonschema", "xgrammar"}
    if config["quantization"]["format"] == "nf4":
        required.add("bitsandbytes")
    require(required.issubset(packages) and all(isinstance(v, str) and v for v in packages.values()),
            "candidate_package_versions_missing")
    require(packages["xgrammar"] == POLICY["version"], "candidate_xgrammar_version_unpinned")
    require(config.get("host") in {"0.0.0.0", "127.0.0.1"}
            and type(config.get("port")) is int and 1024 <= config["port"] <= 65535,
            "candidate_listen_config_invalid")
    limits = config.get("limits", {})
    for key, maximum in (("max_images", 64), ("max_pixels_per_image", 16_777_216),
                         ("max_source_pixels_per_image", 64_000_000),
                         ("max_context_tokens", 262144), ("max_body_bytes", 128 * 1024 * 1024),
                         ("request_timeout_seconds", 600)):
        require(type(limits.get(key)) is int and 0 < limits[key] <= maximum,
                "candidate_limits_invalid")
    require(limits["max_context_tokens"] > generation["max_new_tokens"], "candidate_context_budget_invalid")
    require(config.get("device") == "cuda:0", "candidate_cuda_device_required")
    shapes = config.get("prewarm_shapes")
    require(isinstance(shapes, list) and 1 <= len(shapes) <= 4, "candidate_prewarm_shapes_invalid")
    for shape in shapes:
        require(isinstance(shape, dict) and set(shape) == {"frames", "width", "height"}
                and all(type(v) is int and v > 0 for v in shape.values())
                and shape["frames"] <= limits["max_images"]
                and shape["width"] * shape["height"] <= limits["max_source_pixels_per_image"],
                "candidate_prewarm_shapes_invalid")


def load_expected_runtime_identity(config_path: str | Path) -> dict:
    """Read expected public pins; never claim remote files have been verified.

    API/worker containers need only this bound config, not base weights or the
    private training data. Compare every returned field against actual readyz.
    """
    config = load_json(config_path)
    validate_config(config)
    return {"model_id": config["model_id"], "model_revision": config["model_revision"],
            "processor_revision": config["processor_revision"], "backend": config["backend"],
            "model_manifest_sha256": config["base_manifest"]["sha256"],
            "tokenizer_sha256": config["tokenizer_sha256"],
            "chat_template_sha256": config["chat_template_sha256"],
            "adapter_sha256": config["adapter_sha256"],
            "adapter_config_sha256": config["adapter_config_sha256"],
            "training_run_sha256": config["training_run"]["sha256"],
            "quantization_sha256": canonical_sha256(config["quantization"]),
            "generation_sha256": canonical_sha256(config["generation"]),
            "config_sha256": canonical_sha256(config),
            "image_sha256": config["runtime_image_sha256"],
            "source_sha256": config["source_archive"]["sha256"]}


def verify_candidate(config: dict) -> dict:
    """Hash all base artifacts, exact adapter tree, run and source archive."""
    validate_config(config)
    manifest_path = _bound_file(config.get("base_manifest"))
    source_path = _bound_file(config.get("source_archive"))
    run_path = _bound_file(config.get("training_run"))
    base = verify_candidate_base(config["model_path"], manifest_path)
    manifest = load_json(manifest_path)
    entries = {row["path"]: row for row in manifest["files"]}
    require(config["tokenizer_sha256"] == entries["tokenizer.json"]["sha256"]
            and config["chat_template_sha256"] == entries["chat_template.jinja"]["sha256"],
            "candidate_processor_component_pin_mismatch")
    run = load_json(run_path)
    training_source = config.get("training_source_archive", config["source_archive"])
    training_source_path = _bound_file(training_source)
    from serving.qwen38_source_lineage import SourceLineageError, verify_source_lineage
    try:
        lineage = verify_source_lineage(runtime_archive=source_path,
            runtime_sha256=config["source_archive"]["sha256"], training_archive=training_source_path,
            training_sha256=training_source["sha256"], recorded_training_sha256=run.get("code_archive_sha256"))
    except SourceLineageError as error:
        raise CandidateError(str(error)) from None
    if "training_source_lineage" in config:
        require(config["training_source_lineage"] == lineage, "candidate_training_source_lineage_receipt_mismatch")
    require(run.get("runtime_image_sha256") == config["runtime_image_sha256"],
            "candidate_training_runtime_image_mismatch")
    require(run.get("schema_version") == "forte.qwen38.silver-run.v1"
            and run.get("status") == "candidate_unselected"
            and run.get("dataset_origin") == "local_teacher_silver" and run.get("human_verified") is False,
            "candidate_training_provenance_invalid")
    require(run.get("base_revision") == REVISION and run.get("processor_revision") == REVISION
            and run.get("base_manifest_sha256") == config["base_manifest"]["sha256"],
            "candidate_trained_base_mismatch")
    require(type(run.get("optimizer_steps")) is int and run["optimizer_steps"] >= 2,
            "candidate_two_real_optimizer_steps_required")
    require(run.get("gpu_executed") is True and run.get("seed") == config["generation"]["seed"],
            "candidate_training_execution_or_seed_mismatch")
    for key in ("finite_loss_verified", "gradients_verified", "frozen_tensors_unchanged",
                "peft_load_verified", "reload_forward_verified"):
        require(run.get(key) is True, "candidate_training_evidence_incomplete")
    changed = run.get("changed_adapter_tensors")
    require((isinstance(changed, list) and bool(changed)) or (type(changed) is int and changed > 0),
            "candidate_changed_adapter_tensors_missing")
    require(run.get("quantization") == config["quantization"], "candidate_training_quantization_mismatch")
    require(isinstance(run.get("processor_limits"), dict)
            and run["processor_limits"].get("max_pixels_per_image") == config["limits"]["max_pixels_per_image"],
            "candidate_training_processor_budget_mismatch")
    checksums = run.get("adapter_checksums")
    require(isinstance(checksums, dict) and {"adapter_config.json", "adapter_model.safetensors"}.issubset(checksums),
            "candidate_adapter_inventory_missing")
    require(all(isinstance(k, str) and k and "\\" not in k and ":" not in k
                and not PurePosixPath(k).is_absolute() and ".." not in PurePosixPath(k).parts
                and _hash(v) for k, v in checksums.items()), "candidate_adapter_inventory_invalid")
    adapter = _local(config["adapter_path"]).resolve()
    require(adapter.is_dir(), "candidate_adapter_missing")
    actual = {p.relative_to(adapter).as_posix() for p in adapter.rglob("*") if p.is_file()}
    require(actual == set(checksums), "candidate_adapter_extra_or_missing_files")
    for name, expected in checksums.items():
        item = adapter / name
        require(not item.is_symlink() and item.resolve().is_relative_to(adapter)
                and file_sha256(item) == expected, "candidate_adapter_checksum_mismatch")
    tree = canonical_sha256(checksums)
    require(tree == run.get("adapter_tree_sha256") and tree == config.get("adapter_sha256"),
            "candidate_adapter_tree_mismatch")
    require(checksums["adapter_config.json"] == config["adapter_config_sha256"],
            "candidate_adapter_config_pin_mismatch")
    adapter_config = load_json(adapter / "adapter_config.json")
    require(adapter_config.get("peft_type") == "LORA"
            and adapter_config.get("base_model_name_or_path") in {MODEL_ID, config["model_path"]}
            and adapter_config.get("revision") in {None, REVISION}
            and not adapter_config.get("modules_to_save")
            and adapter_config.get("target_modules"), "candidate_adapter_config_invalid")
    require(not any(word in str(target).lower() for target in adapter_config["target_modules"]
                    for word in ("visual", "vision", "lm_head", "embed_tokens")),
            "candidate_adapter_language_only_required")
    require(run.get("changed_adapter_groups") == {"self_attn": True, "linear_attn": True, "mlp": True},
            "candidate_hybrid_adapter_inventory_mismatch")
    from serving.qwen38_adapter_contract import AdapterContractError, verify_adapter_contract
    try:
        adapter_proof = verify_adapter_contract(run.get("adapted_module_inventory"), adapter_config,
            adapter / "adapter_model.safetensors")
    except AdapterContractError as error:
        raise CandidateError(str(error)) from None
    return {"verified": True, "model_id": MODEL_ID, "model_revision": REVISION,
            "processor_revision": REVISION, "backend": "transformers_peft",
            "model_manifest_sha256": config["base_manifest"]["sha256"],
            "tokenizer_sha256": entries["tokenizer.json"]["sha256"],
            "chat_template_sha256": entries["chat_template.jinja"]["sha256"],
            "adapter_sha256": tree, "adapter_config_sha256": checksums["adapter_config.json"],
            "training_run_sha256": config["training_run"]["sha256"],
            "quantization_sha256": canonical_sha256(config["quantization"]),
            "generation_sha256": canonical_sha256(config["generation"]),
            "image_sha256": config["runtime_image_sha256"],
            "source_sha256": file_sha256(source_path), "config_sha256": canonical_sha256(config),
            "files_verified": base.get("files_verified", len(entries)),
            "model_files_verified": True, "model_loaded": False,
            "training_source_lineage": lineage, "adapter_topology": adapter_proof,
            "quality_or_latency_measured": False, "production_selected": False}

