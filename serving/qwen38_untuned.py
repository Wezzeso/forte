"""Explicit private base-only Qwen3.8 development diagnostic, never production.

No adapter, fake training receipt or alternate candidate fallback is accepted.
The trained candidate verifier and public API gate are deliberately unchanged.
--verify-only hashes local artifacts without initializing CUDA or making calls.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import threading
import time

from serving.qwen38_comparison_profile import require_matching_profiles, runtime_profile_sha256
from serving.qwen38_grammar import POLICY
from serving.qwen38_identity import (CandidateError, MODEL_ID, REVISION, _bound_file, _hash,
                                    _local, canonical_sha256, load_json, require,
                                    verify_candidate_base)
from serving.qwen38_service import CandidateRuntime, load_processor_and_base


def validate_untuned_config(config: dict) -> None:
    require(config.get("schema_version") == "forte.qwen38.untuned-runtime.v1"
            and config.get("backend") == "transformers_base_diagnostic"
            and config.get("arm") == "qwen38_untuned"
            and config.get("diagnostic_only") is True
            and config.get("production_selected") is False, "untuned_diagnostic_scope_required")
    require(config.get("model_id") == MODEL_ID and config.get("model_revision") == REVISION
            and config.get("processor_revision") == REVISION, "untuned_base_identity_invalid")
    require(all(key in config and config[key] is None for key in
                ("adapter_sha256", "adapter_config_sha256", "training_run_sha256"))
            and not any(key in config for key in ("adapter_path", "training_run")),
            "untuned_adapter_or_training_identity_forbidden")
    _local(config.get("model_path"))
    require(isinstance(config.get("served_model_name"), str)
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", config["served_model_name"]) is not None,
            "untuned_alias_invalid")
    for key in ("runtime_image_sha256", "tokenizer_sha256", "chat_template_sha256", "runtime_profile_sha256"):
        require(_hash(config.get(key)), "untuned_component_pin_missing")
    for key in ("base_manifest", "source_archive"):
        entry = config.get(key)
        require(isinstance(entry, dict) and _hash(entry.get("sha256")), "untuned_file_pin_missing")
        _local(entry.get("path"))
    require(config.get("quantization") in (
        {"format": "nf4", "compute_dtype": "bfloat16", "double_quant": True},
        {"format": "none", "compute_dtype": "bfloat16", "double_quant": False}),
        "untuned_quantization_unsupported")
    generation = config.get("generation", {})
    require(isinstance(generation, dict) and set(generation) == {
        "do_sample", "temperature", "seed", "max_new_tokens", "enable_thinking", "preserve_thinking", "num_beams"}
        and generation.get("do_sample") is False and generation.get("temperature") == 0
        and generation.get("seed") == 20261017 and generation.get("num_beams") == 1
        and generation.get("enable_thinking") is False and generation.get("preserve_thinking") is False
        and type(generation.get("max_new_tokens")) is int and 1 <= generation["max_new_tokens"] <= 8192,
        "untuned_generation_invalid")
    packages = config.get("package_versions", {})
    require(config.get("structured_decoding") == POLICY, "untuned_structured_decoding_invalid")
    required = {"torch", "transformers", "peft", "fastapi", "uvicorn", "pillow", "jsonschema", "xgrammar"}
    if config["quantization"]["format"] == "nf4":
        required.add("bitsandbytes")
    require(isinstance(packages, dict) and required.issubset(packages)
            and all(isinstance(value, str) and value for value in packages.values()), "untuned_packages_unpinned")
    require(packages["xgrammar"] == "0.2.7", "untuned_structured_decoding_package_mismatch")
    require(config.get("device") == "cuda:0" and config.get("host") in {"127.0.0.1", "0.0.0.0"}
            and type(config.get("port")) is int and 1024 <= config["port"] <= 65535,
            "untuned_device_or_listen_invalid")
    limits = config.get("limits", {})
    require(isinstance(limits, dict), "untuned_limits_invalid")
    for key, maximum in (("max_images", 64), ("max_pixels_per_image", 16_777_216),
                         ("max_source_pixels_per_image", 64_000_000), ("max_context_tokens", 262144),
                         ("max_body_bytes", 128 * 1024 * 1024), ("request_timeout_seconds", 600)):
        require(type(limits.get(key)) is int and 0 < limits[key] <= maximum, "untuned_limits_invalid")
    require(limits["max_context_tokens"] > generation["max_new_tokens"], "untuned_context_budget_invalid")
    shapes = config.get("prewarm_shapes")
    require(isinstance(shapes, list) and 1 <= len(shapes) <= 4, "untuned_prewarm_invalid")
    for shape in shapes:
        require(isinstance(shape, dict) and set(shape) == {"frames", "width", "height"}
                and all(type(value) is int and value > 0 for value in shape.values())
                and shape["frames"] <= limits["max_images"]
                and shape["width"] * shape["height"] <= limits["max_source_pixels_per_image"],
                "untuned_prewarm_invalid")
    require(config["runtime_profile_sha256"] == runtime_profile_sha256(config), "untuned_runtime_profile_mismatch")


def expected_untuned_identity(config: dict) -> dict:
    validate_untuned_config(config)
    return {"model_id": MODEL_ID, "model_revision": REVISION, "processor_revision": REVISION,
            "backend": "transformers_base_diagnostic", "arm": "qwen38_untuned", "diagnostic_only": True,
            "production_selected": False, "trained": False, "adapter_sha256": None,
            "adapter_config_sha256": None, "training_run_sha256": None,
            "model_manifest_sha256": config["base_manifest"]["sha256"],
            "tokenizer_sha256": config["tokenizer_sha256"], "chat_template_sha256": config["chat_template_sha256"],
            "quantization_sha256": canonical_sha256(config["quantization"]),
            "generation_sha256": canonical_sha256(config["generation"]),
            "runtime_profile_sha256": config["runtime_profile_sha256"],
            "image_sha256": config["runtime_image_sha256"], "source_sha256": config["source_archive"]["sha256"],
            "config_sha256": canonical_sha256(config)}


def verify_untuned(config: dict) -> dict:
    """Verify actual full local base/source bytes, never adapter/training files."""
    identity = expected_untuned_identity(config)
    manifest_path = _bound_file(config["base_manifest"])
    _bound_file(config["source_archive"])
    proof = verify_candidate_base(config["model_path"], manifest_path)
    manifest = load_json(manifest_path)
    files = {entry["path"]: entry for entry in manifest["files"]}
    require(files["tokenizer.json"]["sha256"] == config["tokenizer_sha256"]
            and files["chat_template.jinja"]["sha256"] == config["chat_template_sha256"],
            "untuned_processor_pin_mismatch")
    require(proof.get("verified") is True, "untuned_base_snapshot_unverified")
    return {**identity, "verified": True, "model_files_verified": True,
            "files_verified": proof.get("files_verified", len(files)), "model_loaded": False,
            "quality_or_latency_measured": False}


def validate_untuned_readiness(ready: dict, config: dict) -> None:
    expected = expected_untuned_identity(config)
    require(isinstance(ready, dict) and all(key in ready and ready[key] == value for key, value in expected.items())
            and ready.get("diagnostic_only") is True and ready.get("trained") is False
            and ready.get("production_selected") is False
            and all(ready.get(key) is True for key in ("verified", "ready", "model_files_verified",
                "model_loaded", "prewarm_completed", "schema_smoke_completed", "namespace_default_route_absent"))
            and type(ready.get("adapter_loaded_tensor_count")) is int
            and ready["adapter_loaded_tensor_count"] == 0, "untuned_readiness_identity_mismatch")
    receipt = ready.get("runtime_load")
    require(isinstance(receipt, dict) and receipt.get("model_gpu_loaded") is True
            and receipt.get("model_type") == "qwen3_5" and receipt.get("attn_implementation") == "sdpa"
            and receipt.get("cuda_version") == "12.9" and receipt.get("gpu_compute_capability") == [8, 0]
            and receipt.get("quantization") == config["quantization"], "untuned_actual_load_receipt_mismatch")
    count = receipt.get("nf4_module_count")
    require(type(count) is int and (count > 0 and receipt.get("actual_nf4_state_verified") is True
            if config["quantization"]["format"] == "nf4"
            else count == 0 and receipt.get("actual_nf4_state_verified") is False),
            "untuned_actual_quantization_receipt_mismatch")
    grammar = ready.get("structured_decoding")
    require(isinstance(grammar, dict) and all(grammar.get(key) == value for key, value in POLICY.items())
            and grammar.get("fresh_matcher_per_generate") is True
            and grammar.get("strict_post_generation_validation") is True
            and type(grammar.get("requests_bound_to_grammar")) is int
            and grammar["requests_bound_to_grammar"] > 0,
            "untuned_actual_structured_decoding_receipt_mismatch")


class UntunedRuntime(CandidateRuntime):
    """Same parser/processor/generation implementation; a real base-only load."""
    def __init__(self, config: dict, identity: dict):
        validate_untuned_config(config)
        require(all(identity.get(key) == value for key, value in expected_untuned_identity(config).items())
                and identity.get("verified") is True and identity.get("model_files_verified") is True,
                "untuned_verified_identity_required")
        super().__init__(config, identity)
        self.schema_smoke_completed = False
        self.namespace_default_route_absent = False

    def load(self) -> None:
        from serving.qwen38_grammar import NativeGrammar
        self.processor, self.model = load_processor_and_base(self.config)
        require(not getattr(self.model, "peft_config", None)
                and not any(".lora_A." in name or ".lora_B." in name for name, _ in self.model.named_parameters()),
                "untuned_loaded_adapter_forbidden")
        self.model.requires_grad_(False)
        self.model.eval()
        require(all(not parameter.requires_grad for parameter in self.model.parameters()), "untuned_parameters_trainable")
        self.runtime_load = getattr(self.model, "_forte_candidate_load_receipt", None)
        require(isinstance(self.runtime_load, dict) and bool(self.runtime_load), "untuned_actual_load_receipt_missing")
        self.grammar = NativeGrammar(self.processor.tokenizer, self.model, self.config["structured_decoding"])
        self.grammar.precompile_public_schema(max_images=self.config["limits"]["max_images"])

    def readiness(self) -> dict:
        return {**self.identity, "ready": self.ready, "model_loaded": self.model is not None,
                "adapter_loaded_tensor_count": 0, "prewarm_completed": self.ready,
                "schema_smoke_completed": self.schema_smoke_completed,
                "namespace_default_route_absent": self.namespace_default_route_absent,
                "runtime_load": getattr(self, "runtime_load", None),
                "structured_decoding": self.grammar.receipt() if self.grammar is not None else None,
                "quality_or_latency_measured": False, "production_selected": False}


def run(config: dict, identity: dict, state_path: Path) -> int:
    import uvicorn
    from serving.qwen38_launch import private_environment, verify_private_namespace, verify_runtime_packages, write_state
    from serving.qwen38_service import create_app
    from serving.readiness import prewarm, request_local
    from forte.model.smoke import synthetic_smoke

    private_environment()
    verify_private_namespace()
    verify_runtime_packages(config)
    runtime = UntunedRuntime(config, identity)
    runtime.namespace_default_route_absent = True
    runtime.load()
    server = uvicorn.Server(uvicorn.Config(create_app(runtime), host=config["host"], port=config["port"],
                                           access_log=False, log_level="critical"))
    thread = threading.Thread(target=server.run, name="forte-untuned-diagnostic-http", daemon=True)
    thread.start()

    def stop(*_):
        runtime.ready = False
        state_path.unlink(missing_ok=True)
        server.should_exit = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            require(thread.is_alive() and time.monotonic() < deadline, "untuned_http_start_failed")
            time.sleep(.1)
        request_local(config["port"], "/health")
        warmup = prewarm({**config, "request_timeout_seconds": config["limits"]["request_timeout_seconds"],
                          "prewarm_concurrency": 1})
        # Bootstrap only generated media before readiness; private comparison
        # clips must use the separately guarded UntunedModelProvider.
        smoke = asyncio.run(synthetic_smoke(
            base_url=f"http://127.0.0.1:{config['port']}/v1", media_root="/tmp/forte-qwen38-untuned-smoke",
            timeout_sec=config["limits"]["request_timeout_seconds"], api_key=os.environ.get("VLLM_API_KEY"),
            require_readiness=False, model=config["served_model_name"], expected_revision=REVISION,
            seed=20261017, max_tokens=config["generation"]["max_new_tokens"],
            chat_template_kwargs={"enable_thinking": False, "preserve_thinking": False}))
        require(smoke.get("synthetic_only") is True and smoke.get("real_local_inference") is True
                and smoke.get("schema_and_semantic_validation_passed") is True,
                "untuned_actual_schema_smoke_failed")
        require(thread.is_alive(), "untuned_http_stopped_during_warmup")
        runtime.schema_smoke_completed = True
        runtime.ready = True
        write_state(state_path, {**runtime.readiness(), "server_pid": os.getpid(),
                    "ready_at": datetime.now(timezone.utc).isoformat(), "warmup": warmup,
                    "schema_smoke": smoke, "package_versions": config["package_versions"],
                    "external_no_egress_probe_completed": False, "image_digest_independently_inspected": False})
        print(json.dumps({"event": "untuned_diagnostic_ready", "adapter_loaded_tensor_count": 0,
                          "training_performed": False, "production_selected": False,
                          "quality_or_latency_measured": False}), flush=True)
        while thread.is_alive():
            thread.join(timeout=1)
        return 0 if server.should_exit else 1
    finally:
        stop()
        thread.join(timeout=30)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("/run/forte/qwen38-untuned-ready.json"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expectations-only", action="store_true")
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--compare-trained-profile", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_json(args.config)
        identity = expected_untuned_identity(config)
        if args.check_ready:
            from serving.readiness import request_local
            validate_untuned_readiness(request_local(config["port"], "/readyz"), config)
            return 0
        if args.compare_trained_profile:
            shared = require_matching_profiles(config, load_json(args.compare_trained_profile))
            print(json.dumps({"runtime_profile_sha256": shared, "same_runtime_profile": True,
                              "trained_artifacts_verified": False, "model_loaded": False}))
            return 0
        if args.expectations_only:
            print(json.dumps({"expected_runtime_identity": identity, "artifacts_verified": False, "model_loaded": False}))
            return 0
        if not args.verify_only:
            args.state.unlink(missing_ok=True)
        from serving.qwen38_launch import private_environment
        private_environment()
        verified = verify_untuned(config)
        if args.verify_only:
            print(json.dumps(verified, allow_nan=False))
            return 0
        return run(config, verified, args.state)
    except Exception as error:
        print(json.dumps({"event": "untuned_start_failed", "reason_code": str(error)
                          if isinstance(error, CandidateError) else "untuned_runtime_error",
                          "error_type": type(error).__name__, "private_values_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
