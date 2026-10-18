"""Safe, content-bound provenance assembled once at pipeline initialization.

This is a snapshot of configured inputs and installed source files, not proof
of deterministic output, verified GPU execution, or a successful repeat run.
No media, labels, model weights, secrets, remote endpoints or arbitrary config
contents are read or returned. Callers should cache the result for that loaded
pipeline and bind actual source-frame/result identities separately per job.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any

SCHEMA_VERSION = "forte.reproducibility.v1"
_MAX_CONFIG_BYTES = 32 * 1024 * 1024
_CONFIG_SUFFIXES = frozenset({".json", ".yaml", ".yml", ".toml", ".lock"})
_SETTING_FIELDS = (
    "poll_seconds", "lease_seconds", "heartbeat_seconds", "job_timeout_seconds",
    "window_seconds", "stride_seconds", "sample_fps", "max_frames_per_window",
    "max_concurrent_jobs", "max_concurrent_clips", "max_inflight_per_job",
    "compact_frame_ids", "concise_observations", "staged_screening",
    "max_refinement_passes",
)
_DECISION_FIELDS = (
    "max_facts", "max_frames", "max_candidates", "max_output_bytes",
    "compact_frame_ids", "concise_observations",
)
_THRESHOLD_FIELDS = (
    "thresholds", "fact_thresholds", "event_thresholds", "support_thresholds",
    "decoder_thresholds", "present_threshold", "absent_threshold",
    "minimum_score", "minimum_margin", "minimum_support_score",
)
_CONFIG_FILES = {
    "model_manifest": "model_manifest_path",
    "candidate_launch_config": "model_candidate_launch_config_path",
    "taxonomy": "taxonomy_path",
    "rules_file": "rules_path",
    "decision_config": "decision_config_path",
    "thresholds": "thresholds_path",
    "decoder_config": "decoder_config_path",
    "sampling_config": "sampling_config_path",
}
_REQUIRED_CONFIGS = frozenset({"model_manifest", "taxonomy", "rules_file"})
_RUNTIME_DEPENDENCIES = (
    "forte-video-review", "av", "pillow", "pydantic", "fastapi", "httpx",
    "jsonschema", "sqlalchemy", "psycopg", "torch", "torchvision",
    "transformers", "vllm",
)
_REQUIRED_DEPENDENCIES = frozenset({"av", "pillow", "pydantic"})
_SKIP_SOURCE_DIRS = frozenset({
    "__pycache__", "private", "data", "videos", "labels", "weights", "benchmarks",
})
_MISSING = object()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _get(value: Any, name: str, default: Any = _MISSING) -> Any:
    try:
        return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
    except Exception:
        return default


def _number_or_bool(value: Any) -> bool:
    return type(value) in (bool, int) or type(value) is float and math.isfinite(value)


def _numeric_fields(value: Any, fields: tuple[str, ...], prefix: str,
                    missing: set[str]) -> dict[str, Any]:
    result = {}
    for name in fields:
        actual = _get(value, name)
        if not _number_or_bool(actual):
            result[name] = None
            missing.add(f"{prefix}.{name}")
        else:
            result[name] = actual
    return result


def _stable_file_sha256(path: Any) -> str | None:
    """Hash bounded generic configuration/source files without echoing paths."""
    try:
        value = Path(path)
        if value.is_symlink() or not value.is_file():
            return None
        before = value.stat()
        if before.st_size > _MAX_CONFIG_BYTES:
            return None
        with value.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        after = value.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return None
        return checksum
    except (OSError, ValueError, TypeError):
        return None


def _installed_code_files() -> tuple[list[tuple[str, Path]], list[str]]:
    """Enumerate only the installed inference package and two explicit modules."""
    import forte

    result: list[tuple[str, Path]] = []
    problems: list[str] = []
    try:
        source = Path(forte.__file__ or "")
        if source.suffix != ".py" or not source.is_file():
            raise ValueError
        root = source.resolve().parent
        def unavailable(_: OSError) -> None:
            problems.append("code.package_source_unavailable")

        for directory, children, files in os.walk(root, followlinks=False, onerror=unavailable):
            if any((Path(directory) / name).is_symlink() for name in children):
                problems.append("code.package_source_unavailable")
            children[:] = sorted(name for name in children
                                 if not name.startswith(".") and name not in _SKIP_SOURCE_DIRS
                                 and not (Path(directory) / name).is_symlink())
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix != ".py":
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    problems.append("code.package_source_unavailable")
                    continue
                logical = "forte/" + path.relative_to(root).as_posix()
                result.append((logical, path))
    except (OSError, TypeError, ValueError):
        problems.append("code.package_source_unavailable")
    for name in ("rules.catalog", "serving.scheduler"):
        try:
            spec = importlib.util.find_spec(name)
            path = Path(spec.origin) if spec is not None and spec.origin else None
            if path is None or path.suffix != ".py" or path.is_symlink() or not path.is_file():
                raise ValueError
            result.append((name.replace(".", "/") + ".py", path))
        except (ImportError, AttributeError, OSError, TypeError, ValueError):
            problems.append("code." + name + "_source_unavailable")
    return sorted(result), problems


def _code_binding(missing: set[str]) -> dict[str, Any]:
    try:
        files, problems = _installed_code_files()
    except Exception:
        files, problems = [], ["code.package_source_unavailable"]
    missing.update(problems)
    records = []
    for logical, path in files:
        checksum = _stable_file_sha256(path)
        if checksum is None:
            missing.add("code.file_unavailable_or_changing")
        else:
            records.append({"module": logical, "sha256": checksum})
    if not records:
        missing.add("code.source_set_empty")
    # File/module names participate in the digest but are not exposed in output.
    return {"scope": "installed_forte_python_plus_rules_catalog_and_serving_scheduler",
            "source_sha256": _digest(records) if records else None,
            "source_file_count": len(records), "loaded_bytecode_attested": False}


def _runtime(missing: set[str]) -> dict[str, Any]:
    versions = {}
    for distribution in _RUNTIME_DEPENDENCIES:
        try:
            value = importlib.metadata.version(distribution)
            # Version metadata is bounded and must not contain paths or URLs.
            version = value if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_!-]{0,127}", value) else None
        except Exception:
            version = None
        versions[distribution] = version
        if version is None and distribution in _REQUIRED_DEPENDENCIES:
            missing.add("runtime." + distribution + "_version_unavailable")
    return {"scope": "current_python_process", "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine()},
            "dependency_versions": versions, "remote_model_runtime_attested": False}


def _provider_binding(provider_config: Any, missing: set[str]) -> dict[str, Any]:
    # A foreign/injected provider must never inherit the production defaults.
    from forte.model.adapter import LocalModelConfig

    if type(provider_config) is not LocalModelConfig:
        missing.add("provider.actual_configuration_unknown")
        return {"kind": "unknown_or_injected", "generation": None, "limits": None}
    generation = {"temperature": 0.0}
    for name in ("seed", "max_tokens"):
        value = _get(provider_config, name)
        if type(value) is not int or name == "max_tokens" and value < 1:
            missing.add("provider." + name)
            value = None
        generation[name] = value
    limits = _numeric_fields(provider_config, (
        "max_frames", "max_image_bytes", "max_clip_bytes", "max_source_pixels",
        "max_output_bytes", "request_timeout_sec", "connect_timeout_sec",
        "max_attempts", "retry_delay_sec", "cleanup_grace_sec", "require_readiness",
        "readiness_cache_sec",
    ), "provider", missing)
    result = {"kind": "local_model_config", "scope": "configured_values_not_execution_attestation",
              "generation": generation, "limits": limits}
    if provider_config.expected_runtime_identity:
        # LocalModelConfig validates this exact allowlist; secrets, URLs and
        # paths cannot enter an identity. This remains configured provenance,
        # with actual execution checked separately on every model connection.
        result["candidate_runtime_identity"] = dict(provider_config.expected_runtime_identity)
        result["generation"]["chat_template_kwargs"] = dict(provider_config.chat_template_kwargs)
    return result


def build_reproducibility_bundle(settings: Any, catalog: Any, decision_config: Any,
                                 provider_config: Any = None) -> dict[str, Any]:
    """Build a safe canonical snapshot without model/network/media operations.

    ``catalog.payload`` is the exact already-loaded object; its configured file
    is separately hashed and is never substituted for the loaded-content claim.
    Optional threshold maps are content-hashed, not rendered. Unknown provider
    configuration and unavailable/changing inputs remain explicit missing
    bindings. Cache once per loaded pipeline, never rebuild on every job.
    """
    missing: set[str] = set()
    configured = _numeric_fields(settings, _SETTING_FIELDS, "settings", missing)
    context = _get(settings, "trusted_scene_context")
    if context not in (None, "bank_cash_workstation"):
        context = None
        missing.add("settings.trusted_scene_context")
    configured["trusted_scene_context"] = context
    decision = _numeric_fields(decision_config, _DECISION_FIELDS, "decision", missing)
    config_version = _get(decision_config, "config_version")
    if isinstance(config_version, str) and re.fullmatch(r"forte\.decision\.v[0-9.]+", config_version):
        decision["config_version"] = config_version
    else:
        decision["config_version"] = None
        missing.add("decision.config_version")
    thresholds = {}
    for name in _THRESHOLD_FIELDS:
        value = _get(decision_config, name)
        if value is _MISSING:
            continue
        try:
            thresholds[name] = _digest(value)
        except (TypeError, ValueError, OverflowError, RecursionError):
            thresholds[name] = None
            missing.add("decision." + name + "_binding_unavailable")
    decision["threshold_content_sha256"] = thresholds
    payload = _get(catalog, "payload")
    try:
        if not isinstance(payload, dict) or not payload:
            raise ValueError
        catalog_sha256 = _digest(payload)
    except (TypeError, ValueError, OverflowError, RecursionError):
        catalog_sha256 = None
        missing.add("catalog.loaded_payload_unavailable")
    config_files = {}
    for label, attribute in _CONFIG_FILES.items():
        path = _get(settings, attribute)
        if path is _MISSING or path is None:
            config_files[label] = {"state": "not_configured", "sha256": None}
            if label in _REQUIRED_CONFIGS:
                missing.add("config." + label + "_unavailable")
            continue
        try:
            is_config = Path(path).suffix.lower() in _CONFIG_SUFFIXES
        except (TypeError, ValueError):
            is_config = False
        checksum = _stable_file_sha256(path) if is_config else None
        config_files[label] = {"state": "bound" if checksum else "unavailable_or_changing",
                               "sha256": checksum}
        if checksum is None:
            missing.add("config." + label + "_unavailable")
    result = {
        "schema_version": SCHEMA_VERSION,
        "settings": configured,
        "decision": decision,
        "loaded_catalog_sha256": catalog_sha256,
        "config_files": config_files,
        "code": _code_binding(missing),
        "runtime": _runtime(missing),
        "provider": _provider_binding(provider_config, missing),
        "claim_limits": {
            "source_media_or_labels_read": False, "model_calls_made": False,
            "model_weights_verified_by_this_bundle": False,
            "deterministic_output_proven": False, "repeatability_measured": False,
            "human_review_required_to_build": False,
        },
    }
    result["missing_bindings"] = sorted(missing)
    result["bindings_complete"] = not missing
    result["bundle_sha256"] = _digest(result)
    return result


__all__ = ["SCHEMA_VERSION", "build_reproducibility_bundle"]
