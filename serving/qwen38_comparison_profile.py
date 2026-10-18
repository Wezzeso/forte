"""Shared immutable runtime conditions for untuned/tuned Qwen diagnostics."""

from serving.qwen38_identity import canonical_sha256, require


PROFILE_FIELDS = ("model_id", "model_revision", "processor_revision", "tokenizer_sha256",
                  "chat_template_sha256", "quantization", "generation", "structured_decoding", "limits",
                  "prewarm_shapes", "device", "package_versions", "runtime_image_sha256")


def runtime_profile(config: dict) -> dict:
    require(isinstance(config, dict) and all(key in config for key in PROFILE_FIELDS),
            "comparison_runtime_profile_incomplete")
    require(isinstance(config.get("base_manifest"), dict)
            and isinstance(config.get("source_archive"), dict), "comparison_runtime_profile_incomplete")
    return {"schema_version": "forte.qwen38.comparison-runtime-profile.v1",
            **{key: config[key] for key in PROFILE_FIELDS},
            "base_manifest_sha256": config["base_manifest"].get("sha256"),
            "source_archive_sha256": config["source_archive"].get("sha256"),
            "generation_implementation": "serving.qwen38_service.CandidateRuntime.generate"}


def runtime_profile_sha256(config: dict) -> str:
    return canonical_sha256(runtime_profile(config))


def require_matching_profiles(untuned: dict, trained: dict) -> str:
    require(untuned.get("schema_version") == "forte.qwen38.untuned-runtime.v1"
            and untuned.get("backend") == "transformers_base_diagnostic"
            and trained.get("schema_version") == "forte.qwen38.runtime.v1"
            and trained.get("backend") == "transformers_peft", "comparison_runtime_roles_invalid")
    left, right = runtime_profile_sha256(untuned), runtime_profile_sha256(trained)
    require(left == right, "comparison_runtime_profile_mismatch")
    return left
