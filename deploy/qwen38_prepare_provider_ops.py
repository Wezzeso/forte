"""CPU-only actual installed worker/provider construction; zero HTTP/model calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys


class PreparationError(RuntimeError):
    pass


def need(value, code):
    if not value:
        raise PreparationError(code)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def prepare(args):
    sys.path[:0] = [str(args.source_root), str(args.source_root / "src")]
    manifest = json.loads(args.source_manifest.read_text())
    need(re.fullmatch(r"[a-f0-9]{64}", args.source_manifest_sha256) is not None
         and sha(args.source_manifest) == args.source_manifest_sha256,
         "provider_prepare_manifest_hash_mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    need(manifest["sha256"] == config["source_archive"]["sha256"],
         "provider_prepare_runtime_source_mismatch")
    expected_modules = {
        "forte.config": "src/forte/config.py",
        "forte.pipeline": "src/forte/pipeline.py",
        "forte.model.adapter": "src/forte/model/adapter.py",
        "serving.qwen38_identity": "serving/qwen38_identity.py",
        "serving.qwen38_comparison_profile": "serving/qwen38_comparison_profile.py",
        "serving.qwen38_untuned_provider": "serving/qwen38_untuned_provider.py",
    }
    module_proofs = {}
    for name, member in expected_modules.items():
        module = importlib.import_module(name)
        actual = sha(module.__file__)
        need(actual == manifest["files"][member], "provider_prepare_loaded_source_mismatch")
        module_proofs[name] = actual
    from forte.config import WorkerSettings
    from forte.pipeline import _provider_configuration
    from forte.model.adapter import LocalModelProvider
    from serving.qwen38_comparison_profile import runtime_profile_sha256
    from serving.qwen38_identity import load_expected_runtime_identity
    from serving.qwen38_untuned_provider import UntunedModelProvider

    worker = config["quality_family_binding"]["worker_settings"]
    settings = WorkerSettings(database_url="postgresql+psycopg://unused/no-connection",
        data_dir=Path("/tmp/provider-prepare"), model_base_url="http://model-3:8000/v1",
        model_name=config["served_model_name"], model_api_key=None,
        worker_id="cpu-provider-prepare-only", heartbeat_file=Path("/tmp/provider-prepare/heartbeat"),
        rules_path=args.source_root / "rules/catalog.json",
        taxonomy_path=args.source_root / "annotations/taxonomy.v1.json",
        model_manifest_path=args.source_root / "models/manifest.json",
        model_candidate_launch_config_path=args.config if config["backend"] == "transformers_peft" else None,
        **worker)
    if config["backend"] == "transformers_peft":
        # Exercise the actual product worker construction, rather than a
        # hand-written equivalent that could hide an inherited default.
        provider = LocalModelProvider(_provider_configuration(settings))
        expected = load_expected_runtime_identity(args.config)
        need(dict(provider.config.expected_runtime_identity) == expected,
             "provider_prepare_candidate_identity_mismatch")
        path = "WorkerSettings -> _provider_configuration -> LocalModelProvider"
    else:
        provider = UntunedModelProvider(config, media_root=settings.work_dir,
                                       base_url=settings.model_base_url)
        path = "WorkerSettings -> UntunedModelProvider"
    try:
        actual = provider.config
        need(actual.request_timeout_sec == config["limits"]["request_timeout_seconds"] == 240,
             "provider_prepare_actual_timeout_mismatch")
        need(actual.max_tokens == config["generation"]["max_new_tokens"] == 4096
             and actual.seed == 20261017 and actual.max_frames == 16
             and dict(actual.chat_template_kwargs) == {"enable_thinking": False, "preserve_thinking": False},
             "provider_prepare_actual_generation_mismatch")
        profile = runtime_profile_sha256(config)
        need(profile == args.expected_runtime_profile_sha256, "provider_prepare_profile_mismatch")
        return {"schema_version": "forte.qwen38.actual-worker-provider-cpu-preparation.v1",
            "status": "passed", "backend": config["backend"], "construction_path": path,
            "source_archive_sha256": config["source_archive"]["sha256"],
            "source_manifest_sha256": args.source_manifest_sha256,
            "config_file_sha256": sha(args.config), "runtime_profile_sha256": profile,
            "actual_client_timeout_seconds": actual.request_timeout_sec,
            "actual_max_tokens": actual.max_tokens, "actual_max_frames": actual.max_frames,
            "actual_seed": actual.seed, "loaded_module_sha256": module_proofs,
            "interpreter": sys.executable, "http_calls": 0, "model_posts": 0,
            "gpu_access": False, "database_connection_opened": False, "private_media_read": False}
    finally:
        await provider.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/app"))
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--expected-runtime-profile-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(prepare(args))))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "reason_code": str(error)
                          if isinstance(error, PreparationError) else "provider_prepare_error",
                          "error_type": type(error).__name__,
                          "missing_module": error.name if isinstance(error, ModuleNotFoundError) else None,
                          "http_calls": 0, "model_posts": 0, "gpu_access": False}))
        raise SystemExit(2)
