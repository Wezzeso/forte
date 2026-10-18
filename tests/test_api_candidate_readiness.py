"""Candidate public readiness compares complete actual private runtime pins."""

import json
from dataclasses import replace

import httpx
import pytest

from forte.api.config import Settings
from forte.api.main import _model_alive
from serving.qwen38_identity import REVISION, load_expected_runtime_identity
from serving.qwen38_grammar import POLICY


@pytest.fixture
def candidate_settings(tmp_path):
    bound = {"path": "/private/synthetic/bound.json", "sha256": "a" * 64}
    config = {
        "schema_version": "forte.qwen38.runtime.v1", "backend": "transformers_peft",
        "structured_decoding": dict(POLICY),
        "model_id": "Qwen/Qwen3.8-27B", "model_revision": REVISION, "processor_revision": REVISION,
        "production_selected": False, "served_model_name": "Qwen3.8-27B-forte-silver",
        "model_path": "/models/synthetic", "adapter_path": "/adapters/synthetic",
        "runtime_image_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64, "adapter_sha256": "d" * 64,
        "adapter_config_sha256": "e" * 64, "base_manifest": bound,
        "training_run": bound, "source_archive": bound,
        "quantization": {"format": "none", "compute_dtype": "bfloat16", "double_quant": False},
        "generation": {"do_sample": False, "temperature": 0, "seed": 20261017,
            "max_new_tokens": 4096, "enable_thinking": False, "preserve_thinking": False, "num_beams": 1},
        "package_versions": {key: "synthetic-pinned" for key in (
            "torch", "transformers", "peft", "fastapi", "uvicorn", "pillow", "jsonschema")},
        "host": "0.0.0.0", "port": 8000, "device": "cuda:0",
        "prewarm_shapes": [{"frames": 3, "width": 448, "height": 448}],
        "limits": {"max_images": 16, "max_pixels_per_image": 1_048_576, "max_context_tokens": 16384,
            "max_source_pixels_per_image": 64_000_000,
            "max_body_bytes": 64 * 1024 * 1024, "request_timeout_seconds": 120},
    }
    config["package_versions"]["xgrammar"] = POLICY["version"]
    path = tmp_path / "candidate-runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return Settings(data_dir=tmp_path / "data", model_candidate_launch_config_path=path,
        model_api_key="synthetic-model-token")


def install_model_transport(monkeypatch, candidate_settings, updates=None):
    identity = load_expected_runtime_identity(candidate_settings.model_candidate_launch_config_path)
    attestation = {**identity, "ready": True, "model_files_verified": True,
        "model_loaded": True, "prewarm_completed": True, "adapter_loaded_tensor_count": 8,
        **(updates or {})}
    paths = []
    original = httpx.AsyncClient
    def route(request):
        paths.append(request.url.path)
        assert request.headers["authorization"] == "Bearer synthetic-model-token"
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/readyz"
        return httpx.Response(200, json=attestation)
    monkeypatch.setattr("forte.api.main.httpx.AsyncClient", lambda **kwargs:
        original(transport=httpx.MockTransport(route), **kwargs))
    return paths


async def test_transformers_candidate_ready_without_baseline_manifest_or_vllm_version(
        candidate_settings, monkeypatch):
    settings = replace(candidate_settings, model_manifest_path=candidate_settings.data_dir / "absent.json")
    paths = install_model_transport(monkeypatch, settings)
    assert await _model_alive(settings) is True
    assert sorted(paths) == ["/health", "/readyz"]
    assert "synthetic-model-token" not in repr(settings)


@pytest.mark.parametrize("field", [
    "model_id", "model_revision", "processor_revision", "backend", "model_manifest_sha256",
    "tokenizer_sha256", "chat_template_sha256", "adapter_sha256", "adapter_config_sha256",
    "training_run_sha256", "quantization_sha256", "generation_sha256", "image_sha256", "source_sha256", "config_sha256",
])
async def test_every_candidate_identity_field_is_required(candidate_settings, monkeypatch, field):
    install_model_transport(monkeypatch, candidate_settings, {field: "wrong"})
    assert await _model_alive(candidate_settings) is False


@pytest.mark.parametrize("updates", [
    {"ready": False}, {"model_files_verified": False}, {"model_loaded": False},
    {"prewarm_completed": False}, {"model_loaded": 1}, {"prewarm_completed": 1},
    {"adapter_loaded_tensor_count": 0}, {"adapter_loaded_tensor_count": True},
])
async def test_artifact_pins_without_actual_load_and_prewarm_are_not_ready(
        candidate_settings, monkeypatch, updates):
    install_model_transport(monkeypatch, candidate_settings, updates)
    assert await _model_alive(candidate_settings) is False


async def test_missing_candidate_config_cannot_fall_back_to_baseline(candidate_settings, monkeypatch):
    candidate_settings.model_candidate_launch_config_path.unlink()
    monkeypatch.setattr("forte.api.main.httpx.AsyncClient", lambda **_:
        pytest.fail("Missing candidate pins must block all model requests"))
    assert await _model_alive(candidate_settings) is False


def test_api_candidate_selection_is_explicit_environment_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("FORTE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG", raising=False)
    assert Settings.from_env().model_candidate_launch_config_path is None
    monkeypatch.setenv("FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG", str(tmp_path / "candidate.json"))
    assert Settings.from_env().model_candidate_launch_config_path == tmp_path / "candidate.json"

