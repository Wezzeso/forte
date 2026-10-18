"""Candidate worker configuration must bind every separately selected runtime pin."""

from dataclasses import replace
import json
import time

import httpx
import pytest
from PIL import Image

from forte.config import WorkerSettings
from forte.pipeline import AnalysisIncomplete, _audit_provenance, _provider_configuration
from serving.qwen38_identity import CandidateError, REVISION, load_expected_runtime_identity
from serving.qwen38_grammar import POLICY
from forte.model import LocalModelProvider
from forte.shared.schemas import ClipRequest, FrameRef


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("FORTE_DATA_DIR", str(tmp_path / "runtime"))
    for key in ("FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG", "FORTE_MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)
    settings = WorkerSettings.from_env()
    pin = "a" * 64
    config = {
        "schema_version": "forte.qwen38.runtime.v1", "backend": "transformers_peft",
        "structured_decoding": dict(POLICY),
        "model_id": "Qwen/Qwen3.8-27B", "model_revision": REVISION,
        "processor_revision": REVISION, "production_selected": False,
        "served_model_name": "Qwen3.8-27B-forte-silver",
        "model_path": str(tmp_path / "base"), "adapter_path": str(tmp_path / "adapter"),
        "runtime_image_sha256": pin, "tokenizer_sha256": pin,
        "chat_template_sha256": pin, "adapter_sha256": pin, "adapter_config_sha256": pin,
        **{key: {"path": str(tmp_path / (key + ".json")), "sha256": pin}
           for key in ("base_manifest", "training_run", "source_archive")},
        "quantization": {"format": "nf4", "compute_dtype": "bfloat16", "double_quant": True},
        "generation": {"do_sample": False, "temperature": 0, "seed": 20261017,
                       "max_new_tokens": 1024, "num_beams": 1,
                       "enable_thinking": False, "preserve_thinking": False},
        "package_versions": {key: "synthetic" for key in
                             ("torch", "transformers", "peft", "fastapi", "uvicorn", "pillow",
                              "jsonschema", "bitsandbytes")},
        "host": "127.0.0.1", "port": 8001, "device": "cuda:0",
        "prewarm_shapes": [{"frames": 1, "width": 64, "height": 64}],
        "limits": {"max_images": 16, "max_pixels_per_image": 65536,
                   "max_source_pixels_per_image": 64000000,
                   "max_context_tokens": 4096, "max_body_bytes": 64 * 1024 * 1024,
                   "request_timeout_seconds": 120},
    }
    config["package_versions"]["xgrammar"] = POLICY["version"]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return replace(settings, model_candidate_launch_config_path=path,
                   model_name=config["served_model_name"]), config, path


def test_candidate_provider_binds_exact_identity_and_generation(candidate):
    settings, config, path = candidate
    provider = _provider_configuration(settings)
    assert dict(provider.expected_runtime_identity) == load_expected_runtime_identity(path)
    assert provider.expected_revision == REVISION
    assert provider.max_tokens == 1024 and provider.seed == 20261017
    assert dict(provider.chat_template_kwargs) == {"enable_thinking": False, "preserve_thinking": False}
    assert provider.model == config["served_model_name"]
    assert provider.request_timeout_sec == config["limits"]["request_timeout_seconds"] == 120
    provenance = _audit_provenance(settings)
    assert provenance["actual_snapshot_verified"] is None
    assert provenance["revision_source"] == "configured_candidate_manifest"
    assert provenance["candidate_runtime_identity"] == dict(provider.expected_runtime_identity)


@pytest.mark.parametrize("changes", [{"model_name": "unbound-alias"}, {"max_frames_per_window": 17}])
def test_candidate_alias_or_frame_budget_cannot_silently_fallback(candidate, changes):
    with pytest.raises(AnalysisIncomplete, match="candidate_worker_configuration_mismatch"):
        _provider_configuration(replace(candidate[0], **changes))


def test_missing_candidate_config_fails_closed(candidate):
    candidate[2].unlink()
    with pytest.raises(CandidateError):
        _provider_configuration(candidate[0])


def test_baseline_defaults_remain_strict_without_opt_in(candidate):
    settings = replace(candidate[0], model_candidate_launch_config_path=None,
                       model_name="Qwen3-VL-8B-Instruct")
    provider = _provider_configuration(settings)
    assert provider.expected_revision == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert provider.expected_runtime_identity == () and provider.chat_template_kwargs == ()
    assert provider.max_tokens == 4096
    assert provider.request_timeout_sec == 60


@pytest.mark.parametrize("timeout", [120, 240, 600])
def test_candidate_http_timeout_follows_its_validated_runtime_budget(candidate, timeout):
    settings, config, path = candidate
    config["limits"]["request_timeout_seconds"] = timeout
    path.write_text(json.dumps(config), encoding="utf-8")
    assert _provider_configuration(settings).request_timeout_sec == timeout


@pytest.mark.parametrize("timeout", [0, -1, 601, True, 240.5])
def test_invalid_candidate_timeout_is_not_forwarded(candidate, timeout):
    settings, config, path = candidate
    config["limits"]["request_timeout_seconds"] = timeout
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CandidateError, match="limits_invalid"):
        _provider_configuration(settings)


async def test_candidate_http_transport_receives_full_configured_timeout(candidate):
    settings, config, path = candidate
    config["limits"]["request_timeout_seconds"] = 240
    path.write_text(json.dumps(config), encoding="utf-8")
    provider_config = replace(_provider_configuration(settings), require_readiness=False)
    owned = provider_config.media_root / "generated_timeout_job"
    owned.mkdir(parents=True)
    image = owned / "generated.jpg"
    Image.new("RGB", (16, 12), (220, 20, 20)).save(image, "JPEG")
    clip = ClipRequest(job_id=owned.name, clip_id="generated_timeout_clip", frame_refs=[FrameRef(
        frame_id="generated-source-frame", pts_sec=0, path=image, width=16, height=12)])
    def handler(request):
        assert request.extensions["timeout"]["read"] == 240
        assert request.extensions["timeout"]["connect"] == 3
        return httpx.Response(200, json={"model": provider_config.model, "choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": '{"state":"unclear"}'}}]})
    async with LocalModelProvider(provider_config, transport=httpx.MockTransport(handler)) as provider:
        result = await provider.infer(clip, prompt="Generated test only.",
            response_schema={"type": "object"}, deadline=time.monotonic()+300)
    assert result.data == {"state": "unclear"}


def test_candidate_worker_environment_opt_in(candidate, monkeypatch):
    monkeypatch.setenv("FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG", str(candidate[2]))
    assert WorkerSettings.from_env().model_candidate_launch_config_path == candidate[2].resolve()
