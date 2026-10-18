"""Synthetic checks for strict observations and opt-in candidate identity."""

import hashlib
import json
import time
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from forte.model import InvalidModelOutput, LocalModelConfig, LocalModelProvider, ModelUnavailable
from forte.model.adapter import strict_json_object
from forte.shared.schemas import ClipRequest, FrameRef


def candidate_identity():
    revision = "b" * 40
    generation = {"do_sample": False, "temperature": 0, "seed": 20261017,
        "max_new_tokens": 4096, "enable_thinking": False, "preserve_thinking": False, "num_beams": 1}
    identity = {key: "a" * 64 for key in (
        "tokenizer_sha256", "chat_template_sha256", "model_manifest_sha256", "adapter_sha256",
        "adapter_config_sha256", "training_run_sha256", "quantization_sha256", "generation_sha256",
        "image_sha256", "source_sha256", "config_sha256")}
    identity.update(model_id="Qwen/Qwen3.8-27B", model_revision=revision,
        processor_revision=revision, backend="transformers_peft")
    identity["generation_sha256"] = hashlib.sha256(json.dumps(generation, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return identity


@pytest.fixture
def candidate_config(tmp_path):
    identity = candidate_identity()
    return LocalModelConfig(media_root=tmp_path, base_url="http://127.0.0.1:8000/v1",
        model="forte-qwen38-candidate", expected_revision=identity["model_revision"], seed=20261017,
        expected_runtime_identity=tuple(identity.items()),
        chat_template_kwargs=(("enable_thinking", False), ("preserve_thinking", False)))


@pytest.fixture
def clip(tmp_path):
    owned = tmp_path / "synthetic_job"
    owned.mkdir()
    path = owned / "frame.jpg"
    Image.new("RGB", (16, 12), (200, 20, 20)).save(path, "JPEG")
    return ClipRequest(job_id=owned.name, clip_id="clip", frame_refs=[FrameRef(
        frame_id="frame", pts_sec=0, path=path, width=16, height=12, job_id=owned.name)])


def envelope(model, **message):
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": '{"state":"unclear"}', **message}}]}


@pytest.mark.parametrize("raw", ['{"v":1e999}', '{"v":[{"x":-1e999}]}'])
def test_json_exponent_overflow_cannot_become_infinite_fact(raw):
    with pytest.raises(InvalidModelOutput, match="json_invalid"):
        strict_json_object(raw)


def test_representable_json_numbers_remain_accepted():
    assert strict_json_object('{"v":1e300,"small":1e-300}') == {"v": 1e300, "small": 1e-300}


@pytest.mark.parametrize("message", [
    {"role": "user"}, {"role": None}, {"function_call": {"name": "private_tool"}},
    {"reasoning_content": "private reasoning"}, {"reasoning": "private reasoning"},
])
async def test_candidate_accepts_only_assistant_observation_without_tools_or_thinking(
        candidate_config, clip, message):
    config = replace(candidate_config, require_readiness=False)
    async with LocalModelProvider(config, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=envelope(config.model, **message)))) as provider:
        with pytest.raises(InvalidModelOutput):
            await provider.infer(clip, prompt="Synthetic observations only.",
                response_schema={"type": "object"}, deadline=time.monotonic() + 2)


async def test_candidate_controls_and_all_identity_pins_are_verified(candidate_config, clip):
    calls = []
    identity = dict(candidate_config.expected_runtime_identity)
    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"ready": True, "model_files_verified": True,
                "model_loaded": True, "prewarm_completed": True, "adapter_loaded_tensor_count": 8, **identity})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": candidate_config.model}]})
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}
        assert body["temperature"] == 0 and body["seed"] == 20261017 and body["max_tokens"] == 4096
        return httpx.Response(200, json=envelope(candidate_config.model))
    async with LocalModelProvider(candidate_config, transport=httpx.MockTransport(handler)) as provider:
        response = await provider.infer(clip, prompt="Synthetic observations only.",
            response_schema={"type": "object"}, deadline=time.monotonic() + 2)
    assert response.data == {"state": "unclear"}
    assert all(response.metrics.serving_attestation[key] == value for key, value in identity.items())
    assert calls == ["/health", "/readyz", "/v1/models", "/v1/chat/completions"]


@pytest.mark.parametrize("field", tuple(candidate_identity()))
async def test_any_candidate_identity_mismatch_blocks_media(candidate_config, clip, field):
    identity = dict(candidate_config.expected_runtime_identity)
    identity[field] = "wrong"
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"ready": True, "model_files_verified": True, **identity})
    async with LocalModelProvider(candidate_config, transport=httpx.MockTransport(handler)) as provider:
        with pytest.raises(ModelUnavailable, match="identity_mismatch"):
            await provider.infer(clip, prompt="Synthetic observations only.",
                response_schema={"type": "object"}, deadline=time.monotonic() + 2)
    assert calls == ["/health", "/readyz"]


def test_candidate_generation_cannot_drift_from_pinned_identity(candidate_config):
    with pytest.raises(ValueError, match="generation configuration identity"):
        replace(candidate_config, seed=18)
    with pytest.raises(ValueError, match="generation configuration identity"):
        replace(candidate_config, max_tokens=128)
    with pytest.raises(ValueError, match="disabled thinking"):
        replace(candidate_config, chat_template_kwargs=())
    with pytest.raises(ValueError, match="complete and pinned"):
        replace(candidate_config, expected_runtime_identity=(("backend", "transformers_peft"),))


async def test_malformed_model_alias_does_not_escape_as_unsanitized_type_error(tmp_path):
    config = LocalModelConfig(media_root=tmp_path)
    async with LocalModelProvider(config, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"id": {"private": "value"}}]}))) as provider:
        with pytest.raises(ModelUnavailable, match="identity_not_served"):
            await provider.check_available(deadline=time.monotonic() + 2)

