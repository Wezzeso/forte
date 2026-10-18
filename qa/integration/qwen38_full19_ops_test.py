"""Actual frozen prompt/schema/media/parser path over in-memory mock HTTP."""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from deploy import qwen38_full19_generated_gate as gate
from serving.qwen38_comparison_profile import runtime_profile_sha256
from test_qwen38_untuned import config_fixture, ready_fixture

ROOT = Path(__file__).resolve().parents[2]


def prepare_config(tmp_path, *, trained=False):
    config = config_fixture()
    if trained:
        config.update(schema_version="forte.qwen38.runtime.v1", backend="transformers_peft",
            adapter_path="/generated/adapter", adapter_sha256="1" * 64,
            adapter_config_sha256="2" * 64,
            training_run={"path": "/generated/training-run.json", "sha256": "3" * 64})
    config["limits"]["max_pixels_per_image"] = 262144
    config["runtime_profile_sha256"] = runtime_profile_sha256(config)
    path = tmp_path / "generated-runtime.json"
    path.write_text(json.dumps(config))
    return config, path


def mock_endpoint(tmp_path, monkeypatch, *, behavior="valid", use_actual_profile=False, trained=False):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    config, config_path = prepare_config(tmp_path, trained=trained)
    _, fact_ids, _, schema = gate.schema_definition(config, ROOT)
    if trained:
        from serving.qwen38_identity import load_expected_runtime_identity
        ready = {**load_expected_runtime_identity(config_path), "ready": True,
            "model_files_verified": True, "model_loaded": True, "prewarm_completed": True,
            "adapter_loaded_tensor_count": 992,
            "structured_decoding": {**config["structured_decoding"], "fresh_matcher_per_generate": True,
                "strict_post_generation_validation": True, "requests_bound_to_grammar": 3}}
        if behavior == "unloaded":
            ready["adapter_loaded_tensor_count"] = 0
    else:
        ready = ready_fixture(config)
    ready["structured_decoding"].update(model_vocab_size=248320, stop_token_ids=[248046, 248044],
        public_schema_warmup={"fact_count": 19, "rule_count": 4, "event_type_count": 3,
            "frame_count": 16, "model_calls": 0, "private_media_read": False})
    # Only the fixed production recipe digest is substituted in this synthetic
    # fixture. The unpatched guard is separately exercised below. Identity,
    # prompt/schema, JPEG ownership, provider, parser and semantic validator are real.
    if not use_actual_profile:
        monkeypatch.setattr("serving.qwen38_comparison_profile.runtime_profile_sha256",
            lambda value: "4bbe4f79281d1b30ab2d45af539308bc57a4f911e5aad5d845c7b41eff9f07ab")
    posts, requests = [], []
    if behavior == "identity":
        ready["model_revision"] = "0" * 40
    if behavior == "grammar":
        ready["structured_decoding"]["model_vocab_size"] = 1
    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/readyz":
            response = deepcopy(ready)
            response["structured_decoding"]["requests_bound_to_grammar"] = 3 + (0 if behavior == "counter" else len(posts))
            return httpx.Response(200, json=response)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": config["served_model_name"]}]})
        assert request.url.path == "/v1/chat/completions" and request.method == "POST"
        body = json.loads(request.content)
        posts.append(body)
        assert body["structured_outputs"]["json"] == schema
        assert body["max_tokens"] == 4096 and body["seed"] == 20261017
        assert body["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}
        images = [block for message in body["messages"] if isinstance(message.get("content"), list)
                  for block in message["content"] if block.get("type") == "image_url"]
        assert len(images) == 16 and all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in images)
        if behavior == "http500":
            return httpx.Response(500, text="SYNTHETIC_RAW_MODEL_TEXT_MUST_NOT_BE_LOGGED")
        data = {"facts": [{"fact_id": fact, "state": "unclear", "supporting_frame_ids": [], "notes": None}
                           for fact in fact_ids], "candidates": []}
        if behavior == "duplicate":
            data["facts"][-1]["fact_id"] = data["facts"][0]["fact_id"]
        return httpx.Response(200, json={"model": config["served_model_name"],
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(data)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100}})
    original = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=transport, **{key: value for key, value in kwargs.items() if key != "transport"}))
    output = tmp_path / "private-output"
    output.mkdir()
    return config, config_path, output, posts, requests


@pytest.mark.asyncio
async def test_real_full19_path_is_one_post_with_original_schema_semantics_and_owned_jpegs(tmp_path, monkeypatch):
    config, path, output, posts, _ = mock_endpoint(tmp_path, monkeypatch)
    result = await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT)
    assert result["status"] == "passed" and result["model_post_attempts"] == len(posts) == 1
    assert result["facts_validated"] == 19 and result["canonical_fact_count"] == 19
    assert result["strict_json_schema_passed"] and result["semantic_validation_passed"]
    assert result["actual_grammar_requests_after"] == result["actual_grammar_requests_before"] + 1
    assert result["private_quality_measured"] is False and result["visual_truth_measured"] is False
    # Exclusive one-use claim forbids reusing the output even after success.
    with pytest.raises(FileExistsError):
        await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT)
    assert len(posts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["http500", "duplicate", "counter"])
async def test_one_failed_post_never_retries_or_becomes_a_structural_pass(tmp_path, monkeypatch, behavior):
    config, path, output, posts, _ = mock_endpoint(tmp_path, monkeypatch, behavior=behavior)
    from forte.decision import DecisionOutputError
    from forte.model.adapter import ModelUnavailable
    failure = {"http500": ModelUnavailable, "duplicate": DecisionOutputError, "counter": gate.GateError}[behavior]
    with pytest.raises(failure):
        await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT)
    receipt = json.loads((output / "full19generated.gate.receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["model_post_attempts"] == len(posts) == 1
    assert receipt["automatic_retry"] is False
    assert "SYNTHETIC_RAW_MODEL_TEXT" not in json.dumps(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["identity", "grammar"])
async def test_wrong_identity_or_actual_grammar_blocks_before_any_post(tmp_path, monkeypatch, behavior):
    config, path, output, posts, _ = mock_endpoint(tmp_path, monkeypatch, behavior=behavior)
    with pytest.raises(gate.GateError):
        await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT)
    assert posts == [] and not (output / "generated.gate.attempt.json").exists()


@pytest.mark.asyncio
async def test_unpatched_production_profile_guard_rejects_synthetic_recipe_before_http(tmp_path, monkeypatch):
    config, path = prepare_config(tmp_path)
    output = tmp_path / "private-output"
    output.mkdir()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("profile mismatch must precede HTTP"))
    with pytest.raises(gate.GateError, match="comparison_profile_mismatch"):
        await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT)


@pytest.mark.asyncio
async def test_explicit_actual_profile_joins_new_source_without_profile_function_patch(tmp_path, monkeypatch):
    config, path, output, posts, _ = mock_endpoint(tmp_path, monkeypatch, use_actual_profile=True)
    expected_profile = runtime_profile_sha256(config)
    result = await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT,
                                expected_profile_sha256=expected_profile)
    assert result["runtime_profile_sha256"] == expected_profile
    assert result["status"] == "passed" and result["model_post_attempts"] == len(posts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_profile", ["", "A" * 64, "0" * 63, "0" * 64])
async def test_explicit_wrong_or_malformed_profile_rejects_before_http(tmp_path, monkeypatch, expected_profile):
    config, path = prepare_config(tmp_path)
    output = tmp_path / "private-output"
    output.mkdir()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("profile mismatch must precede HTTP"))
    with pytest.raises(gate.GateError, match="comparison_profile_mismatch"):
        await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT,
                           expected_profile_sha256=expected_profile)
    assert not (output / "generated.gate.attempt.json").exists()


@pytest.mark.asyncio
async def test_trained_full19_provider_path_with_exact_identity_and_explicit_profile(tmp_path, monkeypatch):
    config, path, output, posts, _ = mock_endpoint(tmp_path, monkeypatch, use_actual_profile=True, trained=True)
    result = await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT,
                                expected_profile_sha256=runtime_profile_sha256(config))
    assert result["status"] == "passed" and result["model_post_attempts"] == len(posts) == 1
    assert len(result["expected_runtime_identity"]) == 15
    assert result["expected_runtime_identity"]["adapter_sha256"] == config["adapter_sha256"]


@pytest.mark.asyncio
async def test_trained_provider_requires_actual_loaded_adapter_before_post(tmp_path, monkeypatch):
    from forte.model.adapter import ModelUnavailable
    config, path, output, posts, _ = mock_endpoint(tmp_path, monkeypatch, use_actual_profile=True,
                                                 trained=True, behavior="unloaded")
    with pytest.raises(ModelUnavailable, match="readiness_identity_mismatch"):
        await gate.execute(config, path, "http://127.0.0.1:8000/v1", output, "generated-release", ROOT,
                           expected_profile_sha256=runtime_profile_sha256(config))
    assert posts == []
    receipt = json.loads((output / "full19generated.gate.receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["model_post_attempts"] == 0
