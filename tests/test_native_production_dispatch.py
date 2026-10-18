"""Generated CPU consumers: standard Worker dispatch and native API readiness."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import sys
import threading
import time

import httpx
import pytest
import serving

from forte.api.config import Settings
from forte.api.main import _model_alive
from forte.config import WorkerSettings
from forte.native_dispatch import make_native_worker_pipeline
from forte.model import ModelCancelled, ModelDeadlineExceeded, ModelUnavailable
from forte.worker import Worker

PREPARED = Path(__file__).resolve().parents[1] / "training/private/qwen38-vllm-integration-a10/prepared"
sys.path.insert(0, str(PREPARED))
if str(PREPARED / "serving") not in serving.__path__:
    serving.__path__.append(str(PREPARED / "serving"))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


SERIAL_FIXTURE = _module("generated_native_serial_fixture", PREPARED / "tests/test_qwen38_vllm_provider.py")
PARALLEL_FIXTURE = _module("generated_native_parallel_fixture", PREPARED / "experimental_parallel4/tests/fixtures.py")


def fixture_config(mode):
    from serving.qwen38_vllm_identity import validate_native_config
    from experimental_parallel4.profile import parallel_runtime_profile_sha256, validate_parallel_config
    if mode == "serial":
        result = SERIAL_FIXTURE.synthetic_config()
        validate_native_config(result)
    else:
        result = json.loads((PREPARED / "experimental_parallel4/runtime.example.json").read_bytes())
        result["source_archive"]["sha256"] = "a" * 64
        result["package_versions"] = {key: value or "1.2.3-generated" for key, value in result["package_versions"].items()}
        result["runtime_profile_sha256"] = parallel_runtime_profile_sha256(result)
        validate_parallel_config(result)
    return result


def fixture_ready(config, role, mode):
    result = (SERIAL_FIXTURE.synthetic_ready(config, role) if mode == "serial"
              else PARALLEL_FIXTURE.synthetic_parallel_ready(config, role))
    result["live_admission"] = {"schema_version": ("forte.native.live-admission.v1" if mode == "serial"
        else "forte.native.parallel4.live-admission.v1"),
        "engine_instance_id": result["engine_instance_id"],
        "runtime_profile_sha256": config["runtime_profile_sha256"], "admission_open": True,
        "boundary_active_requests": 0, "engine_unfinished_requests": 0,
        "own_requests_tracked": 0, "drain_state": "complete" if mode == "serial" else "tracked",
        "observed_at_monotonic_ns": 123456789}
    return result


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("FORTE_"):
            monkeypatch.delenv(name)


def consumer_settings(tmp_path, mode="serial", role="fit_candidate"):
    from serving.qwen38_vllm_profile import PIPELINE
    from experimental_parallel4.profile import PIPELINE as PARALLEL_PIPELINE
    config = fixture_config(mode)
    recipe = dict(PIPELINE if mode == "serial" else PARALLEL_PIPELINE)
    recipe.setdefault("max_concurrent_jobs", 1)
    path = tmp_path / (mode + ".runtime.generated.json")
    path.write_text(json.dumps(config), encoding="utf8")
    settings = replace(WorkerSettings.from_env(), data_dir=tmp_path / "data",
        model_base_url="http://model-3:8000/v1", model_api_key="generated-only-model-token",
        model_name=config["roles"][role]["served_model_name"],
        model_native_launch_config_path=path, model_native_role=role, model_native_mode=mode,
        trusted_scene_context="bank_cash_workstation", **recipe)
    return settings, config


@pytest.mark.parametrize("missing", ["CONFIG", "ROLE", "MODE"])
@pytest.mark.parametrize("kind", [WorkerSettings, Settings])
def test_partial_native_selection_is_rejected_from_environment(monkeypatch, tmp_path, missing, kind):
    values = {"CONFIG": str(tmp_path / "native.json"), "ROLE": "fit_candidate", "MODE": "serial"}
    for suffix, value in values.items():
        if suffix != missing:
            name = "FORTE_MODEL_NATIVE_LAUNCH_CONFIG" if suffix == "CONFIG" else "FORTE_MODEL_NATIVE_" + suffix
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="native selection"):
        kind.from_env()


@pytest.mark.parametrize("mode,role", [("serial", "fit_candidate"), ("parallel4", "base_diagnostic")])
def test_explicit_environment_and_hf_exclusion(monkeypatch, tmp_path, mode, role):
    monkeypatch.setenv("FORTE_MODEL_NATIVE_LAUNCH_CONFIG", str(tmp_path / "native.json"))
    monkeypatch.setenv("FORTE_MODEL_NATIVE_ROLE", role)
    monkeypatch.setenv("FORTE_MODEL_NATIVE_MODE", mode)
    assert WorkerSettings.from_env().model_native_mode == mode
    assert Settings.from_env().model_native_role == role
    monkeypatch.setenv("FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG", str(tmp_path / "hf.json"))
    for kind in (WorkerSettings, Settings):
        with pytest.raises(ValueError, match="without HF"):
            kind.from_env()


@pytest.mark.parametrize("mode,role", [("serial", "base_diagnostic"), ("serial", "fit_candidate"),
                                      ("parallel4", "base_diagnostic"), ("parallel4", "fit_candidate")])
async def test_standard_worker_constructs_actual_native_factory(tmp_path, mode, role):
    settings, config = consumer_settings(tmp_path, mode, role)
    worker = Worker(settings, store=SimpleNamespace())
    try:
        provider = worker.pipeline.provider.delegate
        assert provider.native_role == role
        assert provider.native_launch_config == config
        assert len(worker.pipeline.rules) == 4 and len(worker.pipeline.facts) == 19
        assert worker.settings.job_timeout_seconds == 285
        assert provider.config.request_timeout_sec == 240
        assert provider.config.max_attempts == 1
        assert len(provider.members) == 4 if mode == "parallel4" else not hasattr(provider, "members")
    finally:
        await worker.aclose()


@pytest.mark.parametrize("change", ["wrong_mode", "wrong_alias", "concurrent_jobs", "extended_deadline"])
def test_native_dispatch_cannot_fall_back_on_configuration_errors(tmp_path, change):
    settings, _ = consumer_settings(tmp_path)
    replacements = {
        "wrong_mode": {"model_native_mode": "parallel4"},
        "wrong_alias": {"model_name": "Qwen3-VL-8B-Instruct"},
        "concurrent_jobs": {"max_concurrent_jobs": 4},
        "extended_deadline": {"job_timeout_seconds": 960.0},
    }[change]
    with pytest.raises((ValueError, RuntimeError)):
        make_native_worker_pipeline(replace(settings, **replacements))


def test_unselected_worker_keeps_original_factory(monkeypatch, tmp_path):
    settings = replace(WorkerSettings.from_env(), data_dir=tmp_path)
    marker = object()
    monkeypatch.setattr("forte.worker.AnalysisPipeline", lambda supplied: marker)
    assert Worker(settings, store=SimpleNamespace()).pipeline is marker


def install_transport(monkeypatch, ready, settings, status=200):
    calls = []
    original = httpx.AsyncClient
    def route(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.headers["Authorization"] == "Bearer generated-only-model-token"
        assert request.headers["X-Forte-Native-Role"] == settings.model_native_role
        if request.url.path == "/health":
            return httpx.Response(200, json={"healthy": True})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": ready["served_model_name"]}]})
        assert request.url.path == "/readyz"
        return httpx.Response(status, json=ready)
    def client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(route)
        return original(**kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    return calls


@pytest.mark.parametrize("mode", ["serial", "parallel4"])
@pytest.mark.parametrize("role", ["base_diagnostic", "fit_candidate"])
async def test_api_uses_actual_native_identity_role_and_full_gate(tmp_path, monkeypatch, mode, role):
    worker, config = consumer_settings(tmp_path, mode, role)
    settings = Settings(data_dir=tmp_path, model_base_url=worker.model_base_url,
        model_native_launch_config_path=worker.model_native_launch_config_path,
        model_native_role=role, model_native_mode=mode, model_api_key=worker.model_api_key)
    ready = fixture_ready(config, role, mode)
    calls = install_transport(monkeypatch, ready, settings)
    assert await _model_alive(settings) is True
    assert len(calls) == 2 and "generated-only-model-token" not in repr(settings)


@pytest.mark.parametrize("mode", ["serial", "parallel4"])
@pytest.mark.parametrize("change", ["identity", "role", "gate", "drain", "arithmetic", "config_missing"])
async def test_api_rejects_stale_or_incomplete_native_proofs(tmp_path, monkeypatch, mode, change):
    worker, config = consumer_settings(tmp_path, mode)
    settings = Settings(data_dir=tmp_path, model_base_url=worker.model_base_url,
        model_native_launch_config_path=worker.model_native_launch_config_path,
        model_native_role="fit_candidate", model_native_mode=mode, model_api_key=worker.model_api_key)
    ready = fixture_ready(config, "fit_candidate", mode)
    if change == "identity": ready["source_sha256"] = "0" * 64
    if change == "role": ready["native_role"] = "base_diagnostic"
    if change == "gate": ready["generated_http_gate"]["attempts"] = 5
    if change == "drain": ready["draining"] = True
    if change == "arithmetic": ready["loaded_engine"]["actual_runtime_state"]["quantization"] = "nf4"
    if change == "config_missing": worker.model_native_launch_config_path.unlink()
    calls = install_transport(monkeypatch, ready, settings)
    assert await _model_alive(settings) is False
    if change == "config_missing": assert calls == []


async def test_worker_wrapper_retains_actual_native_proof_and_original_deadline(tmp_path, monkeypatch):
    settings, config = consumer_settings(tmp_path)
    ready = fixture_ready(config, "fit_candidate", "serial")
    install_transport(monkeypatch, ready, settings)
    pipeline = make_native_worker_pipeline(settings)
    deadline, cancelled, marker = time.monotonic() + 285, threading.Event(), object()
    async def analyze(job, *, stage, deadline: float, cancel_event):
        assert deadline == target_deadline and cancel_event is cancelled
        return marker
    target_deadline = deadline
    monkeypatch.setattr(pipeline.pipeline, "analyze", analyze)
    try:
        assert await pipeline.analyze(SimpleNamespace(), stage=lambda _: None, deadline=deadline, cancel_event=cancelled) is marker
        proof = pipeline.audit_provenance["native_runtime_attestation"]
        assert proof["actual_ready_verified"] is True
        assert proof["native_role"] == "fit_candidate" and proof["hf_nf4_equivalent"] is False
        assert proof["generated_http_gate"]["attempts"] == 6
    finally:
        await pipeline.aclose()


@pytest.mark.parametrize("failure", ["cancel", "deadline", "config_changed"])
async def test_native_wrapper_stops_before_http_and_keeps_original_cancellation(tmp_path, monkeypatch, failure):
    settings, config = consumer_settings(tmp_path)
    calls = install_transport(monkeypatch, fixture_ready(config, "fit_candidate", "serial"), settings)
    pipeline = make_native_worker_pipeline(settings)
    cancel = threading.Event()
    deadline = time.monotonic() + 285
    if failure == "cancel": cancel.set()
    if failure == "deadline": deadline = time.monotonic() - 1
    if failure == "config_changed": settings.model_native_launch_config_path.write_text("{}")
    try:
        expected = {"cancel": ModelCancelled, "deadline": ModelDeadlineExceeded, "config_changed": ModelUnavailable}[failure]
        with pytest.raises(expected):
            await pipeline.analyze(SimpleNamespace(), stage=lambda _: None, deadline=deadline, cancel_event=cancel)
        assert calls == []
    finally:
        await pipeline.aclose()


@pytest.mark.parametrize("failure", ["auth_missing", "redirect", "server_error", "oversized"])
async def test_native_api_fails_closed_without_changing_hf_boundary(tmp_path, monkeypatch, failure):
    worker, config = consumer_settings(tmp_path)
    settings = Settings(data_dir=tmp_path, model_base_url=worker.model_base_url,
        model_native_launch_config_path=worker.model_native_launch_config_path,
        model_native_role="fit_candidate", model_native_mode="serial", model_api_key=worker.model_api_key)
    ready = fixture_ready(config, "fit_candidate", "serial")
    if failure == "auth_missing": settings = replace(settings, model_api_key=None)
    if failure == "oversized": ready["generated_test_padding"] = "x" * (1024 * 1024)
    calls = install_transport(monkeypatch, ready, settings,
        status={"redirect": 302, "server_error": 500}.get(failure, 200))
    assert await _model_alive(settings) is False
    if failure == "auth_missing": assert calls == []


def test_native_admission_pause_does_not_contact_model_or_change_public_response(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from forte.api.main import create_app
    worker, _ = consumer_settings(tmp_path)
    settings = Settings(data_dir=tmp_path, admission_paused=True,
        model_native_launch_config_path=worker.model_native_launch_config_path,
        model_native_role="fit_candidate", model_native_mode="serial", api_keys=("generated-jury-key",))
    monkeypatch.setattr("forte.api.main._model_alive", lambda _: pytest.fail("paused API must not query model"))
    with TestClient(create_app(settings)) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "components": {"admission": False}}


@pytest.mark.parametrize("mode", ["serial", "parallel4"])
@pytest.mark.parametrize("failure", ["missing", "closed", "unknown", "wrong_instance", "wrong_profile", "untracked"])
async def test_api_requires_fresh_live_native_admission_proof(tmp_path, monkeypatch, mode, failure):
    worker, config = consumer_settings(tmp_path, mode)
    settings = Settings(data_dir=tmp_path, model_base_url=worker.model_base_url,
        model_native_launch_config_path=worker.model_native_launch_config_path,
        model_native_role="fit_candidate", model_native_mode=mode, model_api_key=worker.model_api_key)
    ready = fixture_ready(config, "fit_candidate", mode)
    live = ready["live_admission"]
    if failure == "missing": ready.pop("live_admission")
    if failure == "closed": live["admission_open"] = False
    if failure == "unknown": live["drain_state"] = "unknown"
    if failure == "wrong_instance": live["engine_instance_id"] = "different_generated_instance"
    if failure == "wrong_profile": live["runtime_profile_sha256"] = "0" * 64
    if failure == "untracked": live["engine_unfinished_requests"] = 1
    install_transport(monkeypatch, ready, settings)
    assert await _model_alive(settings) is False


@pytest.mark.parametrize("mode,expected", [("serial", False), ("parallel4", True)])
async def test_parallel_live_sibling_is_not_mislabeled_as_serial_idle(tmp_path, monkeypatch, mode, expected):
    worker, config = consumer_settings(tmp_path, mode)
    settings = Settings(data_dir=tmp_path, model_base_url=worker.model_base_url,
        model_native_launch_config_path=worker.model_native_launch_config_path,
        model_native_role="fit_candidate", model_native_mode=mode, model_api_key=worker.model_api_key)
    ready = fixture_ready(config, "fit_candidate", mode)
    ready["live_admission"].update(boundary_active_requests=1, own_requests_tracked=1, engine_unfinished_requests=1)
    install_transport(monkeypatch, ready, settings)
    assert await _model_alive(settings) is expected


@pytest.mark.parametrize("field", [
    "backend", "model_id", "revision", "processor_revision", "model_manifest_sha256",
    "tokenizer_sha256", "chat_template_sha256", "source_sha256", "image_sha256", "config_sha256",
    "runtime_profile_sha256", "generation_sha256", "arithmetic_sha256", "grammar_sha256",
    "package_versions", "native_role", "served_model_name", "diagnostic_only", "trained", "lora_id",
    "adapter_sha256", "adapter_config_sha256", "training_run_sha256",
])
async def test_every_native_runtime_identity_pin_is_required(tmp_path, monkeypatch, field):
    worker, config = consumer_settings(tmp_path)
    settings = Settings(data_dir=tmp_path, model_base_url=worker.model_base_url,
        model_native_launch_config_path=worker.model_native_launch_config_path,
        model_native_role="fit_candidate", model_native_mode="serial", model_api_key=worker.model_api_key)
    ready = fixture_ready(config, "fit_candidate", "serial")
    ready[field] = "wrong-generated-pin"
    install_transport(monkeypatch, ready, settings)
    assert await _model_alive(settings) is False


@pytest.mark.parametrize("kind", [WorkerSettings, Settings])
def test_native_job_budget_cannot_silently_become_diagnostic_960_seconds(tmp_path, kind):
    base = kind.from_env()
    with pytest.raises(ValueError, match="285"):
        replace(base, model_native_launch_config_path=tmp_path / "generated.json",
                model_native_role="fit_candidate", model_native_mode="serial", job_timeout_seconds=960)


@pytest.mark.parametrize("change", [None, "missing", "mode", "native_role", "config_sha256",
                                   "runtime_profile_sha256", "source_sha256", "image_sha256"])
def test_native_api_requires_matching_worker_heartbeat_not_only_fresh_timestamp(tmp_path, change):
    from forte.api.main import _worker_alive
    from forte.native_dispatch import native_worker_binding
    worker, _ = consumer_settings(tmp_path)
    heartbeat = tmp_path / "generated-heartbeat.json"
    settings = Settings(data_dir=tmp_path, worker_heartbeat_file=heartbeat,
        model_native_launch_config_path=worker.model_native_launch_config_path,
        model_native_role="fit_candidate", model_native_mode="serial")
    record = {"updated_at": datetime.now(timezone.utc).isoformat(), "worker_id": "generated-worker",
              "native_consumer": native_worker_binding(settings)}
    if change == "missing": record.pop("native_consumer")
    elif change is not None: record["native_consumer"][change] = "wrong-generated-binding"
    heartbeat.write_text(json.dumps(record))
    assert _worker_alive(settings) is (change is None)
    # Baseline/HF heartbeat behavior remains timestamp-based without promotion
    # into a native route unless its full matching binding was supplied.
    assert _worker_alive(Settings(data_dir=tmp_path, worker_heartbeat_file=heartbeat)) is True


async def test_standard_native_worker_emits_bound_safe_heartbeat(tmp_path, monkeypatch):
    from forte.native_dispatch import native_worker_binding
    settings, _ = consumer_settings(tmp_path)
    worker = Worker(settings, store=SimpleNamespace())
    captured = []
    def write(path, record):
        captured.append(record)
        worker.stop.set()
    monkeypatch.setattr("forte.worker.atomic_private_json", write)
    try:
        await worker._heartbeat()
        assert captured[0]["native_consumer"] == native_worker_binding(settings)
        assert "generated-only-model-token" not in json.dumps(captured)
    finally:
        await worker.aclose()


async def test_worker_heartbeat_cannot_adopt_a_config_the_loaded_pipeline_did_not_load(tmp_path, monkeypatch):
    from forte.api.main import _worker_alive
    settings, _ = consumer_settings(tmp_path)
    worker = Worker(settings, store=SimpleNamespace())
    original = worker.pipeline.native_consumer_binding
    # Even a whitespace-only rewrite must not advertise a new raw config: the
    # loaded pipeline's own drift guard rejects these new bytes before HTTP.
    path = settings.model_native_launch_config_path
    path.write_bytes(path.read_bytes() + b"\n")
    heartbeat = tmp_path / "drift-heartbeat.json"
    def write(_, record):
        heartbeat.write_text(json.dumps(record))
        worker.stop.set()
    monkeypatch.setattr("forte.worker.atomic_private_json", write)
    try:
        await worker._heartbeat()
        assert json.loads(heartbeat.read_text())["native_consumer"] == original
        assert worker.pipeline.native_consumer_binding == original
        api = Settings(data_dir=tmp_path, worker_heartbeat_file=heartbeat,
            model_native_launch_config_path=path, model_native_role="fit_candidate", model_native_mode="serial")
        assert _worker_alive(api) is False
    finally:
        await worker.aclose()
