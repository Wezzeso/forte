"""Synthetic cross-owner API/queue/result checks, with no inference claims."""

from __future__ import annotations

import base64
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import shutil
from types import SimpleNamespace

import av
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from forte.api.config import Settings
from forte.api.main import create_app
from forte.jobs import JobStore
from forte.config import WorkerSettings
from forte.pipeline import AnalysisPipeline
from forte.shared.schemas import CompletedResponse
from forte.worker import Worker
from rules.catalog import load_catalog


pytestmark = pytest.mark.integration
_KEY = "synthetic-integration-key-not-a-production-secret"


def _synthetic_mov(path: Path) -> bytes:
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=4)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for index in range(4):
            image = Image.new("RGB", (64, 48), (index * 40, 60, 100))
            frame = av.VideoFrame.from_image(image)
            frame.pts = index
            frame.time_base = Fraction(1, 4)
            for packet in stream.encode(frame):
                output.mux(packet)
            image.close()
        for packet in stream.encode():
            output.mux(packet)
    return path.read_bytes()


@pytest.fixture
def service(tmp_path: Path):
    settings = Settings(
        api_keys=(_KEY,),
        data_dir=tmp_path / "runtime",
        database_url="sqlite:///" + (tmp_path / "jobs.db").as_posix(),
        web_dist=tmp_path / "no-web-build",
    )
    store = JobStore(settings.database_url, settings.result_root)
    app = create_app(settings, store=store, readiness=lambda: True)
    with TestClient(app) as client:
        yield client, store, settings, _synthetic_mov(tmp_path / "synthetic.mov")
    store.close()


def _accept(client: TestClient, media: bytes) -> str:
    response = client.post(
        "/api/v1/analyze", headers={"X-API-Key": _KEY},
        files={"file": ("synthetic.mov", media, "video/quicktime")},
    )
    assert response.status_code == 202
    assert set(response.json()) == {"job_id"}
    return response.json()["job_id"]


def _event(timestamp: float, label: str) -> dict:
    output = BytesIO()
    with Image.new("RGB", (64, 48), (80, 120, 180)) as image:
        image.save(output, "JPEG")
    return {
        "type": label,
        "timecode_sec": timestamp,
        "description": "Synthetic contract fixture; no real observation or legal claim.",
        "evidence_frames": [{
            "timestamp_sec": timestamp,
            "image_base64": base64.b64encode(output.getvalue()).decode("ascii"),
        }],
    }


def test_upload_queue_publication_and_immutable_polling(service) -> None:
    client, store, settings, media = service
    job_id = _accept(client, media)
    record = store.get_job(job_id)
    assert record is not None
    assert Path(record.input_path).is_relative_to(settings.upload_root)
    assert Path(record.input_path).read_bytes() == media
    assert record.input_sha256 == sha256(media).hexdigest()
    headers = {"X-API-Key": _KEY}
    assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).json() == {
        "job_id": job_id, "status": "processing",
    }
    attempt = store.claim("synthetic-worker")
    assert attempt is not None and attempt.job_id == job_id
    store.set_stage(job_id, "synthetic-worker", attempt.attempt, "assembling")
    # Explicit fixture publication exercises the API/queue contract. It does
    # not exercise the learned detector or establish evidence correspondence.
    result = CompletedResponse(
        job_id=job_id, verdict="violation",
        violations=[_event(0.0, "Synthetic event A"), _event(0.5, "Synthetic event B")],
    )
    store.publish(job_id, "synthetic-worker", attempt.attempt, result.model_dump())
    first = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    second = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content == store.get_result_bytes(job_id)
    parsed = CompletedResponse.model_validate(first.json())
    assert parsed.verdict == "violation" and len(parsed.violations) == 2
    assert first.headers["cache-control"] == "no-store"
    assert client.get(f"/api/ui/jobs/{job_id}/video", headers=headers).content == media


def test_negative_and_failed_jobs_keep_distinct_public_results(service) -> None:
    client, store, _, media = service
    headers = {"X-API-Key": _KEY}
    normal_id = _accept(client, media)
    claim = store.claim("synthetic-worker")
    assert claim is not None and claim.job_id == normal_id
    store.publish(normal_id, "synthetic-worker", claim.attempt, {
        "job_id": normal_id, "status": "done", "verdict": "no_violation", "violations": [],
    })
    assert client.get(f"/api/v1/jobs/{normal_id}", headers=headers).json() == {
        "job_id": normal_id, "status": "done", "verdict": "no_violation", "violations": [],
    }
    failed_id = _accept(client, media)
    claim = store.claim("synthetic-worker")
    assert claim is not None and claim.job_id == failed_id
    store.fail(failed_id, "synthetic-worker", claim.attempt, "synthetic internal detail", error_code="synthetic_failure")
    failed = client.get(f"/api/v1/jobs/{failed_id}", headers=headers)
    assert failed.status_code == 200
    assert failed.json() == {"job_id": failed_id, "status": "failed"}
    assert "synthetic internal detail" not in failed.text


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-synthetic-key"}])
def test_both_jury_routes_require_key_before_acceptance(service, headers: dict) -> None:
    client, store, settings, media = service
    upload = client.post("/api/v1/analyze", headers=headers, files={"file": ("synthetic.mov", media)})
    polling = client.get("/api/v1/jobs/unknown", headers=headers)
    assert upload.status_code == polling.status_code == 401
    assert store.counts() == {}
    assert list(settings.upload_root.iterdir()) == []


def test_unknown_job_and_wrong_multipart_field_are_rejected(service) -> None:
    client, store, settings, media = service
    headers = {"X-API-Key": _KEY}
    assert client.get("/api/v1/jobs/unknown", headers=headers).status_code == 404
    response = client.post("/api/v1/analyze", headers=headers, files={"video": ("synthetic.mov", media)})
    assert response.status_code == 400
    assert store.counts() == {}
    assert list(settings.upload_root.iterdir()) == []


class _SyntheticDecisionProvider:
    """Test-only observations over generated colors, never a detector fallback."""

    def __init__(self, catalog, mode: str):
        self.catalog, self.mode, self.calls = catalog, mode, 0

    async def infer(self, clip, **kwargs):
        self.calls += 1
        if self.mode == "provider_failure":
            raise RuntimeError("synthetic private model diagnostic must not reach public output")
        rule = self.catalog.enabled_rules()[0]
        binding = self.catalog.bindings[rule.rule_id]
        states = binding["required_fact_states"] if self.mode == "positive" else {}
        supporting = [frame.frame_id for frame in clip.frame_refs]
        facts = [{
            "fact_id": fact.fact_id,
            "state": "unclear" if self.mode == "unclear" else states.get(fact.fact_id, "absent"),
            "supporting_frame_ids": [] if self.mode == "unclear" else supporting,
            "notes": None,
        } for fact in clip.fact_definitions]
        candidates = []
        if self.mode == "positive":
            candidates.append({
                "candidate_id": "synthetic-event",
                "event_type_id": binding["event_type_id"],
                "start_frame_id": supporting[0], "end_frame_id": supporting[-1],
                "supporting_fact_ids": list(states),
                "supporting_frame_ids": supporting,
                "visibility_notes": [],
            })
        return SimpleNamespace(data={"facts": facts, "candidates": candidates}, metrics={}, finish_reason="stop")


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="Real pipeline requires FFprobe")
@pytest.mark.parametrize("mode", ["positive", "normal", "unclear", "provider_failure"])
def test_real_worker_pipeline_with_explicit_synthetic_provider(service, mode: str) -> None:
    """Exercise decoder/sampling/decision/rules/evidence/queue/API together."""
    import asyncio

    client, store, settings, media = service
    job_id = _accept(client, media)
    catalog = load_catalog()
    provider = _SyntheticDecisionProvider(catalog, mode)
    worker_settings = WorkerSettings(
        database_url=settings.database_url, data_dir=settings.data_dir,
        model_base_url="http://127.0.0.1:1/v1", model_name="synthetic-test-provider",
        model_api_key=None, worker_id="synthetic-worker", heartbeat_file=settings.worker_heartbeat_file,
        rules_path=Path("rules/catalog.json"), taxonomy_path=Path("annotations/taxonomy.v1.json"),
        model_manifest_path=Path("models/manifest.json"), trusted_scene_context="bank_cash_workstation",
    )
    pipeline = AnalysisPipeline(worker_settings, provider=provider, catalog=catalog)
    synthetic_diagnostics: list[str] = []
    original_analyze = pipeline.analyze

    async def diagnosed_analyze(*args, **kwargs):
        try:
            return await original_analyze(*args, **kwargs)
        except Exception as error:
            # This fixture is generated here; its diagnostic contains no
            # organizer video facts or private model output.
            synthetic_diagnostics.append(f"{type(error).__name__}: {error}")
            raise

    pipeline.analyze = diagnosed_analyze
    worker = Worker(worker_settings, store=store, pipeline=pipeline)

    async def run_worker():
        try:
            return await worker.run_once()
        finally:
            await worker.aclose()

    summary = asyncio.run(run_worker())
    assert provider.calls >= 1
    response = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": _KEY})
    assert response.status_code == 200
    if mode in {"unclear", "provider_failure"}:
        assert summary["status"] == "failed"
        assert response.json() == {"job_id": job_id, "status": "failed"}
        assert "synthetic private model diagnostic" not in response.text
        return
    assert summary["status"] == "done", synthetic_diagnostics
    completed = CompletedResponse.model_validate(response.json())
    assert completed.verdict == ("violation" if mode == "positive" else "no_violation")
    assert bool(completed.violations) == (mode == "positive")
    if mode == "positive":
        assert provider.calls >= 2  # The candidate receives a refinement pass.
        for event in completed.violations:
            assert "https://adilet.zan.kz/" in event.description
            for frame in event.evidence_frames:
                with Image.open(BytesIO(base64.b64decode(frame.image_base64))) as image:
                    assert image.size == (64, 48)
                    expected = (round(frame.timestamp_sec * 4) * 40, 60, 100)
                    assert max(abs(a - b) for a, b in zip(image.getpixel((32, 24)), expected)) < 12
    assert client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": _KEY}).content == response.content
