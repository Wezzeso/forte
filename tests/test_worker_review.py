"""Synthetic worker lifecycle regressions with an explicitly injected pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forte.config import WorkerSettings
from forte.decision import DecisionOutputError
from forte.jobs import JobStore, LeaseLost
from forte.observability import aggregate_inference_metrics
from forte.worker import Worker


class _Store:
    def __init__(self, *, lose_lease: bool = False):
        self.lose_lease = lose_lease
        self.published: list[bytes] = []
        self.failures: list[tuple[str, str]] = []

    def heartbeat(self, *args):
        if self.lose_lease:
            raise LeaseLost("synthetic lease was superseded")

    def set_stage(self, *args):
        return None

    def publish(self, job_id, worker_id, attempt, payload):
        self.published.append(payload)

    def fail(self, job_id, worker_id, attempt, message, *, error_code, retryable=False):
        self.failures.append((message, error_code))


def _settings(tmp_path: Path) -> WorkerSettings:
    return WorkerSettings(
        database_url="sqlite://", data_dir=tmp_path,
        model_base_url="http://127.0.0.1:1/v1", model_name="synthetic-pipeline",
        model_api_key=None, worker_id="synthetic-worker",
        heartbeat_file=tmp_path / "heartbeat.json", rules_path=tmp_path / "unused-rules.json",
        taxonomy_path=tmp_path / "unused-taxonomy.json", model_manifest_path=tmp_path / "unused-model.json",
        heartbeat_seconds=0.005, lease_seconds=1.0,
    )


def _job(remaining: float = 1.0):
    return SimpleNamespace(job_id="synthetic-job", attempt=1, remaining_seconds=lambda: remaining)


@pytest.mark.asyncio
async def test_heartbeat_lease_loss_cancels_only_that_job(tmp_path: Path) -> None:
    """FairScheduler propagates a set cancellation signal as CancelledError."""
    store = _Store(lose_lease=True)

    class CancelOnOwnershipLoss:
        async def analyze(self, job, *, stage, deadline, cancel_event):
            await stage("screening")
            while not cancel_event.is_set():
                await asyncio.sleep(0.001)
            raise asyncio.CancelledError

    worker = Worker(_settings(tmp_path), store=store, pipeline=CancelOnOwnershipLoss())
    summary = await worker.process_claimed(_job())
    assert summary["status"] == "ownership_lost"
    assert not store.published
    assert not store.failures


@pytest.mark.asyncio
async def test_external_task_cancellation_preserves_recoverable_attempt(tmp_path: Path) -> None:
    store = _Store()
    started = asyncio.Event()
    cancellation_signals = []

    class WaitingPipeline:
        async def analyze(self, job, *, stage, deadline, cancel_event):
            cancellation_signals.append(cancel_event)
            started.set()
            await asyncio.Event().wait()

    worker = Worker(_settings(tmp_path), store=store, pipeline=WaitingPipeline())
    task = asyncio.create_task(worker.process_claimed(_job()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancellation_signals[0].is_set()
    assert not store.published
    assert not store.failures


@pytest.mark.asyncio
async def test_provider_cancellation_without_external_task_cancel_is_a_failed_job(tmp_path: Path) -> None:
    store = _Store()

    class ProviderCancelledPipeline:
        async def analyze(self, job, *, stage, deadline, cancel_event):
            raise asyncio.CancelledError

    worker = Worker(_settings(tmp_path), store=store, pipeline=ProviderCancelledPipeline())
    summary = await worker.process_claimed(_job())
    assert summary["status"] == "failed"
    assert summary["error_code"] == "model_cancelled"
    assert store.failures == [("model_cancelled", "model_cancelled")]
    assert not store.published


@pytest.mark.asyncio
async def test_worker_timeout_cannot_publish_clean_negative(tmp_path: Path) -> None:
    store = _Store()
    signals = []

    class WaitingPipeline:
        async def analyze(self, job, *, stage, deadline, cancel_event):
            signals.append(cancel_event)
            await asyncio.Event().wait()

    worker = Worker(_settings(tmp_path), store=store, pipeline=WaitingPipeline())
    summary = await worker.process_claimed(_job(remaining=0.02))
    assert summary["status"] == "failed"
    assert summary["error_code"] == "deadline_exceeded"
    assert signals[0].is_set()
    assert not store.published
    assert store.failures == [("deadline_exceeded", "deadline_exceeded")]


@pytest.mark.asyncio
async def test_private_exception_message_never_enters_summary_or_stored_diagnostics(tmp_path: Path) -> None:
    store = _Store()
    private_marker = "synthetic-private-caption-and-path-never-log-this"

    class BrokenPipeline:
        async def analyze(self, job, *, stage, deadline, cancel_event):
            raise RuntimeError(private_marker)

    worker = Worker(_settings(tmp_path), store=store, pipeline=BrokenPipeline())
    summary = await worker.process_claimed(_job())
    assert summary["error_code"] == "processing_failed"
    assert private_marker not in json.dumps(summary)
    assert store.failures == [("processing_failed", "processing_failed")]
    assert not store.published


@pytest.mark.asyncio
@pytest.mark.parametrize("subcode,expected", [("duplicate_fact_id", "duplicate_fact_id"),
                                               ({"private": "synthetic-caption"}, None)])
async def test_rejected_decision_retains_only_static_code_and_numeric_metrics(tmp_path: Path, subcode, expected) -> None:
    store = _Store()

    class RejectedDecisionPipeline:
        async def analyze(self, job, *, stage, deadline, cancel_event):
            error = DecisionOutputError("duplicate_fact_id")
            error.forte_decision_subcode = subcode
            error.forte_model_metrics = aggregate_inference_metrics([
                {"elapsed_sec": .25, "generated_tokens": 17, "private": "synthetic-caption"},
            ])
            error.forte_model_metrics["private"] = "synthetic-caption"
            raise error

    summary = await Worker(_settings(tmp_path), store=store, pipeline=RejectedDecisionPipeline()).process_claimed(_job())
    assert summary["status"] == "failed"
    assert summary["error_code"] == "invalid_model_output"
    assert summary.get("decision_error_subcode") == expected
    assert summary["rejected_model_request_metrics"]["request_record_count"] == 1
    assert summary["rejected_model_request_metrics"]["numeric_counts"]["generated_tokens"]["sum"] == 17
    assert "synthetic-caption" not in json.dumps(summary)
    assert not store.published


@pytest.mark.asyncio
async def test_preserved_publication_bytes_must_include_required_status(tmp_path: Path) -> None:
    """Default injection into a parsed model must not validate unchanged bad bytes."""
    source = tmp_path / "synthetic.mov"
    source.write_bytes(b"synthetic input for an injected fake pipeline")
    settings = _settings(tmp_path)
    store = JobStore("sqlite:///" + (tmp_path / "jobs.db").as_posix(), settings.result_dir)
    store.create_job("synthetic-job", source, hashlib.sha256(source.read_bytes()).hexdigest(),
                     source.stat().st_size, "synthetic.mov")

    class MissingStatusPipeline:
        async def analyze(self, job, *, stage, deadline, cancel_event):
            return SimpleNamespace(json_bytes=json.dumps({
                "job_id": job.job_id, "verdict": "no_violation", "violations": [],
            }).encode("utf-8"))

    try:
        worker = Worker(settings, store=store, pipeline=MissingStatusPipeline())
        summary = await worker.run_once()
        assert summary["status"] == "failed"
        assert store.get_job("synthetic-job").status == "failed"
        assert store.get_result_bytes("synthetic-job") is None
    finally:
        store.close()


@pytest.mark.parametrize("name,value", [
    ("FORTE_MAX_CONCURRENT_JOBS", "0.5"),
    ("FORTE_WORKER_POLL_SECONDS", "nan"),
    ("FORTE_WORKER_POLL_SECONDS", "inf"),
    ("FORTE_WINDOW_SECONDS", "nan"),
    ("FORTE_WORKER_HEARTBEAT_SECONDS", "nan"),
])
def test_worker_rejects_silent_stall_and_nonfinite_configuration(monkeypatch, tmp_path: Path, name: str, value: str) -> None:
    monkeypatch.setenv("FORTE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        WorkerSettings.from_env()
