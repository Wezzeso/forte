"""Opt-in private trace around the unchanged worker; never selected implicitly.

Root must bind the new wrapper/source/CPU image and exact A09 HTTP model slots.
No old audit is upgraded into a fresh witness. Public result bytes are returned
unchanged. One job and one model call are required for unambiguous tracing.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import signal
import time

from evaluation.io import digest, read_json, write_json
from evaluation.qwen38_comparison import MODEL_FIELDS
from evaluation.qwen38_pipeline import (ManifestBoundBaselineProvider, file_sha,
                                       require_released_runtime, trace_pipeline_stages)
from evaluation.qwen38_server_export import write_private_new
from forte.config import WorkerSettings
from forte.model.adapter import LocalModelProvider
from forte.pipeline import AnalysisPipeline, _provider_configuration
from training.qwen38_preflight import FIXED_STOP_UTC, validate_attempt_reservation

TRACE_BINDING_SCHEMA = "forte.qwen38-worker-trace-binding.v1"


def require(value, code):
    if not value:
        raise RuntimeError(code)


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "trace_release_timezone_missing")
    return parsed.astimezone(timezone.utc)


def bound_json(entry):
    require(file_sha(entry["path"]) == entry["sha256"], "trace_bound_artifact_changed")
    return read_json(entry["path"])


class TraceSession:
    def __init__(self, binding: dict, settings: WorkerSettings, witness_root: Path):
        import forte.pipeline

        self.binding, self.settings, self.root = binding, settings, Path(witness_root)
        require(binding.get("schema_version") == TRACE_BINDING_SCHEMA, "trace_binding_schema_invalid")
        require(settings.max_concurrent_jobs == settings.max_concurrent_clips == settings.max_inflight_per_job == 1
                and settings.staged_screening is False, "trace_single_job_nonstaged_profile_required")
        require(all(getattr(settings, key) == value for key, value in binding["worker_settings"].items()),
                "trace_actual_worker_recipe_mismatch")
        require(binding["wrapper_sha256"] == file_sha(__file__)
                and binding["pipeline_sha256"] == file_sha(inspect.getfile(forte.pipeline))
                and binding["trace_source_archive"]["sha256"] == file_sha(binding["trace_source_archive"]["path"])
                and os.environ.get("FORTE_TRACE_APP_IMAGE_SHA256") == binding["trace_runtime_image_sha256"],
                "trace_wrapper_source_or_cpu_image_pin_mismatch")
        self.plan = bound_json(binding["model_attempt_plan"])
        self.release = bound_json(binding["model_release"])
        self.reservation = bound_json(binding["model_reservation"])
        self.rows = [{**slot, "status": "not_attempted"} for slot in self.plan["attempts"]]
        self.sources = [dict(row) for row in binding["source_attempts"]]
        require(len(self.sources) == 6 and len({row["attempt_id"] for row in self.sources}) == 6
                and all(row["split"] in {"train", "validation"} for row in self.sources)
                and binding["provider_max_attempts"] == 2
                and binding["maximum_model_requests"] == len(self.rows), "trace_bound_source_or_model_budget_invalid")
        source_by_id = {row["attempt_id"]: row for row in self.sources}
        require(all(row["source_attempt_id"] in source_by_id
            and all(row[key] == source_by_id[row["source_attempt_id"]][key] for key in ("content_sha256", "split"))
            for row in self.rows), "trace_model_slot_source_mismatch")
        self.claimed = set()
        self.wall_deadline = None
        self.guard()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        # A reservation cannot be reused by selecting a different witness root.
        write_private_new(settings.data_dir / "trace-execution-claims" / (binding["model_attempt_plan"]["sha256"] + ".json"),
                          {"binding_sha256": digest(binding), "claimed_at_utc": datetime.now(timezone.utc).isoformat()})
        self.flush_slots()

    def guard(self):
        now = datetime.now(timezone.utc)
        release = self.release
        require(release.get("schema_version") == "forte.qwen38-root-release.v1"
            and release.get("released_by_root") is True and release.get("phase") == "acceptance"
            and release.get("attempt_plan_sha256") == self.binding["model_attempt_plan"]["sha256"]
            and release.get("maximum_attempts") == len(self.rows)
            and release.get("expected_model_identity") == self.binding["expected_model_identity"]
            and release.get("exclusive_idle_gpu_handoff_verified") is True,
            "trace_exact_model_root_release_required")
        require(utc(release["not_before_utc"]) <= now < utc(release["expires_at_utc"]) <= FIXED_STOP_UTC,
                "trace_model_release_expired")
        validate_attempt_reservation(self.reservation, self.plan,
            plan_sha256=self.binding["model_attempt_plan"]["sha256"], phase="acceptance", now=now)
        require(self.reservation["release_sha256"] == self.binding["model_release"]["sha256"]
            and self.reservation["config_sha256"] == release["config_sha256"]
            and all(self.reservation[key] == release[key] for key in
                ("source_archive_sha256", "runtime_image_sha256", "base_manifest_sha256")),
            "trace_model_reservation_runtime_mismatch")
        require(type(release.get("maximum_execution_seconds")) is int and release["maximum_execution_seconds"] > 0,
                "trace_model_wall_bound_missing")
        if self.wall_deadline is None:
            self.wall_deadline = time.monotonic() + release["maximum_execution_seconds"]
        require(time.monotonic() < self.wall_deadline, "trace_model_execution_wall_exhausted")

    def flush_slots(self):
        write_json(self.root / "model-slot-terminals.json", {"records": self.rows})

    def claim_source(self, job):
        self.guard()
        choices = [row for row in self.sources if row["content_sha256"] == job.content_hash
                   and row["attempt_id"] not in self.claimed]
        require(bool(choices), "trace_job_outside_reserved_sources")
        selected = min(choices, key=lambda row: row["repeat_index"])
        self.claimed.add(selected["attempt_id"])
        return selected

    def claim_model_slot(self, source_id):
        self.guard()
        choices = [row for row in self.rows if row["source_attempt_id"] == source_id and row["status"] == "not_attempted"]
        require(bool(choices), "trace_model_call_budget_exhausted")
        slot = choices[0]
        slot.update(status="attempted_unfinished", started_at_utc=datetime.now(timezone.utc).isoformat())
        self.flush_slots()
        return slot


class TracedPipeline:
    """Delegate all decisions, source decode and evidence to AnalysisPipeline."""
    def __init__(self, settings, session: TraceSession):
        self.session = session
        config = _provider_configuration(settings)
        config = replace(config, readiness_cache_sec=0)
        require(config.max_attempts == session.binding["provider_max_attempts"] and config.require_readiness,
                "trace_actual_provider_retry_or_readiness_mismatch")
        raw = (ManifestBoundBaselineProvider(config, settings.model_manifest_path)
               if settings.model_candidate_launch_config_path is None else LocalModelProvider(config))
        self.delegate = AnalysisPipeline(settings, provider=raw)
        self.provider = self.delegate.provider
        self.rules = self.delegate.rules
        self.raw = raw

    async def analyze(self, job, **kwargs):
        import forte.model.adapter as adapter

        source = self.session.claim_source(job)
        destination = self.session.root / f"{job.job_id}.attempt-{job.attempt}.json"
        attempt_root = self.session.settings.work_dir / job.job_id / f"attempt-{job.attempt}"
        fresh = not attempt_root.exists()
        require(fresh, "trace_existing_attempt_output_forbidden")
        actual_identity = None
        preprocessing_calls = 0
        model_calls = 0
        original_request, original_prepare = self.raw._request, adapter._prepare_media
        original_result = None
        audit_path = None
        status = "processing"
        stage_counts = Counter()
        evidence_sources = []
        projection = {"schema_version": "forte.qwen38-fresh-stage-witness.v1", "job_id": job.job_id,
            "attempt": job.attempt, "source_attempt_id": source["attempt_id"], "content_sha256": job.content_hash,
            "status": status, "fresh_output_directory_created": fresh,
            "wrapper_sha256": self.session.binding["wrapper_sha256"],
            "pipeline_sha256": self.session.binding["pipeline_sha256"],
            "trace_source_archive_sha256": self.session.binding["trace_source_archive"]["sha256"],
            "trace_runtime_image_sha256": self.session.binding["trace_runtime_image_sha256"],
            "trace_binding_sha256": digest(self.session.binding), "model_slots_reserved": len(self.session.rows),
            "result_bytes_changed_by_wrapper": False}
        def flush():
            projection.update(status=status, stage_calls={"decode": stage_counts["decode"],
                "preprocessing": preprocessing_calls, "model": model_calls, "evidence": stage_counts["evidence"]},
                actual_model_identity=actual_identity, evidence_sources=evidence_sources)
            write_json(destination, projection)
        def preprocess(*args, **kw):
            nonlocal preprocessing_calls
            preprocessing_calls += 1
            return original_prepare(*args, **kw)
        async def request(method, endpoint, **kw):
            nonlocal model_calls, actual_identity
            slot = None
            if method == "POST" and endpoint == "/chat/completions":
                proof = self.raw.serving_attestation
                identity = {key: proof.get(key) for key in MODEL_FIELDS} if isinstance(proof, dict) else None
                require(identity == self.session.binding["expected_model_identity"], "trace_actual_model_identity_mismatch")
                require_released_runtime(proof, identity, self.session.release)
                slot = self.session.claim_model_slot(source["attempt_id"])
                model_calls += 1
                actual_identity = identity
                flush()
            try:
                response = await original_request(method, endpoint, **kw)
                if slot is not None:
                    slot.update(status="done" if response[0] == 200 else "failed", http_status=response[0])
                return response
            except BaseException as error:
                if slot is not None:
                    slot.update(status="failed", error_type=type(error).__name__)
                raise
            finally:
                if slot is not None:
                    self.session.flush_slots()
        self.raw._request, adapter._prepare_media = request, preprocess
        flush()
        try:
            with trace_pipeline_stages() as (stage_counts, evidence_sources):
                original_result = await self.delegate.analyze(job, **kwargs)
            status = "done"
            audit_path = original_result.metadata_path
            projection.update(result_sha256=__import__("hashlib").sha256(original_result.json_bytes).hexdigest(),
                audit_sha256=digest(read_json(audit_path)), audit_path=str(audit_path),
                cache_hits={"decode": 0, "model": 0, "evidence": 0, "whole_result": 0},
                cache_scope="fresh unchanged pipeline decoder/provider/evidence invocations; backend attention KV internals not independently measured")
            require(stage_counts["decode"] > 0 and preprocessing_calls > 0 and model_calls > 0
                    and stage_counts["evidence"] > 0, "trace_required_fresh_stage_missing")
            return original_result
        except BaseException as error:
            status = "failed"
            projection["error_type"] = type(error).__name__
            raise
        finally:
            self.raw._request, adapter._prepare_media = original_request, original_prepare
            flush()

    async def aclose(self):
        await self.delegate.aclose()


async def run(binding_path, witness_root, *, once=False):
    from forte.worker import Worker

    settings = WorkerSettings.from_env()
    session = TraceSession(read_json(binding_path), settings, witness_root)
    pipeline = TracedPipeline(settings, session)
    worker = Worker(settings, pipeline=pipeline)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, worker.stop.set)
        except NotImplementedError:
            pass
    try:
        await worker.run(once=once)
    finally:
        await worker.aclose()
        session.flush_slots()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--witness-root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args.binding, args.witness_root, once=args.once))
        return 0
    except Exception as error:
        print(json.dumps({"status": "worker_trace_failed", "error_type": type(error).__name__,
                          "raw_predictions_or_pixels_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
