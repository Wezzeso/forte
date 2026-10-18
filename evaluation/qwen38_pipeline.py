"""Bounded private full-pipeline quality runner; prepare is CPU/metadata only.

Execute is operator-invoked only after exact root release and A09 reservation.
This direct runner does not measure upload/network/usable-browser latency.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace

from evaluation.io import digest, read_json, write_json
from evaluation.qwen38_comparison import MODEL_FIELDS, validate_plan
from evaluation.qwen38_server_export import export_artifacts, write_private_new
from forte.config import WorkerSettings
from forte.model.adapter import LocalModelConfig, LocalModelProvider
from forte.pipeline import AnalysisPipeline, _provider_configuration
from forte.video.probe import content_sha256, probe_video
from forte.video.sampling import coverage_windows
from training.qwen38_preflight import FIXED_STOP_UTC, validate_attempt_reservation


WORKER_FIELDS = ("window_seconds", "stride_seconds", "sample_fps", "max_frames_per_window",
    "max_concurrent_jobs", "max_concurrent_clips", "max_inflight_per_job", "compact_frame_ids",
    "concise_observations", "staged_screening", "max_refinement_passes", "job_timeout_seconds",
    "trusted_scene_context")


def file_sha(path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _require(value, code):
    if not value:
        raise ValueError(code)


def _now():
    return datetime.now(timezone.utc)


def _utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "evaluation_release_timezone_required")
    return parsed.astimezone(timezone.utc)


def prepare(comparison: dict, spec: dict, *, role_config_sha256: str) -> dict:
    """Probe/hash only the six explicitly eligible validation sources; no model."""
    records, expected = validate_plan(comparison)
    arm = spec["arm"]
    _require(arm in comparison["arms"] and len(records) == 6
        and all(row["split"] == "validation" for row in records.values()), "evaluation_exact_six_validation_required")
    frozen_worker = comparison["recipe"].get("worker_settings", comparison["recipe"]["sampling"])
    _require(set(spec["worker"]) == set(WORKER_FIELDS)
        and frozen_worker == spec["worker"], "evaluation_frozen_worker_recipe_mismatch")
    selected = {row["sample_id"]: row for row in expected.values() if row["arm"] == arm}
    _require(len(selected) == 6 and all(row["repeat_index"] == 0 for row in selected.values()), "evaluation_one_frozen_source_attempt_each")
    sources = {row["sample_id"]: row for row in spec["sources"]}
    _require(set(sources) == set(records), "evaluation_source_map_mismatch")
    cases, slots = [], []
    for sid, record in records.items():
        source = sources[sid]
        _require(source["content_sha256"] == record["content_sha256"] and source["split"] == "validation", "evaluation_source_role_mismatch")
        path = Path(source["owned_path"])
        _require(path.is_file() and not path.is_symlink(), "evaluation_source_unavailable")
        asset = probe_video(path, job_id="prepare-" + record["content_sha256"][:20])
        _require(asset.content_hash == record["content_sha256"], "evaluation_source_hash_changed")
        worker = spec["worker"]
        windows = coverage_windows(asset.duration_sec, window_sec=worker["window_seconds"],
            stride_sec=worker["stride_seconds"], sample_fps=worker["sample_fps"],
            max_frames_per_window=worker["max_frames_per_window"], max_windows=512, max_total_frames=8192)
        logical_upper = len(windows) * (2 if worker["staged_screening"] else 1) + worker["max_refinement_passes"]
        retries = comparison["recipe"].get("provider_max_attempts", 2)
        _require(retries in (1, 2, 3), "evaluation_provider_retry_budget_invalid")
        upper = logical_upper * retries
        source_attempt = selected[sid]
        cases.append({**source_attempt, "owned_path": str(path.resolve()), "source_bytes": path.stat().st_size,
            "duration_sec": asset.duration_sec, "coarse_windows": len(windows),
            "maximum_pipeline_inferences": logical_upper, "maximum_model_requests": upper})
        for index in range(upper):
            slots.append({"attempt_id": f"{source_attempt['attempt_id']}-model-{index:04d}",
                "source_attempt_id": source_attempt["attempt_id"], "content_sha256": record["content_sha256"],
                "split": "validation", "slot_index": index})
    plan = {"schema_version": "forte.qwen38-attempt-plan.v1", "kind": "fullpipeline_quality",
        "phase": "evaluation", "arm": arm, "comparison_plan_sha256": digest(comparison),
        "recipe_sha256": comparison["recipe_sha256"], "recipe": comparison["recipe"],
        "expected_model_identity": comparison["arms"][arm], "worker": spec["worker"],
        "provider": spec["provider"], "role_config_sha256": role_config_sha256,
        "runtime_bindings": spec["runtime_bindings"], "cases": cases,
        "source_attempts": [{key: row[key] for key in ("attempt_id", "arm", "sample_id", "content_sha256", "split", "repeat_index")} for row in cases],
        "attempts": slots, "prepared_at_utc": _now().isoformat(),
        "call_bound_policy": "(N coarse + N possible primary when staged + maximum refinement passes) times frozen provider max_attempts; every attempted generation HTTP POST consumes one slot",
        "source_attempts_required": 6, "maximum_private_model_requests": len(slots),
        "quality_model_calls_during_prepare": 0, "final_check_used": False,
        "upload_start_or_browser_latency_measured": False}
    return plan


def validate_execution(plan: dict, release: dict, reservation: dict, *, plan_sha256: str,
                       release_sha256: str, now=None) -> None:
    now = now or _now()
    _require(plan.get("kind") == "fullpipeline_quality" and plan.get("phase") == "evaluation"
        and len(plan["source_attempts"]) == 6 and all(row["split"] == "validation" for row in plan["attempts"]),
        "evaluation_plan_scope_invalid")
    _require(release.get("schema_version") == "forte.qwen38-root-release.v1"
        and release.get("released_by_root") is True and release.get("phase") == "evaluation"
        and release.get("attempt_plan_sha256") == plan_sha256
        and release.get("maximum_attempts") == len(plan["attempts"])
        and release.get("expected_model_identity") == plan["expected_model_identity"]
        and release.get("config_sha256") == plan["role_config_sha256"]
        and release.get("exclusive_idle_gpu_handoff_verified") is True,
        "evaluation_exact_root_release_required")
    _require(_utc(release["not_before_utc"]) <= now < _utc(release["expires_at_utc"]) <= FIXED_STOP_UTC,
             "evaluation_root_release_expired")
    _require(type(release.get("maximum_execution_seconds")) is int and release["maximum_execution_seconds"] > 0,
             "evaluation_wall_bound_missing")
    validate_attempt_reservation(reservation, plan, plan_sha256=plan_sha256, phase="evaluation", now=now)
    _require(reservation.get("release_sha256") == release_sha256
        and reservation.get("config_sha256") == plan["role_config_sha256"]
        and all(release.get(key) == reservation.get(key) == plan["runtime_bindings"][key]
            for key in ("source_archive_sha256", "runtime_image_sha256", "base_manifest_sha256")),
        "evaluation_release_reservation_runtime_mismatch")


def require_released_runtime(attestation, expected_identity, runtime_bindings):
    """Candidate readiness pins must match the exact released GPU artifacts."""
    _require(isinstance(attestation, dict) and attestation.get("model_files_verified") is True,
             "evaluation_actual_runtime_unverified")
    _require(attestation.get("model_manifest_sha256") == runtime_bindings["base_manifest_sha256"],
             "evaluation_actual_base_manifest_mismatch")
    if expected_identity["model_id"] == "Qwen/Qwen3.8-27B":
        _require(attestation.get("source_sha256") == runtime_bindings["source_archive_sha256"]
            and attestation.get("image_sha256") == runtime_bindings["runtime_image_sha256"],
            "evaluation_actual_source_or_image_mismatch")


class ReservedProvider:
    """Consume an exact reserved slot before every bounded raw provider call."""
    def __init__(self, delegate, slots, *, guard, progress_path, runtime_bindings=None):
        self.delegate, self.slots = delegate, slots
        self.config = delegate.config
        self.guard, self.progress_path = guard, progress_path
        self.rows = [{**row, "status": "not_attempted"} for row in slots]
        self.calls = 0
        self.actual_identity = None
        self.runtime_bindings = runtime_bindings
        self.original_request = delegate._request
        delegate._request = self._request

    async def _request(self, method, endpoint, **kwargs):
        if method != "POST" or endpoint != "/chat/completions":
            return await self.original_request(method, endpoint, **kwargs)
        self.guard()
        if self.calls >= len(self.rows):
            raise RuntimeError("evaluation_reserved_call_budget_exhausted")
        index = self.calls
        self.calls += 1
        row = self.rows[index]
        row.update(status="attempted_unfinished", started_at_utc=_now().isoformat())
        write_json(self.progress_path, {"records": self.rows})
        try:
            expected = self.guard()
            attestation = self.delegate.serving_attestation
            actual = {key: attestation.get(key) for key in MODEL_FIELDS} if isinstance(attestation, dict) else None
            _require(actual == expected, "evaluation_actual_provider_identity_mismatch")
            if self.runtime_bindings is not None:
                require_released_runtime(attestation, expected, self.runtime_bindings)
            self.actual_identity = actual
            response = await self.original_request(method, endpoint, **kwargs)
            row.update(status="done" if response[0] == 200 else "failed", finished_at_utc=_now().isoformat(),
                       serving_attestation_sha256=digest(attestation), http_status=response[0])
            return response
        except BaseException as error:
            row.update(status="failed", finished_at_utc=_now().isoformat(), error_type=type(error).__name__)
            raise
        finally:
            write_json(self.progress_path, {"records": self.rows})

    async def infer(self, clip, **kwargs):
        self.guard()
        return await self.delegate.infer(clip, **kwargs)

    async def aclose(self):
        await self.delegate.aclose()


class ManifestBoundBaselineProvider(LocalModelProvider):
    """Extend the baseline's genuine manifest attestation with its bound ID."""
    def __init__(self, config, manifest_path):
        from forte.model.manifest import load_manifest
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        super().__init__(config)

    async def check_ready(self, **kwargs):
        await super().check_ready(**kwargs)
        proof = self.serving_attestation
        _require(proof["model_manifest_sha256"] == file_sha(self.manifest_path)
            and proof["model_revision"] == self.manifest["revision"]
            and self.config.model == self.manifest["served_model_name"]
            and self.manifest["selected_adapter"] is None, "evaluation_baseline_manifest_identity_mismatch")
        self._serving_attestation.update(model_id=self.manifest["model_id"],
            adapter_sha256=None, backend="vllm")


@contextmanager
def trace_pipeline_stages():
    """Count actual unchanged decoder/assembler invocations and evidence joins."""
    import forte.pipeline as module
    import forte.model.adapter as adapter
    original_decode, original_assemble = module.decode_frames, module.assemble_result
    original_prepare = adapter._prepare_media
    counts, evidence_sources = Counter(), []
    lock = threading.Lock()
    def decode(*args, **kwargs):
        with lock:
            counts["decode"] += 1
        return original_decode(*args, **kwargs)
    def assemble(*args, **kwargs):
        with lock:
            counts["evidence"] += 1
        result = original_assemble(*args, **kwargs)
        for frame in result.frames:
            evidence_sources.append({**asdict(frame), "original_jpeg_path": str(frame.original_jpeg_path)})
        return result
    def prepare(*args, **kwargs):
        with lock:
            counts["preprocessing"] += 1
        return original_prepare(*args, **kwargs)
    module.decode_frames, module.assemble_result = decode, assemble
    adapter._prepare_media = prepare
    try:
        yield counts, evidence_sources
    finally:
        module.decode_frames, module.assemble_result = original_decode, original_assemble
        adapter._prepare_media = original_prepare


def make_settings(plan, output):
    return WorkerSettings(database_url="sqlite://", data_dir=output / "runtime",
        model_base_url=plan["provider"]["base_url"], model_name=plan["provider"]["served_model_name"],
        model_api_key=os.environ.get(plan["provider"].get("api_key_env", "VLLM_API_KEY")),
        worker_id="private-quality-evaluator", heartbeat_file=output / "heartbeat",
        rules_path=Path(plan["provider"].get("rules_path", "rules/catalog.json")),
        taxonomy_path=Path(plan["provider"].get("taxonomy_path", "annotations/taxonomy.v1.json")),
        model_manifest_path=Path(plan["provider"].get("baseline_manifest_path", "models/manifest.json")),
        **plan["worker"])


def make_provider(plan, settings):
    provider = plan["provider"]
    if plan["arm"] == "qwen38_untuned":
        from serving.qwen38_untuned_provider import UntunedModelProvider
        raw = UntunedModelProvider(provider["launch_config"], media_root=settings.work_dir,
                                  base_url=settings.model_base_url, api_key=settings.model_api_key)
    elif plan["arm"] == "qwen38_tuned":
        settings = replace(settings, model_candidate_launch_config_path=Path(provider["launch_config"]))
        raw = LocalModelProvider(_provider_configuration(settings))
    else:
        raw = ManifestBoundBaselineProvider(LocalModelConfig(media_root=settings.work_dir,
            base_url=settings.model_base_url, model=settings.model_name,
            expected_revision=plan["expected_model_identity"]["model_revision"], api_key=settings.model_api_key,
            max_frames=settings.max_frames_per_window, max_tokens=plan["recipe"]["generation"]["max_new_tokens"],
            seed=plan["recipe"]["generation"]["seed"], request_timeout_sec=settings.job_timeout_seconds), settings.model_manifest_path)
    raw.config = replace(raw.config, max_attempts=plan["recipe"].get("provider_max_attempts", 2), readiness_cache_sec=0)
    _require(raw.config.max_tokens == plan["recipe"]["generation"]["max_new_tokens"]
        and raw.config.seed == plan["recipe"]["generation"]["seed"], "evaluation_generation_budget_mismatch")
    return raw


async def execute(plan_path, release_path, reservation_path, output):
    plan, release, reservation = read_json(plan_path), read_json(release_path), read_json(reservation_path)
    plan_sha, release_sha = file_sha(plan_path), file_sha(release_path)
    validate_execution(plan, release, reservation, plan_sha256=plan_sha, release_sha256=release_sha)
    # Claim the reservation once even if another output directory is requested.
    write_private_new(Path(str(plan_path) + ".execution-claim.json"), {"plan_sha256": plan_sha, "claimed_at_utc": _now().isoformat()})
    output = Path(output)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    stop_at = min(time.monotonic() + release["maximum_execution_seconds"],
                  time.monotonic() + (_utc(release["expires_at_utc"]) - _now()).total_seconds())
    source_rows = [{**attempt, "comparison_plan_sha256": plan["comparison_plan_sha256"],
        "recipe_sha256": plan["recipe_sha256"], "expected_model_identity": plan["expected_model_identity"],
        "status": "not_attempted", "public_contract_validated": False,
        "upload_start_to_full_result_ms": None, "timing_origin": None,
        "upload_start_to_usable_result_ms": None, "ui_timing_origin": None} for attempt in plan["source_attempts"]]
    all_slots = [{**slot, "status": "not_attempted"} for slot in plan["attempts"]]
    write_json(output / "source-terminals.json", {"records": source_rows})
    write_json(output / "model-slot-terminals.json", {"records": all_slots})
    settings = make_settings(plan, output)
    for source_index, case in enumerate(plan["cases"]):
        attempt = {key: case[key] for key in ("attempt_id", "arm", "sample_id", "content_sha256", "split", "repeat_index")}
        row = {**attempt, "comparison_plan_sha256": plan["comparison_plan_sha256"], "recipe_sha256": plan["recipe_sha256"],
            "expected_model_identity": plan["expected_model_identity"], "status": "not_attempted", "public_contract_validated": False,
            "upload_start_to_full_result_ms": None, "timing_origin": None,
            "upload_start_to_usable_result_ms": None, "ui_timing_origin": None}
        slots = [slot for slot in plan["attempts"] if slot["source_attempt_id"] == case["attempt_id"]]
        folder = output / case["attempt_id"]
        folder.mkdir(mode=0o700)
        def guard():
            validate_execution(plan, release, reservation, plan_sha256=plan_sha, release_sha256=release_sha)
            _require(time.monotonic() < stop_at, "evaluation_execution_wall_bound")
            return plan["expected_model_identity"]
        budget = pipeline = None
        started = time.monotonic()
        try:
            guard()
            source = Path(case["owned_path"])
            _require(source.is_file() and source.stat().st_size == case["source_bytes"]
                and content_sha256(source) == case["content_sha256"], "evaluation_source_changed")
            job_id = hashlib.sha256(case["attempt_id"].encode()).hexdigest()[:32]
            source_copy = folder / "source.mov"
            shutil.copyfile(source, source_copy)
            source_copy.chmod(0o600)
            job = SimpleNamespace(job_id=job_id, attempt=1, owned_path=str(source_copy), content_hash=case["content_sha256"])
            raw = make_provider(plan, settings)
            budget = ReservedProvider(raw, slots, guard=guard, progress_path=folder / "model-slots.json",
                                      runtime_bindings=plan["runtime_bindings"])
            pipeline = AnalysisPipeline(settings, provider=budget)
            if plan["arm"] == "qwen38_untuned":
                pipeline.audit_provenance.update(diagnostic_only=True, arm="qwen38_untuned", trained=False)
            async def stage(name):
                write_json(folder / "stage.json", {"stage": name})
            with trace_pipeline_stages() as (counts, evidence_sources):
                result = await pipeline.analyze(job, stage=stage,
                    deadline=min(stop_at, time.monotonic() + settings.job_timeout_seconds), cancel_event=threading.Event())
            result_path = folder / "result.json"
            with result_path.open("xb") as handle:
                result_path.chmod(0o600)
                handle.write(result.json_bytes)
            parsed = json.loads(result.json_bytes)
            row.update(status="done", public_contract_validated=True, job_id=job_id,
                model_identity=budget.actual_identity, complete_result=parsed,
                verdict=parsed["verdict"], violations=parsed["violations"],
                pipeline_start_to_result_ms=(time.monotonic() - started) * 1000)
            witness = {"schema_version": "forte.qwen38-fresh-stage-witness.v1", "status": "done",
                "job_id": job_id, "source_attempt_id": case["attempt_id"],
                "result_sha256": hashlib.sha256(result.json_bytes).hexdigest(),
                "audit_sha256": digest(read_json(result.metadata_path)), "actual_model_identity": budget.actual_identity,
                "stage_calls": {"decode": counts["decode"], "preprocessing": counts["preprocessing"],
                    "model": budget.calls, "evidence": counts["evidence"]},
                "cache_hits": {"decode": 0, "model": 0, "evidence": 0, "whole_result": 0},
                "fresh_output_directory_created": True, "evidence_sources": evidence_sources}
            binding = {"attempt_id": case["attempt_id"], "job_id": job_id, "content_sha256": case["content_sha256"],
                "conditions_sha256": plan_sha, "recipe_sha256": plan["recipe_sha256"],
                "expected_model_identity": plan["expected_model_identity"]}
            write_private_new(folder / "fresh-stage-witness.json", witness)
            write_private_new(folder / "server-binding.json", binding)
            try:
                export_artifacts(audit_path=result.metadata_path, result_path=result_path, binding=binding,
                    witness=witness, output=folder / "server-receipt.json")
                row.update(server_receipt_verified=True, server_receipt_sha256=file_sha(folder / "server-receipt.json"))
            except Exception as error:
                row.update(server_receipt_verified=False, server_receipt_error_type=type(error).__name__)
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__)
            if isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
        finally:
            if pipeline is not None:
                await pipeline.aclose()
            updates = {slot["attempt_id"]: slot for slot in budget.rows} if budget is not None else {}
            all_slots = [updates.get(slot["attempt_id"], slot) for slot in all_slots]
            source_rows[source_index] = row
            write_json(folder / "terminal.json", row)
            write_json(output / "source-terminals.json", {"records": source_rows})
            write_json(output / "model-slot-terminals.json", {"records": all_slots})
    counts = Counter(row["status"] for row in source_rows)
    summary = {"source_attempts_required": len(plan["cases"]), "source_terminal_counts": dict(counts),
        "reserved_private_model_slots": len(plan["attempts"]), "model_slot_terminal_counts": dict(Counter(row["status"] for row in all_slots)),
        "private_artifacts_retained_on_execution_host": True, "raw_predictions_or_pixels_printed": False,
        "upload_start_or_usable_ui_latency_claim": False, "heldout_accuracy_claim": False}
    write_private_new(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("comparison-plan", "spec", "role-config", "output"):
        prep.add_argument("--" + name, required=True, type=Path)
    run = sub.add_parser("execute")
    for name in ("plan", "release", "reservation", "output"):
        run.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            plan = prepare(read_json(args.comparison_plan), read_json(args.spec), role_config_sha256=file_sha(args.role_config))
            write_private_new(args.output, plan)
            result = {"status": "prepared_not_executed", "source_attempts": 6,
                "maximum_private_model_requests": len(plan["attempts"]), "model_calls_made": 0}
        else:
            result = asyncio.run(execute(args.plan, args.release, args.reservation, args.output))
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "fullpipeline_evaluation_failed", "error_type": type(error).__name__,
                          "raw_predictions_or_pixels_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
