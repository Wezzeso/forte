"""Worker configuration. Secrets are read at runtime and are never logged."""

from __future__ import annotations

import os
import math
import socket
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


def _positive(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_integer(name: str, default: int) -> int:
    value = _positive(name, default)
    if not value.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _boolean(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


@dataclass(frozen=True)
class WorkerSettings:
    database_url: str = field(repr=False)
    data_dir: Path
    model_base_url: str
    model_name: str
    model_api_key: str | None = field(repr=False)
    worker_id: str
    heartbeat_file: Path
    rules_path: Path
    taxonomy_path: Path
    model_manifest_path: Path
    model_candidate_launch_config_path: Path | None = None
    trusted_scene_context: str | None = None
    poll_seconds: float = 1.0
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 8.0
    job_timeout_seconds: float = 285.0
    window_seconds: float = 8.0
    stride_seconds: float = 4.0
    sample_fps: float = 2.0
    max_frames_per_window: int = 16
    max_concurrent_jobs: int = 4
    max_concurrent_clips: int = 2
    max_inflight_per_job: int = 1
    compact_frame_ids: bool = False
    concise_observations: bool = False
    staged_screening: bool = False
    max_refinement_passes: int = 8
    model_native_launch_config_path: Path | None = None
    model_native_role: str | None = None
    model_native_mode: str | None = None

    def __post_init__(self) -> None:
        from forte.native_dispatch import validate_native_selection
        validate_native_selection(self.model_native_launch_config_path, self.model_native_role,
                                  self.model_native_mode, self.model_candidate_launch_config_path,
                                  job_timeout=self.job_timeout_seconds)
        for name in ("poll_seconds", "lease_seconds", "heartbeat_seconds", "job_timeout_seconds",
                     "window_seconds", "stride_seconds", "sample_fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_frames_per_window", "max_concurrent_jobs", "max_concurrent_clips",
                     "max_inflight_per_job", "max_refinement_passes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("compact_frame_ids", "concise_observations", "staged_screening"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.max_inflight_per_job > self.max_concurrent_clips:
            raise ValueError("max_inflight_per_job cannot exceed max_concurrent_clips")
        if self.heartbeat_seconds >= self.lease_seconds / 2:
            raise ValueError("worker heartbeat must be less than half the lease duration")
        if self.stride_seconds > self.window_seconds:
            raise ValueError("stride cannot leave uncovered video intervals")
        if self.trusted_scene_context not in (None, "bank_cash_workstation"):
            raise ValueError("unsupported trusted deployment context")

    @classmethod
    def from_env(cls) -> "WorkerSettings":
        data_dir = Path(os.environ.get("FORTE_DATA_DIR", "var/forte")).resolve()
        default_database = "sqlite:///" + (data_dir / "jobs.db").as_posix()
        settings = cls(
            database_url=os.environ.get("FORTE_DATABASE_URL", default_database),
            data_dir=data_dir,
            model_base_url=os.environ.get("FORTE_MODEL_BASE_URL", "http://127.0.0.1:8001/v1"),
            model_name=os.environ.get("FORTE_MODEL_NAME", "Qwen3-VL-8B-Instruct"),
            model_api_key=os.environ.get("FORTE_MODEL_API_KEY") or None,
            worker_id=os.environ.get("FORTE_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"),
            heartbeat_file=Path(os.environ.get("FORTE_WORKER_HEARTBEAT_FILE", str(data_dir / "worker-heartbeat.json"))).resolve(),
            rules_path=Path(os.environ.get("FORTE_RULES_PATH", "rules/catalog.json")).resolve(),
            taxonomy_path=Path(os.environ.get("FORTE_TAXONOMY_PATH", "annotations/taxonomy.v1.json")).resolve(),
            model_manifest_path=Path(os.environ.get("FORTE_MODEL_MANIFEST", "models/manifest.json")).resolve(),
            model_candidate_launch_config_path=(Path(os.environ["FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG"]).resolve()
                if os.environ.get("FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG") else None),
            model_native_launch_config_path=(Path(os.environ["FORTE_MODEL_NATIVE_LAUNCH_CONFIG"]).resolve()
                if os.environ.get("FORTE_MODEL_NATIVE_LAUNCH_CONFIG") else None),
            model_native_role=os.environ.get("FORTE_MODEL_NATIVE_ROLE") or None,
            model_native_mode=os.environ.get("FORTE_MODEL_NATIVE_MODE") or None,
            trusted_scene_context=os.environ.get("FORTE_TRUSTED_SCENE_CONTEXT") or None,
            poll_seconds=_positive("FORTE_WORKER_POLL_SECONDS", 1.0),
            lease_seconds=_positive("FORTE_LEASE_SECONDS", 30.0),
            heartbeat_seconds=_positive("FORTE_WORKER_HEARTBEAT_SECONDS", 8.0),
            job_timeout_seconds=_positive("FORTE_JOB_TIMEOUT_SECONDS", 285.0),
            window_seconds=_positive("FORTE_WINDOW_SECONDS", 8.0),
            stride_seconds=_positive("FORTE_STRIDE_SECONDS", 4.0),
            sample_fps=_positive("FORTE_SAMPLE_FPS", 2.0),
            max_frames_per_window=_positive_integer("FORTE_MAX_FRAMES_PER_WINDOW", 16),
            max_concurrent_jobs=_positive_integer("FORTE_MAX_CONCURRENT_JOBS", 4),
            max_concurrent_clips=_positive_integer("FORTE_MAX_CONCURRENT_CLIPS", 2),
            max_inflight_per_job=_positive_integer("FORTE_MAX_INFLIGHT_PER_JOB", 1),
            compact_frame_ids=_boolean("FORTE_COMPACT_FRAME_IDS", False),
            concise_observations=_boolean("FORTE_CONCISE_OBSERVATIONS", False),
            staged_screening=_boolean("FORTE_STAGED_SCREENING", False),
            max_refinement_passes=_positive_integer("FORTE_MAX_REFINEMENT_PASSES", 8),
        )
        return settings

    @property
    def result_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

