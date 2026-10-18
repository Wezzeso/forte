"""Environment configuration for the CPU-only public API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def _boolean_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _https_origin(value: str) -> str:
    """Canonicalize an explicit HTTPS origin; reject URL paths and wildcards."""
    if value != value.strip() or any(ord(char) < 33 for char in value) or "*" in value:
        raise ValueError("An explicit HTTPS origin is required")
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("An explicit HTTPS origin without credentials or a path is required")
        port = parsed.port
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        else:
            host = host.encode("idna").decode("ascii")
        return "https://" + host + (f":{port}" if port not in (None, 443) else "")
    except ValueError as exc:
        raise ValueError("An explicit HTTPS origin is required") from exc


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    database_url: str = field(default="", repr=False)
    data_dir: Path = Path("var/forte")
    max_upload_bytes: int = 256 * 1024 * 1024
    multipart_overhead_bytes: int = 64 * 1024
    web_dist: Path = Path("web/dist")
    model_base_url: str = "http://127.0.0.1:8001/v1"
    model_manifest_path: Path = Path("models/manifest.json")
    # Separate candidate profile, never inferred from a model alias.
    model_candidate_launch_config_path: Path | None = None
    model_api_key: str | None = field(default=None, repr=False)
    rules_path: Path = Path("rules/catalog.json")
    worker_heartbeat_file: Path | None = None
    worker_heartbeat_max_age_seconds: float = 30.0
    readiness_timeout_seconds: float = 2.0
    job_timeout_seconds: int = 285
    cors_allowed_origins: tuple[str, ...] = ()
    public_api_base_url: str = ""
    admission_paused: bool = False
    model_native_launch_config_path: Path | None = None
    model_native_role: str | None = None
    model_native_mode: str | None = None

    def __post_init__(self) -> None:
        from forte.native_dispatch import validate_native_selection
        validate_native_selection(self.model_native_launch_config_path, self.model_native_role,
                                  self.model_native_mode, self.model_candidate_launch_config_path,
                                  job_timeout=self.job_timeout_seconds)
        object.__setattr__(self, "data_dir", Path(self.data_dir).resolve())
        object.__setattr__(self, "web_dist", Path(self.web_dist).resolve())
        object.__setattr__(self, "model_manifest_path", Path(self.model_manifest_path).resolve())
        if self.model_candidate_launch_config_path is not None:
            object.__setattr__(self, "model_candidate_launch_config_path",
                               Path(self.model_candidate_launch_config_path).resolve())
        if self.model_native_launch_config_path is not None:
            object.__setattr__(self, "model_native_launch_config_path",
                               Path(self.model_native_launch_config_path).resolve())
        object.__setattr__(self, "rules_path", Path(self.rules_path).resolve())
        object.__setattr__(self, "cors_allowed_origins", tuple(dict.fromkeys(_https_origin(origin) for origin in self.cors_allowed_origins)))
        if self.public_api_base_url:
            object.__setattr__(self, "public_api_base_url", _https_origin(self.public_api_base_url))
        if not self.database_url:
            object.__setattr__(self, "database_url", "sqlite:///" + (self.data_dir / "jobs.db").as_posix())
        if self.worker_heartbeat_file is None:
            object.__setattr__(self, "worker_heartbeat_file", self.data_dir / "worker-heartbeat.json")
        else:
            object.__setattr__(self, "worker_heartbeat_file", Path(self.worker_heartbeat_file).resolve())
        if (type(self.max_upload_bytes) is not int or self.max_upload_bytes < 1
                or type(self.multipart_overhead_bytes) is not int or self.multipart_overhead_bytes < 1024):
            raise ValueError("Upload limits must be positive")
        if self.job_timeout_seconds < 1 or self.worker_heartbeat_max_age_seconds <= 0:
            raise ValueError("Job and readiness timeouts must be positive")
        if type(self.admission_paused) is not bool:
            raise ValueError("admission_paused must be a boolean")

    @property
    def upload_root(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def result_root(self) -> Path:
        return self.data_dir / "results"

    @classmethod
    def from_env(cls) -> "Settings":
        key_text = os.getenv("FORTE_API_KEYS", os.getenv("FORTE_API_KEY", ""))
        key_file = os.getenv("FORTE_API_KEYS_FILE")
        if key_file:
            key_text += "," + Path(key_file).read_text(encoding="utf-8")
        keys = tuple(dict.fromkeys(key.strip() for key in key_text.replace("\n", ",").split(",") if key.strip()))
        data_dir = Path(os.getenv("FORTE_DATA_DIR", "var/forte")).resolve()
        default_db = "sqlite:///" + (data_dir / "jobs.db").as_posix()
        return cls(
            api_keys=keys,
            database_url=os.getenv("FORTE_DATABASE_URL", default_db),
            data_dir=data_dir,
            max_upload_bytes=int(os.getenv("FORTE_MAX_UPLOAD_BYTES", str(256 * 1024 * 1024))),
            web_dist=Path(os.getenv("FORTE_WEB_DIST", "web/dist")),
            model_base_url=os.getenv("FORTE_MODEL_BASE_URL", "http://127.0.0.1:8001/v1"),
            model_manifest_path=Path(os.getenv("FORTE_MODEL_MANIFEST", "models/manifest.json")),
            model_candidate_launch_config_path=(Path(os.environ["FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG"])
                if os.getenv("FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG") else None),
            model_native_launch_config_path=(Path(os.environ["FORTE_MODEL_NATIVE_LAUNCH_CONFIG"])
                if os.getenv("FORTE_MODEL_NATIVE_LAUNCH_CONFIG") else None),
            model_native_role=os.getenv("FORTE_MODEL_NATIVE_ROLE") or None,
            model_native_mode=os.getenv("FORTE_MODEL_NATIVE_MODE") or None,
            model_api_key=os.getenv("FORTE_MODEL_API_KEY") or None,
            rules_path=Path(os.getenv("FORTE_RULES_PATH", "rules/catalog.json")),
            worker_heartbeat_file=Path(os.environ["FORTE_WORKER_HEARTBEAT_FILE"]) if os.getenv("FORTE_WORKER_HEARTBEAT_FILE") else None,
            worker_heartbeat_max_age_seconds=float(os.getenv("FORTE_WORKER_HEARTBEAT_MAX_AGE_SECONDS", "30")),
            readiness_timeout_seconds=float(os.getenv("FORTE_READINESS_TIMEOUT_SECONDS", "2")),
            job_timeout_seconds=int(os.getenv("FORTE_JOB_TIMEOUT_SECONDS", "285")),
            cors_allowed_origins=tuple(origin.strip() for origin in os.getenv("FORTE_CORS_ALLOWED_ORIGINS", "").split(",") if origin.strip()),
            public_api_base_url=os.getenv("FORTE_PUBLIC_API_BASE_URL", "").strip(),
            admission_paused=_boolean_env("FORTE_ADMISSION_PAUSED", False),
        )
