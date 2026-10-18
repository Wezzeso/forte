"""Immutable worker-side evidence metadata, bounds, and measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    """Hard resource bounds, independent of public response schemas."""

    max_events: int = 256
    max_frame_bytes: int = 8 * 1024 * 1024
    max_total_jpeg_bytes: int = 64 * 1024 * 1024
    max_payload_bytes: int = 96 * 1024 * 1024
    max_jpeg_pixels: int = 40_000_000
    max_description_bytes: int = 32 * 1024
    max_label_bytes: int = 1024
    timestamp_tolerance_sec: float = 1e-6

    def __post_init__(self) -> None:
        for value in (
            self.max_events,
            self.max_frame_bytes,
            self.max_total_jpeg_bytes,
            self.max_payload_bytes,
            self.max_jpeg_pixels,
            self.max_description_bytes,
            self.max_label_bytes,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Evidence resource bounds must be positive integers")
        if not 0 <= self.timestamp_tolerance_sec <= 0.001:
            raise ValueError("Evidence timestamp tolerance must be between zero and 1 ms")


@dataclass(frozen=True, slots=True)
class EvidenceFrameMetadata:
    frame_id: str
    timestamp_sec: float
    original_jpeg_path: Path
    width: int
    height: int
    sha256: str
    jpeg_bytes: int


@dataclass(frozen=True, slots=True)
class EventEvidenceMetadata:
    event_id: str
    event_type_id: str
    rule_id: str
    anchor_frame_id: str
    evidence_frame_ids: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceStats:
    event_count: int
    unique_frame_count: int
    jpeg_bytes: int
    payload_bytes: int
    extraction_sec: float
    encoding_sec: float
    description_sec: float
    validation_sec: float
    total_sec: float
