"""Bounded original JPEG evidence and immutable completed-result assembly."""

from ._jpeg import EvidenceError
from .assembly import AssemblyResult, assemble_result
from .types import EventEvidenceMetadata, EvidenceConfig, EvidenceFrameMetadata, EvidenceStats

__all__ = [
    "AssemblyResult", "EventEvidenceMetadata", "EvidenceConfig", "EvidenceError", "EvidenceFrameMetadata",
    "EvidenceStats", "assemble_result",
]
