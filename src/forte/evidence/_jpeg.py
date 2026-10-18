"""Bounded validation of original decoder-produced JPEGs.

This module never edits, enhances, resizes, or draws on evidence. Validation is
performed once in the worker before the immutable public projection is stored.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


class EvidenceError(ValueError):
    """Evidence cannot be published safely as a completed response."""


def read_validated_jpeg(
    path: Path | str,
    *,
    max_bytes: int,
    max_pixels: int,
    expected_width: int,
    expected_height: int,
) -> bytes:
    """Read within a hard byte bound and fully decode an unmodified JPEG.

    Header inspection precedes pixel allocation. Neither paths nor image
    contents are included in errors, because they can identify private media.
    """
    try:
        with Path(path).open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise EvidenceError("Evidence JPEG is unavailable") from exc
    if not data or len(data) > max_bytes:
        raise EvidenceError("Evidence JPEG exceeds the configured byte limit")
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise EvidenceError("Evidence must be a complete original JPEG")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "JPEG":
                raise EvidenceError("Evidence image is not a JPEG")
            if image.width * image.height > max_pixels:
                raise EvidenceError("Evidence JPEG exceeds the configured pixel limit")
            if image.size != (expected_width, expected_height):
                raise EvidenceError("Evidence dimensions differ from the frame reference")
            image.load()
    except EvidenceError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise EvidenceError("Evidence JPEG cannot be fully decoded") from exc
    return data
