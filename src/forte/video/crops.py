"""Local adaptive views and inexpensive motion cues; originals stay unchanged."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from forte.shared.schemas import FrameRef


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("Crop coordinates must be finite")
        if not (0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1):
            raise ValueError("Crop coordinates must define a nonempty normalized box")

    @property
    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        return (math.floor(self.left * width), math.floor(self.top * height),
                math.ceil(self.right * width), math.ceil(self.bottom * height))


FULL_SCENE = NormalizedBox(0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class CropView:
    frame_ref: FrameRef
    source_frame_id: str
    box: NormalizedBox
    is_full_scene: bool


@dataclass(frozen=True, slots=True)
class MotionCue:
    start_sec: float
    end_sec: float
    score: float
    region: NormalizedBox


def _expand(box: NormalizedBox, margin: float, min_area: float) -> NormalizedBox:
    width = box.right - box.left
    height = box.bottom - box.top
    width = min(1.0, width + 2 * margin)
    height = min(1.0, height + 2 * margin)
    if width * height < min_area:
        scale = math.sqrt(min_area / (width * height))
        width, height = min(1.0, width * scale), min(1.0, height * scale)
        # A very thin region can hit one boundary before reaching minimum area.
        if width * height < min_area:
            width = max(width, min_area / height)
            height = max(height, min_area / width)
    left = min(max(0.0, (box.left + box.right - width) / 2), 1 - width)
    top = min(max(0.0, (box.top + box.bottom - height) / 2), 1 - height)
    return NormalizedBox(left, top, left + width, top + height)


def adaptive_crops(
    regions: Sequence[NormalizedBox] = (),
    *,
    context_margin: float = 0.12,
    min_area_fraction: float = 0.20,
    max_crops: int = 2,
) -> list[NormalizedBox]:
    """Always retain global context, then add expanded content-derived regions.

    Coordinates come from local detection, motion, or human work-area selection;
    this function never assumes a fixed camera viewpoint. ``max_crops`` counts
    extra views, so zero still returns the full scene.
    """
    if not math.isfinite(context_margin) or not 0 <= context_margin <= 1:
        raise ValueError("context_margin must be in [0, 1]")
    if not math.isfinite(min_area_fraction) or not 0 < min_area_fraction <= 1:
        raise ValueError("min_area_fraction must be in (0, 1]")
    if isinstance(max_crops, bool) or not isinstance(max_crops, int) or max_crops < 0:
        raise ValueError("max_crops cannot be negative")
    result = [FULL_SCENE]
    for region in regions:
        if len(result) - 1 >= max_crops:
            break
        expanded = _expand(region, context_margin, min_area_fraction)
        if expanded not in result:
            result.append(expanded)
    return result


def materialize_crops(
    frame: FrameRef,
    output_dir: str | Path,
    regions: Sequence[NormalizedBox] = (),
    *,
    context_margin: float = 0.12,
    min_area_fraction: float = 0.20,
    max_crops: int = 2,
    jpeg_quality: int = 92,
) -> list[CropView]:
    """Write optional local detail JPEGs and retain the unchanged full-scene ref."""
    from PIL import Image
    from forte.shared.schemas import FrameRef

    if not 70 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must preserve detail (70..100)")
    boxes = adaptive_crops(regions, context_margin=context_margin,
                           min_area_fraction=min_area_fraction, max_crops=max_crops)
    views = [CropView(frame, frame.frame_id, FULL_SCENE, True)]
    if len(boxes) == 1:
        return views
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # IDs are hashed to avoid introducing source path characters into filenames.
    import hashlib
    with Image.open(frame.path) as source:
        if source.size != (frame.width, frame.height):
            raise ValueError("Decoded frame dimensions differ from FrameRef")
        for index, box in enumerate(boxes[1:]):
            opaque = hashlib.sha256(f"{frame.frame_id}:{box!r}".encode()).hexdigest()[:24]
            pixels = box.pixels(frame.width, frame.height)
            crop = source.crop(pixels).convert("RGB")
            path = destination / f"{opaque}-crop-{index}.jpg"
            crop.save(path, "JPEG", quality=jpeg_quality, subsampling=0)
            ref = FrameRef(frame_id=f"crop-{opaque}-{index}", pts_sec=frame.pts_sec,
                           path=str(path.resolve()), width=crop.width, height=crop.height,
                           job_id=frame.job_id, rotation_deg=0, source_pts=frame.source_pts)
            views.append(CropView(ref, frame.frame_id, box, False))
    return views


def detect_motion_cues(
    frames: Sequence[FrameRef],
    *,
    threshold: float = 0.08,
    thumbnail_size: tuple[int, int] = (96, 54),
    max_cues: int = 32,
    min_interval_sec: float = 0.5,
) -> list[MotionCue]:
    """Compute local grayscale change cues; baseline coverage remains mandatory.

    Scores are mean absolute luminance differences in [0,1], not probabilities.
    Only downsampled grayscale buffers are retained, never whole videos.
    """
    from PIL import Image, ImageChops, ImageStat

    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if (len(thumbnail_size) != 2 or min(thumbnail_size) < 1
            or max(thumbnail_size) > 512 or not isinstance(max_cues, int) or max_cues < 0
            or not math.isfinite(min_interval_sec) or min_interval_sec < 0):
        raise ValueError("Invalid bounded motion settings")
    if max_cues == 0:
        return []
    previous = None
    previous_pts = None
    cues: list[MotionCue] = []
    for frame in sorted(frames, key=lambda item: item.pts_sec):
        with Image.open(frame.path) as source:
            current = source.convert("L").resize(thumbnail_size, Image.Resampling.BILINEAR)
        if previous is not None and previous_pts is not None and frame.pts_sec > previous_pts:
            difference = ImageChops.difference(previous, current)
            score = ImageStat.Stat(difference).mean[0] / 255.0
            if score >= threshold and (not cues or frame.pts_sec - cues[-1].end_sec >= min_interval_sec):
                mask = difference.point(lambda value: 255 if value >= threshold * 255 else 0)
                bounds = mask.getbbox()
                if bounds:
                    width, height = thumbnail_size
                    box = NormalizedBox(bounds[0] / width, bounds[1] / height,
                                        bounds[2] / width, bounds[3] / height)
                    cues.append(MotionCue(previous_pts, frame.pts_sec, score, box))
                    if len(cues) >= max_cues:
                        break
        previous, previous_pts = current, frame.pts_sec
    return cues
