"""Verify downloaded private audits and map POSIX media paths to local files.

No network, file writes, model inference, review decisions or source mutation
occur here. Paths change; source/frame hashes, IDs and timestamps do not.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .proposals import _sha256, _under_roots
from .store import ReviewError

_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def _remote_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//") or "\\" in value or any(ord(character) < 32 for character in value):
        raise ReviewError("remote frame references must be explicit absolute POSIX paths")
    if any(part in {".", ".."} for part in value.split("/")):
        raise ReviewError("remote frame references cannot contain traversal components")
    return PurePosixPath(value)


def _portable_relative(path: PurePosixPath, root: PurePosixPath) -> tuple[str, ...]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ReviewError("remote frame reference is outside its explicit mapping root") from None
    if not relative.parts:
        raise ReviewError("remote frame reference must identify a file below its mapping root")
    for part in relative.parts:
        if any(character in part for character in '<>:"|?*\\\x00') or any(ord(character) < 32 for character in part) or part.endswith((".", " ")) or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise ReviewError("remote relative path is not safe for the local filesystem")
    return relative.parts


def rebase_proposal(
    audit: dict[str, Any],
    local_source_by_hash: Mapping[str, str | Path],
    remote_frame_root: str,
    local_frame_root: str | Path,
) -> dict[str, Any]:
    """Return a verified deep copy with source/frame paths mapped locally.

    ``local_frame_root`` must contain an unchanged private download preserving
    paths relative to ``remote_frame_root``. A declared SHA-256 is mandatory for
    every top-level frame, so existing or substituted local files cannot gain
    the authority of a remote source reference by basename alone. Embedded
    ClipRequest references must match the same registered frame before mapping.
    The original audit object and all source files remain unchanged.
    """
    from forte.shared.schemas import FrameRef, VideoAsset

    if not isinstance(audit, dict) or audit.get("schema_version") != "forte.local-proposal.v1":
        raise ReviewError("unsupported private proposal schema for rebasing")
    try:
        asset = VideoAsset.model_validate(audit.get("asset"))
    except Exception:
        raise ReviewError("rebasing requires a valid source asset contract") from None
    if audit.get("job_id") not in {None, asset.job_id}:
        raise ReviewError("remote audit job identity differs from its source asset")
    if asset.content_hash not in local_source_by_hash:
        raise ReviewError("rebasing requires an explicit local source for the declared content hash")
    local_source = Path(local_source_by_hash[asset.content_hash])
    if not local_source.is_absolute():
        raise ReviewError("source mapping must use an absolute local input path")
    try:
        local_source = local_source.resolve(strict=True)
    except OSError:
        raise ReviewError("mapped local source is unavailable") from None
    if not local_source.is_file() or _sha256(local_source) != asset.content_hash:
        raise ReviewError("mapped local source bytes differ from the original content hash")
    remote_root = _remote_path(remote_frame_root)
    local_root = Path(local_frame_root)
    if not local_root.is_absolute():
        raise ReviewError("frame mapping must use an absolute local root")
    try:
        local_root = local_root.resolve(strict=True)
    except OSError:
        raise ReviewError("mapped local frame root is unavailable") from None
    if not local_root.is_dir():
        raise ReviewError("mapped local frame root must be a directory")
    result = deepcopy(audit)
    result["asset"]["owned_path"] = str(local_source)
    raw_frames = result.get("frames", result.get("frame_map"))
    if isinstance(raw_frames, dict):
        if any(not isinstance(frame, dict) or frame.get("frame_id") != key for key, frame in raw_frames.items()):
            raise ReviewError("rebased frame map keys must equal their registered IDs")
        frames = list(raw_frames.values())
    elif isinstance(raw_frames, list):
        frames = raw_frames
    else:
        raise ReviewError("rebasing requires a registered source frame collection")
    if not frames:
        raise ReviewError("rebasing requires at least one original source frame")
    original_frames: dict[str, Any] = {}
    mapped_paths: dict[str, str] = {}
    used_paths: set[Path] = set()
    for raw_frame in frames:
        try:
            frame = FrameRef.model_validate(raw_frame)
            frame.validate_timeline(asset)
        except Exception:
            raise ReviewError("rebased frame violates its shared source contract") from None
        if frame.job_id != asset.job_id or frame.frame_id in original_frames:
            raise ReviewError("rebased frames require unique IDs bound to the source job")
        if frame.jpeg_sha256 is None:
            raise ReviewError("rebasing requires the original declared SHA-256 for every JPEG")
        relative = _portable_relative(_remote_path(frame.path), remote_root)
        candidate = local_root.joinpath(*relative)
        path = _under_roots(candidate, [local_root], description="rebased source frame")
        if path in used_paths:
            raise ReviewError("different source frame IDs cannot alias one mapped local file")
        if _sha256(path) != frame.jpeg_sha256:
            raise ReviewError("downloaded JPEG bytes differ from the original frame checksum")
        try:
            from PIL import Image
            with Image.open(path) as image:
                if image.format != "JPEG" or image.size != (frame.width, frame.height):
                    raise ReviewError("downloaded JPEG format or dimensions differ from its source reference")
                image.verify()
        except ReviewError:
            raise
        except Exception:
            raise ReviewError("downloaded source JPEG is not decodable") from None
        used_paths.add(path)
        original_frames[frame.frame_id] = frame
        mapped_paths[frame.frame_id] = str(path)
        raw_frame["path"] = str(path)
    for window in result.get("window_decisions", []):
        if not isinstance(window, dict):
            raise ReviewError("rebased window decisions must be objects")
        request = window.get("request")
        if request is None:
            continue
        if not isinstance(request, dict) or request.get("job_id") != asset.job_id or not isinstance(request.get("frame_refs"), list):
            raise ReviewError("rebased embedded request must belong to the source job")
        for raw_frame in request["frame_refs"]:
            try:
                frame = FrameRef.model_validate(raw_frame)
            except Exception:
                raise ReviewError("invalid embedded frame reference during rebasing") from None
            original = original_frames.get(frame.frame_id)
            if original is None or any(getattr(frame, key) != getattr(original, key) for key in ("path", "pts_sec", "width", "height", "job_id", "rotation_deg", "source_pts")):
                raise ReviewError("embedded request frame differs from its registered remote source")
            if frame.jpeg_sha256 not in {None, original.jpeg_sha256}:
                raise ReviewError("embedded request frame checksum differs from its source")
            raw_frame["path"] = mapped_paths[frame.frame_id]
    result["local_rebase"] = {
        "source_content_hash": asset.content_hash,
        "original_source_path": asset.owned_path,
        "remote_frame_root": str(remote_root),
        "local_frame_root": str(local_root),
        "mapped_frame_count": len(frames),
        "source_and_frame_hashes_verified": True,
    }
    return result
