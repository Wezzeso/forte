"""Optional exact original-resolution JPEG evidence from grouped keyframe seeks.

No feature-sized RGB batch is used as evidence. This module leaves the existing
decoder and frozen feature preprocessing code unchanged and has no implicit
full-forward fallback. Caller-supplied identities must come from a completed
native feature pass over the same immutable owned video.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from fractions import Fraction
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from forte.shared.schemas import FrameRef, VideoAsset
from forte.video.batches import (
    SourceFrameIdentity, _Config, _Progress, _configuration, _identity,
    _open_stream, _pairs, _source_index, validate_source_identity,
)
from forte.video.decode import _write_frame, validate_frame_ref
from forte.video.probe import (
    CancelSignal, VideoCancelledError, VideoDeadlineError, VideoDecodeError,
    VideoError, VideoLimitError, check_cancelled, content_sha256,
    local_media_path, operation_slot,
)


def _file_stat(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _validate_and_group(
    asset: VideoAsset, selected: Sequence[SourceFrameIdentity],
    source_index: Sequence[SourceFrameIdentity], config: _Config,
    cancel: CancelSignal, max_preroll_sec: float,
) -> tuple[list[int], dict[int, SourceFrameIdentity], list[tuple[int, list[int]]]]:
    pts, keyframes = _source_index(None, None, asset, config, cancel, source_index)
    if not keyframes:
        raise VideoDecodeError("Native source index has no original keyframe anchors")
    ledger = {entry.source_pts: entry for entry in source_index}
    grouped: dict[int, list[int]] = {}
    base = Fraction(asset.time_base)
    for source in selected:
        check_cancelled(config.deadline_monotonic, cancel)
        validate_source_identity(asset, source)
        if ledger.get(source.source_pts) != source:
            raise VideoDecodeError("Selected source identity differs from the completed native ledger")
        anchor = max((point for point in keyframes if point <= source.source_pts), default=None)
        if anchor is None:
            raise VideoDecodeError("Selected evidence has no preceding native keyframe")
        if float((source.source_pts - anchor) * base) > max_preroll_sec:
            raise VideoLimitError("Evidence keyframe preroll exceeds the configured bound")
        grouped.setdefault(anchor, []).append(source.source_pts)
    return pts, ledger, [(anchor, sorted(set(targets))) for anchor, targets in sorted(grouped.items())]


def _seek_local(
    asset: VideoAsset, selected: Sequence[SourceFrameIdentity],
    source_index: Sequence[SourceFrameIdentity], directory: Path, config: _Config,
    cancel: CancelSignal, *, jpeg_quality: int, max_decoded_frames: int,
    max_preroll_sec: float, max_jpeg_bytes: int, expected_stat: tuple[int, int, int],
) -> tuple[list[FrameRef], dict[str, float | int]]:
    import av
    started = time.monotonic()
    pts, ledger, groups = _validate_and_group(asset, selected, source_index, config, cancel, max_preroll_sec)
    progress = _Progress()
    refs: dict[int, FrameRef] = {}
    source_path = local_media_path(asset.owned_path)
    if _file_stat(source_path) != expected_stat:
        raise VideoDecodeError("Owned source changed before evidence decoding")
    try:
        with av.open(str(source_path), options={"protocol_whitelist": "file,pipe"}) as container:
            stream, origin, base = _open_stream(container, asset, config)
            for anchor, targets in groups:
                check_cancelled(config.deadline_monotonic, cancel)
                # One seek per keyframe group. Never retry from the source start.
                container.seek(anchor, stream=stream, backward=True, any_frame=False)
                expected = [point for point in pts if targets[0] <= point <= targets[-1]]
                expected_position = 0
                target_set = set(targets)
                observed_any = False
                for frame, next_pts in _pairs(container, stream, asset, config, progress, cancel):
                    check_cancelled(config.deadline_monotonic, cancel)
                    if progress.decoded > max_decoded_frames:
                        raise VideoLimitError("Evidence decoded frame count exceeds the configured bound")
                    if not observed_any and frame.pts > targets[0]:
                        raise VideoDecodeError("Evidence seek skipped an exact required source frame")
                    if not observed_any and float((targets[0] - frame.pts) * base) > max_preroll_sec:
                        raise VideoLimitError("Actual evidence seek preroll exceeds the configured bound")
                    observed_any = True
                    if frame.pts < targets[0]:
                        continue
                    if frame.pts > targets[-1]:
                        break
                    if expected_position >= len(expected) or frame.pts != expected[expected_position]:
                        raise VideoDecodeError("Evidence seek failed to reproduce the native PTS interval")
                    expected_position += 1
                    observed = _identity(frame, next_pts, asset, origin, base)
                    if observed != ledger[frame.pts]:
                        raise VideoDecodeError("Evidence source metadata differs from the native feature pass")
                    if frame.pts in target_set:
                        # Original decoded pixels go directly to the established
                        # full-resolution JPEG writer. No resize/letterbox/crop.
                        reference = _write_frame(frame, asset, directory, jpeg_quality)
                        destination = Path(reference.path)
                        destination.chmod(0o600)
                        if destination.stat().st_size > max_jpeg_bytes:
                            raise VideoLimitError("Original evidence JPEG exceeds the configured byte limit")
                        validate_frame_ref(asset, reference)
                        if reference.frame_id != ledger[frame.pts].frame_id:
                            raise VideoDecodeError("Written evidence identity differs from the selected source")
                        refs[int(frame.pts)] = reference
                if expected_position != len(expected) or any(target not in refs for target in targets):
                    raise VideoDecodeError("Evidence seek omitted an exact required source frame")
        if _file_stat(source_path) != expected_stat:
            raise VideoDecodeError("Owned source changed during evidence decoding")
        check_cancelled(config.deadline_monotonic, cancel)
    except VideoError:
        raise
    except Exception as exc:
        raise VideoDecodeError("Original evidence keyframe decoding failed") from exc
    return [refs[key] for key in sorted(refs)], {
        "decode_jpeg_sec": time.monotonic() - started,
        "decoded_source_frames": progress.decoded,
        "keyframe_groups": len(groups), "selected_frame_count": len(refs),
        "full_forward_fallback_used": 0,
    }


def _seek_process_entry(connection: Any, arguments: dict) -> None:
    def send(message: tuple) -> None:
        try:
            connection.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass
    try:
        refs, metrics = _seek_local(cancel=None, **arguments)
        send(("ok", ([ref.model_dump(mode="json") for ref in refs], metrics)))
    except VideoError as exc:
        send((type(exc).__name__, str(exc)))
    except Exception:
        send(("VideoDecodeError", "Original evidence seek subprocess failed"))
    finally:
        connection.close()


def _discard_owned_operation(directory: Path, parent: Path) -> None:
    # Only our newly created flat nonce directory is cleaned. Never recurse into
    # a caller's existing tree or follow a symlink out of the owned output root.
    if directory.resolve().parent != parent.resolve() or not directory.name.startswith(".evidence-seek-"):
        return
    try:
        for child in directory.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
        directory.rmdir()
    except OSError:
        # Unexpected nested state stays private for the caller's job cleanup.
        pass


def materialize_evidence_frames_seek(
    asset: VideoAsset, selected_sources: Sequence[SourceFrameIdentity],
    output_dir: str | os.PathLike[str], *, source_index: Sequence[SourceFrameIdentity],
    deadline_monotonic: float | None = None, cancel: CancelSignal = None,
    max_runtime_sec: float = 5.0, thread_count: int = 4, jpeg_quality: int = 95,
    max_pixels: int = 33_554_432, max_index_frames: int = 100_000,
    max_decoded_frames: int = 1000, max_preroll_sec: float = 5.0,
    max_jpeg_bytes: int = 16 * 1024 * 1024, use_subprocess: bool = True,
    metrics: MutableMapping[str, float | int] | None = None,
) -> list[FrameRef]:
    """Write at most three exact original JPEGs using a completed native index.

    Selected frames sharing their preceding keyframe are decoded together in
    one group. Source SHA-256, original units/origin/identity, interval PTS and
    source metadata must all agree. Returned paths exist in a private nonce
    directory below ``output_dir`` and each JPEG was atomically written. The
    caller owns successful output cleanup. Failed operations clean only their
    newly created files. No automatic full-forward fallback exists.

    Default subprocess supervision hard-stops a blocked codec. Set
    ``use_subprocess=False`` only when an outer supervised process enforces the
    deadline. Metrics contain aggregate operation counts/times, never pixels.
    """
    started = time.monotonic()
    check_cancelled(deadline_monotonic, cancel)
    if len(selected_sources) > 3:
        raise VideoLimitError("Exact evidence seek accepts at most three selected source frames")
    if not selected_sources:
        return []
    if (type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100
        or any(type(value) is not int or value < 1 for value in (max_index_frames, max_decoded_frames, max_jpeg_bytes))
        or not math.isfinite(max_preroll_sec) or max_preroll_sec <= 0):
        raise ValueError("Invalid exact evidence seek limits")
    config = _configuration(
        batch_size=1, target_size=(1, 1), max_batch_bytes=3,
        deadline_monotonic=deadline_monotonic, max_runtime_sec=max_runtime_sec,
        max_source_frames=max_index_frames, max_output_frames=3,
        max_pixels=max_pixels, thread_count=thread_count,
    )
    selected = tuple(selected_sources)
    native_index = tuple(source_index)
    _validate_and_group(asset, selected, native_index, config, cancel, max_preroll_sec)
    source_path = local_media_path(asset.owned_path)
    expected_stat = _file_stat(source_path)
    hash_started = time.monotonic()
    if content_sha256(source_path, deadline_monotonic=config.deadline_monotonic, cancel=cancel) != asset.content_hash:
        raise VideoDecodeError("Owned video content has changed")
    hash_elapsed = time.monotonic() - hash_started
    if _file_stat(source_path) != expected_stat:
        raise VideoDecodeError("Owned source changed while its identity was checked")
    parent = Path(output_dir).expanduser().resolve()
    if parent == source_path:
        raise VideoDecodeError("Evidence output directory overlaps the original video")
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    operation_directory = Path(tempfile.mkdtemp(prefix=".evidence-seek-", dir=parent))
    arguments = dict(asset=asset, selected=selected, source_index=native_index,
                     directory=operation_directory, config=config, jpeg_quality=jpeg_quality,
                     max_decoded_frames=max_decoded_frames, max_preroll_sec=max_preroll_sec,
                     max_jpeg_bytes=max_jpeg_bytes, expected_stat=expected_stat)
    succeeded = False
    try:
        with operation_slot(config.deadline_monotonic, cancel):
            if not use_subprocess:
                refs, observed = _seek_local(cancel=cancel, **arguments)
            else:
                context = multiprocessing.get_context("spawn")
                receiving, sending = context.Pipe(duplex=False)
                process = context.Process(target=_seek_process_entry, args=(sending, arguments), daemon=True)
                try:
                    process.start()
                    sending.close()
                    while not receiving.poll(.05):
                        check_cancelled(config.deadline_monotonic, cancel)
                        if not process.is_alive():
                            raise VideoDecodeError("Evidence seek subprocess exited without a result")
                    check_cancelled(config.deadline_monotonic, cancel)
                    kind, value = receiving.recv()
                    if kind != "ok":
                        error_type = {"VideoDeadlineError": VideoDeadlineError, "VideoCancelledError": VideoCancelledError,
                                      "VideoLimitError": VideoLimitError}.get(kind, VideoDecodeError)
                        raise error_type(value)
                    dumped, observed = value
                    refs = [FrameRef.model_validate(reference) for reference in dumped]
                except EOFError as exc:
                    raise VideoDecodeError("Evidence seek subprocess failed") from exc
                finally:
                    receiving.close()
                    sending.close()
                    if process.pid is not None:
                        process.join(timeout=.1)
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=1)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=1)
                        process.close()
        if {reference.frame_id for reference in refs} != {source.frame_id for source in selected}:
            raise VideoDecodeError("Exact evidence output set differs from the selected source identities")
        for reference in refs:
            validate_frame_ref(asset, reference)
            path = Path(reference.path).resolve(strict=True)
            if path.parent != operation_directory.resolve():
                raise VideoDecodeError("Evidence path escaped its private operation directory")
            if content_sha256(path, deadline_monotonic=config.deadline_monotonic, cancel=cancel) != reference.jpeg_sha256:
                raise VideoDecodeError("Written evidence JPEG hash does not match its frame reference")
        check_cancelled(config.deadline_monotonic, cancel)
        if metrics is not None:
            metrics.update(observed)
            metrics.update(source_hash_sec=hash_elapsed, total_sec=time.monotonic() - started)
        succeeded = True
        return refs
    finally:
        if not succeeded:
            _discard_owned_operation(operation_directory, parent)
__all__ = ["materialize_evidence_frames_seek"]
