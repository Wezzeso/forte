"""Synthetic media only: PTS, rotation, source binding and bounded failure."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import shutil
import threading
import time

import av
from PIL import Image
import pytest

from forte.video import (
    VideoCancelledError, VideoDeadlineError, VideoDecodeError, VideoLimitError,
    VideoProbeError, decode_frames, extract_evidence_frames, probe_video,
    validate_frame_ref,
)


pytestmark = pytest.mark.skipif(not shutil.which("ffprobe"), reason="FFprobe is required")


def make_video(
    path: Path, timestamps_ms: list[int], *, rotation: int = 0,
    time_base: Fraction = Fraction(1, 1000),
) -> Path:
    """MPEG4 MOV with explicit presentation timestamps and synthetic colors."""
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=25)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.codec_context.max_b_frames = 0
        if rotation:
            stream.set_display_rotation(rotation)
        for index, pts in enumerate(timestamps_ms):
            image = Image.new("RGB", (64, 32), (20 + index * 20, 30, 100))
            # An asymmetrical synthetic marker tests rotation direction.
            image.paste((230, 20, 30), (0, 0, 20, 12))
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = pts, time_base
            for packet in stream.encode(frame):
                output.mux(packet)
            image.close()
        for packet in stream.encode():
            output.mux(packet)
    return path


@pytest.fixture
def vfr_video(tmp_path: Path) -> Path:
    return make_video(tmp_path / "synthetic-vfr.mov", [0, 40, 160, 200, 480])


def test_vfr_sampling_uses_real_pts_and_handles_tail(vfr_video: Path, tmp_path: Path) -> None:
    asset = probe_video(vfr_video, job_id="synthetic-vfr")
    refs = decode_frames(asset, tmp_path / "frames", timestamps_sec=[0, 0.11, asset.duration_sec], use_subprocess=False)
    assert [ref.pts_sec for ref in refs] == pytest.approx([0, 0.16, 0.48])
    assert len({ref.frame_id for ref in refs}) == 3
    for ref in refs:
        validate_frame_ref(asset, ref)
        with Image.open(ref.path) as image:
            assert image.format == "JPEG"
            assert image.size == (64, 32)
        assert ref.jpeg_sha256 is not None


def test_nonzero_stream_origin_normalizes_to_video_start(tmp_path: Path) -> None:
    video = make_video(tmp_path / "offset.mov", [5000, 5040, 5160])
    asset = probe_video(video)
    assert asset.source_start_sec == pytest.approx(5)
    refs = decode_frames(asset, tmp_path / "offset-frames", timestamps_sec=[0, 0.16], use_subprocess=False)
    assert [ref.pts_sec for ref in refs] == pytest.approx([0, 0.16])
    assert all(ref.source_pts > 0 for ref in refs)


def test_integer_start_pts_avoids_decimal_timestamp_rounding(tmp_path: Path) -> None:
    video = make_video(tmp_path / "fractional-origin.mov", [1, 1921], time_base=Fraction(1, 48000))
    asset = probe_video(video)
    refs = decode_frames(asset, tmp_path / "fractional-frames", timestamps_sec=[0], use_subprocess=False)
    assert refs[0].pts_sec == 0


def test_matroska_nonzero_origin_container_duration(tmp_path: Path) -> None:
    video = make_video(tmp_path / "offset.mkv", [5000, 5040, 5160])
    asset = probe_video(video)
    assert asset.duration_sec == pytest.approx(0.2)
    refs = decode_frames(asset, tmp_path / "mkv-frames", timestamps_sec=[asset.duration_sec], use_subprocess=False)
    assert refs[-1].pts_sec == pytest.approx(0.16)


def test_rotation_preserves_full_source_pixels_and_orientation(tmp_path: Path) -> None:
    video = make_video(tmp_path / "rotated.mov", [0, 40], rotation=90)
    asset = probe_video(video)
    assert asset.rotation_deg == 90
    assert (asset.width, asset.height) == (32, 64)
    frame = decode_frames(asset, tmp_path / "rotated-frames", timestamps_sec=[0], use_subprocess=False)[0]
    with Image.open(frame.path) as image:
        assert image.size == (32, 64)
        # Top-left marker moves bottom-left under a 90-degree CCW rotation.
        red, green, blue = image.getpixel((5, 55))
        assert red > 150 and green < 80 and blue < 80


def test_exact_evidence_binds_identity_and_never_reads_request_image(vfr_video: Path, tmp_path: Path) -> None:
    asset = probe_video(vfr_video, job_id="exact-test")
    refs = decode_frames(asset, tmp_path / "screen", timestamps_sec=[0.16, 0.48], use_subprocess=False)
    for ref in refs:
        Path(ref.path).write_bytes(b"modified-screen-image")
    evidence = extract_evidence_frames(asset, refs, tmp_path / "evidence", use_subprocess=False)
    assert [ref.frame_id for ref in evidence] == [ref.frame_id for ref in refs]
    assert [ref.pts_sec for ref in evidence] == [ref.pts_sec for ref in refs]
    assert all(Path(ref.path).read_bytes()[:2] == b"\xff\xd8" for ref in evidence)
    wrong_job = refs[0].model_copy(update={"job_id": "another-job"})
    with pytest.raises(VideoDecodeError, match="different job|this job"):
        extract_evidence_frames(asset, [wrong_job], tmp_path / "bad-job", use_subprocess=False)
    wrong_timestamp = refs[0].model_copy(update={"pts_sec": refs[0].pts_sec + 0.001})
    with pytest.raises(VideoDecodeError, match="timestamp"):
        extract_evidence_frames(asset, [wrong_timestamp], tmp_path / "bad-time", use_subprocess=False)


def test_changed_source_is_rejected_for_evidence(vfr_video: Path, tmp_path: Path) -> None:
    asset = probe_video(vfr_video)
    refs = decode_frames(asset, tmp_path / "screen", timestamps_sec=[0], use_subprocess=False)
    with vfr_video.open("ab") as source:
        source.write(b"synthetic-mutation")
    with pytest.raises(VideoDecodeError, match="content has changed"):
        extract_evidence_frames(asset, refs, tmp_path / "changed", use_subprocess=False)


def test_invalid_media_is_failure_not_clean_result(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.mov"
    invalid.write_bytes(b"not a media container")
    with pytest.raises(VideoProbeError):
        probe_video(invalid)
    with pytest.raises(VideoProbeError):
        decode_frames(invalid, tmp_path / "invalid-frames")


def test_cancel_deadline_and_resource_bounds(vfr_video: Path, tmp_path: Path) -> None:
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(VideoCancelledError):
        decode_frames(vfr_video, tmp_path / "cancelled", cancel=cancelled)
    with pytest.raises(VideoDeadlineError):
        probe_video(vfr_video, deadline_monotonic=time.monotonic() - 1)
    asset = probe_video(vfr_video)
    with pytest.raises(VideoLimitError):
        decode_frames(asset, tmp_path / "too-many", timestamps_sec=[0, 0.1], max_frames=1)
    with pytest.raises(ValueError):
        decode_frames(asset, tmp_path / "bad-time", timestamps_sec=[asset.duration_sec + 1])


def test_subprocess_decodes_and_returns_only_metadata(vfr_video: Path, tmp_path: Path) -> None:
    asset = probe_video(vfr_video)
    refs = decode_frames(asset, tmp_path / "isolated", timestamps_sec=[0, 0.16])
    assert [ref.pts_sec for ref in refs] == pytest.approx([0, 0.16])


def test_single_frame_short_clip_and_duplicate_requests(tmp_path: Path) -> None:
    video = make_video(tmp_path / "single.mov", [0])
    asset = probe_video(video)
    refs = decode_frames(asset, tmp_path / "single-frames", timestamps_sec=[0, 0, asset.duration_sec], use_subprocess=False)
    assert len(refs) == 1
    assert refs[0].pts_sec == 0


def _blocked_decoder(connection: object, payload: dict) -> None:
    # Deliberately uncooperative child verifies the parent kills stuck codecs.
    time.sleep(30)


def test_hard_deadline_kills_uncooperative_codec(vfr_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import forte.video.decode as decoder
    asset = probe_video(vfr_video)
    monkeypatch.setattr(decoder, "_decode_process_entry", _blocked_decoder)
    started = time.monotonic()
    with pytest.raises(VideoDeadlineError):
        decode_frames(asset, tmp_path / "stuck", timestamps_sec=[0], max_runtime_sec=0.5)
    assert time.monotonic() - started < 3


def test_cancellation_kills_running_codec(vfr_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import forte.video.decode as decoder
    asset = probe_video(vfr_video)
    monkeypatch.setattr(decoder, "_decode_process_entry", _blocked_decoder)
    cancelled = threading.Event()
    timer = threading.Timer(0.5, cancelled.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(VideoCancelledError):
            decode_frames(asset, tmp_path / "stuck-cancel", timestamps_sec=[0], cancel=cancelled)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3
