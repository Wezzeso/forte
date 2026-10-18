"""Small generated fixtures compare seek JPEGs with full-forward source pixels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

from PIL import Image
import pytest

from forte.video.batches import iter_rgb_batches, materialize_evidence_frames
from forte.video.evidence_seek import materialize_evidence_frames_seek
from forte.video.probe import VideoCancelledError, VideoDeadlineError, VideoDecodeError, VideoLimitError, probe_video
from test_video_batches import make_fixture


def native_index(asset):
    return [frame for batch in iter_rgb_batches(asset, target_size=(8, 8), use_subprocess=False)
            for frame in batch.frames]


@pytest.mark.parametrize("rotation,origin", [(0, 0), (90, 0), (0, 5000), (270, 5000)])
def test_seek_original_vfr_pixels_equal_full_forward_jpeg(tmp_path: Path, rotation: int, origin: int) -> None:
    video = make_fixture(tmp_path / "synthetic-vfr-original.mov", rotation=rotation,
                         timestamps=tuple(origin + value for value in (0, 40, 160, 200, 480)))
    asset = probe_video(video, job_id="exact-seek")
    index = native_index(asset)
    selected = [index[0], index[2], index[-1]]
    forward = materialize_evidence_frames(asset, selected, tmp_path / "forward", use_subprocess=False)
    metrics = {}
    sought = materialize_evidence_frames_seek(asset, selected, tmp_path / "seek", source_index=index,
                                              use_subprocess=False, metrics=metrics)
    assert [frame.frame_id for frame in sought] == [frame.frame_id for frame in forward]
    assert [frame.pts_sec for frame in sought] == [frame.pts_sec for frame in forward]
    assert [frame.jpeg_sha256 for frame in sought] == [frame.jpeg_sha256 for frame in forward]
    assert len(list((tmp_path / "seek").rglob("*.jpg"))) == 3
    for frame in sought:
        assert Path(frame.path).parent.name.startswith(".evidence-seek-")
        with Image.open(frame.path) as image:
            assert image.size == (asset.width, asset.height)
            assert image.size != (8, 8)
    assert metrics["full_forward_fallback_used"] == 0
    assert metrics["selected_frame_count"] == 3


def test_groups_shared_keyframes_and_preserves_b_frame_source_pixels(tmp_path: Path) -> None:
    asset = probe_video(make_fixture(tmp_path / "synthetic-b-frame-gops.mov", b_frames=2,
                                    timestamps=tuple(5000 + index * 40 for index in range(90))))
    index = native_index(asset)
    gops = []
    for source in index:
        if source.key_frame:
            gops.append([])
        gops[-1].append(source)
    shared = next(group for group in gops if len(group) >= 3)
    distinct = next(group for group in reversed(gops) if group is not shared)
    selected = [shared[1], shared[2], distinct[0]]
    forward = materialize_evidence_frames(asset, selected, tmp_path / "forward-b", use_subprocess=False)
    metrics = {}
    sought = materialize_evidence_frames_seek(asset, selected, tmp_path / "seek-b", source_index=index,
                                              use_subprocess=False, metrics=metrics)
    assert [frame.frame_id for frame in sought] == [frame.frame_id for frame in forward]
    assert [frame.jpeg_sha256 for frame in sought] == [frame.jpeg_sha256 for frame in forward]
    assert metrics["keyframe_groups"] == 2
    assert metrics["decoded_source_frames"] < len(index)


def test_bad_identity_ledger_and_modified_source_fail_without_fallback(tmp_path: Path) -> None:
    asset = probe_video(make_fixture(tmp_path / "source.mov"))
    index = native_index(asset)
    with pytest.raises(VideoDecodeError, match="another source"):
        materialize_evidence_frames_seek(asset, [replace(index[1], job_id="wrong")], tmp_path / "wrong", source_index=index)
    with pytest.raises(VideoDecodeError, match="source start|declared native"):
        materialize_evidence_frames_seek(asset, [index[1]], tmp_path / "partial", source_index=index[1:])
    no_keys = [replace(frame, key_frame=False) for frame in index]
    with pytest.raises(VideoDecodeError, match="no original keyframe"):
        materialize_evidence_frames_seek(asset, [no_keys[1]], tmp_path / "no-key", source_index=no_keys)
    with Path(asset.owned_path).open("ab") as source:
        source.write(b"synthetic changed content")
    with pytest.raises(VideoDecodeError, match="content has changed"):
        materialize_evidence_frames_seek(asset, [index[1]], tmp_path / "changed", source_index=index)


def test_limits_cleanup_private_partial_jpegs(tmp_path: Path) -> None:
    asset = probe_video(make_fixture(tmp_path / "limits.mov"))
    index = native_index(asset)
    with pytest.raises(VideoLimitError, match="at most three"):
        materialize_evidence_frames_seek(asset, index[:4], tmp_path / "too-many", source_index=index)
    with pytest.raises(VideoLimitError, match="JPEG"):
        materialize_evidence_frames_seek(asset, [index[0]], tmp_path / "too-big", source_index=index,
                                          use_subprocess=False, max_jpeg_bytes=1)
    assert not list((tmp_path / "too-big").rglob("*.jpg"))
    with pytest.raises(VideoLimitError, match="preroll"):
        materialize_evidence_frames_seek(asset, [index[1]], tmp_path / "preroll", source_index=index,
                                          max_preroll_sec=.001)
    with pytest.raises(VideoLimitError, match="decoded frame count"):
        materialize_evidence_frames_seek(asset, [index[1]], tmp_path / "decode-limit", source_index=index,
                                          use_subprocess=False, max_decoded_frames=1)
    assert not list((tmp_path / "decode-limit").rglob("*.jpg"))


def test_default_subprocess_materializes_original_jpeg(tmp_path: Path) -> None:
    asset = probe_video(make_fixture(tmp_path / "isolated.mov"))
    index = native_index(asset)
    refs = materialize_evidence_frames_seek(asset, [index[1], index[-1]], tmp_path / "isolated", source_index=index)
    assert [frame.frame_id for frame in refs] == [index[1].frame_id, index[-1].frame_id]
    assert all(Path(frame.path).read_bytes()[:2] == b"\xff\xd8" for frame in refs)


def _blocked_seek(connection, arguments) -> None:
    time.sleep(30)


def test_hard_deadline_and_running_cancel_kill_codec_and_cleanup(tmp_path: Path, monkeypatch) -> None:
    import forte.video.evidence_seek as implementation
    asset = probe_video(make_fixture(tmp_path / "blocked.mov"))
    index = native_index(asset)
    monkeypatch.setattr(implementation, "_seek_process_entry", _blocked_seek)
    started = time.monotonic()
    with pytest.raises(VideoDeadlineError):
        materialize_evidence_frames_seek(asset, [index[0]], tmp_path / "deadline", source_index=index, max_runtime_sec=.4)
    assert time.monotonic() - started < 3
    assert not list((tmp_path / "deadline").glob(".evidence-seek-*"))
    cancelled = threading.Event()
    timer = threading.Timer(.4, cancelled.set)
    timer.start()
    try:
        with pytest.raises(VideoCancelledError):
            materialize_evidence_frames_seek(asset, [index[0]], tmp_path / "cancel", source_index=index, cancel=cancelled)
    finally:
        timer.cancel()
    assert not list((tmp_path / "cancel").glob(".evidence-seek-*"))
