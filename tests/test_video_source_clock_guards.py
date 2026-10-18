"""Source-clock/evidence regressions using only generated local video."""

from __future__ import annotations

import hashlib
from pathlib import Path

import av
import pytest

from forte.video.batches import iter_rgb_batches
from forte.video.decode import decode_frames
from forte.video.evidence_seek import materialize_evidence_frames_seek
from forte.video.probe import VideoDecodeError, probe_video
from test_video_batches import make_fixture


@pytest.mark.parametrize("field,value", [("stream_index", 1), ("source_start_sec", -0.04)])
def test_jpeg_decoder_rejects_different_probed_stream_before_writing(
    tmp_path: Path, field: str, value: int | float,
) -> None:
    asset = probe_video(make_fixture(tmp_path / "stream.mov"))
    altered = asset.model_copy(update={field: value})
    with pytest.raises(VideoDecodeError, match="stream|origin"):
        decode_frames(altered, tmp_path / "frames", timestamps_sec=[0.04], use_subprocess=False)
    assert not list((tmp_path / "frames").rglob("*.jpg"))


def test_jpeg_decoder_rejects_corrupt_frame_before_encoding(tmp_path: Path, monkeypatch) -> None:
    asset = probe_video(make_fixture(tmp_path / "corrupt-flag.mov"))
    original_open = av.open

    class CorruptFrame:
        is_corrupt = True

        def __init__(self, frame):
            self.frame = frame

        def __getattr__(self, name):
            return getattr(self.frame, name)

        def to_image(self):
            pytest.fail("Corrupt decoded pixels must not reach JPEG encoding")

    class FlaggedContainer:
        def __init__(self, *args, **kwargs):
            self.container = original_open(*args, **kwargs)
            self.streams = self.container.streams

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.container.close()

        def decode(self, stream):
            for frame in self.container.decode(stream):
                yield CorruptFrame(frame)

    monkeypatch.setattr(av, "open", FlaggedContainer)
    with pytest.raises(VideoDecodeError, match="trustworthy"):
        decode_frames(asset, tmp_path / "frames", timestamps_sec=[0], use_subprocess=False)
    assert not list((tmp_path / "frames").rglob("*.jpg"))


def test_failed_evidence_seek_does_not_substitute_nearest_frame_or_forward_fallback(
    tmp_path: Path, monkeypatch,
) -> None:
    import forte.video.evidence_seek as implementation

    video = make_fixture(tmp_path / "wrong-seek.mov", timestamps=tuple(range(0, 480, 40)))
    source_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    asset = probe_video(video)
    index = [frame for batch in iter_rgb_batches(asset, target_size=(8, 8), use_subprocess=False)
             for frame in batch.frames]
    original_pairs = implementation._pairs
    calls = 0

    def skipped_target(*args, **kwargs):
        nonlocal calls
        calls += 1
        for frame, next_pts in original_pairs(*args, **kwargs):
            if frame.pts != index[1].source_pts:
                yield frame, next_pts

    monkeypatch.setattr(implementation, "_pairs", skipped_target)
    with pytest.raises(VideoDecodeError, match="native PTS interval|exact required"):
        materialize_evidence_frames_seek(
            asset, [index[1]], tmp_path / "evidence", source_index=index, use_subprocess=False,
        )
    assert calls == 1
    assert not list((tmp_path / "evidence").glob(".evidence-seek-*"))
    assert hashlib.sha256(video.read_bytes()).hexdigest() == source_hash


def test_three_fresh_decodes_reproduce_native_ledger_and_original_evidence(tmp_path: Path) -> None:
    video = make_fixture(
        tmp_path / "fresh-repeat.mov", rotation=90, b_frames=2,
        timestamps=tuple(5000 + index * 40 for index in range(90)),
    )
    original_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    observed = []
    for attempt in range(3):
        # Probe, decode and encode are repeated; no pixels/index/JPEG cache is
        # passed from an earlier attempt. This exercises codecs, not a model.
        asset = probe_video(video, job_id=f"repeat-{attempt}")
        index = [frame for batch in iter_rgb_batches(
            asset, target_size=(8, 8), verify_source_hash=True, use_subprocess=False,
        ) for frame in batch.frames]
        selected = [index[1], index[len(index) // 2], index[-1]]
        metrics = {}
        refs = materialize_evidence_frames_seek(
            asset, selected, tmp_path / f"repeat-{attempt}", source_index=index,
            use_subprocess=False, metrics=metrics,
        )
        observed.append((
            [(frame.frame_id, frame.source_pts, frame.time_base, frame.source_origin_pts,
              frame.pts_sec, frame.next_source_pts, frame.next_pts_sec, frame.source_end_bound_sec)
             for frame in index],
            [(frame.frame_id, frame.pts_sec, frame.jpeg_sha256) for frame in refs],
        ))
        assert metrics["full_forward_fallback_used"] == 0
    assert observed[0] == observed[1] == observed[2]
    assert hashlib.sha256(video.read_bytes()).hexdigest() == original_hash
