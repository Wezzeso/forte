"""Small synthetic correctness checks; no throughput, private pixels or GPU."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from fractions import Fraction
import hashlib
import multiprocessing
from pathlib import Path
import shutil
import threading
import time

import av
from PIL import Image
import pytest

from forte.video.batches import (
    BoundaryReadRequest, group_boundary_requests, iter_boundary_batches,
    iter_rgb_batches, materialize_evidence_frames, rgb_transform_recipe, source_frame_id, validate_source_identity,
)
from forte.video.decode import frame_identity
from forte.video.probe import VideoCancelledError, VideoDeadlineError, VideoDecodeError, VideoLimitError, probe_video


pytestmark = pytest.mark.skipif(not shutil.which("ffprobe"), reason="FFprobe is required")


def make_fixture(path: Path, *, rotation: int = 0, b_frames: int = 0,
                 timestamps: tuple[int, ...] = (0, 40, 160, 200, 480)) -> Path:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.max_b_frames = b_frames
        stream.codec_context.gop_size = 30 if b_frames else 2
        stream.options = {"preset": "veryfast", "crf": "18"}
        if rotation:
            stream.set_display_rotation(rotation)
        for index, pts in enumerate(timestamps):
            with Image.new("RGB", (64, 32), ((20 + 25 * index) % 256, 30, 100)) as image:
                image.paste((235, 20, 30), (0, 0, 20, 12))
                frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def asset(tmp_path: Path):
    return probe_video(make_fixture(tmp_path / "synthetic-batch.mov"), job_id="batch-test")


def read_native(asset, **options):
    batches = list(iter_rgb_batches(asset, batch_size=2, target_size=(32, 32), use_subprocess=False, **options))
    return batches, [frame for batch in batches for frame in batch.frames]


def test_native_vfr_batches_flush_b_frames_and_retain_native_bounds(asset, tmp_path: Path) -> None:
    batches, frames = read_native(asset)
    assert [frame.pts_sec for frame in frames] == pytest.approx([0, .04, .16, .2, .48])
    assert [frame.next_pts_sec for frame in frames[:-1]] == pytest.approx([.04, .16, .2, .48])
    assert frames[-1].next_pts_sec is None
    assert frames[-1].next_source_pts is None
    assert frames[-1].source_end_bound_sec == asset.duration_sec
    assert [batch.segment_complete for batch in batches] == [False, False, True]
    assert batches[-1].source_complete and batches[-1].tail_included
    assert batches[-1].requested_count == batches[-1].delivered_count == len(frames)
    assert batches[-1].decoded_source_frame_count == len(frames)
    assert [len(batch.frames) for batch in batches] == [2, 2, 1]
    for batch in batches:
        assert len(batch.pixels) == len(batch.frames) * 32 * 32 * 3
        assert len(batch.frame_pixels(0)) == 32 * 32 * 3
        assert batch.shape == (len(batch.frames), 32, 32, 3)
        assert "pixels=" not in repr(batch)
        for frame in batch.frames:
            validate_source_identity(asset, frame)
            assert frame.frame_id == frame_identity(asset, frame.source_pts)
            assert not hasattr(frame, "path")
    assert not list(tmp_path.rglob("*.jpg"))


def test_rate_selection_nearest_tie_earlier_and_real_tail(asset) -> None:
    batches = list(iter_rgb_batches(asset, sample_fps=10, target_size=(16, 16), batch_size=2, use_subprocess=False))
    frames = [frame for batch in batches for frame in batch.frames]
    assert [frame.pts_sec for frame in frames] == pytest.approx([0, .04, .2, .48])
    assert len({frame.frame_id for frame in frames}) == len(frames)
    assert batches[-1].requested_count == 7
    assert batches[-1].delivered_count == 4
    assert batches[-1].tail_included and batches[-1].source_complete
    assert frames[1].next_pts_sec == pytest.approx(.16)  # native neighbor, not next sample


def test_frame_threading_flushes_delayed_h264_b_frames(tmp_path: Path) -> None:
    asset = probe_video(make_fixture(tmp_path / "b-frames.mov", b_frames=2, timestamps=tuple(range(0, 480, 40))))
    with av.open(asset.owned_path) as container:
        assert container.streams.video[0].codec_context.has_b_frames
    batches = list(iter_rgb_batches(asset, target_size=(16, 16), batch_size=5, use_subprocess=False))
    frames = [frame for batch in batches for frame in batch.frames]
    assert len(frames) == 12
    assert [frame.pts_sec for frame in frames] == pytest.approx([index / 25 for index in range(12)])
    assert batches[-1].source_complete and batches[-1].tail_included


def test_absolute_nonzero_origin_keeps_normalized_times_and_boundary_ticks(tmp_path: Path) -> None:
    asset = probe_video(make_fixture(tmp_path / "source-origin.mov", timestamps=(5000, 5040, 5160, 5200, 5480)))
    _, native = read_native(asset)
    assert asset.source_start_sec == 5
    assert [frame.pts_sec for frame in native] == pytest.approx([0, .04, .16, .2, .48])
    assert all(frame.source_origin_pts == native[0].source_pts for frame in native)
    request = [BoundaryReadRequest("absolute-ticks", native[1].source_pts, native[2].source_pts)]
    batches = list(iter_boundary_batches(asset, request, source_index=native, target_size=(16, 16), use_subprocess=False))
    assert [frame.frame_id for batch in batches for frame in batch.frames] == [frame.frame_id for frame in native[1:3]]


def test_b_frame_keyframe_seek_reproduces_complete_native_interval(tmp_path: Path) -> None:
    asset = probe_video(make_fixture(tmp_path / "b-frame-gops.mov", b_frames=2,
                                    timestamps=tuple(range(0, 3600, 40))))
    batches = list(iter_rgb_batches(asset, target_size=(16, 16), use_subprocess=False))
    native = [frame for batch in batches for frame in batch.frames]
    assert len([frame for frame in native if frame.key_frame]) >= 2
    request = [BoundaryReadRequest("after-second-keyframe", native[33].source_pts, native[41].source_pts)]
    for index in (None, native):
        refined = list(iter_boundary_batches(asset, request, source_index=index,
                                             target_size=(16, 16), use_subprocess=False))
        assert [frame.frame_id for batch in refined for frame in batch.frames] == [frame.frame_id for frame in native[33:42]]


def test_geometry_recipe_is_bound_to_actual_batch(asset) -> None:
    recipe = rgb_transform_recipe(target_size=(16, 16), aspect_policy="stretch")
    batch = next(iter_rgb_batches(asset, target_size=(16, 16), aspect_policy="stretch", use_subprocess=False))
    assert batch.transform_sha256 == recipe["sha256"]
    assert recipe["sha256"] != rgb_transform_recipe(target_size=(16, 16), aspect_policy="letterbox")["sha256"]
    assert recipe["normalization"].startswith("none")


def test_tight_rgb_fast_copy_has_same_pixels_as_pillow_surface(asset) -> None:
    batch = next(iter_rgb_batches(asset, target_size=(64, 32), aspect_policy="stretch", use_subprocess=False))
    with av.open(asset.owned_path) as container:
        first = next(container.decode(video=0))
        with first.reformat(width=64, height=32, format="rgb24", threads=1).to_image() as expected:
            assert batch.frame_pixels(0).tobytes() == expected.tobytes()


def test_wrong_declared_frame_count_cannot_emit_complete_coverage(asset) -> None:
    wrong = asset.model_copy(update={"frame_count": 6})
    reader = iter_rgb_batches(wrong, target_size=(16, 16), batch_size=1, use_subprocess=False)
    delivered = []
    with pytest.raises(VideoDecodeError, match="declared native frame count"):
        for batch in reader:
            delivered.append(batch)
    assert delivered and not any(batch.source_complete for batch in delivered)


def test_rotation_applied_and_letterbox_transform_metadata(tmp_path: Path) -> None:
    asset = probe_video(make_fixture(tmp_path / "rotated-batch.mov", rotation=90))
    batches = list(iter_rgb_batches(asset, target_size=(32, 64), aspect_policy="stretch", use_subprocess=False))
    with Image.frombytes("RGB", (32, 64), batches[0].frame_pixels(0).tobytes()) as image:
        red, green, blue = image.getpixel((5, 55))
        assert red > 150 and green < 80 and blue < 80
    frame = batches[0].frames[0]
    assert (frame.width, frame.height, frame.applied_rotation_deg) == (32, 64, 90)
    box = list(iter_rgb_batches(asset, target_size=(64, 64), aspect_policy="letterbox", use_subprocess=False))[0]
    assert box.transforms[0].resized_width == 32
    assert box.transforms[0].resized_height == 64
    assert box.transforms[0].pad_left == 16


def test_grouped_boundary_reads_exact_native_pts_with_and_without_index(asset) -> None:
    _, native = read_native(asset)
    requests = [BoundaryReadRequest("left", native[1].source_pts, native[3].source_pts),
                BoundaryReadRequest("overlap", native[2].source_pts, native[4].source_pts)]
    groups = group_boundary_requests(asset, requests)
    assert len(groups) == 1
    assert groups[0].request_ids == ("left", "overlap")
    for index in (None, native):
        batches = list(iter_boundary_batches(asset, requests, source_index=index, batch_size=2,
                                             target_size=(16, 16), use_subprocess=False))
        actual = [frame.source_pts for batch in batches for frame in batch.frames]
        assert actual == [frame.source_pts for frame in native[1:]]
        assert batches[-1].segment_complete
        assert not batches[-1].source_complete
        assert batches[-1].tail_included
        assert batches[-1].delivered_count == batches[-1].requested_count == 4


def test_separate_boundary_groups_keep_local_and_total_coverage(asset) -> None:
    _, native = read_native(asset)
    requests = [BoundaryReadRequest("first", native[0].source_pts, native[1].source_pts),
                BoundaryReadRequest("last", native[-1].source_pts, native[-1].source_pts)]
    batches = list(iter_boundary_batches(asset, requests, source_index=native, batch_size=4,
                                         target_size=(16, 16), use_subprocess=False))
    assert len(batches) == 2
    assert [batch.delivered_count for batch in batches] == [2, 1]
    assert [batch.total_delivered_count for batch in batches] == [2, 3]
    assert batches[0].request_ids == ("first",)
    assert batches[1].request_ids == ("last",)


def test_partial_rate_index_cannot_claim_native_boundary_coverage(asset) -> None:
    _, native = read_native(asset)
    request = [BoundaryReadRequest("boundary", native[1].source_pts, native[2].source_pts)]
    with pytest.raises(VideoDecodeError, match="source start|declared native"):
        list(iter_boundary_batches(asset, request, source_index=native[1:], target_size=(16, 16), use_subprocess=False))
    with pytest.raises(VideoDecodeError, match="native-PTS chain"):
        list(iter_boundary_batches(asset, request, source_index=[native[0], *native[2:]], target_size=(16, 16), use_subprocess=False))


def test_source_identity_and_canonical_rational_helper(asset) -> None:
    _, native = read_native(asset)
    frame = native[0]
    assert source_frame_id(asset.content_hash, frame.source_pts, frame.time_base_num, frame.time_base_den) == frame.frame_id
    assert source_frame_id("a" * 64, 3, 1, 1) == hashlib.sha256(f"{'a' * 64}:3:1/1".encode()).hexdigest()
    with pytest.raises(ValueError, match="reduced"):
        source_frame_id(asset.content_hash, frame.source_pts, 2, 2000)
    with pytest.raises(VideoDecodeError, match="another source"):
        validate_source_identity(asset, replace(frame, job_id="another-job"))
    with pytest.raises(VideoDecodeError, match="next-PTS"):
        validate_source_identity(asset, replace(frame, next_pts_sec=100.0))
    with pytest.raises(VideoDecodeError, match="integers"):
        validate_source_identity(asset, replace(frame, source_pts=float(frame.source_pts)))


def test_materializes_actual_owned_evidence_jpegs_only_when_selected(asset, tmp_path: Path) -> None:
    _, native = read_native(asset)
    refs = materialize_evidence_frames(asset, [native[0], native[-1]], tmp_path / "evidence", use_subprocess=False)
    assert {frame.frame_id for frame in refs} == {native[0].frame_id, native[-1].frame_id}
    for frame in refs:
        assert Path(frame.path).is_file()
        with Image.open(frame.path) as image:
            assert image.format == "JPEG" and image.size == (64, 32)


def test_changed_source_is_rejected_before_feature_read(asset) -> None:
    with Path(asset.owned_path).open("ab") as source:
        source.write(b"synthetic mutation")
    with pytest.raises(VideoDecodeError, match="content has changed"):
        list(iter_rgb_batches(asset, verify_source_hash=True, use_subprocess=False))


def test_batch_byte_frame_and_boundary_budgets(asset) -> None:
    with pytest.raises(VideoLimitError, match="memory limit"):
        iter_rgb_batches(asset, batch_size=32, target_size=(256, 256), max_batch_bytes=100)
    with pytest.raises(VideoLimitError, match="source frame count"):
        list(iter_rgb_batches(asset, max_source_frames=2, target_size=(16, 16), use_subprocess=False))
    with pytest.raises(VideoLimitError, match="Selected RGB frame count"):
        list(iter_rgb_batches(asset, max_output_frames=2, target_size=(16, 16), use_subprocess=False))
    _, native = read_native(asset)
    with pytest.raises(VideoLimitError, match="refinement"):
        group_boundary_requests(asset, [BoundaryReadRequest("wide", native[0].source_pts, native[-1].source_pts)], max_refine_span_sec=.1)


def test_cancellation_between_batches_and_expired_deadline(asset) -> None:
    cancelled = threading.Event()
    reader = iter_rgb_batches(asset, target_size=(16, 16), batch_size=1, cancel=cancelled, use_subprocess=False)
    next(reader)
    cancelled.set()
    with pytest.raises(VideoCancelledError):
        next(reader)
    with pytest.raises(VideoDeadlineError):
        iter_rgb_batches(asset, deadline_monotonic=time.monotonic() - 1)


def test_subprocess_batches_and_early_close_clean_up(asset) -> None:
    baseline = {process.pid for process in multiprocessing.active_children()}
    with closing(iter_rgb_batches(asset, target_size=(16, 16), batch_size=1)) as reader:
        assert next(reader).shape == (1, 16, 16, 3)
    assert {process.pid for process in multiprocessing.active_children()} <= baseline
    batches = list(iter_rgb_batches(asset, target_size=(16, 16), batch_size=2))
    assert batches[-1].source_complete
    assert sum(len(batch.frames) for batch in batches) == 5


def test_subprocess_boundary_and_running_cancel(asset) -> None:
    _, native = read_native(asset)
    requests = [BoundaryReadRequest("spawned-boundary", native[1].source_pts, native[3].source_pts)]
    batches = list(iter_boundary_batches(asset, requests, source_index=native, target_size=(16, 16)))
    assert batches[-1].segment_complete and batches[-1].delivered_count == 3
    cancelled = threading.Event()
    with closing(iter_rgb_batches(asset, target_size=(16, 16), batch_size=1, cancel=cancelled)) as reader:
        next(reader)
        cancelled.set()
        with pytest.raises(VideoCancelledError):
            next(reader)


def _blocked_batch_process(connection, asset, config, arguments) -> None:
    time.sleep(30)


def test_subprocess_hard_deadline_kills_blocked_codec(asset, monkeypatch) -> None:
    import forte.video.batches as batches
    monkeypatch.setattr(batches, "_batch_process_entry", _blocked_batch_process)
    started = time.monotonic()
    with pytest.raises(VideoDeadlineError):
        list(iter_rgb_batches(asset, target_size=(16, 16), max_runtime_sec=.4))
    assert time.monotonic() - started < 3
