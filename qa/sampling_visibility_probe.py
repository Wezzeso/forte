"""Generated-pixel counterexample for sparse visibility; no model/private media.

Run: .venv/Scripts/python.exe -m qa.sampling_visibility_probe --output <json>
The marker is a synthetic visibility oracle, never a bank-rule annotation.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile

import av
from PIL import Image, ImageStat

from forte.video.batches import iter_rgb_batches
from forte.video.crops import detect_motion_cues
from forte.video.decode import decode_frames
from forte.video.probe import probe_video
from forte.video.sampling import (
    context_window, coverage_report, coverage_windows, make_clip_requests, record_inspection, sampling_timestamps,
)


def _marker_visible(image: Image.Image) -> bool:
    red, green, blue = ImageStat.Stat(image.convert("RGB")).mean
    return red > 100 and red > green * 2 and red > blue * 2


def run_probe() -> dict:
    with tempfile.TemporaryDirectory(prefix="forte-synthetic-visibility-") as temporary:
        directory = Path(temporary)
        video = directory / "generated-marker.mov"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=25)
            stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
            stream.codec_context.max_b_frames = 2
            stream.options = {"preset": "veryfast", "crf": "18"}
            for index in range(200):
                color = (240, 20, 20) if 5 <= index <= 8 else (60, 60, 60)
                with Image.new("RGB", (96, 64), color) as image:
                    frame = av.VideoFrame.from_image(image)
                    frame.pts, frame.time_base = index, Fraction(1, 25)
                    for packet in stream.encode(frame):
                        container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        original_sha256 = hashlib.sha256(video.read_bytes()).hexdigest()
        asset = probe_video(video, job_id="synthetic-visibility")
        native_markers, native_count = [], 0
        for batch in iter_rgb_batches(asset, target_size=(32, 32), verify_source_hash=True, use_subprocess=False):
            for index, source in enumerate(batch.frames):
                native_count += 1
                with Image.frombytes("RGB", (32, 32), batch.frame_pixels(index).tobytes()) as image:
                    if _marker_visible(image):
                        native_markers.append(source.pts_sec)
        profiles = []
        for name, window, stride, fps in (("default_sparse", 8.0, 4.0, 2.0),
                                          ("diagnostic_denser", 4.0, 2.0, 4.0)):
            windows = coverage_windows(asset.duration_sec, window_sec=window, stride_sec=stride,
                                       sample_fps=fps, max_frames_per_window=16)
            refs = decode_frames(asset, directory / name, timestamps_sec=sampling_timestamps(windows),
                                 use_subprocess=False)
            clips = make_clip_requests(asset.job_id, refs, windows)
            visible = []
            for reference in refs:
                with Image.open(reference.path) as image:
                    if _marker_visible(image):
                        visible.append(reference.pts_sec)
            report = coverage_report(asset.duration_sec, windows,
                                     [record_inspection(w, c) for w, c in zip(windows, clips)])
            points = [frame.pts_sec for frame in refs]
            profiles.append({"profile": name, "window_seconds": window, "stride_seconds": stride,
                "sample_fps": fps, "requests": len(clips), "unique_selected_frames": len(refs),
                "model_frame_uses": sum(len(clip.frame_refs) for clip in clips),
                "maximum_frames_per_request": max(len(clip.frame_refs) for clip in clips),
                "selected_source_pts_sec": points,
                "maximum_selected_gap_sec": max(b - a for a, b in zip(points, points[1:])),
                "marker_selected_frames": len(visible), "marker_selected_pts_sec": visible,
                "interval_coverage_complete_if_requests_succeed": report.complete,
                "sampled_motion_cues": len(detect_motion_cues(refs))})
        assert native_markers and profiles[0]["marker_selected_frames"] == 0
        assert profiles[0]["interval_coverage_complete_if_requests_succeed"]
        assert profiles[1]["marker_selected_frames"] > 0
        refinement_schedules = []
        for span in (2.0, 4.0, 8.0, 16.0, 32.0):
            fps = min(4.0, 16 / span)
            window = context_window(32.0, 0.0, span, context_sec=0.0, sample_fps=fps, max_frames=16)
            refinement_schedules.append({"span_sec": span, "configured_sample_fps": fps,
                "requested_frames": len(window.timestamps_sec),
                "maximum_target_gap_sec": max(b - a for a, b in
                    zip(window.timestamps_sec, window.timestamps_sec[1:]))})
        return {"schema_version": "forte.synthetic-sampling-visibility.v1",
            "synthetic_only": True, "model_calls": 0, "gpu_used": False,
            "claims_real_bank_violation_or_model_accuracy": False,
            "duration_sec": asset.duration_sec, "source_dimensions": [asset.width, asset.height],
            "native_source_frames": native_count, "native_marker_frame_count": len(native_markers),
            "native_marker_pts_sec": native_markers, "marker_duration_sec": len(native_markers) / 25,
            "original_bytes_unchanged": hashlib.sha256(video.read_bytes()).hexdigest() == original_sha256,
            "profiles": profiles, "current_refinement_schedules": refinement_schedules}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_probe()
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
