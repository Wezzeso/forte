"""CPU-only synthetic H.264 throughput experiment, never a GPU/model test.

Generates a public test-pattern fixture. Results characterize only this host,
fixture and byte/Pillow output path; no supplied pixels are decoded or displayed.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import av
from PIL import Image


def fixture(path: Path, frame_count: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=464x848:rate=30", "-frames:v", str(frame_count),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2",
        "-g", "30", "-keyint_min", "30", "-sc_threshold", "0", "-bf", "2",
        "-pix_fmt", "yuv420p", "-video_track_timescale", "19200", str(path),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    return time.monotonic() - started


def run_case(path: Path, *, threads: int, thread_type: str, output: str,
             sample_fps: int | None = None, jpeg_dir: Path | None = None) -> dict:
    started = time.monotonic()
    decoded = converted = output_bytes = 0
    next_target = Fraction(0)
    pending: list[bytes] = []
    max_batch_bytes = 0
    with av.open(str(path), options={"protocol_whitelist": "file,pipe"}) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = threads
        stream.codec_context.thread_type = thread_type
        origin = Fraction(stream.start_time * stream.time_base)
        for frame in container.decode(stream):
            decoded += 1
            seconds = Fraction(frame.pts * frame.time_base) - origin
            if output == "pts_only":
                continue
            if sample_fps is not None and seconds < next_target:
                continue
            if sample_fps is not None:
                while next_target <= seconds:
                    next_target += Fraction(1, sample_fps)
            if output == "rgb256_bytes":
                resized = frame.reformat(width=256, height=256, format="rgb24", threads=1)
                pixels = bytes(resized.planes[0])
                output_bytes += len(pixels)
                pending.append(pixels)
                if len(pending) == 32:
                    max_batch_bytes = max(max_batch_bytes, sum(map(len, pending)))
                    pending.clear()
            elif output == "letterbox256_bytes":
                resized = frame.reformat(width=140, height=256, format="rgb24", threads=1)
                image = resized.to_image()
                with Image.new("RGB", (256, 256)) as canvas:
                    canvas.paste(image, (58, 0))
                    pixels = canvas.tobytes()
                image.close()
                output_bytes += len(pixels)
                pending.append(pixels)
                if len(pending) == 32:
                    max_batch_bytes = max(max_batch_bytes, sum(map(len, pending)))
                    pending.clear()
            elif output == "original_jpeg95":
                image = frame.to_image()
                destination = jpeg_dir / f"{converted:05d}.jpg"
                image.save(destination, format="JPEG", quality=95, subsampling=0)
                image.close()
                output_bytes += destination.stat().st_size
            else:
                raise ValueError("unknown_benchmark_mode")
            converted += 1
    max_batch_bytes = max(max_batch_bytes, sum(map(len, pending)))
    elapsed = time.monotonic() - started
    return {
        "elapsed_sec": elapsed, "decoded_frames": decoded, "converted_frames": converted,
        "decoded_fps": decoded / elapsed, "converted_output_bytes": output_bytes,
        "maximum_materialized_batch_bytes": max_batch_bytes,
    }


def run_seek_case(path: Path) -> dict:
    """Verify grouped local seek refinement against the synthetic packet index."""
    brackets = [(5.1, 5.6), (21.2, 21.7), (61.9, 62.4), (120.1, 120.6)]
    with av.open(str(path)) as indexing:
        stream = indexing.streams.video[0]
        expected_pts = sorted(packet.pts for packet in indexing.demux(stream)
                              if packet.size and packet.pts is not None)
    started = time.monotonic()
    decoded = refined = 0
    exact = True
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        stream.codec_context.thread_type = "FRAME"
        base = Fraction(stream.time_base.numerator, stream.time_base.denominator)
        origin = stream.start_time
        for lower, upper in brackets:
            lower_pts = origin + int(Fraction(str(lower)) / base)
            upper_pts = origin + int(Fraction(str(upper)) / base)
            expected = [pts for pts in expected_pts if lower_pts <= pts <= upper_pts]
            actual = []
            container.seek(lower_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                decoded += 1
                if frame.pts > upper_pts:
                    break
                if frame.pts >= lower_pts:
                    actual.append(frame.pts)
                    resized = frame.reformat(width=256, height=256, format="rgb24", threads=1)
                    pixels = bytes(resized.planes[0])
                    refined += 1
            exact = exact and actual == expected
    return {"elapsed_sec": time.monotonic() - started,
            "brackets": len(brackets), "total_bracket_seconds": 2.0,
            "decoded_with_keyframe_preroll": decoded, "refined_native_frames": refined,
            "exact_pts_sets_equal_to_packet_index": exact}


def benchmark(directory: Path, frame_count: int, repeats: int) -> dict:
    source = directory / "generated-h264-464x848.mov"
    generation = fixture(source, frame_count)
    modes = [
        ("decode_pts_slice1", dict(threads=1, thread_type="SLICE", output="pts_only")),
        ("decode_pts_frame2", dict(threads=2, thread_type="FRAME", output="pts_only")),
        ("decode_pts_frame4", dict(threads=4, thread_type="FRAME", output="pts_only")),
        ("dense_rgb256_frame4", dict(threads=4, thread_type="FRAME", output="rgb256_bytes")),
        ("sample8_rgb256_frame4", dict(threads=4, thread_type="FRAME", output="rgb256_bytes", sample_fps=8)),
        ("sample8_letterbox256_frame4", dict(threads=4, thread_type="FRAME", output="letterbox256_bytes", sample_fps=8)),
        ("sample2_original_jpeg95_slice2", dict(threads=2, thread_type="SLICE", output="original_jpeg95", sample_fps=2)),
    ]
    results = {}
    for name, arguments in modes:
        jpeg_dir = directory / name
        if arguments["output"] == "original_jpeg95":
            jpeg_dir.mkdir(exist_ok=True)
        passes = [run_case(source, jpeg_dir=jpeg_dir, **arguments) for _ in range(repeats)]
        results[name] = {
            "parameters": arguments, "passes": passes,
            "median_elapsed_sec": statistics.median(row["elapsed_sec"] for row in passes),
        }
        print(json.dumps({"completed_case": name, "median_elapsed_sec": results[name]["median_elapsed_sec"]}), flush=True)
    seek_passes = [run_seek_case(source) for _ in range(repeats)]
    results["four_half_second_boundary_seeks_frame4"] = {
        "parameters": {"threads": 4, "thread_type": "FRAME", "output": "rgb256_bytes"},
        "passes": seek_passes,
        "median_elapsed_sec": statistics.median(row["elapsed_sec"] for row in seek_passes),
    }
    return {
        "schema_version": "1.0", "measurement": "cpu_synthetic_only_no_gpu_or_model",
        "host": {"platform": platform.platform(), "processor": platform.processor(),
                 "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
                 "pyav": av.__version__, "libavcodec": list(av.library_versions["libavcodec"])},
        "fixture": {"generated": True, "width": 464, "height": 848, "fps": 30,
                    "frames": frame_count, "duration_sec": frame_count / 30,
                    "keyframe_period_frames": 30, "b_frames": 2,
                    "file_bytes": source.stat().st_size, "generation_sec": generation},
        "repeats": repeats, "cases": results,
        "limitations": [
            "Synthetic local CPU microbenchmark; no user-upload, private-pixel, GPU, encoder-model, temporal-head, upload or serving latency was measured.",
            "The direct-resize mode measures throughput, not acceptance of a production aspect-ratio policy.",
            "Byte-batch conversion includes CPU copies, but excludes pinned staging and host-to-device transfer.",
            "Only generated synthetic media is decoded; native scene complexity may change throughput.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("tmp/a05-synthetic-decode"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/video/synthetic-decode-cpu.json"))
    parser.add_argument("--frames", type=int, default=4276)
    parser.add_argument("--repeats", type=int, default=2)
    arguments = parser.parse_args()
    result = benchmark(arguments.directory, arguments.frames, arguments.repeats)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(arguments.output), "case_count": len(result["cases"])}))
