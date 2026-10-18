"""Generated-only comparison of exact seek JPEGs with full-forward evidence.

The fixture is produced by synthetic_decode_bench.py. No GPU/model is invoked;
the report contains only operation timing, counts and equality booleans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from forte.video.batches import iter_rgb_batches, materialize_evidence_frames
from forte.video.evidence_seek import materialize_evidence_frames_seek
from forte.video.probe import probe_video


def benchmark(directory: Path) -> dict:
    source = Path("tmp/a05-synthetic-decode/generated-h264-464x848.mov")
    if not source.is_file():
        raise RuntimeError("Generate the synthetic fixture with synthetic_decode_bench.py first")
    asset = probe_video(source, job_id="synthetic-seek-benchmark")
    started = time.monotonic()
    # The native index is normally retained from the required feature pass.
    # This isolated setup pass is measured separately and is not a seek cost.
    native = [frame for batch in iter_rgb_batches(asset, target_size=(1, 1),
              aspect_policy="stretch", batch_size=128, use_subprocess=False)
              for frame in batch.frames]
    index_setup_sec = time.monotonic() - started
    selected = [min(native, key=lambda frame: abs(frame.pts_sec - target))
                for target in (10.2, 10.6, 125.4)]
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    forward = materialize_evidence_frames(asset, selected, directory / "forward", use_subprocess=False)
    forward_elapsed = time.monotonic() - started
    forward_hashes = {reference.frame_id: reference.jpeg_sha256 for reference in forward}
    observations = []
    for _ in range(2):
        measured = {}
        refs = materialize_evidence_frames_seek(asset, selected, directory / "seek", source_index=native,
                                                 use_subprocess=False, metrics=measured)
        measured["exact_ids_and_jpeg_sha_equal_full_forward"] = {
            reference.frame_id: reference.jpeg_sha256 for reference in refs
        } == forward_hashes
        observations.append(measured)
    isolated = {}
    refs = materialize_evidence_frames_seek(asset, selected, directory / "isolated", source_index=native,
                                             use_subprocess=True, metrics=isolated)
    isolated["exact_ids_and_jpeg_sha_equal_full_forward"] = {
        reference.frame_id: reference.jpeg_sha256 for reference in refs
    } == forward_hashes
    return {
        "schema_version": "1.0", "measurement": "generated_cpu_exact_seek_evidence_only",
        "implementation_sha256": hashlib.sha256(Path("src/forte/video/evidence_seek.py").read_bytes()).hexdigest(),
        "fixture": {"generated": True, "duration_sec": asset.duration_sec,
                    "native_frames": len(native), "width": asset.width, "height": asset.height},
        "selected_evidence_frames": len(selected), "native_index_setup_sec": index_setup_sec,
        "full_forward_evidence_sec": forward_elapsed,
        "grouped_seek_passes": observations,
        "grouped_seek_median_sec": statistics.median(row["total_sec"] for row in observations),
        "grouped_seek_spawn_isolation": isolated,
        "limitations": ["Generated local CPU fixture only; no supplied pixels, events, GPU or model calls.",
                       "The native index is reused from the feature pass in production; setup cost is separately recorded.",
                       "This is evidence materialization timing, not upload-to-result latency."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("tmp/a05-synthetic-seek-evidence"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/video/synthetic-seek-evidence-cpu.json"))
    options = parser.parse_args()
    result = benchmark(options.directory)
    options.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
