"""Read source packet timestamps locally; write aggregate metadata only.

No decoder, image conversion, model invocation, labels, or media writes occur.
Packet PTS are a metadata measurement, not an annotation or event prediction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from functools import reduce
import json
import math
from pathlib import Path
import statistics
import time

import av


def _ratio(value: object) -> Fraction:
    return Fraction(value.numerator, value.denominator)  # type: ignore[attr-defined]


def _stats(values: list[float]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else {}


def report_sources(directory: Path) -> dict:
    started = time.monotonic()
    sources = sorted(directory.glob("*.mov"))
    time_bases: Counter[str] = Counter()
    rates: Counter[str] = Counter()
    period_counts: Counter[str] = Counter()
    source_periods: Counter[str] = Counter()
    gcd_ticks: dict[str, Counter[int]] = defaultdict(Counter)
    durations: list[float] = []
    frame_counts: list[float] = []
    periods: list[float] = []
    keyframe_gaps: list[float] = []
    keyframe_count: list[float] = []
    millisecond_errors: list[float] = []
    duplicate_pts = missing_pts = pts_before_start = mismatched_frame_counts = 0
    failures: Counter[str] = Counter()
    unchanged = 0
    packet_count = 0
    last_pts_within_duration = 0
    for path in sources:
        before = path.stat()
        try:
            with av.open(str(path), options={"protocol_whitelist": "file,pipe"}) as container:
                stream = container.streams.video[0]
                base = _ratio(stream.time_base)
                origin = stream.start_time
                if origin is None:
                    raise ValueError("missing_video_start")
                pts: list[int] = []
                keyframes: list[int] = []
                for packet in container.demux(stream):
                    if packet.size == 0:
                        continue
                    if packet.pts is None:
                        missing_pts += 1
                        continue
                    pts.append(packet.pts)
                    if packet.is_keyframe:
                        keyframes.append(packet.pts)
                ordered = sorted(set(pts))
                if not ordered:
                    raise ValueError("missing_pts")
                deltas = [right - left for left, right in zip(ordered, ordered[1:])]
                ordered_keys = sorted(set(keyframes))
                time_bases[str(base)] += 1
                rates[str(_ratio(stream.average_rate))] += 1
                duplicate_pts += len(pts) - len(ordered)
                pts_before_start += sum(value < origin for value in ordered)
                if stream.frames and stream.frames != len(ordered):
                    mismatched_frame_counts += 1
                packet_count += len(pts)
                frame_counts.append(len(ordered))
                if stream.duration:
                    durations.append(float(stream.duration * base))
                    last_pts_within_duration += ordered[-1] - origin <= stream.duration
                keyframe_count.append(len(ordered_keys))
                keyframe_gaps.extend(float((right - left) * base) for left, right in zip(ordered_keys, ordered_keys[1:]))
                periods.extend(float(delta * base) for delta in deltas)
                period_counts.update(str(delta * base) for delta in deltas)
                source_periods[",".join(str(delta * base) for delta in sorted(set(deltas)))] += 1
                if deltas:
                    gcd_ticks[str(base)][reduce(math.gcd, deltas)] += 1
                for value in ordered:
                    seconds = float((value - origin) * base)
                    millisecond_errors.append(abs(seconds - round(seconds, 3)))
        except Exception as error:
            failures[type(error).__name__] += 1
        after = path.stat()
        unchanged += (before.st_size, before.st_mtime_ns, before.st_ino) == (
            after.st_size, after.st_mtime_ns, after.st_ino
        )
    return {
        "schema_version": "1.0",
        "measurement": "local_video_packet_pts_metadata_only",
        "source_file_count": len(sources),
        "successful_files": len(sources) - sum(failures.values()),
        "failure_types": dict(failures),
        "source_stat_unchanged_count": unchanged,
        "time_base_file_counts": dict(time_bases),
        "average_frame_rate_file_counts": dict(rates),
        "source_duration_sec": _stats(durations),
        "presentation_packet_count": packet_count,
        "files_with_last_presentation_pts_within_declared_duration": last_pts_within_duration,
        "source_presentation_packet_count": _stats(frame_counts),
        "packet_pts_period_fraction_counts": dict(period_counts),
        "source_unique_periods_file_counts": dict(source_periods),
        "packet_pts_period_sec": _stats(periods),
        "pts_delta_gcd_ticks_file_counts": {base: dict(counts) for base, counts in gcd_ticks.items()},
        "duplicate_packet_pts": duplicate_pts,
        "missing_packet_pts": missing_pts,
        "packet_pts_before_stream_start": pts_before_start,
        "files_with_declared_frames_differing_from_unique_packet_pts": mismatched_frame_counts,
        "keyframe_count_per_source": _stats(keyframe_count),
        "consecutive_keyframe_gap_sec": _stats(keyframe_gaps),
        "millisecond_format_rounding_error_sec": _stats(millisecond_errors),
        "elapsed_sec": round(time.monotonic() - started, 6),
        "limitations": [
            "Packet PTS are measured without decoding pixels; source-frame content and event boundaries were not evaluated.",
            "Millisecond string precision does not imply millisecond event localization accuracy.",
            "No event, label, prediction, filename, source hash, image or caption is included.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("videos"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/video/source-timing-aggregate.json"))
    arguments = parser.parse_args()
    report = report_sources(arguments.sources)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
