"""Whitelist numeric model telemetry; observations and attestation text stay private."""

from collections.abc import Iterable, Mapping
import math
import statistics


CLIENT_SECONDS = ("elapsed_sec", "http_elapsed_sec", "preprocessing_sec", "readiness_sec")
SERVER_MS = ("time_to_first_token_ms", "generation_time_ms", "queue_time_ms", "mean_itl_ms")
COUNTS = ("attempts", "input_frames", "encoded_media_bytes", "prompt_tokens", "generated_tokens", "total_tokens", "visual_tokens")
SCOPE = "per-request observations; request sums can overlap and are not job wall time"
SEMANTICS = {"server_time_to_first_token_ms": "TTFT, not pure GPU prefill",
             "server_generation_time_ms": "server-reported decode generation time",
             "server_queue_time_ms": "model request queue, not durable job queue"}


def number(value):
    if type(value) not in {int, float}:
        return None
    try:
        return float(value) if math.isfinite(value) and value >= 0 else None
    except (OverflowError, ValueError):
        return None


def numeric_summary(values, unit):
    values = sorted(cleaned for value in values if (cleaned := number(value)) is not None)
    try:
        total = math.fsum(values)
    except OverflowError:
        total = None
    return {"unit": unit, "count": len(values), "sum": total,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "p50": number(statistics.median(values)) if values else None,
            "p95": values[math.ceil(.95 * len(values)) - 1] if len(values) >= 20 else None}


def aggregate_inference_metrics(records: Iterable[Mapping]) -> dict:
    records = [item for item in records if isinstance(item, Mapping)]
    durations, counts = {}, {}
    for key in CLIENT_SECONDS:
        values = [value * 1000 for item in records if (value := number(item.get(key))) is not None]
        durations[key.removesuffix("_sec") + "_ms"] = numeric_summary(values, "ms")
    for key in SERVER_MS:
        values = [value for item in records if isinstance(server := item.get("server_metrics"), Mapping)
                  and (value := number(server.get(key))) is not None]
        durations["server_" + key] = numeric_summary(values, "ms")
    for key in COUNTS:
        values = [value for item in records if (value := number(item.get(key))) is not None]
        counts[key] = numeric_summary(values, "bytes" if key == "encoded_media_bytes" else "count")
    rates = [value for item in records if isinstance(server := item.get("server_metrics"), Mapping)
             and (value := number(server.get("tokens_per_second"))) is not None]
    dimensions = {}
    for item in records:
        pairs = item.get("input_dimensions")
        if not isinstance(pairs, (tuple, list)):
            continue
        for pair in pairs:
            if isinstance(pair, (tuple, list)) and len(pair) == 2 and all(type(value) is int and 0 < value <= 65536 for value in pair):
                dimensions[tuple(pair)] = dimensions.get(tuple(pair), 0) + 1
    return {"schema_version": "1.0", "request_record_count": len(records),
            "request_durations": durations, "numeric_counts": counts,
            "server_tokens_per_second": numeric_summary(rates, "tokens/s"),
            "encoded_input_frame_resolutions": [{"width": width, "height": height, "frame_count": count}
                                                 for (width, height), count in sorted(dimensions.items())],
            "scope": SCOPE,
            "timing_semantics": dict(SEMANTICS),
            "processor_output_dimensions": "unmeasured by standard response",
            "p95_minimum_request_observations": 20}


def sanitize_model_metrics(value) -> dict | None:
    """Rebuild an aggregated metric shape; never trust notes, units or extra keys."""
    if not isinstance(value, Mapping):
        return None
    count = value.get("request_record_count")
    if type(count) is not int or count < 0:
        return None

    def summary(raw, unit):
        if not isinstance(raw, Mapping) or type(raw.get("count")) is not int or not 0 <= raw["count"] <= count:
            return None
        return {"unit": unit, "count": raw["count"],
                **{name: number(raw.get(name)) for name in ("sum", "min", "max", "p50")},
                "p95": number(raw.get("p95")) if raw["count"] >= 20 else None}

    durations, counts = {}, {}
    raw_durations, raw_counts = value.get("request_durations"), value.get("numeric_counts")
    expected = tuple(key.removesuffix("_sec") + "_ms" for key in CLIENT_SECONDS) + tuple("server_" + key for key in SERVER_MS)
    if isinstance(raw_durations, Mapping):
        durations = {key: cleaned for key in expected if (cleaned := summary(raw_durations.get(key), "ms")) is not None}
    if isinstance(raw_counts, Mapping):
        counts = {key: cleaned for key in COUNTS
                  if (cleaned := summary(raw_counts.get(key), "bytes" if key == "encoded_media_bytes" else "count")) is not None}
    dimensions = []
    raw_dimensions = value.get("encoded_input_frame_resolutions")
    if isinstance(raw_dimensions, (list, tuple)):
        for pair in raw_dimensions:
            if (isinstance(pair, Mapping) and all(type(pair.get(key)) is int and pair[key] > 0 for key in ("width", "height", "frame_count"))
                    and pair["width"] <= 65536 and pair["height"] <= 65536):
                dimensions.append({key: pair[key] for key in ("width", "height", "frame_count")})
    return {"schema_version": "1.0", "request_record_count": count,
            "request_durations": durations, "numeric_counts": counts,
            "server_tokens_per_second": summary(value.get("server_tokens_per_second"), "tokens/s"),
            "encoded_input_frame_resolutions": dimensions,
            "scope": SCOPE, "timing_semantics": dict(SEMANTICS),
            "processor_output_dimensions": "unmeasured by standard response",
            "p95_minimum_request_observations": 20}
