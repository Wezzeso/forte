"""Optional tracked-feature Transformer, loaded only through a separate type gate.

This is not the older frozen-frame-feature head. Unknown channels have explicit
masks, 2-D hand proximity is not physical contact, and scores are proposals that
still need independent VLM verification and the analyzer's procedural proof rules.
Importing this module neither imports torch nor opens a checkpoint or video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

SUPPORTED_TYPES = frozenset({"PERSONAL_ITEMS", "KEY_LEFT_IN_SAFE", "NO_CLIENT_ID",
    "NO_AUTHENTICITY_CHECK", "NO_RECOUNT", "CASH_HANDOVER_IRREGULAR"})
OMISSION_TYPES = frozenset({"NO_CLIENT_ID", "NO_AUTHENTICITY_CHECK", "NO_RECOUNT"})
STEP_SECONDS = 0.5
_SHA = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _number(value: Any, lo: float, hi: float) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and lo <= value <= hi


def _names(values: Any) -> bool:
    return (isinstance(values, list) and bool(values) and len(values) == len(set(values))
        and all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_]+", v) for v in values))


def default_feature_schema(*, classes: Sequence[str], zones: Sequence[str],
        contact_classes: Sequence[str], seats: Sequence[str],
        producer: Mapping[str, Any],
        embedding: Mapping[str, Any] | None = None, max_count: int = 8) -> dict:
    schema = {"schema_version": "forte.tracked-head.features.v1", "step_seconds": STEP_SECONDS,
        "classes": list(classes), "zones": list(zones), "contact_classes": list(contact_classes),
        "seats": list(seats), "max_count": max_count, "contact_signal": "proximity_2d",
        "producer": dict(producer), "embedding": dict(embedding) if embedding is not None else None}
    validate_feature_schema(schema)
    return schema


def validate_feature_schema(schema: Mapping[str, Any]) -> None:
    _require(schema.get("schema_version") == "forte.tracked-head.features.v1"
        and schema.get("step_seconds") == STEP_SECONDS, "tracked_feature_schema_invalid")
    _require(all(_names(schema.get(k)) for k in ("classes", "zones", "contact_classes", "seats")),
        "tracked_feature_axes_invalid")
    _require(set(schema["contact_classes"]) <= set(schema["classes"])
        and type(schema.get("max_count")) is int and 1 <= schema["max_count"] <= 128
        and schema.get("contact_signal") == "proximity_2d", "tracked_feature_channels_invalid")
    producer = schema.get("producer")
    _require(isinstance(producer, dict) and producer.get("local_only") is True
        and all(_SHA.fullmatch(str(producer.get(k, ""))) for k in
            ("detector_checkpoint_sha256", "detector_gate_sha256", "tracking_config_sha256",
             "layout_sha256", "hand_model_sha256")), "tracked_producer_identity_required")
    embedding = schema.get("embedding")
    if embedding is not None:
        _require(isinstance(embedding, dict) and embedding.get("local_only") is True
            and embedding.get("reviewed_by") and embedding.get("reviewed_at")
            and isinstance(embedding.get("model_id"), str) and embedding["model_id"]
            and all(_SHA.fullmatch(str(embedding.get(k, ""))) for k in
                ("checkpoint_sha256", "extractor_config_sha256"))
            and type(embedding.get("dimension")) is int and 1 <= embedding["dimension"] <= 8192,
            "tracked_embedding_identity_required")


def feature_names(schema: Mapping[str, Any]) -> list[str]:
    validate_feature_schema(schema)
    names = [f"count:{z}:{c}" for z in schema["zones"] for c in schema["classes"]]
    names += [f"proximity_2d:{z}:{c}" for z in schema["zones"] for c in schema["contact_classes"]]
    names += [f"seat:{s}" for s in schema["seats"]]
    if schema.get("embedding"):
        names += [f"embedding:{i}" for i in range(schema["embedding"]["dimension"])]
    return names


@dataclass(frozen=True)
class TrackedFeatureSequence:
    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    observed: tuple[tuple[bool, ...], ...]

    def __post_init__(self):
        _require(bool(self.times) and len(self.times) == len(self.values) == len(self.observed),
            "tracked_sequence_shape_invalid")
        dimension = len(self.values[0])
        _require(dimension > 0 and all(len(v) == len(m) == dimension
            and all(type(value) is bool for value in m)
            and all(_number(value, -1e6, 1e6) for value in v)
            for v, m in zip(self.values, self.observed)), "tracked_sequence_shape_invalid")
        _require(all(_number(t, 0, 100000) and abs(t - i * STEP_SECONDS) < 1e-7
            for i, t in enumerate(self.times)), "tracked_sequence_grid_invalid")


def featurize_sequence(rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> TrackedFeatureSequence:
    """Encode exact half-second observations; absent/missing fields stay unknown.

    Counts are normalized by a fixed clip value. A known zero must be explicitly
    supplied by a calibrated producer with valid visibility; an empty detection
    list cannot supply it. Seats accept strict booleans, never truthy strings.
    Optional embeddings must match the pinned producer identity on every row.
    """
    names = feature_names(schema)
    _require(bool(rows), "tracked_sequence_empty")
    values, masks, times = [], [], []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping) and _number(row.get("t"), 0, 100000)
            and abs(row["t"] - index * STEP_SECONDS) < 1e-7, "tracked_sequence_grid_invalid")
        _require(row.get("producer_identity_sha256") == canonical_sha256(schema["producer"]),
            "tracked_producer_identity_mismatch")
        encoded, known = [], []
        for category in ("counts", "contacts", "seats"):
            _require(isinstance(row.get(category, {}), Mapping), "tracked_channel_map_invalid")
        allowed = {"counts": {f"{z}/{c}" for z in schema["zones"] for c in schema["classes"]},
            "contacts": {f"{z}/{c}" for z in schema["zones"] for c in schema["contact_classes"]},
            "seats": set(schema["seats"])}
        _require(all(set(row.get(k, {})) <= v for k, v in allowed.items()), "tracked_unknown_channel")
        for z in schema["zones"]:
            for c in schema["classes"]:
                value = row.get("counts", {}).get(f"{z}/{c}")
                _require(value is None or type(value) is int and value >= 0, "tracked_count_invalid")
                encoded.append(min(value, schema["max_count"]) / schema["max_count"] if value is not None else 0.0)
                known.append(value is not None)
        for z in schema["zones"]:
            for c in schema["contact_classes"]:
                value = row.get("contacts", {}).get(f"{z}/{c}")
                _require(value is None or type(value) is bool, "tracked_contact_invalid")
                encoded.append(float(value) if value is not None else 0.0)
                known.append(value is not None)
        for seat in schema["seats"]:
            value = row.get("seats", {}).get(seat)
            _require(value is None or type(value) is bool, "tracked_seat_state_invalid")
            encoded.append(float(value) if value is not None else 0.0)
            known.append(value is not None)
        embedding = schema.get("embedding")
        vector = row.get("embedding")
        if embedding:
            _require(vector is None or (row.get("embedding_identity_sha256") == canonical_sha256(embedding)
                and isinstance(vector, list) and len(vector) == embedding["dimension"]
                and all(_number(v, -1e6, 1e6) for v in vector)), "tracked_embedding_mismatch")
            encoded.extend(vector if vector is not None else [0.0] * embedding["dimension"])
            known.extend([vector is not None] * embedding["dimension"])
        else:
            _require(vector is None and row.get("embedding_identity_sha256") is None,
                "tracked_unconfigured_embedding")
        _require(len(encoded) == len(names), "tracked_feature_dimension_invalid")
        values.append(tuple(encoded)); masks.append(tuple(known)); times.append(float(row["t"]))
    return TrackedFeatureSequence(tuple(times), tuple(values), tuple(masks))


def validate_model_config(config: Mapping[str, Any]) -> None:
    _require(config.get("architecture") == "tracked_transformer_v1", "tracked_architecture_invalid")
    for field, low, high in (("hidden_dim", 8, 512), ("num_heads", 1, 16),
            ("num_layers", 1, 6), ("max_steps", 2, 4096)):
        _require(type(config.get(field)) is int and low <= config[field] <= high,
            "tracked_model_shape_invalid")
    _require(config["hidden_dim"] % config["num_heads"] == 0
        and _number(config.get("dropout"), 0, .5), "tracked_model_shape_invalid")
    types = config.get("types")
    _require(isinstance(types, list) and bool(types) and len(types) == len(set(types))
        and set(types) <= SUPPORTED_TYPES, "tracked_model_types_invalid")


def build_transformer(config: Mapping[str, Any], dimension: int):
    """Import torch on explicit use only; randomly initialize a genuine small head."""
    validate_model_config(config)
    _require(type(dimension) is int and dimension > 0, "tracked_feature_dimension_invalid")
    import torch
    from torch import nn

    class TrackedTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = config["hidden_dim"]
            self.project = nn.Sequential(nn.Linear(dimension * 2, hidden), nn.LayerNorm(hidden), nn.GELU())
            self.position = nn.Embedding(config["max_steps"], hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=config["num_heads"],
                dim_feedforward=hidden * 2, dropout=config["dropout"], batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=config["num_layers"], enable_nested_tensor=False)
            self.output = nn.Linear(hidden, len(config["types"]))

        def forward(self, values, observed, step_mask=None):
            _require(values.ndim == 3 and values.shape[-1] == dimension
                and observed.shape == values.shape and observed.dtype == torch.bool,
                "tracked_tensor_shape_invalid")
            batch, steps = values.shape[:2]
            _require(0 < steps <= config["max_steps"], "tracked_sequence_too_long")
            if step_mask is None:
                step_mask = torch.ones((batch, steps), dtype=torch.bool, device=values.device)
            _require(step_mask.shape == values.shape[:2] and step_mask.dtype == torch.bool
                and bool(step_mask.any(dim=1).all()), "tracked_step_mask_invalid")
            lengths = step_mask.sum(dim=1)
            _require(bool((step_mask == (torch.arange(steps, device=values.device)[None, :] < lengths[:, None])).all()),
                "tracked_padding_must_be_right_contiguous")
            effective = observed & step_mask.unsqueeze(-1)
            clean = values.masked_fill(~effective, 0).to(self.project[0].weight.dtype)
            _require(bool(torch.isfinite(clean).all()), "tracked_nonfinite_observed_feature")
            encoded = self.project(torch.cat((clean, effective.to(clean.dtype)), dim=-1))
            encoded = encoded + self.position(torch.arange(steps, device=values.device))[None, :, :]
            result = self.output(self.temporal(encoded, src_key_padding_mask=~step_mask))
            return result.masked_fill(~step_mask.unsqueeze(-1), 0)

    return TrackedTransformer()


@dataclass(frozen=True)
class TrackedHeadProposal:
    type_id: str
    t: float
    probability: float
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "trained_tracked_temporal_head"


def _pinned_json(path: Path, expected: str) -> dict:
    _require(bool(_SHA.fullmatch(str(expected))) and path.is_file()
        and file_sha256(path) == expected, "tracked_artifact_hash_mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(result, dict), "tracked_artifact_object_required")
    return result


def _artifact_file(root: Path, name: Any, digest: Any) -> Path:
    _require(isinstance(name, str) and bool(name) and Path(name).name == name
        and ":" not in name and "\\" not in name, "tracked_artifact_path_invalid")
    path = (root / name).resolve()
    _require(path.is_relative_to(root.resolve()) and path.is_file()
        and bool(_SHA.fullmatch(str(digest))) and file_sha256(path) == digest,
        "tracked_artifact_hash_mismatch")
    return path


def validate_full_video_gate(gate: Mapping[str, Any], artifact: Mapping[str, Any], artifact_sha256: str,
        model: Mapping[str, Any], schema: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, list[str]]:
    """A training validation receipt alone never enables an inference type."""
    _require(gate.get("schema_version") == "forte.tracked-head.full-video-gate.v1"
        and gate.get("state") == "passed" and gate.get("reviewed_by") and gate.get("reviewed_at")
        and gate.get("full_video_type_evaluation") is True
        and gate.get("independent_vlm_verification_required") is True
        and gate.get("procedural_proof_rules_required") is True
        and gate.get("artifact_sha256") == artifact_sha256, "tracked_full_video_gate_required")
    for key in ("checkpoint", "model_config", "feature_schema", "thresholds"):
        _require(gate.get(key + "_sha256") == artifact["files"][key]["sha256"], "tracked_gate_identity_mismatch")
    _require(all(_SHA.fullmatch(str(gate.get(k, ""))) for k in
        ("evaluation_report_sha256", "evaluation_data_manifest_sha256")), "tracked_gate_evaluation_identity_missing")
    enabled = gate.get("enabled_types")
    _require(isinstance(enabled, list) and bool(enabled) and len(enabled) == len(set(enabled))
        and set(enabled) <= set(model["types"]), "tracked_gate_types_invalid")
    channels = set(feature_names(schema))
    required = gate.get("required_channels_by_type", {})
    _require(isinstance(required, dict), "tracked_gate_required_channels_missing")
    for type_id in enabled:
        calibration = thresholds.get("per_type", {}).get(type_id, {})
        metrics = gate.get("per_type", {}).get(type_id, {})
        fields = required.get(type_id)
        _require(calibration.get("supported") is True and _number(calibration.get("threshold"), 0, 1),
            "tracked_type_calibration_missing")
        _require(isinstance(metrics, dict) and metrics.get("passed") is True
            and type(metrics.get("clean_videos")) is int and metrics["clean_videos"] > 0
            and type(metrics.get("positive_videos")) is int and metrics["positive_videos"] > 0
            and type(metrics.get("false_positive_videos")) is int and metrics["false_positive_videos"] == 0
            and _number(metrics.get("recall"), .95, 1)
            and metrics.get("strict_improvement") is True
            and metrics.get("no_type_regressions") is True
            and metrics.get("timecode_gate_passed") is True,
            "tracked_full_video_type_evidence_missing")
        _require(isinstance(fields, list) and bool(fields) and len(fields) == len(set(fields))
            and set(fields) <= channels, "tracked_gate_required_channels_missing")
    return {type_id: list(required[type_id]) for type_id in enabled}


class TrackedTemporalHead:
    """Exact local artifact loader. It has no auto-discovery or environment toggle."""
    @classmethod
    def load(cls, artifact_path: Path, *, expected_artifact_sha256: str,
            gate_path: Path, expected_gate_sha256: str, device: str = "cpu") -> "TrackedTemporalHead":
        artifact_path, gate_path = Path(artifact_path).resolve(), Path(gate_path).resolve()
        artifact = _pinned_json(artifact_path, expected_artifact_sha256)
        _require(artifact.get("schema_version") == "forte.tracked-head.artifact.v1"
            and artifact.get("state") == "trained_unselected"
            and artifact.get("requires_independent_verification") is True,
            "tracked_training_artifact_invalid")
        files = artifact.get("files", {})
        _require(isinstance(files, dict) and set(files) == {"checkpoint", "model_config", "feature_schema", "thresholds", "validation_metrics"},
            "tracked_artifact_inventory_invalid")
        paths = {key: _artifact_file(artifact_path.parent, entry.get("path"), entry.get("sha256"))
            for key, entry in files.items()}
        model = _pinned_json(paths["model_config"], files["model_config"]["sha256"])
        schema = _pinned_json(paths["feature_schema"], files["feature_schema"]["sha256"])
        thresholds = _pinned_json(paths["thresholds"], files["thresholds"]["sha256"])
        validate_model_config(model); validate_feature_schema(schema)
        _require(thresholds.get("schema_version") == "forte.tracked-head.thresholds.v1"
            and thresholds.get("feature_schema_sha256") == files["feature_schema"]["sha256"],
            "tracked_threshold_schema_mismatch")
        gate = _pinned_json(gate_path, expected_gate_sha256)
        required = validate_full_video_gate(gate, artifact, expected_artifact_sha256, model, schema, thresholds)
        _require(device == "cpu" or device == "cuda" or re.fullmatch(r"cuda:[0-9]+", device), "tracked_device_invalid")
        expected_torch = artifact.get("runtime_versions", {}).get("torch")
        _require(isinstance(expected_torch, str) and bool(expected_torch), "tracked_torch_version_pin_required")
        try:
            installed_torch = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            raise ValueError("tracked_torch_not_installed") from None
        _require(installed_torch == expected_torch, "tracked_torch_version_mismatch")
        import torch
        instance = cls()
        instance.schema, instance.config, instance.thresholds = schema, model, thresholds
        instance.required, instance.device = required, device
        instance.artifact_sha256, instance.gate_sha256 = expected_artifact_sha256, expected_gate_sha256
        instance.feature_schema_sha256 = files["feature_schema"]["sha256"]
        instance.model = build_transformer(model, len(feature_names(schema)))
        # Restricted tensor-only loading; no arbitrary checkpoint pickle objects.
        state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
        instance.model.load_state_dict(state, strict=True)
        instance.model.to(device).eval()
        return instance

    def propose(self, rows: Sequence[Mapping[str, Any]], *, duration_sec: float) -> list[TrackedHeadProposal]:
        _require(_number(duration_sec, 0.001, 100000)
            and len(rows) == math.ceil(duration_sec / STEP_SECONDS), "tracked_full_video_grid_incomplete")
        sequence = featurize_sequence(rows, self.schema)
        _require(len(rows) <= self.config["max_steps"], "tracked_sequence_too_long")
        import torch
        values = torch.tensor(sequence.values, dtype=torch.float32, device=self.device)[None]
        observed = torch.tensor(sequence.observed, dtype=torch.bool, device=self.device)[None]
        with torch.inference_mode():
            probabilities = self.model(values, observed).sigmoid()[0].cpu().tolist()
        return self.proposals_from_probabilities(sequence, probabilities)

    def proposals_from_probabilities(self, sequence: TrackedFeatureSequence,
            probabilities: Sequence[Sequence[float]]) -> list[TrackedHeadProposal]:
        """Pure decoding seam also used for CPU-only contract tests."""
        sequence.__post_init__()
        _require(len(sequence.values[0]) == len(feature_names(self.schema)), "tracked_sequence_shape_invalid")
        _require(len(probabilities) == len(sequence.times) and all(len(p) == len(self.config["types"])
            and all(_number(v, 0, 1) for v in p) for p in probabilities), "tracked_probability_shape_invalid")
        names = feature_names(self.schema)
        result = []
        for type_id, required in self.required.items():
            columns = [names.index(name) for name in required]
            # Unknown anywhere in a required full-video channel causes abstention.
            if not all(all(mask[column] for column in columns) for mask in sequence.observed):
                continue
            index = self.config["types"].index(type_id)
            peak = max(range(len(probabilities)), key=lambda t: probabilities[t][index])
            probability = probabilities[peak][index]
            if probability < self.thresholds["per_type"][type_id]["threshold"]:
                continue
            result.append(TrackedHeadProposal(type_id, sequence.times[peak], probability, {
                "artifact_sha256": self.artifact_sha256, "gate_sha256": self.gate_sha256,
                "feature_schema_sha256": self.feature_schema_sha256,
                "frame_times": [sequence.times[peak]], "requires_independent_verification": True,
                "requires_procedural_proof_rules": True, "hand_signal": "proximity_2d",
                "required_channels": list(required)}))
        return result
