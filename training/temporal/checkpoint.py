"""Pinned safetensors checkpoint IO; no arbitrary Python pickle is loaded."""
from __future__ import annotations

import json
from pathlib import Path

from training.guards import HEX64, canonical_sha256, file_sha256, load_json, require

SCHEMA = "forte.temporal-head-checkpoint.v1"
MAX_WEIGHTS_BYTES = 256 * 1024 * 1024


def save_checkpoint(model, destination: Path, *, identity: dict, architecture: dict,
                    provenance: dict) -> dict:
    from forte.decision.head_contract import HeadIdentity
    from safetensors.torch import save_file

    destination = Path(destination).resolve()
    require(not destination.exists(), "weak_checkpoint_destination_exists")
    destination.mkdir(parents=True)
    weights = destination / "head.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(weights), metadata={"schema_version": SCHEMA})
    checksum = file_sha256(weights)
    head_identity = HeadIdentity.model_validate({**identity, "head_sha256": checksum}, strict=False)
    metadata = {"schema_version": SCHEMA, "head_identity": head_identity.model_dump(mode="json"),
                "architecture": architecture, "weights_file": weights.name, "weights_sha256": checksum,
                "state_order": ["absent", "present", "unclear"], "video_score_axis": "violation_log_odds",
                "track": "weak_candidate_unselected", "selected": False, "human_verified": False,
                "provenance": provenance}
    path = destination / "checkpoint.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"manifest_path": str(path), "manifest_sha256": file_sha256(path),
            "weights_sha256": checksum, "head_identity_sha256": canonical_sha256(metadata["head_identity"])}


def load_checkpoint(checkpoint_dir: Path, *, expected_manifest_sha256: str,
                    device: str = "cpu"):
    """Return (model, A08 HeadIdentity, metadata), after exact hash/shape checks."""
    from forte.decision.head_contract import HeadIdentity
    from training.temporal.head import FrozenFeatureWeakMILHead
    from safetensors.torch import load_file
    import torch

    directory = Path(checkpoint_dir).resolve()
    path = directory / "checkpoint.json"
    require(HEX64.fullmatch(str(expected_manifest_sha256)) is not None and path.is_file()
            and path.stat().st_size <= 1024 * 1024 and file_sha256(path) == expected_manifest_sha256,
            "weak_checkpoint_manifest_pin_mismatch")
    metadata = load_json(path)
    require(metadata.get("schema_version") == SCHEMA and metadata.get("weights_file") == "head.safetensors"
            and metadata.get("state_order") == ["absent", "present", "unclear"]
            and metadata.get("video_score_axis") == "violation_log_odds", "weak_checkpoint_schema_invalid")
    identity = HeadIdentity.model_validate(metadata["head_identity"], strict=False)
    weights = directory / "head.safetensors"
    require(weights.is_file() and not weights.is_symlink() and weights.stat().st_size <= MAX_WEIGHTS_BYTES
            and file_sha256(weights) == metadata.get("weights_sha256") == identity.head_sha256,
            "weak_checkpoint_weights_pin_mismatch")
    architecture = dict(metadata["architecture"])
    require(architecture.pop("kind", None) == "FrozenFeatureWeakMILHead"
            and set(architecture) == {"embedding_dim", "hidden_dim", "event_count", "mil_temperature"}
            and type(architecture["embedding_dim"]) is int and 1 <= architecture["embedding_dim"] <= 8192
            and type(architecture["hidden_dim"]) is int and 1 <= architecture["hidden_dim"] <= 1024
            and architecture["event_count"] == len(identity.event_type_ids), "weak_checkpoint_architecture_invalid")
    model = FrozenFeatureWeakMILHead(**architecture)
    state = load_file(str(weights), device="cpu")
    require(all(bool(torch.isfinite(value).all()) for value in state.values()), "weak_checkpoint_weights_nonfinite")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), identity, metadata


def sequence_inputs(sequence, features, *, device="cpu", indices=None) -> dict:
    """Only feature/time/geometry enter the model; provenance and labels do not."""
    import torch
    frames = sequence.frames if indices is None else tuple(sequence.frames[index] for index in indices)
    values = features if indices is None else features[list(indices)]
    return {"features": values.unsqueeze(0).to(device),
            "view_mask": torch.tensor([[frame.view_mask for frame in frames]], dtype=torch.bool, device=device),
            "frame_mask": torch.ones((1, len(frames)), dtype=torch.bool, device=device),
            "pts_sec": torch.tensor([[frame.pts_sec for frame in frames]], dtype=torch.float64, device=device),
            "view_roles": torch.tensor([[[0 if role == "global" else 1 for role in frame.view_roles]
                                          for frame in frames]], dtype=torch.long, device=device),
            "normalized_boxes": torch.tensor([[frame.normalized_boxes for frame in frames]], dtype=torch.float32, device=device)}
