"""Synthetic contract checks; no real videos, downloads or claimed ML performance."""
from __future__ import annotations

import builtins
import copy
import importlib.metadata
import math
from pathlib import Path

import pytest

from forte.fast.tracked_head import (TrackedFeatureSequence, TrackedTemporalHead, build_transformer, canonical_sha256,
    default_feature_schema, feature_names, featurize_sequence, validate_full_video_gate)
from training.guards import TrainingGateError
from training.specialist_temporal import (calibrate_thresholds, interval_targets, preflight,
    train, validate_feature_manifest)
from training.specialists.common import file_sha256, write_json


@pytest.fixture
def feature_schema():
    return default_feature_schema(classes=["person", "id_card"], zones=["desk"],
        contact_classes=["id_card"], seats=["cashier"], producer={"local_only": True,
            **{k: "a" * 64 for k in ("detector_checkpoint_sha256", "detector_gate_sha256",
                "tracking_config_sha256", "layout_sha256", "hand_model_sha256")}})


def frames(schema):
    return [{"t": i * .5, "producer_identity_sha256": canonical_sha256(schema["producer"]),
        "counts": {"desk/person": 1, "desk/id_card": 0}, "contacts": {"desk/id_card": False},
        "seats": {"cashier": True}} for i in range(4)]


def test_missing_is_masked_and_explicit_zero_is_known(feature_schema):
    rows = frames(feature_schema)
    del rows[1]["counts"]["desk/id_card"]
    rows[1]["seats"]["cashier"] = None
    result = featurize_sequence(rows, feature_schema)
    assert result.values[0][1] == result.values[1][1] == 0
    assert result.observed[0][1] is True
    assert result.observed[1][1] is False
    assert result.observed[1][-1] is False


@pytest.mark.parametrize("value", ["false", 0, 1, [], {}, float("nan")])
def test_seat_occupancy_requires_strict_known_boolean(feature_schema, value):
    rows = frames(feature_schema)
    rows[0]["seats"]["cashier"] = value
    with pytest.raises(ValueError, match="tracked_seat_state_invalid"):
        featurize_sequence(rows, feature_schema)


def test_grid_identity_and_embedding_guards(feature_schema):
    rows = frames(feature_schema)
    rows[1]["t"] = .6
    with pytest.raises(ValueError, match="grid"):
        featurize_sequence(rows, feature_schema)
    rows = frames(feature_schema)
    rows[0]["producer_identity_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="producer_identity"):
        featurize_sequence(rows, feature_schema)
    rows = frames(feature_schema)
    rows[0]["embedding"] = [0.1]
    with pytest.raises(ValueError, match="unconfigured_embedding"):
        featurize_sequence(rows, feature_schema)


def test_pinned_embedding_unknown_and_wrong_encoder(feature_schema):
    feature_schema["embedding"] = {"local_only": True, "reviewed_by": "synthetic-test",
        "reviewed_at": "2026-10-18", "model_id": "local/pinned-encoder", "dimension": 2,
        "checkpoint_sha256": "b" * 64, "extractor_config_sha256": "c" * 64}
    rows = frames(feature_schema)
    assert featurize_sequence(rows, feature_schema).observed[0][-2:] == (False, False)
    rows[0]["embedding"] = [.1, -.1]
    rows[0]["embedding_identity_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="embedding_mismatch"):
        featurize_sequence(rows, feature_schema)
    rows[0]["embedding_identity_sha256"] = canonical_sha256(feature_schema["embedding"])
    assert featurize_sequence(rows, feature_schema).observed[0][-2:] == (True, True)


def test_reviewed_intervals_leave_unknowns_and_reject_conflicts():
    interval = {"type_id": "NO_CLIENT_ID", "label": 1, "start_sec": .5, "end_sec": 1,
        "human_reviewed": True, "reviewer": "synthetic-test", "reviewed_at": "2026-10-18"}
    assert interval_targets([interval], (0., .5, 1., 1.5), ["NO_CLIENT_ID"], 2) == [[-100], [1], [-100], [-100]]
    with pytest.raises(TrainingGateError, match="interval_overlap"):
        interval_targets([interval, dict(interval, label=0)], (0., .5, 1., 1.5), ["NO_CLIENT_ID"], 2)
    with pytest.raises(TrainingGateError, match="review_required"):
        interval_targets([dict(interval, human_reviewed=False)], (0., .5), ["NO_CLIENT_ID"], 1)


@pytest.fixture
def inputs(tmp_path, feature_schema):
    schema_path = tmp_path / "schema.json"
    write_json(schema_path, feature_schema)
    schema_sha = file_sha256(schema_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "fixture.bin").write_bytes(b"synthetic-dataset-fixture")
    sources, records = [], []
    for i, split in enumerate(("train", "validation", "holdout")):
        source = {"sample_id": split, "source_recording_sha256": str(i + 1) * 64,
            "content_sha256": str(i + 4) * 64, "scenario_group": "group_" + split,
            "camera_id": "camera_" + split, "origin": "own", "split": split,
            "labels_reviewed": True, "exhaustive_labels": True, "label_reviewer": "synthetic-test",
            "label_reviewed_at": "2026-10-18", "provenance": {"reference": "synthetic-fixture-only",
                "participants_consented": True, "recording_authorized": True, "commercial_use_allowed": True}}
        sources.append(source)
        feature_path = tmp_path / (split + ".json")
        payload = {"schema_version": "forte.tracked-head.sequence.v1", "feature_schema_sha256": schema_sha,
            "source_recording_sha256": source["source_recording_sha256"], "complete_recording": True,
            "duration_sec": 2., "frames": frames(feature_schema), "label_intervals": [{
                "type_id": "NO_CLIENT_ID", "start_sec": 0., "end_sec": 2., "label": i % 2,
                "human_reviewed": True, "reviewer": "synthetic-test", "reviewed_at": "2026-10-18"}]}
        if split == "train":
            payload["label_intervals"][0]["end_sec"] = 1.
            payload["label_intervals"].append(dict(payload["label_intervals"][0],
                start_sec=1., end_sec=2., label=1))
        write_json(feature_path, payload)
        records.append({"sequence_id": split, "source_sample_id": split,
            **{k: source[k] for k in ("source_recording_sha256", "scenario_group", "camera_id", "split", "origin")},
            "feature_path": feature_path.name, "feature_sha256": file_sha256(feature_path),
            "human_reviewed": True, "reviewer": "synthetic-test", "reviewed_at": "2026-10-18"})
    lock = {"schema_version": "forte.specialists.dataset-lock.v1", "state": "frozen", "records": sources,
        "files": {"fixture.bin": file_sha256(dataset / "fixture.bin")}}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(dataset / "dataset-lock.json", lock)
    manifest = {"schema_version": "forte.tracked-head.data.v1", "frozen": True,
        "reviewed_by": "synthetic-test", "reviewed_at": "2026-10-18", "feature_schema_sha256": schema_sha,
        "dataset_lock_sha256": file_sha256(dataset / "dataset-lock.json"), "records": records}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    config = {"schema_version": "forte.tracked-head.train.v1", "dataset_dir": str(dataset),
        "expected_dataset_lock_sha256": manifest["dataset_lock_sha256"],
        "feature_manifest": str(manifest_path), "expected_feature_manifest_sha256": file_sha256(manifest_path),
        "feature_schema": str(schema_path), "expected_feature_schema_sha256": schema_sha,
        "output_dir": str(tmp_path / "run"), "runtime_versions": {"torch": "2.13.0+cu129"},
        "optimizer_steps": 1, "seed": 17, "learning_rate": .0003, "weight_decay": .01,
        "device": "cuda", "allow_cpu_training": False, "model": {"architecture": "tracked_transformer_v1",
            "hidden_dim": 16, "num_heads": 2, "num_layers": 1, "max_steps": 32,
            "dropout": .1, "types": ["NO_CLIENT_ID"]}}
    return config, manifest, lock, feature_schema


def test_preflight_reads_no_torch_or_holdout_feature_bytes(inputs, monkeypatch):
    config, _, _, _ = inputs
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    (Path(config["feature_manifest"]).parent / "holdout.json").unlink()
    result = preflight(config)
    assert result["summary"]["split_counts"] == {"train": 1, "validation": 1}
    assert result["summary"]["holdout_used_for_training_or_thresholds"] is False
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("change,error", [("lineage", "lineage"), ("organizer", "forbidden"),
    ("camera", "holdout_leakage"), ("review", "review_required"), ("hash", "hash_mismatch")])
def test_manifest_rejects_unreviewed_leaked_or_changed_inputs(inputs, change, error):
    config, manifest, lock, schema = inputs
    if change == "lineage":
        manifest["records"][0]["scenario_group"] = "wrong"
    elif change == "organizer":
        lock["records"][0]["origin"] = "organizer"
        manifest["records"][0]["origin"] = "organizer"
    elif change == "camera":
        lock["records"][0]["camera_id"] = lock["records"][2]["camera_id"]
        manifest["records"][0]["camera_id"] = lock["records"][2]["camera_id"]
    elif change == "review":
        manifest["records"][0]["human_reviewed"] = False
    elif change == "hash":
        manifest["records"][0]["feature_sha256"] = "f" * 64
    with pytest.raises(TrainingGateError, match=error):
        validate_feature_manifest(manifest, lock, config["expected_dataset_lock_sha256"], schema,
            config["expected_feature_schema_sha256"], Path(config["feature_manifest"]).parent, config["model"])


def test_train_fails_closed_without_torch_and_has_no_output(inputs, monkeypatch):
    config = inputs[0]
    def missing(_):
        raise importlib.metadata.PackageNotFoundError()
    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(TrainingGateError, match="torch_not_installed"):
        train(config)
    assert not Path(config["output_dir"]).exists()


def test_untrained_type_cannot_be_calibrated_from_validation_alone(inputs):
    config = inputs[0]
    config["model"]["types"].append("KEY_LEFT_IN_SAFE")
    with pytest.raises(TrainingGateError, match="per_type_reviewed_training_support_required"):
        preflight(config)


def test_duplicate_complete_recording_cannot_inflate_video_support(inputs):
    config, manifest, lock, schema = inputs
    duplicate = dict(manifest["records"][1], sequence_id="duplicate_validation")
    manifest["records"].append(duplicate)
    with pytest.raises(TrainingGateError, match="duplicate_complete_recording"):
        validate_feature_manifest(manifest, lock, config["expected_dataset_lock_sha256"], schema,
            config["expected_feature_schema_sha256"], Path(config["feature_manifest"]).parent, config["model"])


def test_thresholds_use_only_fully_reviewed_videos_and_zero_fp():
    rows = [{"sequence_id": "clean", "times": [0., .5], "probabilities": [[.7], [.2]], "targets": [[0], [0]]},
        {"sequence_id": "positive", "times": [0., .5], "probabilities": [[.2], [.9]], "targets": [[0], [1]]},
        {"sequence_id": "unknown", "times": [0., .5], "probabilities": [[.99], [.1]], "targets": [[-100], [0]]}]
    thresholds, metrics = calibrate_thresholds(rows, ["NO_CLIENT_ID"])
    assert .7 < thresholds["NO_CLIENT_ID"]["threshold"] < .71
    assert metrics["NO_CLIENT_ID"]["video_false_positive"] == 0
    assert metrics["NO_CLIENT_ID"]["video_recall"] == 1
    assert metrics["NO_CLIENT_ID"]["unknown_frames_excluded"] == 1
    thresholds, _ = calibrate_thresholds(rows[:1], ["NO_CLIENT_ID"])
    assert thresholds["NO_CLIENT_ID"]["supported"] is False


def test_runtime_type_gate_and_unknown_abstention(feature_schema):
    artifact = {"files": {k: {"sha256": "a" * 64} for k in
        ("checkpoint", "model_config", "feature_schema", "thresholds")}}
    thresholds = {"per_type": {"NO_CLIENT_ID": {"supported": True, "threshold": .8}}}
    model = {"types": ["NO_CLIENT_ID"]}
    gate = {"schema_version": "forte.tracked-head.full-video-gate.v1", "state": "passed",
        "reviewed_by": "synthetic-test", "reviewed_at": "2026-10-18", "artifact_sha256": "b" * 64,
        "full_video_type_evaluation": True, "independent_vlm_verification_required": True,
        "procedural_proof_rules_required": True, "evaluation_report_sha256": "c" * 64,
        "evaluation_data_manifest_sha256": "d" * 64, "enabled_types": ["NO_CLIENT_ID"],
        **{k + "_sha256": "a" * 64 for k in artifact["files"]},
        "required_channels_by_type": {"NO_CLIENT_ID": ["count:desk:id_card"]},
        "per_type": {"NO_CLIENT_ID": {"passed": True, "clean_videos": 19, "positive_videos": 2,
            "false_positive_videos": 0, "recall": 1., "strict_improvement": True,
            "no_type_regressions": True, "timecode_gate_passed": True}}}
    required = validate_full_video_gate(gate, artifact, "b" * 64, model, feature_schema, thresholds)
    damaged = copy.deepcopy(gate)
    damaged["per_type"]["NO_CLIENT_ID"]["timecode_gate_passed"] = False
    with pytest.raises(ValueError, match="type_evidence"):
        validate_full_video_gate(damaged, artifact, "b" * 64, model, feature_schema, thresholds)
    runtime = TrackedTemporalHead()
    runtime.config, runtime.schema, runtime.thresholds, runtime.required = model, feature_schema, thresholds, required
    runtime.artifact_sha256, runtime.gate_sha256, runtime.feature_schema_sha256 = "b" * 64, "e" * 64, "a" * 64
    rows = frames(feature_schema)
    proposals = runtime.proposals_from_probabilities(featurize_sequence(rows, feature_schema), [[.1], [.9], [.2], [.1]])
    assert len(proposals) == 1 and proposals[0].t == .5
    assert proposals[0].evidence["requires_independent_verification"] is True
    assert proposals[0].evidence["requires_procedural_proof_rules"] is True
    rows[3]["counts"]["desk/id_card"] = None
    assert runtime.proposals_from_probabilities(featurize_sequence(rows, feature_schema), [[.1], [.9], [.2], [.1]]) == []


def test_runtime_rejects_artifact_before_importing_torch(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    write_json(artifact, {"schema_version": "forte.tracked-head.artifact.v1", "state": "preflight_only"})
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] != "torch"
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="training_artifact_invalid"):
        TrackedTemporalHead.load(artifact, expected_artifact_sha256=file_sha256(artifact),
            gate_path=tmp_path / "missing.json", expected_gate_sha256="a" * 64)


def test_incomplete_full_video_rejected_before_torch(feature_schema):
    runtime = TrackedTemporalHead()
    with pytest.raises(ValueError, match="full_video_grid_incomplete"):
        runtime.propose(frames(feature_schema), duration_sec=3.)


def test_decoder_sequence_cannot_hide_missing_observations():
    with pytest.raises(ValueError, match="sequence_shape"):
        TrackedFeatureSequence((0.,), (), ())
    with pytest.raises(ValueError, match="sequence_shape"):
        TrackedFeatureSequence((0., .5), ((0.,), (0.,)), ((True,),))
    with pytest.raises(ValueError, match="sequence_shape"):
        TrackedFeatureSequence((0.,), ((0.,),), ((1,),))


def test_torch_forward_masks_unknown_values_and_has_real_gradients(feature_schema):
    torch = pytest.importorskip("torch", reason="optional ML runtime is not installed")
    model = build_transformer({"architecture": "tracked_transformer_v1", "hidden_dim": 16,
        "num_heads": 2, "num_layers": 1, "max_steps": 8, "dropout": 0.,
        "types": ["NO_CLIENT_ID"]}, len(feature_names(feature_schema)))
    sequence = featurize_sequence(frames(feature_schema), feature_schema)
    values = torch.tensor(sequence.values)[None]
    observed = torch.tensor(sequence.observed)[None]
    observed[:, 2, 1] = False
    model.eval()
    baseline = model(values, observed)
    values[:, 2, 1] = float("nan")
    assert torch.allclose(baseline, model(values, observed))
    model.train()
    model(values, observed).square().mean().backward()
    assert model.project[0].weight.grad is not None
    assert bool(torch.isfinite(model.project[0].weight.grad).all())
    bad_observed = observed.clone()
    bad_observed[:, 2, 1] = True
    with pytest.raises(ValueError, match="nonfinite_observed"):
        model(values, bad_observed)


def test_optional_cpu_training_exports_unselected_tensor_artifact(inputs):
    pytest.importorskip("torch", reason="optional ML runtime is not installed")
    config = inputs[0]
    config.update(device="cpu", allow_cpu_training=True,
        runtime_versions={"torch": importlib.metadata.version("torch")})
    result = train(config)
    assert result["state"] == "trained_unselected" and result["approved_types"] == []
    output = Path(config["output_dir"])
    assert (output / "head.pt").is_file()
    assert not (output / "full-video-gate.json").exists()
