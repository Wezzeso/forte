"""Specialist artifact, geometry and streaming checks without GPU or real media."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from forte.fast.specialists import (
    Detection, HandKeypoint, HandObservation, PromotionError, RFDETRByteTrackBackend,
    RTMPoseHandBackend, SpecialistError, SpecialistRuntime, Zone, artifact_sha256,
    hand_object_contacts, load_promotion_gate, normalize_pixel_box, zones_from_layout,
)
from forte.fast.video import DecodedVideo, Frame


class Detector:
    def __init__(self, detections=()):
        self.detections = detections
        self.frame_rates = []

    def start_video(self, frame_rate):
        self.frame_rates.append(frame_rate)

    def detect(self, image, t):
        return self.detections


def artifacts(tmp_path):
    checkpoint = tmp_path / "trained.pth"
    checkpoint.write_bytes(b"synthetic checkpoint, not a model")
    gate = {
        "schema_version": "forte.specialists.gate.v1", "eligible": True,
        "checkpoint_sha256": artifact_sha256(checkpoint), "dataset_lock_sha256": "1" * 64,
        "config_sha256": "2" * 64, "run_sha256": "3" * 64, "metrics_sha256": "4" * 64,
        "model_class": "RFDETRSmall", "class_names": {"0": "id_card", "1": "cup"},
        "rfdetr_version": "1.11.2", "resolution": 640,
        "approved_classes": ["id_card", "cup"], "thresholds": {"id_card": 0.8, "cup": 0.6},
        "approved_types": [],
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    config = {"enabled": True, "checkpoint": str(checkpoint), "gate_path": str(gate_path),
              "class_names": gate["class_names"], "model_class": "RFDETRSmall"}
    return config, gate, gate_path


def test_disabled_runtime_does_not_import_optional_models(monkeypatch):
    monkeypatch.setattr("forte.fast.specialists.importlib.import_module",
                        lambda name: pytest.fail(f"unexpected optional import: {name}"))
    assert SpecialistRuntime.from_config({}) is None
    assert SpecialistRuntime.from_config({"enabled": False}) is None


@pytest.mark.parametrize("mutation", [
    {"eligible": False}, {"eligible": "true"}, {"checkpoint_sha256": "0" * 64},
    {"approved_classes": []}, {"approved_classes": ["untrained"]},
    {"class_names": {"1": "id_card", "2": "cup"}}, {"model_class": "RFDETRBase"},
    {"thresholds": {"id_card": 0.8}}, {"thresholds": {"id_card": True, "cup": 0.5}},
    {"metrics_sha256": None}, {"approved_types": "NO_CLIENT_ID"},
    {"scope": "object_classes_only", "approved_types": ["NO_CLIENT_ID"]},
    {"approved_types": ["NO_CLIENT_ID"]},
    {"approved_types": ["NO_CLIENT_ID"], "analyzer_policy_sha256": "invalid"},
])
def test_invalid_gate_is_rejected_before_backend_loading(tmp_path, monkeypatch, mutation):
    config, gate, path = artifacts(tmp_path)
    path.write_text(json.dumps({**gate, **mutation}), encoding="utf-8")
    monkeypatch.setattr("forte.fast.specialists.importlib.import_module",
                        lambda name: pytest.fail(f"invalid model must not load {name}"))
    with pytest.raises(PromotionError):
        SpecialistRuntime.from_config(config)


def test_checkpoint_change_and_threshold_override_invalidate_gate(tmp_path):
    config, _, _ = artifacts(tmp_path)
    with pytest.raises(PromotionError, match="thresholds"):
        SpecialistRuntime.from_config({**config, "thresholds": {"id_card": 0.1}}, backend=Detector())
    with open(config["checkpoint"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(PromotionError, match="SHA256"):
        SpecialistRuntime.from_config(config, backend=Detector())


def test_explicit_prediction_id_map_preserves_zero_based_labels(tmp_path):
    config, _, _ = artifacts(tmp_path)
    result = load_promotion_gate(config["checkpoint"], config["gate_path"],
                                 model_class="RFDETRSmall", class_names={0: "id_card", 1: "cup"})
    assert result["class_names"]["0"] == "id_card"


def test_video_gate_policy_hash_is_required_and_preserved_in_provenance(tmp_path):
    config, gate, path = artifacts(tmp_path)
    gate.update(scope="video_types", approved_types=["PERSONAL_ITEMS"], analyzer_policy_sha256="a" * 64)
    path.write_text(json.dumps(gate), encoding="utf-8")
    runtime = SpecialistRuntime.from_config(config, backend=Detector())
    runtime.start_video()
    runtime.process_frame(Frame(0, Image.new("RGB", (16, 16))))
    result = runtime.finish_video()
    assert result.approved_types == ("PERSONAL_ITEMS",)
    assert result.provenance["analyzer_policy_sha256"] == "a" * 64


@pytest.mark.parametrize("box", [(0.5, 0, 0.4, 1), (0, 0, 0, 1), (-0.1, 0, 1, 1), (0, 0, float("nan"), 1)])
def test_invalid_normalized_boxes_are_rejected(box):
    with pytest.raises(ValueError):
        Detection("cup", 0.9, box)


def test_pixel_boxes_clamp_to_image_boundaries():
    assert normalize_pixel_box((-5, 20, 120, 90), 100, 100) == (0.0, 0.2, 1.0, 0.9)
    with pytest.raises(ValueError):
        normalize_pixel_box((100, 5, 120, 90), 100, 100)


def test_explicit_zones_include_boundaries_and_reject_bad_layout():
    zone = zones_from_layout({"desk": [100, 200, 500, 600]})[0]
    assert zone.contains((0.1, 0.2))
    assert zone.contains((0.3, 0.4))
    assert not zone.contains((0.9, 0.4))
    with pytest.raises(ValueError):
        zones_from_layout({"desk": [0, 0, 1100, 1000]})
    with pytest.raises(ValueError):
        Zone("bad", ((0, 0), (1, 1), (0, 1), (1, 0)))


def test_proximity_is_not_physical_contact_or_visibility():
    cup = Detection("cup", 0.9, (0.4, 0.4, 0.6, 0.6), 2)
    person = Detection("person", 0.9, (0, 0, 1, 1), 3)
    hand = HandObservation("h1", (HandKeypoint("index_tip", 0.5, 0.5, 0.8),), 0.9)
    contacts = hand_object_contacts([hand], [cup, person])
    assert len(contacts) == 1 and contacts[0].object_track_id == 2
    assert contacts[0].signal == "proximity_2d"
    assert contacts[0].confidence == 0.8
    assert hand_object_contacts([], [cup]) == ()


def test_streaming_keeps_only_observations_and_zones_can_be_assigned_later(tmp_path):
    config, _, _ = artifacts(tmp_path)
    detector = Detector((Detection("cup", 0.9, (0.3, 0.3, 0.4, 0.4), 1),
                         Detection("id_card", 0.7, (0.5, 0.5, 0.6, 0.6), 2)))
    runtime = SpecialistRuntime.from_config(config, backend=detector)
    image = Image.new("RGB", (100, 200))
    runtime.start_video(29.97)
    for t in (0.0, 0.033, 0.067):
        runtime.process_frame(Frame(t, image))
    with pytest.raises(SpecialistError, match="active video"):
        runtime.start_video(30)
    result = runtime.finish_video([Zone.from_box("desk", (0, 0, 0.5, 0.5))])
    assert detector.frame_rates == [29.97]
    assert len(result.frames) == 3
    assert result.hand_status == "not_configured"
    assert result.approved_types == ()
    assert result.frames[0].detections[0].zones == ("desk",)
    assert len(result.frames[0].detections) == 1  # Gate threshold retained.
    assert "image" not in result.to_dict()["frames"][0]
    assert result.provenance["absence_proves_visibility"] is False
    runtime.start_video(30)
    runtime.abort_video()


@pytest.mark.parametrize("failure", [RuntimeError("GPU failure"), None])
def test_inference_failure_never_becomes_empty_detections_and_releases_session(tmp_path, failure):
    config, _, _ = artifacts(tmp_path)
    class Broken(Detector):
        def detect(self, image, t):
            if failure is None:
                return None
            raise failure
    runtime = SpecialistRuntime.from_config(config, backend=Broken())
    runtime.start_video()
    with pytest.raises(SpecialistError):
        runtime.process_frame(Frame(0.0, Image.new("RGB", (16, 16))))
    runtime.start_video()
    runtime.abort_video()


def test_explicit_empty_observation_and_missing_hand_model_are_distinct(tmp_path):
    config, _, _ = artifacts(tmp_path)
    runtime = SpecialistRuntime.from_config(config, backend=Detector(()))
    video = DecodedVideo(1, 30, 16, 16, [Frame(0, Image.new("RGB", (16, 16)))])
    result = runtime.analyze(video)
    assert result.frames[0].detections == ()
    assert result.frames[0].hands == ()
    assert result.hand_status == "not_configured"


def test_duplicate_timestamp_fails_and_sampled_analyze_uses_observation_rate(tmp_path):
    config, _, _ = artifacts(tmp_path)
    detector = Detector()
    runtime = SpecialistRuntime.from_config(config, backend=detector)
    image = Image.new("RGB", (16, 16))
    video = DecodedVideo(2, 30, 16, 16, [Frame(0, image), Frame(0.5, image), Frame(1, image)])
    runtime.analyze(video)
    assert detector.frame_rates == [2.0]
    runtime.start_video()
    runtime.process_frame(Frame(0, image))
    with pytest.raises(SpecialistError, match="strictly increasing"):
        runtime.process_frame(Frame(0, image))
    runtime.start_video()
    runtime.abort_video()


def test_hand_backend_observations_and_contacts_are_propagated(tmp_path):
    config, _, _ = artifacts(tmp_path)
    class Hands:
        def start_video(self, fps):
            self.fps = fps
        def observe(self, image, t, detections):
            return [HandObservation("h", (HandKeypoint("tip", 0.5, 0.5, 0.9),), 0.9)]
    hands = Hands()
    detector = Detector([Detection("id_card", 0.95, (0.4, 0.4, 0.6, 0.6), 5)])
    runtime = SpecialistRuntime.from_config(config, backend=detector, hand_backend=hands)
    runtime.start_video(25)
    runtime.process_frame(Frame(0, Image.new("RGB", (100, 100))))
    result = runtime.finish_video([Zone.from_box("desk", (0, 0, 1, 1))])
    assert result.hand_status == "ok" and hands.fps == 25
    assert result.frames[0].contacts[0].zones == ("desk",)


def test_real_backend_missing_dependencies_raises_without_a_fallback(tmp_path, monkeypatch):
    config, _, _ = artifacts(tmp_path)
    def unavailable(name):
        raise ImportError("missing optional package")
    monkeypatch.setattr("forte.fast.specialists.importlib.import_module", unavailable)
    with pytest.raises(SpecialistError, match="cannot load local"):
        SpecialistRuntime.from_config(config)


def test_rtmpose_gate_is_checked_before_optional_imports(tmp_path, monkeypatch):
    monkeypatch.setattr("forte.fast.specialists.importlib.import_module",
                        lambda name: pytest.fail(f"invalid hand model must not load {name}"))
    gate_path = tmp_path / "hand_gate.json"
    gate_path.write_text(json.dumps({"schema_version": "forte.specialists.hands.gate.v1", "eligible": False}))
    with pytest.raises(PromotionError, match="did not pass"):
        RTMPoseHandBackend({"gate_path": str(gate_path)})


def test_rtmpose_uses_local_artifacts_and_frame_local_hand_ids(tmp_path, monkeypatch):
    config = {"backend": "rtmpose"}
    gate = {"schema_version": "forte.specialists.hands.gate.v1", "eligible": True,
            "keypoint_threshold": 0.7, "approved_signals": ["proximity_2d"]}
    for field in RTMPoseHandBackend._ARTIFACTS:
        path = tmp_path / field
        path.write_bytes(field.encode())
        config[field] = str(path)
        gate[field + "_sha256"] = artifact_sha256(path)
    gate_path = tmp_path / "hand_gate.json"
    gate_path.write_text(json.dumps(gate))
    config["gate_path"] = str(gate_path)
    constructed = {}
    class Array:
        def __getitem__(self, index):
            return self
        def copy(self):
            return self
    class Inferencer:
        def __init__(self, **kwargs):
            constructed.update(kwargs)
        def __call__(self, *args, **kwargs):
            yield {"predictions": [[{"keypoints": [[50, 40]] * 21, "keypoint_scores": [0.9] * 21}]]}
    modules = {"numpy": SimpleNamespace(asarray=lambda image: Array()),
               "mmpose.apis": SimpleNamespace(MMPoseInferencer=Inferencer)}
    monkeypatch.setattr("forte.fast.specialists.importlib.import_module", modules.__getitem__)
    backend = RTMPoseHandBackend(config)
    image = Image.new("RGB", (100, 200))
    first, second = backend.observe(image, 0, ())[0], backend.observe(image, 1, ())[0]
    assert first.keypoints[0].x == 0.5 and first.keypoints[0].y == 0.2
    assert first.hand_id != second.hand_id and first.person_track_id is None
    assert constructed["pose2d_weights"] == config["pose_checkpoint"]
    assert constructed["det_weights"] == config["detector_checkpoint"]


@pytest.mark.parametrize("loaded_names", [{0: "id_card", 1: "cup"}, ["id_card", "cup"]])
def test_rfdetr_tracks_each_class_separately_and_preserves_unconfirmed_detections(monkeypatch, tmp_path, loaded_names):
    class Vector(list):
        def __eq__(self, other):
            return Mask([x == other for x in self])
        def __ge__(self, other):
            return Mask([x >= other for x in self])
    class Mask(list):
        def __and__(self, other):
            return Mask([a and b for a, b in zip(self, other)])
    class Predictions:
        def __init__(self, rows):
            self.rows = rows
            self.class_id = Vector([row[0] for row in rows])
            self.confidence = Vector([row[1] for row in rows])
            self.xyxy = [row[2] for row in rows]
            self.tracker_id = None
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, mask):
            return Predictions([row for row, include in zip(self.rows, mask) if include])
    class Model:
        class_names = loaded_names
        def __init__(self, **kwargs):
            assert kwargs["pretrain_weights"] == str((tmp_path / "weights.pth").resolve())
        def predict(self, image, **kwargs):
            assert kwargs["threshold"] == 0.1
            return Predictions([(0, 0.9, (10, 10, 20, 20)), (1, 0.8, (30, 30, 50, 50))])
    trackers = []
    class Tracker:
        def __init__(self, **kwargs):
            trackers.append(self)
        def update_with_detections(self, predictions):
            if predictions.class_id and predictions.class_id[0] == 1:
                return Predictions([])  # Class cup still has an observation.
            predictions.tracker_id = [1] * len(predictions)
            return predictions
    modules = {"rfdetr": SimpleNamespace(RFDETRSmall=Model),
               "supervision": SimpleNamespace(ByteTrack=Tracker)}
    monkeypatch.setattr("forte.fast.specialists.importlib.import_module", modules.__getitem__)
    backend = RFDETRByteTrackBackend(tmp_path / "weights.pth", class_names={0: "id_card", 1: "cup"},
                                     thresholds={"id_card": 0.5, "cup": 0.6})
    backend.start_video(30)
    result = backend.detect(Image.new("RGB", (100, 100)), 0)
    assert len(trackers) == 2
    assert result[0].track_id == 1 and result[0].box == (0.1, 0.1, 0.2, 0.2)
    assert result[1].class_name == "cup" and result[1].track_id is None


@pytest.mark.parametrize("loaded_names", [["cup", "id_card"], ["id_card", "cup", "unpromoted"], None])
def test_rfdetr_rejects_wrong_or_missing_prediction_names(monkeypatch, tmp_path, loaded_names):
    class Model:
        class_names = loaded_names
        def __init__(self, **kwargs):
            pass
    modules = {"rfdetr": SimpleNamespace(RFDETRSmall=Model), "supervision": SimpleNamespace()}
    monkeypatch.setattr("forte.fast.specialists.importlib.import_module", modules.__getitem__)
    with pytest.raises(SpecialistError, match="class.names|class-name"):
        RFDETRByteTrackBackend(tmp_path / "weights.pth", class_names={0: "id_card", 1: "cup"},
                               thresholds={"id_card": 0.5, "cup": 0.6})
