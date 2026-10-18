"""Optional, promotion-gated specialist perception on original video frames.

This module never treats an unavailable model as a successful empty detection.
All public geometry is normalized to [0, 1]. Hand/object proximity is a 2-D
signal, not proof of physical contact, identity, or complete scene visibility.
RF-DETR and supervision are imported only after a local checkpoint passes its
promotion gate. No cloud inference or automatic weight download is requested.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from forte.fast.video import DecodedVideo, Frame

Box = tuple[float, float, float, float]
Point = tuple[float, float]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_CLASSES = frozenset({"RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase"})


class SpecialistError(RuntimeError):
    """Configured specialist perception could not produce a valid observation."""


class PromotionError(SpecialistError):
    """A model has no matching, explicitly passed promotion gate."""


def _unit(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def validate_box(box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError("box must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = (_unit(v, "box coordinate") for v in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive area")
    return x1, y1, x2, y2


def normalize_pixel_box(box: Sequence[float], width: int, height: int) -> Box:
    """Clamp pixel predictions to the image, then reject degenerate geometry."""
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError("invalid frame dimensions or box")
    if not all(math.isfinite(float(v)) for v in box):
        raise ValueError("non-finite pixel box")
    return validate_box((max(0.0, min(float(box[0]) / width, 1.0)),
                         max(0.0, min(float(box[1]) / height, 1.0)),
                         max(0.0, min(float(box[2]) / width, 1.0)),
                         max(0.0, min(float(box[3]) / height, 1.0))))


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    box: Box
    track_id: int | None = None
    class_id: int | None = None
    zones: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.class_name:
            raise ValueError("detection class_name is required")
        _unit(self.confidence, "detection confidence")
        object.__setattr__(self, "box", validate_box(self.box))
        if self.track_id is not None and (type(self.track_id) is not int or self.track_id < 0):
            raise ValueError("track_id must be a nonnegative integer")
        if self.class_id is not None and (type(self.class_id) is not int or self.class_id < 0):
            raise ValueError("class_id must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class HandKeypoint:
    name: str
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("keypoint name is required")
        _unit(self.x, "keypoint x")
        _unit(self.y, "keypoint y")
        _unit(self.confidence, "keypoint confidence")


@dataclass(frozen=True, slots=True)
class HandObservation:
    hand_id: str
    keypoints: tuple[HandKeypoint, ...]
    confidence: float
    person_track_id: int | None = None

    def __post_init__(self) -> None:
        if not self.hand_id or not self.keypoints:
            raise ValueError("hand_id and observed keypoints are required")
        _unit(self.confidence, "hand confidence")
        if len({p.name for p in self.keypoints}) != len(self.keypoints):
            raise ValueError("hand keypoint names must be unique")


@dataclass(frozen=True, slots=True)
class HandContact:
    hand_id: str
    class_name: str
    object_track_id: int | None
    confidence: float
    zones: tuple[str, ...] = ()
    signal: str = "proximity_2d"


@dataclass(frozen=True, slots=True)
class Zone:
    """Explicit camera zone; membership uses a detection's center point."""

    name: str
    polygon: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.name or len(self.polygon) < 3:
            raise ValueError("zone needs a name and at least three vertices")
        points = tuple((_unit(x, "zone x"), _unit(y, "zone y")) for x, y in self.polygon)
        area = sum(x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
                   for i, (x, y) in enumerate(points))
        if abs(area) < 1e-10 or len(set(points)) != len(points):
            raise ValueError("zone polygon must have positive area and unique vertices")
        # Reject crossings: ambiguous zone geometry must not assign people roles.
        def side(a: Point, b: Point, c: Point) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        for i, a in enumerate(points):
            b = points[(i + 1) % len(points)]
            for j in range(i + 1, len(points)):
                if j == i + 1 or (i == 0 and j == len(points) - 1):
                    continue
                c, d = points[j], points[(j + 1) % len(points)]
                if side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0:
                    raise ValueError("zone polygon must not intersect itself")
        object.__setattr__(self, "polygon", points)

    @classmethod
    def from_box(cls, name: str, box: Sequence[float]) -> Zone:
        x1, y1, x2, y2 = validate_box(box)
        return cls(name, ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))

    def contains(self, point: Point) -> bool:
        x, y = point
        _unit(x, "point x")
        _unit(y, "point y")
        inside = False
        for i, (ax, ay) in enumerate(self.polygon):
            bx, by = self.polygon[(i + 1) % len(self.polygon)]
            cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
            if abs(cross) < 1e-12 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
                return True
            if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
                inside = not inside
        return inside


def zones_from_layout(layout: Mapping[str, Sequence[float]]) -> tuple[Zone, ...]:
    """Convert the VLM layout's 0..1000 boxes without guessing missing zones."""
    return tuple(Zone.from_box(name, tuple(float(v) / 1000 for v in box)) for name, box in layout.items())


def hand_object_contacts(hands: Sequence[HandObservation], detections: Sequence[Detection],
                         *, min_confidence: float = 0.5, margin: float = 0.01) -> tuple[HandContact, ...]:
    """Return observed 2-D proximity; absent hands never mean 'no interaction'."""
    _unit(min_confidence, "minimum contact confidence")
    if not 0 <= margin <= 0.1:
        raise ValueError("contact margin must be in [0, 0.1]")
    contacts: list[HandContact] = []
    for hand in hands:
        if hand.confidence < min_confidence:
            continue
        for obj in detections:
            if obj.class_name == "person" or obj.confidence < min_confidence:
                continue
            x1, y1, x2, y2 = obj.box
            hits = [p.confidence for p in hand.keypoints if p.confidence >= min_confidence
                    and x1 - margin <= p.x <= x2 + margin and y1 - margin <= p.y <= y2 + margin]
            if hits:
                contacts.append(HandContact(hand.hand_id, obj.class_name, obj.track_id,
                                            min(hand.confidence, obj.confidence, max(hits)), obj.zones))
    return tuple(contacts)


@dataclass(frozen=True, slots=True)
class SpecialistFrame:
    t: float
    detections: tuple[Detection, ...]
    hands: tuple[HandObservation, ...] = ()
    contacts: tuple[HandContact, ...] = ()


@dataclass(frozen=True, slots=True)
class SpecialistAnalysis:
    frames: tuple[SpecialistFrame, ...]
    approved_types: tuple[str, ...]
    approved_classes: tuple[str, ...]
    provenance: dict[str, Any]
    hand_status: str = "not_configured"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DetectorBackend(Protocol):
    def start_video(self, frame_rate: float) -> None: ...
    def detect(self, image: Image, t: float) -> Sequence[Detection]: ...


class HandBackend(Protocol):
    """An injected local keypoint backend; no untrained hand model is assumed."""

    def start_video(self, frame_rate: float) -> None: ...
    def observe(self, image: Image, t: float, detections: Sequence[Detection]) -> Sequence[HandObservation]: ...


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_names(raw: Any) -> dict[int, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise PromotionError("class_names must explicitly map prediction class IDs to names")
    try:
        if any(isinstance(key, bool) or str(key) != str(int(key)) or not isinstance(value, str)
               for key, value in raw.items()):
            raise PromotionError("prediction class IDs must be integer keys and names must be strings")
        names = {int(key): value for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise PromotionError("invalid prediction class IDs") from exc
    if len(names) != len(raw) or any(key < 0 or not value for key, value in names.items()):
        raise PromotionError("invalid prediction class ID/name mapping")
    if len(set(names.values())) != len(names):
        raise PromotionError("duplicate class names")
    return names


def load_promotion_gate(checkpoint: str | Path, gate_path: str | Path,
                        *, model_class: str, class_names: Mapping[int | str, str]) -> dict[str, Any]:
    """Fail closed unless the exact artifact and class map have passed a gate."""
    try:
        gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
        if not isinstance(gate, dict) or gate.get("schema_version") != "forte.specialists.gate.v1":
            raise PromotionError("unsupported specialist promotion gate schema")
        if gate.get("eligible") is not True:
            raise PromotionError("specialist promotion gate did not pass")
        for field in ("checkpoint_sha256", "dataset_lock_sha256", "config_sha256", "run_sha256", "metrics_sha256"):
            if not isinstance(gate.get(field), str) or not _SHA256.fullmatch(gate[field]):
                raise PromotionError(f"promotion gate needs {field}")
        if artifact_sha256(checkpoint) != gate["checkpoint_sha256"]:
            raise PromotionError("specialist checkpoint SHA256 does not match promotion gate")
        if gate.get("model_class") != model_class or _class_names(gate.get("class_names")) != _class_names(class_names):
            raise PromotionError("model architecture or class map does not match promotion gate")
        if not isinstance(gate.get("rfdetr_version"), str) or not gate["rfdetr_version"]:
            raise PromotionError("gate must fix the RF-DETR package version")
        resolution = gate.get("resolution")
        multiple = 56 if model_class == "RFDETRBase" else 32
        if type(resolution) is not int or resolution <= 0 or resolution % multiple:
            raise PromotionError("gate must fix a valid RF-DETR inference resolution")
        approved = gate.get("approved_classes")
        if (not isinstance(approved, list) or not approved or any(not isinstance(c, str) for c in approved)
                or len(set(approved)) != len(approved) or not set(approved) <= set(class_names.values())):
            raise PromotionError("gate approved_classes must be a nonempty subset of model classes")
        thresholds = gate.get("thresholds")
        if (not isinstance(thresholds, dict) or not set(approved) <= set(thresholds)
                or not set(thresholds) <= set(class_names.values())):
            raise PromotionError("gate must fix one threshold per approved class")
        for threshold in thresholds.values():
            _unit(threshold, "promoted threshold")
        types = gate.get("approved_types", [])
        if not isinstance(types, list) or any(not isinstance(t, str) or not t for t in types):
            raise PromotionError("approved_types must be an explicit list of video-validated types")
        if gate.get("scope") == "object_classes_only" and types:
            raise PromotionError("object-class promotion cannot approve whole-video violation types")
        if types and (not isinstance(gate.get("analyzer_policy_sha256"), str)
                      or not _SHA256.fullmatch(gate["analyzer_policy_sha256"])):
            raise PromotionError("video-type promotion must bind analyzer_policy_sha256")
        return gate
    except PromotionError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise PromotionError(f"invalid specialist promotion artifacts: {exc}") from exc


class RFDETRByteTrackBackend:
    """Local RF-DETR inference with class-isolated supervision ByteTrack state.

    Class IDs come from the training manifest; RF-DETR versions have used
    different indexing conventions, so an offset is never guessed.
    """

    def __init__(self, checkpoint: str | Path, *, class_names: Mapping[int, str],
                 thresholds: Mapping[str, float], model_class: str = "RFDETRSmall", device: str = "cuda",
                 resolution: int | None = None, expected_rfdetr_version: str | None = None,
                 expected_supervision_version: str | None = None):
        if model_class not in _MODEL_CLASSES:
            raise SpecialistError("unsupported RF-DETR architecture; choose Nano, Small, Medium, or Base")
        try:
            if expected_rfdetr_version is not None and importlib.metadata.version("rfdetr") != expected_rfdetr_version:
                raise SpecialistError("installed RF-DETR version differs from promotion gate")
            if (expected_supervision_version is not None
                    and importlib.metadata.version("supervision") != expected_supervision_version):
                raise SpecialistError("installed supervision version differs from validated ByteTrack API")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            rfdetr = importlib.import_module("rfdetr")
            self._sv = importlib.import_module("supervision")
            model_type = getattr(rfdetr, model_class)
            options: dict[str, Any] = {"pretrain_weights": str(Path(checkpoint).resolve()), "device": device}
            if resolution is not None:
                options["resolution"] = resolution
            self._model = model_type(**options)
        except Exception as exc:
            raise SpecialistError(f"cannot load local RF-DETR/ByteTrack backend: {exc}") from exc
        self.class_names = dict(class_names)
        self.thresholds = dict(thresholds)
        loaded_names = getattr(self._model, "class_names", None)
        # Pinned RF-DETR 1.11.2 exposes list[str] indexed by prediction ID;
        # older versions used mappings. Normalize both without adding offsets.
        if isinstance(loaded_names, (list, tuple)):
            loaded_names = dict(enumerate(loaded_names))
        try:
            normalized_names = _class_names(loaded_names)
        except PromotionError as exc:
            raise SpecialistError("checkpoint lacks a valid prediction class-name table") from exc
        if normalized_names != self.class_names:
            raise SpecialistError("checkpoint class names differ from promoted prediction IDs")
        self._trackers: dict[int, Any] = {}
        self._track_ids: dict[tuple[int, int], int] = {}
        self._frame_rate: float | None = None

    def start_video(self, frame_rate: float) -> None:
        self._frame_rate = frame_rate
        self._trackers = {}
        self._track_ids = {}

    def detect(self, image: Image, t: float) -> Sequence[Detection]:
        if self._frame_rate is None:
            raise SpecialistError("start_video must run before detector inference")
        try:
            # ByteTrack may associate a low-confidence observation with an
            # existing track. Only final detections meeting gate thresholds
            # leave this backend; low-score boxes never become evidence.
            predictions = self._model.predict(image.convert("RGB"), threshold=min(0.1, *self.thresholds.values()))
            if isinstance(predictions, list):
                raise SpecialistError("single-frame RF-DETR unexpectedly returned a batch")
            if predictions.class_id is None or predictions.confidence is None:
                raise SpecialistError("RF-DETR predictions lack class IDs or confidence")
            unknown = set(int(x) for x in predictions.class_id) - set(self.class_names)
            if unknown:
                raise SpecialistError(f"RF-DETR returned unmapped prediction IDs: {sorted(unknown)}")
            results: list[Detection] = []
            for class_id, name in self.class_names.items():
                if name not in self.thresholds:
                    continue
                # Separate trackers prevent a card track from silently becoming cash.
                if class_id not in self._trackers:
                    self._trackers[class_id] = self._sv.ByteTrack(frame_rate=self._frame_rate)
                class_predictions = predictions[predictions.class_id == class_id]
                tracked = self._trackers[class_id].update_with_detections(class_predictions)
                if len(tracked) and tracked.tracker_id is None:
                    raise SpecialistError("ByteTrack output lacks track IDs")
                track_ids: dict[tuple[float, ...], int] = {}
                for index in range(len(tracked)):
                    local_id = int(tracked.tracker_id[index])
                    key = (class_id, local_id)
                    track_id = self._track_ids.setdefault(key, len(self._track_ids) + 1)
                    track_ids[tuple(float(v) for v in tracked.xyxy[index])] = track_id
                # Preserve unconfirmed detections as track_id=None. A tracker's
                # confirmation delay must not erase a brief object observation.
                selected = class_predictions[class_predictions.confidence >= self.thresholds[name]]
                for index in range(len(selected)):
                    results.append(Detection(name, float(selected.confidence[index]),
                                             normalize_pixel_box(selected.xyxy[index], image.width, image.height),
                                             track_ids.get(tuple(float(v) for v in selected.xyxy[index])), class_id))
            return results
        except SpecialistError:
            raise
        except Exception as exc:
            raise SpecialistError(f"specialist inference failed at {t:.3f}s: {exc}") from exc


class RTMPoseHandBackend:
    """MMPose hand detector + RTMPose keypoints from four approved local files.

    The model's hand IDs are frame-local. This adapter does not fabricate hand
    identity across frames or associate a detected hand with a person's role.
    """

    _ARTIFACTS = ("pose_config", "pose_checkpoint", "detector_config", "detector_checkpoint")

    def __init__(self, config: Mapping[str, Any], *, device: str = "cuda"):
        try:
            gate_path = Path(config["gate_path"])
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if (not isinstance(gate, dict) or gate.get("schema_version") != "forte.specialists.hands.gate.v1"
                    or gate.get("eligible") is not True):
                raise PromotionError("RTMPose hand promotion gate did not pass")
            for name in self._ARTIFACTS:
                digest = gate.get(name + "_sha256")
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise PromotionError(f"RTMPose gate needs {name}_sha256")
                if artifact_sha256(config[name]) != digest:
                    raise PromotionError(f"RTMPose {name} SHA256 differs from promotion gate")
            self._minimum_confidence = _unit(gate["keypoint_threshold"], "promoted keypoint threshold")
            if gate.get("approved_signals") != ["proximity_2d"]:
                raise PromotionError("RTMPose gate must explicitly approve proximity_2d")
            self.provenance = {name + "_sha256": gate[name + "_sha256"] for name in self._ARTIFACTS}
            self.provenance["gate_sha256"] = artifact_sha256(gate_path)
        except PromotionError:
            raise
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise PromotionError(f"invalid RTMPose promotion artifacts: {exc}") from exc
        try:
            self._np = importlib.import_module("numpy")
            pose_api = importlib.import_module("mmpose.apis")
            self._inferencer = pose_api.MMPoseInferencer(
                pose2d=str(Path(config["pose_config"]).resolve()),
                pose2d_weights=str(Path(config["pose_checkpoint"]).resolve()),
                det_model=str(Path(config["detector_config"]).resolve()),
                det_weights=str(Path(config["detector_checkpoint"]).resolve()),
                device=device, show_progress=False)
        except Exception as exc:
            raise SpecialistError(f"cannot load local MMPose hand backend: {exc}") from exc

    def start_video(self, frame_rate: float) -> None:
        pass  # The inferencer operates on independent images and has no tracks.

    def observe(self, image: Image, t: float, detections: Sequence[Detection]) -> Sequence[HandObservation]:
        try:
            # MMPose's numpy-image convention is BGR; PIL images are RGB.
            bgr = self._np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            output = next(self._inferencer(bgr, return_vis=False, show=False))
            predictions = output.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != 1 or not isinstance(predictions[0], list):
                raise SpecialistError("MMPose returned no single-image prediction observation")
            hands: list[HandObservation] = []
            for index, hand in enumerate(predictions[0]):
                coordinates, scores = hand["keypoints"], hand["keypoint_scores"]
                if len(coordinates) != 21 or len(scores) != 21:
                    raise SpecialistError("RTMPose hand model must provide exactly 21 keypoints")
                points = []
                for keypoint_index, (xy, score) in enumerate(zip(coordinates, scores)):
                    confidence = _unit(float(score), "RTMPose keypoint score")
                    if confidence < self._minimum_confidence:
                        continue
                    x, y = float(xy[0]) / image.width, float(xy[1]) / image.height
                    # Off-image or occluded points are not evidence of contact.
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise SpecialistError("RTMPose returned non-finite keypoint coordinates")
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        points.append(HandKeypoint(str(keypoint_index), x, y, confidence))
                if points:
                    hands.append(HandObservation(f"frame:{t:.9f}:hand:{index}", tuple(points),
                                                 max(p.confidence for p in points)))
            return hands
        except SpecialistError:
            raise
        except Exception as exc:
            raise SpecialistError(f"RTMPose inference failed at {t:.3f}s: {exc}") from exc


class SpecialistRuntime:
    """Stream native frames and retain lightweight observations, never pixels.

    A runtime may be reused between videos. An active session is exclusive;
    callers must call finish_video or abort_video after start_video.
    """

    def __init__(self, backend: DetectorBackend, gate: Mapping[str, Any], *,
                 hand_backend: HandBackend | None = None, provenance: Mapping[str, Any] | None = None):
        if gate.get("eligible") is not True:
            raise PromotionError("specialist runtime requires an explicitly passed gate")
        if gate.get("approved_types") and (not isinstance(gate.get("analyzer_policy_sha256"), str)
                                           or not _SHA256.fullmatch(gate["analyzer_policy_sha256"])):
            raise PromotionError("video-type promotion must bind analyzer_policy_sha256")
        self.backend = backend
        self.hand_backend = hand_backend
        self.gate = dict(gate)
        self.approved_types = tuple(gate.get("approved_types", ()))
        self.approved_classes = tuple(gate["approved_classes"])
        self.thresholds = dict(gate["thresholds"])
        self.provenance = dict(provenance or {})
        self._lock = threading.Lock()
        self._active = False
        self._frames: list[SpecialistFrame] = []
        self._zones: tuple[Zone, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, backend: DetectorBackend | None = None,
                    hand_backend: HandBackend | None = None) -> SpecialistRuntime | None:
        if config.get("enabled", False) is False:
            return None
        if config.get("enabled") is not True:
            raise PromotionError("specialist enabled must explicitly be true or false")
        if not config.get("checkpoint") or not config.get("gate_path"):
            raise PromotionError("enabled specialists need checkpoint and gate_path")
        names = _class_names(config.get("class_names"))
        model_class = config.get("model_class", "RFDETRSmall")
        if model_class not in _MODEL_CLASSES:
            raise PromotionError("unsupported specialist model_class")
        gate = load_promotion_gate(config["checkpoint"], config["gate_path"], model_class=model_class,
                                   class_names=names)
        # Promotion fixes thresholds; arbitrary runtime overrides invalidate it.
        if "thresholds" in config and config["thresholds"] != gate["thresholds"]:
            raise PromotionError("runtime thresholds differ from promoted thresholds")
        if config.get("hand_backend") and hand_backend is None:
            hand_config = config["hand_backend"]
            if not isinstance(hand_config, Mapping) or hand_config.get("backend") != "rtmpose":
                raise SpecialistError("unsupported hand_backend; expected a local gated rtmpose config")
            hand_backend = RTMPoseHandBackend(hand_config, device=config.get("device", "cuda"))
        detector = backend if backend is not None else RFDETRByteTrackBackend(
            config["checkpoint"], class_names=names,
            thresholds={name: gate["thresholds"][name] for name in gate["approved_classes"]},
            model_class=model_class, device=config.get("device", "cuda"), resolution=gate["resolution"],
            expected_rfdetr_version=gate["rfdetr_version"], expected_supervision_version="0.29.1")
        return cls(detector, gate, hand_backend=hand_backend, provenance={
            "schema_version": "forte.specialists.observations.v1", "model_class": model_class,
            "checkpoint_sha256": gate["checkpoint_sha256"], "gate_sha256": artifact_sha256(config["gate_path"]),
            "rfdetr_version": gate["rfdetr_version"], "resolution": gate["resolution"],
            "supervision_version": "0.29.1",
            "dataset_lock_sha256": gate["dataset_lock_sha256"], "metrics_sha256": gate["metrics_sha256"],
            "analyzer_policy_sha256": gate.get("analyzer_policy_sha256"),
            "hands": getattr(hand_backend, "provenance", None),
            "contact_signal": "proximity_2d", "absence_proves_visibility": False,
        })

    def start_video(self, frame_rate: float = 30.0, zones: Sequence[Zone] | None = None) -> None:
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be finite and positive")
        if not self._lock.acquire(blocking=False):
            raise SpecialistError("specialist runtime already has an active video")
        self._active = True
        self._frames = []
        self._zones = tuple(zones or ())
        try:
            self.backend.start_video(float(frame_rate))
            if self.hand_backend is not None:
                self.hand_backend.start_video(float(frame_rate))
        except Exception as exc:
            self.abort_video()
            raise SpecialistError(f"specialist video initialization failed: {exc}") from exc

    def process_frame(self, frame: Frame) -> SpecialistFrame:
        if not self._active:
            raise SpecialistError("specialist video has not been started")
        try:
            if not math.isfinite(frame.t) or frame.t < 0 or (self._frames and frame.t <= self._frames[-1].t):
                raise SpecialistError("specialist timestamps must be finite, nonnegative, and strictly increasing")
            detected = self.backend.detect(frame.image, frame.t)
            if detected is None:
                raise SpecialistError("detector returned no observation; empty detections must be explicit")
            objects = tuple(d for d in detected if d.class_name in self.approved_classes
                            and d.confidence >= self.thresholds[d.class_name])
            hands: tuple[HandObservation, ...] = ()
            if self.hand_backend is not None:
                observed = self.hand_backend.observe(frame.image, frame.t, objects)
                if observed is None:
                    raise SpecialistError("hand backend returned no observation")
                hands = tuple(observed)
            result = self._assign_zones(SpecialistFrame(float(frame.t), objects, hands), self._zones)
            self._frames.append(result)
            return result
        except Exception as exc:
            self.abort_video()
            if isinstance(exc, SpecialistError):
                raise
            raise SpecialistError(f"specialist inference failed at {frame.t}s: {exc}") from exc

    @staticmethod
    def _assign_zones(frame: SpecialistFrame, zones: Sequence[Zone]) -> SpecialistFrame:
        detections = tuple(replace(d, zones=tuple(z.name for z in zones
                                                if z.contains(((d.box[0] + d.box[2]) / 2,
                                                               (d.box[1] + d.box[3]) / 2))))
                           for d in frame.detections)
        return replace(frame, detections=detections, contacts=hand_object_contacts(frame.hands, detections))

    def finish_video(self, zones: Sequence[Zone] | None = None) -> SpecialistAnalysis:
        if not self._active:
            raise SpecialistError("specialist video has not been started")
        try:
            if not self._frames:
                raise SpecialistError("specialist video produced no frame observations")
            frames = tuple(self._frames) if zones is None else tuple(self._assign_zones(f, zones) for f in self._frames)
            return SpecialistAnalysis(frames, self.approved_types, self.approved_classes,
                                      {**self.provenance, "observed_frames": len(frames)},
                                      "ok" if self.hand_backend is not None else "not_configured")
        finally:
            self.abort_video()

    def abort_video(self) -> None:
        if self._active:
            self._active = False
            self._frames = []
            self._lock.release()

    def analyze(self, video: DecodedVideo, zones: Sequence[Zone] | None = None) -> SpecialistAnalysis:
        # DecodedVideo.fps is the original stream rate. For sampled convenience
        # input use its actual observation interval for ByteTrack's lost buffer.
        times = [f.t for f in video.frames]
        frame_rate = ((len(times) - 1) / (times[-1] - times[0])
                      if len(times) > 1 and times[-1] > times[0] else video.fps)
        self.start_video(frame_rate, zones)
        try:
            for frame in video.frames:
                self.process_frame(frame)
            return self.finish_video()
        finally:
            self.abort_video()
