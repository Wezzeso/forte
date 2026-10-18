"""Strict local audit ingestion. This module never performs inference or review.

Raw model observations remain private proposals. Importing them cannot mark a
clip reviewed, set a jury verdict, create a negative label, or authorize training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .catalog import event_definitions, fact_definitions
from .store import ReviewError, ReviewStore, canonical

MAX_AUDIT_BYTES = 32 * 1024 * 1024


def _under_roots(path: str | Path, roots: Iterable[Path], *, description: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ReviewError(f"{description} must use an absolute local path")
    try:
        resolved = value.resolve(strict=True)
    except OSError:
        raise ReviewError(f"{description} is not available locally") from None
    approved = [root.resolve(strict=True) for root in roots]
    if not approved or not resolved.is_file() or not any(resolved.is_relative_to(root) for root in approved):
        raise ReviewError(f"{description} is outside the explicit allowed roots")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def import_proposal(
    store: ReviewStore,
    input_path: Path,
    *,
    proposal_roots: list[Path],
    source_roots: list[Path],
    frame_roots: list[Path],
    sample_id: str | None = None,
    expected_revision: int | None = None,
    local_source_by_hash: dict[str, str | Path] | None = None,
    remote_frame_root: str | None = None,
    local_frame_root: str | Path | None = None,
) -> dict[str, int]:
    """Validate private source bindings, then atomically append a proposal.

    All paths are explicitly operator-owned local roots; relative references,
    root escapes, modified source bytes, cross-job frames and foreign sample IDs
    are rejected. Only aggregate counters are returned to terminal callers.
    """
    from forte.shared.schemas import CandidateEvent, ClipRequest, FactDecision, FrameRef, VerifiedEvent, VideoAsset

    audit_path = _under_roots(input_path, proposal_roots, description="proposal audit")
    if audit_path.stat().st_size > MAX_AUDIT_BYTES:
        raise ReviewError("proposal audit exceeds the bounded local size limit")
    raw = audit_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(payload, dict) or payload.get("schema_version") != "forte.local-proposal.v1":
            raise ReviewError("unsupported private proposal schema")
        mapping = (local_source_by_hash, remote_frame_root, local_frame_root)
        if any(value is not None for value in mapping):
            if any(value is None for value in mapping):
                raise ReviewError("proposal rebasing requires source mapping and both frame roots")
            from .rebase import rebase_proposal
            payload = rebase_proposal(payload, local_source_by_hash, remote_frame_root, local_frame_root)
        asset = VideoAsset.model_validate(payload.get("asset"))
    except ReviewError:
        raise
    except Exception:
        raise ReviewError("invalid private proposal or source asset contract") from None
    source = _under_roots(asset.owned_path, source_roots, description="proposal source media")
    if _sha256(source) != asset.content_hash:
        raise ReviewError("proposal source bytes differ from the declared stable identity")
    matches = [record for record in store.list_records() if record["content_hash"] == asset.content_hash]
    if len(matches) != 1:
        raise ReviewError("proposal source must bind exactly one seeded content review unit")
    record = matches[0]
    if sample_id is not None and record["sample_id"] != sample_id:
        raise ReviewError("proposal content does not match the requested review unit")
    if payload.get("sample_id") not in {None, record["sample_id"]}:
        raise ReviewError("proposal carries a conflicting review unit identity")
    if payload.get("job_id") not in {None, asset.job_id}:
        raise ReviewError("proposal job differs from its source asset")
    if record.get("source_role") == "jury_test":
        raise ReviewError("jury test media cannot enter annotation proposals")
    duration = record.get("duration_sec")
    if duration is not None and abs(float(duration) - asset.duration_sec) > max(0.1, 1 / (asset.fps or 30)):
        raise ReviewError("proposal source duration differs from the seeded metadata")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("execution_location") != "team_deployed_service" or provenance.get("external_ai_used") is not False:
        raise ReviewError("proposal requires explicit team-local execution provenance")
    if any(not isinstance(provenance.get(key), str) or not provenance[key].strip() for key in ("model_id", "local_run_id")):
        raise ReviewError("proposal provenance requires model and local run identifiers")
    for key in ("decisions", "candidate_events", "verified_events", "window_decisions"):
        if key in payload and not isinstance(payload[key], list):
            raise ReviewError("proposal decision and event collections must be lists")
    if not isinstance(payload.get("coverage"), dict) or not isinstance(payload["coverage"].get("complete"), bool):
        raise ReviewError("proposal requires an explicit technical coverage state")
    frames_raw = payload.get("frames", payload.get("frame_map", []))
    if isinstance(frames_raw, dict):
        if any(not isinstance(value, dict) or value.get("frame_id") != key for key, value in frames_raw.items()):
            raise ReviewError("proposal frame map keys differ from frame IDs")
        frames_raw = list(frames_raw.values())
    if not isinstance(frames_raw, list) or not frames_raw:
        raise ReviewError("proposal requires original decoded frame references")
    frames: dict[str, Any] = {}
    for raw_frame in frames_raw:
        try:
            frame = FrameRef.model_validate(raw_frame)
            frame.validate_timeline(asset)
        except Exception:
            raise ReviewError("proposal frame violates the shared source timeline contract") from None
        if frame.job_id != asset.job_id or frame.frame_id in frames:
            raise ReviewError("proposal frames require unique IDs bound to the source job")
        path = _under_roots(frame.path, frame_roots, description="proposal frame")
        digest = _sha256(path)
        if frame.jpeg_sha256 is not None and digest != frame.jpeg_sha256:
            raise ReviewError("proposal frame bytes differ from their checksum")
        try:
            from PIL import Image
            with Image.open(path) as image:
                if image.format != "JPEG" or image.size != (frame.width, frame.height):
                    raise ReviewError("proposal frame format or dimensions differ from its reference")
                image.verify()
        except ReviewError:
            raise
        except Exception:
            raise ReviewError("proposal frame is not a decodable source JPEG") from None
        frame.path = str(path)
        frame.jpeg_sha256 = digest
        frames[frame.frame_id] = frame
    known_facts, known_events = fact_definitions(), event_definitions()

    def parse_fact(raw_fact: Any, allowed_frames: set[str] | None = None) -> Any:
        try:
            fact = FactDecision.model_validate(raw_fact)
        except Exception:
            raise ReviewError("proposal contains an invalid shared fact decision") from None
        if fact.fact_id not in known_facts or set(fact.supporting_frame_ids) - (allowed_frames if allowed_frames is not None else frames.keys()):
            raise ReviewError("proposal fact references unknown taxonomy or source frames")
        return fact

    def parse_candidate(raw_event: Any, allowed_frames: set[str] | None = None) -> Any:
        try:
            candidate = CandidateEvent.model_validate(raw_event)
        except Exception:
            raise ReviewError("proposal contains an invalid candidate event") from None
        if candidate.event_type_id not in known_events or candidate.end_sec > asset.duration_sec:
            raise ReviewError("proposal candidate exceeds the taxonomy or source timeline")
        if set(candidate.supporting_frame_ids) - (allowed_frames if allowed_frames is not None else frames.keys()):
            raise ReviewError("proposal candidate references foreign source frames")
        for fact in candidate.supporting_facts:
            parse_fact(fact.model_dump(), allowed_frames)
        return candidate

    decisions = [parse_fact(value) for value in payload.get("decisions", [])]
    candidates = [parse_candidate(value) for value in payload.get("candidate_events", [])]
    verified = []
    for value in payload.get("verified_events", payload.get("events", [])):
        try:
            event = VerifiedEvent.model_validate(value)
        except Exception:
            raise ReviewError("proposal contains an invalid verified-event contract") from None
        if event.event_type_id not in known_events or event.end_sec > asset.duration_sec or set(event.evidence_frame_ids) - frames.keys():
            raise ReviewError("proposal event exceeds its taxonomy, timeline or source frames")
        if not event.start_sec <= frames[event.anchor_frame_id].pts_sec <= event.end_sec:
            raise ReviewError("proposal event anchor lies outside its interval")
        for fact in event.observed_facts:
            parse_fact(fact.model_dump())
        verified.append(event)
    windows = payload.get("window_decisions", [])
    if not isinstance(windows, list):
        raise ReviewError("proposal window decisions must be a list")
    # Request-bearing audit windows bind decisions to the exact inspected frame
    # set. Generic field-only windows are supported after explicit interval and
    # frame-ID validation; neither variant becomes reviewed coverage.
    for window in windows:
        if not isinstance(window, dict):
            raise ReviewError("proposal window decision must be an object")
        if "request" in window:
            try:
                request = ClipRequest.model_validate(window["request"])
                if request.job_id != asset.job_id:
                    raise ValueError("foreign job")
                for frame in request.frame_refs:
                    if frame.frame_id not in frames or frame.model_dump() != frames[frame.frame_id].model_dump():
                        # Added validated digests do not invalidate an original
                        # reference whose optional digest was not provided.
                        parsed = frames.get(frame.frame_id)
                        if parsed is None or any(getattr(frame, key) != getattr(parsed, key) for key in ("job_id", "path", "pts_sec", "width", "height", "rotation_deg")):
                            raise ValueError("foreign frame")
                allowed = {frame.frame_id for frame in request.frame_refs}
            except Exception:
                raise ReviewError("proposal window request differs from registered source frames") from None
        else:
            allowed = set(window.get("frame_ids", window.get("supporting_frame_ids", [])))
            if not allowed or allowed - frames.keys() or window.get("job_id", asset.job_id) != asset.job_id:
                raise ReviewError("proposal window requires its exact job-owned frame IDs")
            start, end = window.get("start_sec"), window.get("end_sec")
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not 0 <= start <= end <= asset.duration_sec:
                raise ReviewError("proposal window lies outside the original timeline")
            if any(not start <= frames[frame_id].pts_sec <= end for frame_id in allowed):
                raise ReviewError("proposal window includes frames outside its interval")
        window_body = window.get("decision", window)
        if not isinstance(window_body, dict):
            raise ReviewError("proposal window decision payload must be an object")
        for value in window_body.get("facts", window_body.get("decisions", [])):
            parse_fact(value, allowed)
        for value in window_body.get("candidate_events", window_body.get("candidates", [])):
            parse_candidate(value, allowed)
    normalized = {
        "schema_version": "forte.local-proposal.v1", "audit_sha256": hashlib.sha256(raw).hexdigest(),
        "audit_path": str(audit_path), "asset": asset.model_dump(mode="json"),
        "frames": [frame.model_dump(mode="json") for frame in frames.values()],
        "decisions": [fact.model_dump(mode="json") for fact in decisions],
        "candidate_events": [event.model_dump(mode="json") for event in candidates],
        "verified_events": [event.model_dump(mode="json") for event in verified],
        "window_decisions": windows, "provenance": provenance,
        "coverage": payload.get("coverage"), "unresolved_intervals": payload.get("unresolved_intervals", []),
        "limitations": payload.get("limitations", []),
        "phase": payload.get("phase"), "error_code": payload.get("error_code"),
        "publication_ready": payload.get("publication_ready", False),
        "refinement_plan": payload.get("refinement_plan", []),
        "analysis_complete": payload.get("analysis_complete", False),
        "reviewed": False,
    }
    if "local_rebase" in payload:
        normalized["local_rebase"] = payload["local_rebase"]
    # Reject nonfinite metadata without ever including its values in an error.
    try:
        canonical(normalized)
    except (ValueError, TypeError):
        raise ReviewError("proposal metadata must be finite JSON data") from None
    return store.append_local_proposal(
        record["sample_id"], normalized,
        expected_revision=record["revision"] if expected_revision is None else expected_revision,
    )
