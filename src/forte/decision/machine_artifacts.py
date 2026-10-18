"""Local, bounded machine-evidence verification; no inference or human claims.

File checks establish integrity/binding to the experiment's pinned artifacts,
not external authorship. A10 verifies provided-label evaluation separately.
Teacher support is recomputed from explicit comparison rows and remains silver.
Missing calibration artifacts fail closed; this module never manufactures them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .machine_contract import (
    MachineCalibrationManifest, MachineEvidenceVerification, machine_manifest_digest,
)
from .temporal_provider import TemporalProviderError

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_PINNED_FILE_BYTES = 256 * 1024 * 1024
HEX64 = re.compile(r"^[a-f0-9]{64}$")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _teacher_records(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise TemporalProviderError("teacher_reference_file_missing_or_oversized")
    try:
        raw = path.read_text(encoding="utf-8-sig")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError
                result[key] = value
            return result
        def parse(value):
            return json.loads(value, object_pairs_hook=pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        try:
            value = parse(raw)
            if isinstance(value, dict) and value.get("schema_version") == "forte.machine-annotation-bundle.v1":
                records = value.get("records")
            else:
                records = [value]
        except json.JSONDecodeError:
            records = [parse(line) for line in raw.splitlines() if line.strip()]
        if not isinstance(records, list) or len(records) > 10000 or any(not isinstance(row, dict) for row in records):
            raise ValueError
        return records
    except Exception:
        raise TemporalProviderError("teacher_reference_file_invalid") from None


def verify_teacher_reference_join(records: list[dict], comparisons: dict,
        manifest: MachineCalibrationManifest, *, validation_contents: set[str]) -> bool:
    """Join every comparison to the exact actual A03 source/window/target.

    Format/hash declarations alone never establish a teacher observation. Each
    reference identifies the complete record, window and original target hashes;
    source PTS, state, support, focal question and emitted prompt must agree.
    """
    from forte.video.batches import source_frame_id
    index = {}
    for record in records:
        # A bound multi-record export may also contain train/final records.
        # They are never calibration references and cannot enter this index.
        if record.get("split") != "validation" or record.get("content_sha256") not in validation_contents:
            continue
        if (record.get("schema_version") != "forte.machine-annotations.v1"
                or record.get("review_status") != "machine_consistency_checked"
                or record.get("annotation_origin") != "local_model" or record.get("human_reviewed") is not False
                or record.get("source_pts_identity_checked") is not True):
            return False
        provenance = record.get("provenance", {})
        attestation = provenance.get("serving_attestation", {})
        if (provenance.get("model_id") != manifest.teacher_identity.model_id
                or provenance.get("model_revision") != manifest.teacher_identity.revision
                or provenance.get("processor_revision") != manifest.teacher_identity.revision
                or provenance.get("model_manifest_sha256") != manifest.teacher_identity.model_artifacts_sha256
                or provenance.get("actual_snapshot_verified") is not True
                or provenance.get("execution_location") != "team_deployed_service"
                or provenance.get("external_ai_used") is not False
                or attestation.get("model_files_verified") is not True
                or attestation.get("model_manifest_sha256") != manifest.teacher_identity.model_artifacts_sha256
                or attestation.get("model_revision") != manifest.teacher_identity.revision
                or attestation.get("processor_revision") != manifest.teacher_identity.revision):
            return False
        record_hash = _canonical_digest(record)
        if record_hash in index:
            return False
        base = Fraction(record["source_time_base"])
        source_start = record["source_start_sec"]
        frames = {}
        for frame in record.get("frames", []):
            pts, frame_id = frame.get("source_pts"), frame.get("frame_id")
            if (type(pts) is not int or frame_id in frames
                    or frame_id != source_frame_id(record["content_sha256"], pts, base.numerator, base.denominator)
                    or abs(frame["pts_sec"] - (float(base * pts) - source_start)) > 1e-6):
                return False
            frames[frame_id] = frame
        windows = {}
        for window in record.get("windows", []):
            if window.get("model_output_complete") is not True or not set(window.get("frame_ids", [])) <= set(frames):
                continue
            window_hash = _canonical_digest(window)
            if window_hash in windows:
                return False
            windows[window_hash] = window
        index[record_hash] = (record, windows)
    for comparison in comparisons.get("fact_comparisons", []) + comparisons.get("event_comparisons", []):
        reference = index.get(comparison.get("teacher_record_sha256"))
        if reference is None:
            return False
        record, windows = reference
        window = windows.get(comparison.get("teacher_window_sha256"))
        if (window is None or comparison.get("content_sha256") != record["content_sha256"]
                or set(comparison.get("source_frame_ids", [])) != set(window["frame_ids"])):
            return False
        provenance = window.get("provenance", {})
        prompt = provenance.get("prompt_sha256")
        if (not isinstance(prompt, str) or not HEX64.fullmatch(prompt) or comparison.get("prompt_sha256") != prompt
                or provenance.get("proposal_config_sha256") != manifest.teacher_identity.proposal_config_sha256):
            return False
        kind = comparison.get("teacher_reference_kind")
        if kind == "frame_fact_target":
            candidates = window.get("frame_fact_targets", [])
        elif kind == "support_review":
            candidates = window.get("support_reviews", [])
        elif kind == "candidate":
            candidates = window.get("target", {}).get("candidates", [])
        elif kind == "event_presence_review":
            candidates = window.get("event_presence_reviews", [])
        else:
            return False
        matched = [target for target in candidates if _canonical_digest(target) == comparison.get("teacher_target_sha256")]
        if len(matched) != 1:
            return False
        target = matched[0]
        if kind == "frame_fact_target":
            focal = target.get("frame_id")
            question = target.get("question_definition")
            question_hash = target.get("provenance", {}).get("focal_question_sha256")
            if (target.get("supervision_mask") is not True or target.get("fact_id") != comparison.get("fact_id")
                    or target.get("human_verified") is not False or target.get("review_status") != "machine_consistency_checked"
                    or target.get("scope") != "exact_anchor_frame" or target.get("derivation") != "direct_anchor_teacher_question"
                    or focal != comparison.get("focal_frame_id") or focal not in window["frame_ids"]
                    or not isinstance(question, dict) or question_hash != _canonical_digest(question)
                    or target.get("question_fact_id") != "ANCHOR__" + comparison["fact_id"]
                    or question.get("fact_id") != target.get("question_fact_id")
                    or comparison.get("focal_question_sha256") != question_hash
                    or target.get("provenance", {}).get("prompt_sha256") != prompt):
                return False
            if comparison.get("supports_state") is True:
                if target.get("state") != comparison.get("state"):
                    return False
                if target["state"] == "absent" and comparison.get("absence_scope") != "exact_anchor_frame":
                    context = window.get("context_masks", {})
                    if context.get("procedure_complete") is not True or context.get("continuous_visibility") is not True:
                        return False
                expected = [] if target["state"] == "unclear" else [focal]
                if (target["state"] != "unclear" and focal not in target.get("supporting_frame_ids", [])
                        or comparison.get("teacher_supporting_frame_ids") != expected):
                    return False
            elif (target.get("state") != "unclear" or comparison.get("teacher_state") != "unclear"
                    or comparison.get("negative_reference_kind") != "focal_frame_explicit_unclear"):
                return False
        elif kind == "support_review":
            if (target.get("review_status") != "machine_consistency_checked" or target.get("supports_state") is not False
                    or target.get("supervision_mask") is not True or target.get("human_verified") is not False
                    or comparison.get("supports_state") is not False
                    or target.get("fact_id") != comparison.get("fact_id")
                    or target.get("frame_id") not in window["frame_ids"]
                    or comparison.get("support_review_sha256") != _canonical_digest(target)):
                return False
            if comparison.get("negative_reference_kind") == "focal_frame_explicit_unclear":
                question = target.get("question_definition")
                question_hash = target.get("provenance", {}).get("focal_question_sha256")
                if (target.get("teacher_state") != "unclear"
                        or target.get("negative_reference_kind", target.get("reference_kind")) != "focal_frame_explicit_unclear"
                        or target.get("frame_id") != comparison.get("focal_frame_id")
                        or not isinstance(question, dict) or question_hash != _canonical_digest(question)
                        or target.get("question_fact_id") != "ANCHOR__" + comparison["fact_id"]
                        or question.get("fact_id") != target.get("question_fact_id")
                        or comparison.get("focal_question_sha256") != question_hash
                        or target.get("provenance", {}).get("prompt_sha256") != prompt):
                    return False
        elif kind == "candidate":
            if (comparison.get("teacher_event_present") is not True or target.get("event_type_id") != comparison.get("event_type_id")
                    or comparison.get("teacher_bracket_frame_ids") != [target.get("start_frame_id"), target.get("end_frame_id")]):
                return False
        elif (target.get("event_type_id") != comparison.get("event_type_id")
                or target.get("review_status") != "machine_consistency_checked"
                or target.get("event_present") is not False or comparison.get("teacher_event_present") is not False):
            return False
    return True


def file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PINNED_FILE_BYTES:
        raise TemporalProviderError("machine_evidence_file_missing")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise TemporalProviderError("machine_evidence_file_missing_or_oversized")
    try:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if not isinstance(value, dict):
            raise ValueError("object required")
        return value
    except Exception:
        raise TemporalProviderError("machine_evidence_json_invalid") from None


@dataclass(frozen=True, repr=False)
class MachineEvidenceFiles:
    split_assignments: Path
    experiment_config: Path
    exposure_manifest: Path
    weak_gate_proof: Path
    provided_evaluation: Path
    selection: Path
    threshold_selection: Path
    teacher_proposals: Path
    teacher_support_agreement: Path
    teacher_comparisons: Path
    teacher_model_manifest: Path | None = None


def teacher_processor_artifacts_sha256(model_manifest: dict) -> str:
    """Versioned digest of processor/shared-config entries in A07's manifest.

    The manifest itself binds every weight and processor artifact. Actual file
    verification at teacher execution is retained in A03's runtime receipt.
    This pure helper needs no private media, model load or manual annotation.
    """
    return _canonical_digest({"schema_version": "forte.teacher-processor-artifacts.v1",
        "processor_revision": model_manifest["processor_revision"],
        "files": sorted((entry for entry in model_manifest["files"]
                         if not entry["path"].endswith(".safetensors")), key=lambda entry: entry["path"])})


def _frame_set(row: dict, key: str) -> set[str]:
    values = row.get(key)
    if (not isinstance(values, list)
            or any(not isinstance(value, str) or not HEX64.fullmatch(value) for value in values)
            or len(values) != len(set(values))):
        raise TemporalProviderError("teacher_comparison_source_ids_invalid")
    return set(values)


def verify_teacher_comparisons(report: dict, comparisons: dict,
                               manifest: MachineCalibrationManifest) -> bool:
    """Recount explicit source-bound state/support matches, never call accuracy.

    A comparison row binds its original source-frame universe, teacher support,
    head support, class, and source-verification receipt. Absence rows require
    explicit complete procedure and adequate visibility. Unmarked pairs do not
    enter the denominator. Exact support agreement is intentionally conservative.
    """
    identity = manifest.identity.model_dump(mode="json")
    teacher = manifest.teacher_identity.model_dump(mode="json")
    if (report.get("schema_version") != "forte.teacher-support-agreement.v1"
            or comparisons.get("schema_version") != "forte.teacher-support-comparisons.v1"
            or report.get("head_identity") != identity or comparisons.get("head_identity") != identity
            or report.get("teacher_identity") != teacher or comparisons.get("teacher_identity") != teacher
            or report.get("teacher_proposals_sha256") != manifest.teacher_proposals_sha256
            or report.get("split_assignments_sha256") != manifest.split_assignments_sha256
            or report.get("reference_kind") != "local_teacher_silver"
            or report.get("teacher_agreement_is_accuracy") is not False):
        return False
    rows = comparisons.get("fact_comparisons")
    events = comparisons.get("event_comparisons")
    if not isinstance(rows, list) or not isinstance(events, list):
        return False
    counts, agreed, class_counts, support_counts, negative_counts = Counter(), Counter(), Counter(), Counter(), Counter()
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        key = (row.get("fact_id"), row.get("state"))
        comparison_id = row.get("comparison_id")
        if (not isinstance(comparison_id, str) or not comparison_id or comparison_id in seen
                or key[0] not in manifest.identity.fact_ids or key[1] not in {"present", "absent", "unclear"}
                or row.get("source_binding_verified") is not True):
            return False
        seen.add(comparison_id)
        source = _frame_set(row, "source_frame_ids")
        expected = _frame_set(row, "teacher_supporting_frame_ids")
        actual = _frame_set(row, "predicted_supporting_frame_ids")
        supports_state = row.get("supports_state")
        if type(supports_state) is not bool:
            return False
        if not supports_state:
            # A locally inspected non-support reference is explicit. Missing
            # teacher labels or arbitrary unmarked frames never enter here.
            if key[1] == "unclear" or expected or not actual <= source or not source:
                return False
            reference = row.get("negative_reference_kind")
            if reference == "focal_frame_explicit_unclear":
                if (row.get("teacher_state") != "unclear" or row.get("focal_frame_id") not in source
                        or not isinstance(row.get("prompt_sha256"), str) or not HEX64.fullmatch(row["prompt_sha256"])
                        or not isinstance(row.get("focal_question_sha256"), str)
                        or not HEX64.fullmatch(row["focal_question_sha256"])):
                    return False
            elif reference == "local_teacher_explicit_non_support":
                receipt = row.get("support_review_sha256")
                if not isinstance(receipt, str) or not HEX64.fullmatch(receipt):
                    return False
            else:
                return False
            negative_counts[key] += 1
            counts[key] += 1
            agreed[key] += not actual
            continue
        if (not expected <= source or not actual <= source
                or key[1] != "unclear" and not expected
                or key[1] == "absent" and (row.get("visibility_adequate") is not True
                    or row.get("absence_scope") != "exact_anchor_frame" and row.get("procedure_context_complete") is not True)):
            return False
        class_counts[key] += 1
        if expected:
            support_counts[key] += 1
        if key[1] != "unclear":
            counts[key] += 1
            agreed[key] += bool(expected and actual == expected and row.get("predicted_state") == key[1])
    for fact_id, calibration in manifest.fact_calibration.items():
        for state in ("present", "absent", "unclear"):
            key = (fact_id, state)
            if calibration.silver_class_counts.get(state, 0) != class_counts[key]:
                return False
            record = getattr(calibration, state, None)
            if record is not None and (record.silver_support_count != support_counts[key]
                    or record.silver_support_negative_count != negative_counts[key]
                    or record.support_agreement.compared_count != counts[key]
                    or record.support_agreement.agreed_count != agreed[key]):
                return False
    # Typed boundary calibration additionally requires explicit exact paired
    # source-frame matches; fact-run proposals do not require positive events.
    event_counts, event_agreed, event_normal = Counter(), Counter(), Counter()
    for row in events:
        if not isinstance(row, dict) or row.get("event_type_id") not in manifest.identity.event_type_ids:
            return False
        event_id = row["event_type_id"]
        comparison_id = row.get("comparison_id")
        if (not isinstance(comparison_id, str) or not comparison_id or comparison_id in seen
                or row.get("source_binding_verified") is not True
                or type(row.get("teacher_event_present")) is not bool):
            return False
        seen.add(comparison_id)
        if not row["teacher_event_present"]:
            event_normal[event_id] += 1
            continue
        source = _frame_set(row, "source_frame_ids")
        expected = row.get("teacher_bracket_frame_ids")
        actual = row.get("predicted_bracket_frame_ids")
        if (not isinstance(expected, list) or len(expected) != 2 or not set(expected) <= source
                or not isinstance(actual, list) or len(actual) not in {0, 2} or not set(actual) <= source):
            return False
        event_counts[event_id] += 1
        event_agreed[event_id] += actual == expected
    for event_id, record in manifest.event_calibration.items():
        if (record.silver_event_count != event_counts[event_id] or record.silver_normal_count != event_normal[event_id]
                or record.support_agreement.compared_count != event_counts[event_id]
                or record.support_agreement.agreed_count != event_agreed[event_id]):
            return False
    return True


class ArtifactMachineEvidenceVerifier:
    """Compose A02 machine freeze, A10 attestation and actual silver/thresholds."""
    def __init__(self, files: MachineEvidenceFiles):
        self.files = files

    def verify(self, manifest: MachineCalibrationManifest) -> MachineEvidenceVerification:
        from evaluation.weak_supervision import (
            weak_assignment_digest, verify_provided_evaluation_for_manifest,
        )
        from evaluation.io import digest

        try:
            files = self.files
            split, config, gate = (_json(path) for path in
                (files.split_assignments, files.experiment_config, files.weak_gate_proof))
            evaluation, selection = _json(files.provided_evaluation), _json(files.selection)
            provided = verify_provided_evaluation_for_manifest(evaluation, manifest, selection)
            authorization = config.get("experiment_authorization", config)
            split_ok = (manifest.assignments_frozen and authorization.get("validation_assignments_frozen") is True
                and weak_assignment_digest(split.get("records", [])) == manifest.split_assignments_sha256
                and split.get("assignments_sha256") == manifest.split_assignments_sha256
                and gate.get("expected_assignments_sha256") == manifest.split_assignments_sha256
                and gate.get("ok") is True and gate.get("label_split", "validation") == "validation"
                and digest(gate) == evaluation.get("provenance", {}).get("weak_gate_proof_sha256")
                and file_sha256(files.exposure_manifest) == manifest.exposure_manifest_sha256
                and gate.get("exposure_registry_file_sha256") == manifest.exposure_manifest_sha256)
            threshold = _json(files.threshold_selection)
            binary = manifest.provided_binary_calibration
            threshold_ok = (file_sha256(files.threshold_selection) == manifest.threshold_selection_sha256
                and threshold.get("schema_version") == "forte.weak_video_threshold.v1"
                and threshold.get("score_axis") == binary.score_axis
                and type(threshold.get("minimum_video_score")) in {int, float}
                and threshold["minimum_video_score"] == binary.minimum_video_score
                and threshold.get("comparison") == "greater_than_or_equal"
                and threshold.get("head_sha256") == manifest.identity.head_sha256
                and threshold.get("head_identity_sha256") == digest(manifest.identity.model_dump(mode="json"))
                and threshold.get("feature_spec_sha256") == manifest.identity.feature_spec_sha256
                and threshold.get("provided_validation_labels_sha256") == manifest.provided_labels_sha256)
            # Any fact/event calibration must appear verbatim in the frozen
            # selection artifact. A binary-only file cannot unlock19 facts.
            has_facts = any(record.present is not None or record.absent is not None
                            for record in manifest.fact_calibration.values())
            if has_facts or manifest.event_calibration:
                threshold_ok = bool(threshold_ok and threshold.get("fact_calibration") == {
                    key: value.model_dump(mode="json") for key, value in manifest.fact_calibration.items()}
                    and threshold.get("event_calibration") == {
                    key: value.model_dump(mode="json") for key, value in manifest.event_calibration.items()}
                    and threshold.get("decoder_sha256") == manifest.decoder_sha256)
            report, comparisons = _json(files.teacher_support_agreement), _json(files.teacher_comparisons)
            teacher_identity_ok = True
            if comparisons.get("fact_comparisons") or comparisons.get("event_comparisons"):
                from forte.model.manifest import load_manifest
                if files.teacher_model_manifest is None:
                    teacher_identity_ok = False
                else:
                    teacher_model = load_manifest(files.teacher_model_manifest)
                    teacher_identity_ok = (
                        file_sha256(files.teacher_model_manifest) == manifest.teacher_identity.model_artifacts_sha256
                        and teacher_model["model_id"] == manifest.teacher_identity.model_id
                        and teacher_model["revision"] == manifest.teacher_identity.revision
                        and teacher_processor_artifacts_sha256(teacher_model) == manifest.teacher_identity.processor_artifacts_sha256)
            teacher_ok = (file_sha256(files.teacher_proposals) == manifest.teacher_proposals_sha256
                and file_sha256(files.teacher_support_agreement) == manifest.teacher_support_agreement_sha256
                and report.get("comparison_records_sha256") == file_sha256(files.teacher_comparisons)
                and teacher_identity_ok
                and verify_teacher_comparisons(report, comparisons, manifest)
                and verify_teacher_reference_join(_teacher_records(files.teacher_proposals), comparisons, manifest,
                    validation_contents={row.get("content_sha256", row.get("content_hash"))
                        for row in split.get("records", []) if row.get("split") == "validation"}))
            return MachineEvidenceVerification(manifest_sha256=machine_manifest_digest(manifest),
                split_and_exposure_verified=bool(split_ok),
                provided_label_evaluation_verified=provided["provided_label_evaluation_verified"] is True,
                teacher_support_agreement_verified=bool(teacher_ok), threshold_selection_verified=bool(threshold_ok),
                provided_binary_target_met=provided.get("provided_binary_target_met"),
                latency_target_met=provided.get("latency_target_met"))
        except TemporalProviderError:
            raise
        except Exception:
            raise TemporalProviderError("machine_evidence_artifact_verification_failed") from None
