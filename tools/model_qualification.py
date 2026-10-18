"""Summarize two collected development runs without opening private reports or media.

Input must be the structured allowlist export from collect_models.remote.sh.
This checks recorded scores and identities; it never infers a release qualification
from two accuracy runs alone. Missing reports remain pending. Output is UTC JSON.

  python tools/model_qualification.py --run collected.json gemma4a \
      --run collected.json gemma4b --out qualification.json
  python tools/model_qualification.py --self-test
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

SCHEMA = "forte.model-benchmark.v1"
Q30_SCHEMA = "forte.q30-model-benchmark.v1"
ATTESTATION_SCHEMAS = {"forte.model-run-attestation.v1", "forte.q30-run-attestation.v1"}
Q30_REFERENCE = "b236180cd0ab3324c521f9f29a154bbcdb32220129c31bd905931ebdc73e08fe"
Q30_REPO = "Qwen/Qwen3-VL-30B-A3B-Instruct"
Q30_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
SHA = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
VIDEO = re.compile(r"X-[1-9][0-9]{0,2}\Z")
REQUIRED_SOURCE = {"src/forte/fast/" + name + ".py" for name in
                   ("analyzer", "prompts", "vlm", "service", "video", "catalog")}
ATTESTED_SOURCE = REQUIRED_SOURCE | {"tools/fast_eval.py", "tools/fast_validation.py"}
COUNTS = ("n", "processed", "failed", "correct", "positive", "negative", "tp", "tn", "fp", "fn")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA.fullmatch(value))


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0
    except ValueError:
        return False


def strict_json(path: Path) -> dict:
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("input_oversize")
    def constant(_: str):
        raise ValueError("nonfinite_json")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("input_not_object")
    return value


def check(status: str, reason: str | None = None, **values: Any) -> dict:
    result = {"status": status, **values}
    if reason:
        result["reason"] = reason
    return result


def comparison(values: list[Any], *, valid=valid_hash) -> dict:
    if any(not valid(value) for value in values):
        return check("pending", "identity_missing_or_invalid")
    return check("passed" if values[0] == values[1] else "failed",
                 None if values[0] == values[1] else "identity_differs",
                 equal=values[0] == values[1])


def evaluate_run(document: dict | None, name: str) -> tuple[dict, dict | None]:
    result = {"run": name, "status": "pending", "reasons": []}
    if document is None:
        result["reasons"] = ["collection_file_missing"]
        return result, None
    if document.get("schema") not in {SCHEMA, Q30_SCHEMA}:
        return {**result, "status": "failed", "reasons": ["structured_collection_schema_required"]}, None
    row = document.get("runs", {}).get(name)
    if not isinstance(row, dict) or row.get("status") == "missing_report":
        result["reasons"] = ["run_report_missing"]
        return result, None
    if row.get("status") != "available":
        return {**result, "status": "failed", "reasons": ["run_report_invalid"]}, row
    reasons = result["reasons"]
    rows = row.get("videos")
    if not isinstance(rows, list) or len(rows) != 40:
        return {**result, "status": "failed", "reasons": ["requires_40_structured_video_rows"]}, row
    names = [video.get("name") if isinstance(video, dict) else None for video in rows]
    if any(not isinstance(name, str) or not VIDEO.fullmatch(name) for name in names) or len(set(names)) != 40:
        reasons.append("video_names_invalid_or_duplicate")
    counts = {key: 0 for key in COUNTS}
    counts["n"] = len(rows)
    valid_rows = True
    for video in rows:
        if not isinstance(video, dict):
            valid_rows = False
            continue
        positive = (video.get("label") or {}).get("violation")
        processed, prediction = video.get("processed"), video.get("prediction")
        if type(positive) is not bool or type(processed) is not bool:
            valid_rows = False
            continue
        counts["positive" if positive else "negative"] += 1
        if not processed:
            counts["failed"] += 1
            if prediction is not None or video.get("status") != "failed":
                valid_rows = False
            continue
        counts["processed"] += 1
        events = video.get("events")
        if (type(prediction) is not bool or not isinstance(events, list)
                or prediction != bool(events) or video.get("status") != "done"
                or video.get("verdict") != ("violation" if prediction else "no_violation")
                or video.get("failure_codes")):
            valid_rows = False
            continue
        if video.get("correct") is not (positive == prediction):
            valid_rows = False
        counts["correct"] += positive == prediction
        counts[("tp" if prediction else "fn") if positive else ("fp" if prediction else "tn")] += 1
    if not valid_rows:
        reasons.append("structured_row_consistency_failed")
    if any(type(row.get("summary", {}).get(key)) is not int or row["summary"][key] != value
           for key, value in counts.items()):
        reasons.append("summary_differs_from_structured_rows")
    if row.get("integrity", {}).get("complete") is not True:
        reasons.append("incomplete_run_integrity")
    if row.get("source_summary_mismatch_fields"):
        reasons.append("source_summary_mismatch")
    for key, expected in (("n", 40), ("processed", 40), ("failed", 0), ("positive", 21), ("negative", 19), ("fp", 0)):
        if counts[key] != expected:
            reasons.append("required_" + key + "_not_met")
    if counts["correct"] < 37:
        reasons.append("correct_below_37")
    for field in ("source_model_matches_metadata", "collected_profile_values_match_reported_config",
                  "reported_config_digest_matches_provenance", "desk_route_matches_requested_profile"):
        if row.get(field) is False:
            reasons.append(field + "_failed")
    attestation = row.get("run_attestation")
    if isinstance(attestation, dict) and attestation.get("schema") in ATTESTATION_SCHEMAS and attestation.get("run") != name:
        reasons.append("attestation_run_name_differs")
    result.update(status="failed" if reasons else "passed", counts=counts,
                  accuracy=counts["correct"] / 40, report_sha256=row.get("private_report_sha256")
                  if valid_hash(row.get("private_report_sha256")) else None)
    return result, row


def attested_identity(row: dict, document: dict) -> tuple[dict, dict | None]:
    value = row.get("run_attestation")
    if not isinstance(value, dict) or value.get("status") in {None, "absent", "missing"}:
        return check("pending", "historical_run_attestation_absent"), None
    if value.get("schema") not in ATTESTATION_SCHEMAS:
        return check("failed", "run_attestation_schema_invalid"), None
    before, after = value.get("before"), value.get("after")
    if value.get("status") != "verified" or value.get("identities_equal") is not True or before != after or not isinstance(before, dict):
        return check("failed", "run_identity_changed_or_not_verified"), None
    source = before.get("source_sha256")
    server = before.get("server")
    if (not isinstance(source, dict) or not ATTESTED_SOURCE <= set(source)
            or not all(valid_hash(item) for item in source.values())
            or not valid_hash(before.get("config_file_sha256")) or not isinstance(server, dict)):
        return check("failed", "attested_source_config_or_server_incomplete"), None
    model = row.get("model") or {}
    q30 = value.get("schema") == "forte.q30-run-attestation.v1"
    server_model = server.get("request_model") if q30 else server.get("served_model")
    if value.get("model") != model.get("served_model") or server_model != model.get("served_model"):
        return check("failed", "attested_served_model_differs"), None
    if (not valid_hash(server.get("container_id")) or not isinstance(server.get("image_id"), str)
            or not server["image_id"].startswith("sha256:") or not valid_hash(server["image_id"][7:])
            or not valid_hash(server.get("base_weight_receipt_sha256"))):
        return check("failed", "attested_server_or_weights_identity_missing"), None
    if value.get("model") == "gemma4ft" and (not valid_hash(server.get("adapter_export_receipt_sha256"))
                                              or not valid_hash(server.get("adapter_file_sha256"))):
        return check("failed", "attested_adapter_identity_missing"), None
    if before["config_file_sha256"] != document.get("config_file_sha256"):
        return check("failed", "attested_config_differs_from_collected_profile"), None
    if q30:
        flags = ("historical_binding_valid", "before_after_exactly_equal", "source_matches_collection",
                 "collected_profile_file_sha256_matches")
        required = ("source_manifest_sha256", "harness_sha256", "evaluation_wrapper_sha256",
                    "source_install_receipt_sha256", "media_inventory_file_sha256", "labels_scorer_file_sha256")
        if any(value.get(flag) is not True for flag in flags) or any(not valid_hash(before.get(key)) for key in required):
            return check("failed", "q30_collected_historical_binding_invalid"), None
        if (source != document.get("source_sha256") or digest(source) != before.get("source_manifest_sha256")
                or before.get("source_manifest_sha256") != document.get("source_manifest_sha256")
                or before.get("labels_scorer_file_sha256") != document.get("labels_file_sha256")
                or before.get("harness_sha256") != document.get("harness_sha256")
                or before.get("source_install_receipt_sha256") != document.get("source_install_receipt_sha256")
                or before.get("reference_profile_file_sha256") != Q30_REFERENCE
                or document.get("reference_profile_file_sha256") != Q30_REFERENCE):
            return check("failed", "q30_attested_frozen_sources_or_receipts_differ"), None
        if (server.get("repo") != Q30_REPO or server.get("revision") != Q30_REVISION
                or server.get("dtype") != "bfloat16" or server.get("quantization") is not None
                or not all(valid_hash(server.get(key)) for key in
                           ("args_sha256", "server_receipt_sha256", "base_weight_manifest_sha256"))
                or server.get("base_weight_manifest_sha256") != model.get("pinned_base_weight_manifest_sha256")
                or server.get("base_weight_receipt_sha256") != model.get("pinned_base_weight_receipt_sha256")
                or before.get("evaluation_args") != {"par": 4, "concurrency": 64}
                or not isinstance(before.get("evaluation_image_id"), str)
                or not before["evaluation_image_id"].startswith("sha256:")
                or not valid_hash(before["evaluation_image_id"][7:])):
            return check("failed", "q30_attested_server_weights_or_eval_runtime_differ"), None
        route = before.get("desk_adapter_model")
        adapter = before.get("adapter")
        if route not in {"", "q30desk"} or route != document.get("desk_adapter_model") or route != model.get("desk_adapter_model"):
            return check("failed", "q30_desk_route_identity_differs"), None
        if route == "q30desk":
            if (not isinstance(adapter, dict) or adapter.get("served_model") != "q30desk"
                    or not all(valid_hash(adapter.get(key)) for key in
                               ("weights_file_sha256", "config_file_sha256", "export_receipt_sha256", "training_run_receipt_sha256"))):
                return check("failed", "q30_attested_adapter_identity_missing"), None
        elif adapter is not None:
            return check("failed", "q30_baseline_unexpected_adapter"), None
        transport = before.get("evaluation_transport") or {}
        if (transport.get("mode") != "model_container_network_namespace" or transport.get("internal_network") is not True
                or transport.get("default_ipv4_route") is not False or transport.get("http_ready") is not True
                or transport.get("aliases_ready") is not True or transport.get("container_id") != server.get("container_id")):
            return check("failed", "q30_offline_transport_identity_invalid"), None
    return check("passed", "before_after_runtime_identities_verified"), before


def identities(documents: list[dict | None], rows: list[dict | None]) -> dict:
    if any(document is None for document in documents) or any(row is None for row in rows):
        return {"status": "pending", "reason": "two_available_runs_required"}
    docs, runs = documents, rows
    attestations = [attested_identity(row, document) for row, document in zip(runs, docs)]
    models = [row.get("model") or {} for row in runs]
    provenance = [row.get("provenance_sha256") or {} for row in runs]
    checks = {
        "model_repo": comparison([model.get("repo") for model in models], valid=lambda value: isinstance(value, str) and bool(REPO.fullmatch(value))),
        "served_model": comparison([model.get("served_model") for model in models], valid=lambda value: isinstance(value, str) and bool(RUN.fullmatch(value))),
        "observed_revision": comparison([model.get("observed_revision") for model in models], valid=lambda value: isinstance(value, str) and bool(REVISION.fullmatch(value))),
        "config": comparison([p.get("config_sha256") for p in provenance]),
        "analyzer_policy_config": comparison([p.get("analyzer_policy_sha256") for p in provenance]),
        "media_inventory": comparison([p.get("input_inventory_sha256") for p in provenance]),
        "labels": comparison([p.get("labels_file_sha256") for p in provenance]),
        "runner": comparison([p.get("runner_code_sha256") for p in provenance]),
        "scorer": comparison([p.get("scorer_code_sha256") for p in provenance]),
    }
    source_maps = [document.get("source_sha256") or {} for document in docs]
    source_valid = [isinstance(mapping, dict) and REQUIRED_SOURCE <= set(mapping)
                    and all(valid_hash(value) for value in mapping.values()) for mapping in source_maps]
    checks["source"] = (comparison(source_maps, valid=lambda value: isinstance(value, dict))
                        if all(source_valid) else check("pending", "analyzer_source_hashes_incomplete"))
    source_digests_valid = all(isinstance(mapping, dict) and digest(mapping) == document.get("source_manifest_sha256")
                               for mapping, document in zip(source_maps, docs))
    if not source_digests_valid:
        checks["source_manifest"] = check("failed", "source_manifest_digest_invalid")
    else:
        checks["source_manifest"] = comparison([document.get("source_manifest_sha256") for document in docs])
    config_projection = [row.get("config") or {} for row in runs]
    if any(config.get("complete_config_json_sha256") != p.get("config_sha256") for config, p in zip(config_projection, provenance)):
        checks["config"] = check("failed", "config_projection_digest_differs_from_provenance")
    observed = [model.get("observed_revision") for model in models]
    if any(isinstance(revision, str) and REVISION.fullmatch(revision)
           and model.get("expected_revision") is not None and model.get("expected_revision") != revision
           for model, revision in zip(models, observed)):
        checks["observed_revision"] = check("failed", "observed_revision_differs_from_expected")
    model_files = [model.get("small_model_files_sha256") or {} for model in models]
    checks["small_model_files"] = comparison(model_files, valid=lambda mapping: isinstance(mapping, dict)
                                             and valid_hash(mapping.get("config.json"))
                                             and all(valid_hash(value) for value in mapping.values()))
    servers = [model.get("server") or {} for model in models]
    checks["serving_image"] = comparison([server.get("image_id") for server in servers],
                                          valid=lambda value: isinstance(value, str) and value.startswith("sha256:") and valid_hash(value[7:]))
    checks["serving_flags"] = comparison([server.get("args") for server in servers], valid=lambda value: isinstance(value, dict) and bool(value))
    hashes = [row.get("private_report_sha256") for row in runs]
    dates = [row.get("source_generated_at_utc") for row in runs]
    if all(valid_hash(value) for value in hashes) and hashes[0] == hashes[1]:
        checks["independent_reports"] = check("failed", "same_private_report_reused")
    elif not all(valid_hash(value) for value in hashes) or not all(utc_timestamp(value) for value in dates):
        checks["independent_reports"] = check("pending", "report_hash_or_run_timestamp_missing")
    elif dates[0] == dates[1]:
        checks["independent_reports"] = check("failed", "same_run_timestamp")
    else:
        checks["independent_reports"] = check("passed", "distinct_report_hashes_and_run_timestamps")
    if all(item[0]["status"] == "passed" for item in attestations):
        checks["historical_run_binding"] = comparison([item[1] for item in attestations], valid=lambda value: isinstance(value, dict))
        checks["source"] = comparison([item[1]["source_sha256"] for item in attestations], valid=lambda value: isinstance(value, dict))
    else:
        checks["historical_run_binding"] = check("failed" if any(item[0]["status"] == "failed" for item in attestations) else "pending",
                                                  "historical_run_binding_incomplete",
                                                  runs=[item[0] for item in attestations])
    status = "failed" if any(value["status"] == "failed" for value in checks.values()) else "pending" if any(value["status"] == "pending" for value in checks.values()) else "passed"
    return {"status": status, "checks": checks,
            "source_scope": "before_after_run_attestation" if checks.get("historical_run_binding", {}).get("status") == "passed"
                            else "collection_time_observation; historical_binding_incomplete",
            "analyzer_policy_scope": "configuration_only; does_not_hash_source_bytes"}


def stage_summary(stage: dict | None) -> dict:
    pending = {key: check("pending", "stage_measurement_required") for key in
               ("schema", "evidence_originality", "six_parallel_within_285s", "candidate_identity_binding")}
    if stage is None:
        return pending
    if stage.get("schema") != "forte.new-stage-check.v1":
        return {key: check("pending", "unsupported_stage_record_schema") for key in pending}
    passed = stage.get("status") == "passed"
    evidence = stage.get("evidence_originality") or {}
    pending["schema"] = check("passed" if passed and stage.get("jobs") == 6 and stage.get("failed") == 0 else "failed",
                              "measured_stage_contract_checks")
    if evidence.get("passed") is True and evidence.get("frames", 0) == 0:
        pending["evidence_originality"] = check("pending", "zero_evidence_frames_exercised")
    else:
        pending["evidence_originality"] = check("passed" if evidence.get("passed") is True and evidence.get("frames", 0) > 0 else "failed")
    pending["six_parallel_within_285s"] = check("passed" if passed and stage.get("parallel_uploads") == 6
                                                and stage.get("jobs") == 6 and stage.get("failed") == 0
                                                and stage.get("deadline_s") == 285
                                                and stage.get("six_long_within_285s") is True else "failed")
    pending["candidate_identity_binding"] = check("pending", "stage_to_candidate_runtime_source_and_weight_binding_requires_review")
    return pending


def summarize(documents: list[dict | None], names: list[str], stage: dict | None = None) -> dict:
    evaluated = [evaluate_run(document, name) for document, name in zip(documents, names)]
    runs, rows = [item[0] for item in evaluated], [item[1] for item in evaluated]
    score_status = "failed" if any(row["status"] == "failed" for row in runs) else "pending" if any(row["status"] == "pending" for row in runs) else "passed"
    identity = identities(documents, rows)
    status = ("not_qualified" if score_status == "failed" or identity["status"] == "failed" else
              "pending_runs_or_identities" if score_status == "pending" or identity["status"] == "pending" else
              "ready_for_release_checks")
    return {"schema": "forte.model-qualification-summary.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
            "scope": "40_unique_development_videos_in_sample; no_unseen_accuracy_claim",
            "qualification_claim": False, "promotion_authorized": False,
            "requirements": {"independent_runs": 2, "n_per_run": 40, "processed_per_run": 40,
                             "failed_per_run": 0, "minimum_correct_per_run": 37, "fp_per_run": 0},
            "score_gate": {"status": score_status, "runs": runs}, "recorded_identity_consistency": identity,
            "release_checks": stage_summary(stage),
            "additional_release_requirements": ["evidence_scene_relevance_no_regression",
                                                 "target_gpu_staging_and_rollback_review", "explicit_user_switch_approval",
                                                 "public_six_upload_check_after_switch"]}


def self_test() -> int:
    source = {name: "1" * 64 for name in REQUIRED_SOURCE}
    provenance = {key: "2" * 64 for key in ("config_sha256", "analyzer_policy_sha256", "input_inventory_sha256",
                 "labels_file_sha256", "runner_code_sha256", "scorer_code_sha256")}
    videos = [{"name": f"X-{i+1}", "label": {"violation": i < 21}, "processed": True,
               "status": "done", "prediction": i < 18, "correct": (i < 21) == (i < 18),
               "verdict": "violation" if i < 18 else "no_violation", "failure_codes": [],
               "events": [{"type_id": "PERSONAL_ITEMS", "t": 1}] if i < 18 else []} for i in range(40)]
    row = {"status": "available", "videos": videos, "summary": dict(zip(COUNTS, (40,40,0,37,21,19,18,19,0,3))),
           "integrity": {"complete": True}, "provenance_sha256": provenance,
           "config": {"complete_config_json_sha256": provenance["config_sha256"]},
           "source_summary_mismatch_fields": [], "model": {"repo": "Qwen/Qwen3.8-27B-FP8", "served_model": "q38",
           "observed_revision": "a" * 40, "small_model_files_sha256": {"config.json": "3" * 64},
           "server": {"image_id": "sha256:" + "4" * 64, "args": {"--seed": 0}}}}
    a, b = copy.deepcopy(row), copy.deepcopy(row)
    a.update(private_report_sha256="a" * 64, source_generated_at_utc="2026-10-18T05:00:00Z")
    b.update(private_report_sha256="b" * 64, source_generated_at_utc="2026-10-18T05:10:00Z")
    base = {"schema": SCHEMA, "source_sha256": source, "source_manifest_sha256": digest(source), "runs": {"a": a, "b": b}}
    assert summarize([base,base],["a","b"])["score_gate"]["status"] == "passed"
    assert summarize([base,base],["a","b"])["recorded_identity_consistency"]["checks"]["historical_run_binding"]["status"] == "pending"
    assert summarize([base,base],["a","b"])["qualification_claim"] is False
    changed = copy.deepcopy(base); changed["runs"]["b"]["summary"]["correct"] = 38
    assert summarize([base,changed],["a","b"])["score_gate"]["status"] == "failed"
    changed = copy.deepcopy(base); changed["runs"]["b"]["videos"][39]["processed"] = False
    assert summarize([base,changed],["a","b"])["score_gate"]["status"] == "failed"
    changed = copy.deepcopy(base); changed["runs"]["b"]["private_report_sha256"] = "a" * 64
    assert summarize([base,changed],["a","b"])["status"] == "not_qualified"
    changed = copy.deepcopy(base); changed["runs"]["b"]["model"]["observed_revision"] = "b" * 40
    assert summarize([base,changed],["a","b"])["status"] == "not_qualified"
    changed = copy.deepcopy(base); changed["source_sha256"] = {}; changed["source_manifest_sha256"] = digest({})
    assert summarize([changed,changed],["a","b"])["status"] == "pending_runs_or_identities"
    assert summarize([base,None],["a","b"])["score_gate"]["status"] == "pending"
    assert stage_summary({"schema":"forte.new-stage-check.v1", "evidence_originality":{"passed":True,"frames":0}})["evidence_originality"]["status"] == "pending"
    attested = copy.deepcopy(base)
    attested["config_file_sha256"] = "c" * 64
    identity = {"source_sha256": {name: "1" * 64 for name in ATTESTED_SOURCE},
                "config_file_sha256": "c" * 64, "server": {"container_id": "5" * 64,
                "image_id": "sha256:" + "4" * 64, "served_model": "q38", "base_weight_receipt_sha256": "6" * 64}}
    for name in ("a", "b"):
        attested["runs"][name]["run_attestation"] = {"schema": "forte.model-run-attestation.v1", "status": "verified",
                "model": "q38", "run": name, "identities_equal": True,
                "before": copy.deepcopy(identity), "after": copy.deepcopy(identity)}
    assert summarize([attested,attested],["a","b"])["status"] == "ready_for_release_checks"
    attested["runs"]["b"]["run_attestation"]["after"]["config_file_sha256"] = "d" * 64
    assert summarize([attested,attested],["a","b"])["status"] == "not_qualified"
    q30 = copy.deepcopy(base)
    q30_source = {name: "1" * 64 for name in ATTESTED_SOURCE}
    q30.update(schema=Q30_SCHEMA, config_file_sha256="c" * 64, desk_adapter_model="",
               source_sha256=q30_source, source_manifest_sha256=digest(q30_source),
               labels_file_sha256="d" * 64, harness_sha256="e" * 64,
               source_install_receipt_sha256="f" * 64, reference_profile_file_sha256=Q30_REFERENCE)
    q30_identity = {"source_sha256": q30_source, "source_manifest_sha256": digest(q30_source),
                   "config_file_sha256": "c" * 64, "labels_scorer_file_sha256": "d" * 64,
                   "harness_sha256": "e" * 64, "source_install_receipt_sha256": "f" * 64,
                   "reference_profile_file_sha256": Q30_REFERENCE,
                   "media_inventory_file_sha256": "7" * 64, "evaluation_wrapper_sha256": "8" * 64,
                   "evaluation_args": {"par": 4, "concurrency": 64}, "evaluation_image_id": "sha256:" + "9" * 64,
                   "desk_adapter_model": "", "adapter": None,
                   "evaluation_transport": {"mode": "model_container_network_namespace", "internal_network": True,
                   "default_ipv4_route": False, "http_ready": True, "aliases_ready": True, "container_id": "5" * 64},
                   "server": {"container_id": "5" * 64, "image_id": "sha256:" + "4" * 64, "request_model": "q30test",
                   "base_weight_receipt_sha256": "6" * 64, "base_weight_manifest_sha256": "7" * 64,
                   "repo": Q30_REPO, "revision": Q30_REVISION, "dtype": "bfloat16", "quantization": None,
                   "args_sha256": "8" * 64, "server_receipt_sha256": "9" * 64}}
    for name in ("a", "b"):
        q30["runs"][name]["model"].update(repo=Q30_REPO, served_model="q30test", observed_revision=Q30_REVISION,
                    pinned_base_weight_receipt_sha256="6" * 64, pinned_base_weight_manifest_sha256="7" * 64, desk_adapter_model="")
        q30["runs"][name]["run_attestation"] = {"schema": "forte.q30-run-attestation.v1", "run": name,
                "model": "q30test", "status": "verified", "identities_equal": True, "historical_binding_valid": True,
                "before_after_exactly_equal": True, "source_matches_collection": True, "collected_profile_file_sha256_matches": True,
                "before": copy.deepcopy(q30_identity), "after": copy.deepcopy(q30_identity)}
    assert summarize([q30,q30],["a","b"])["status"] == "ready_for_release_checks"
    q30["runs"]["b"]["run_attestation"]["historical_binding_valid"] = False
    assert summarize([q30,q30],["a","b"])["status"] == "not_qualified"
    print(json.dumps({"schema": "forte.qualification-self-test.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "checks": 14}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", metavar=("COLLECTED_JSON", "RUN"))
    parser.add_argument("--stage-result", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.run or len(args.run) != 2 or not all(RUN.fullmatch(item[1]) for item in args.run):
        raise ValueError("exactly_two_named_collected_runs_required")
    paths = [Path(item[0]) for item in args.run]
    if args.out and args.out.resolve() in {path.resolve() for path in paths + ([args.stage_result] if args.stage_result else [])}:
        raise ValueError("output_overlaps_input")
    documents = [strict_json(path) if path.is_file() else None for path in paths]
    stage = strict_json(args.stage_result) if args.stage_result and args.stage_result.is_file() else None
    output = summarize(documents, [item[1] for item in args.run], stage)
    output["utility_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output["collection_file_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None for path in paths]
    if args.stage_result and args.stage_result.is_file():
        output["stage_file_sha256"] = hashlib.sha256(args.stage_result.read_bytes()).hexdigest()
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"schema": "forte.model-qualification-summary.v1",
                          "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                          "status": "summary_failed", "qualification_claim": False,
                          "error_type": type(error).__name__}))
        raise SystemExit(2)
