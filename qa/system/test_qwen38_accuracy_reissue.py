"""Generated control artifacts only; no real proof generation or inference."""
from copy import deepcopy
import json

import pytest

from benchmarks.performance import qwen38_accuracy_diagnostic as diagnostic
from benchmarks.performance import qwen38_accuracy_reissue as reissue
from qa.system.test_qwen38_accuracy_diagnostic import inputs


def proof_fixture(tmp_path, monkeypatch):
    family, actual = inputs()
    protocol = diagnostic.freeze_protocol(family, actual)
    old = diagnostic.bind_plan(protocol, actual)
    sources = []
    for index, row in enumerate(old["source_attempts"]):
        source = {**row, "status": "failed" if index == 4 else "not_attempted", "model_posts": 0,
            "source_pipeline_started": False, "public_contract_validated": False, "server_receipt_verified": False,
            "release_state": "released" if index == 4 else "not_released"}
        if index == 4:
            source.update(failure_stage="pre_execution_validation", static_failure_code=reissue.FAILURE_CODE, error_type="ValueError")
        sources.append(source)
    slots = [{**row, "status": "not_attempted"} for row in old["attempts"]]
    paths = {}
    def artifact(name, document):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(document), encoding="utf-8")
        paths[name] = path
        return diagnostic.file_digest(path)
    plan_sha = artifact("prior_plan", old)
    source_sha = artifact("prior_source_terminals", {"records": sources})
    slot_sha = artifact("prior_model_slot_terminals", {"records": slots})
    summary = {"schema_version": "forte.qwen38-accuracy-control-failure-reconciliation.v1",
        "status": "terminal_control_failure_before_pipeline", "plan_raw_sha256": plan_sha, "actual_model_posts": 0,
        "source_pipeline_started": False, "source_failed_control": 1, "source_not_released": 5,
        "source_attempts_required": 6, "static_failure_code": reissue.FAILURE_CODE, "model_slots": 24,
        "model_slot_terminal_counts": {"not_attempted": 24}, "unused_slots_refunded": False,
        "past_stage_witness_inferred": False, "existing_claims_or_results_modified": False,
        "source_terminals_raw_sha256": source_sha, "model_slot_terminals_raw_sha256": slot_sha,
        "reservation_raw_sha256": "1"*64, "release_raw_sha256": "2"*64}
    summary_sha = artifact("prior_reconciliation", summary)
    settled = {"schema_version": "forte.qwen38-accuracy-pilot-control-settlement.v1",
        "status": "spent_control_failed_before_pipeline", "attempt_plan_sha256": plan_sha, "reserved_attempts": 24,
        "actual_model_posts": 0, "actual_video_attempts": 0, "source_pipeline_started": False, "all_slots_consumed": True,
        "exposure_refunded": False, "retry_authorized": False, "new_attempt_authority": False,
        "source_failed_control": 1, "source_not_released": 5, "source_population_denominator": 6,
        "static_failure_code": reissue.FAILURE_CODE, "outcomes": slots,
        "actual_helper_terminal_state": {"Running": False, "Status": "exited", "ExitCode": 1, "OOMKilled": False},
        "reconciliation": {"complete": True, "reserved_attempts": 24, "recorded_attempts": 24, "unresolved_attempts": 0,
            "status_counts": {"not_attempted": 24}, "exposure_reservations_refunded": False, "retry_authorized": False},
        "attempt_reservation_sha256": "1"*64, "root_release_sha256": "2"*64,
        "host_original_metadata_sha256": {"old-helper.summary.json.original": summary_sha,
            "old-helper.source-terminals.json.original": source_sha, "old-helper.model-slot-terminals.json.original": slot_sha}}
    settled_sha = artifact("prior_settlement", settled)
    artifact("root_authorization", {"schema_version": "forte.qwen38-accuracy-control-reissue-authorization.v1",
        "released_by_root": True, "operational_control_attempt": 2, "prior_plan_sha256": plan_sha,
        "prior_reconciliation_sha256": summary_sha, "prior_settlement_sha256": settled_sha,
        "maximum_new_model_requests": 24, "maximum_execution_seconds": 960,
        "maximum_operational_reissues": 1, "actual_source_pipeline_attempts_authorized": 1})
    monkeypatch.setattr(reissue.sys, "version_info", (3, 12, 0))
    proof = reissue.prepare_control_reissue(**paths)
    return protocol, actual, old, proof


def rehash(proof, name):
    # Simulate internally consistent but forbidden claimed metadata. Raw pins in
    # the original authorization must still match; flags cannot authorize retry.
    item = proof["artifacts"][name]
    item["semantic_sha256"] = reissue._digest(item["document"])


def test_one_corrected_control_uses_new_ids_without_changing_protocol_or_budget(tmp_path, monkeypatch):
    protocol, actual, old, proof = proof_fixture(tmp_path, monkeypatch)
    before = diagnostic.digest(old)
    new = diagnostic.bind_plan(protocol, actual, control_reissue=proof)
    assert diagnostic.digest(old) == before
    assert len(new["attempts"]) == 24 and len(new["source_attempts"]) == 6
    assert new["maximum_execution_seconds"] == 960 and new["recipe_sha256"] == old["recipe_sha256"]
    assert new["diagnostic_protocol_sha256"] == old["diagnostic_protocol_sha256"]
    assert new["historical_control_slots_consumed"] == 24
    assert new["maximum_reserved_model_slots_including_failed_control_per_arm"] == 232
    assert new["maximum_actual_model_posts_per_arm"] == 208
    assert not ({row["attempt_id"] for row in new["source_attempts"]} & {row["attempt_id"] for row in old["source_attempts"]})
    release = {"operational_control_reissue_authorized_by_root": True, "operational_control_attempt": 2,
        "prior_settlement_sha256": proof["artifacts"]["prior_settlement"]["raw_sha256"],
        "control_reissue_sha256": diagnostic.digest(proof)}
    assert reissue.validate_reissue_release(new, release) is True
    with pytest.raises(ValueError, match="fresh_exact_root_release"):
        reissue.validate_reissue_release(new, {})


@pytest.mark.parametrize("artifact,field,value", [
    ("prior_reconciliation", "actual_model_posts", 1),
    ("prior_reconciliation", "source_pipeline_started", True),
    ("prior_reconciliation", "past_stage_witness_inferred", True),
    ("prior_settlement", "actual_video_attempts", 1),
    ("prior_settlement", "all_slots_consumed", False),
    ("prior_settlement", "exposure_refunded", True),
    ("root_authorization", "released_by_root", False),
    ("root_authorization", "maximum_operational_reissues", 2),
    ("root_authorization", "maximum_new_model_requests", 25),
    ("prior_reconciliation", "actual_model_posts", False),
    ("prior_settlement", "actual_video_attempts", False),
    ("root_authorization", "actual_source_pipeline_attempts_authorized", True),
])
def test_any_prior_execution_refund_missing_root_or_wider_budget_blocks_reissue(tmp_path, monkeypatch, artifact, field, value):
    _, _, _, proof = proof_fixture(tmp_path, monkeypatch)
    proof["artifacts"][artifact]["document"][field] = value
    rehash(proof, artifact)
    with pytest.raises(ValueError):
        reissue.validate_control_reissue(proof)


@pytest.mark.parametrize("field,value", [("source_pipeline_started", True), ("model_posts", 1),
    ("model_posts", False), ("job_id", "actual-job"), ("failure_stage", "inference"), ("status", "done")])
def test_prior_source_execution_or_success_never_becomes_control_reissue(tmp_path, monkeypatch, field, value):
    _, _, _, proof = proof_fixture(tmp_path, monkeypatch)
    proof["artifacts"]["prior_source_terminals"]["document"]["records"][4][field] = value
    rehash(proof, "prior_source_terminals")
    with pytest.raises(ValueError):
        reissue.validate_control_reissue(proof)


def test_no_missing_settlement_third_attempt_runtime_switch_or_local_ast_generation(tmp_path, monkeypatch):
    protocol, actual, old, proof = proof_fixture(tmp_path, monkeypatch)
    changed = deepcopy(proof); changed["artifacts"].pop("prior_settlement")
    with pytest.raises(ValueError):
        reissue.validate_control_reissue(changed)
    new = diagnostic.bind_plan(protocol, actual, control_reissue=proof)
    new["operational_control_attempt"] = 3
    with pytest.raises(ValueError, match="only_control2"):
        diagnostic.validate_diagnostic_plan(new)
    current = deepcopy(actual); current["provider"]["launch_config"] = "/generated/different.json"
    with pytest.raises(ValueError, match="actual_runtime_or_protocol"):
        diagnostic.bind_plan(protocol, current, control_reissue=proof)
    monkeypatch.setattr(diagnostic.sys, "version_info", (3, 14, 0))
    with pytest.raises(ValueError, match="actual_cpu_python312"):
        diagnostic.bind_plan(protocol, actual, control_reissue=proof)
