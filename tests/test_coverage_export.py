"""Synthetic immutable teacher export plumbing; no private media or inference."""
from pathlib import Path
import json

import pytest

from training.coverage_export import export_coverage
from training.guards import TrainingGateError, file_sha256, canonical_sha256
from tests.test_coverage_expansion import plan_fixture


def export_fixture(tmp_path, monkeypatch):
    result = plan_fixture()
    plan = result["teacher_plan"]
    identity = {"model_id": "synthetic/private", "model_revision": "a" * 40,
        "processor_revision": "a" * 40, "model_manifest_sha256": "b" * 64}
    from tests.test_coverage_expansion import fixture
    sources, split, *_ = fixture()
    paths = {}
    for key, data in (("teacher_plan", plan), ("split", split), ("machine_schema", {"type": "object"})):
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[key] = path
    plan["split_file_sha256"] = file_sha256(paths["split"])
    plan["plan_sha256"] = canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"})
    paths["teacher_plan"].write_text(json.dumps(plan), encoding="utf-8")
    source_path = tmp_path / "source_manifest.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
    paths["source_manifest"] = source_path
    ledger = {"attempts": []}
    raw_root = tmp_path / "raw"
    for record in plan["records"]:
        content = record["content_sha256"]
        windows = []
        for index in record["priority_second_indices"]:
            fid = "frame-" + str(index)
            ledger["attempts"].append({"content_sha256": content, "second_index": index,
                "frames": [{"frame_id": fid}], "anchor_frame_id": fid, "anchor_pts_sec": float(index)})
            windows.append({"second_index": index, "frame_ids": [fid], "anchor_frame_ids": [fid],
                            "anchor_pts_sec": [float(index)]})
        audit = {"schema_version": "forte.local-proposal.v1", "asset": {"content_hash": content},
            "phase": "partial", "annotation_origin": "local_model", "human_reviewed": False,
            "organizer_labels_used_for_facts": False, "provenance": {**identity, "plan_sha256": plan["plan_sha256"]},
            "coverage": {"attempted_model_requests": 2}, "window_decisions": windows}
        directory = raw_root / "train" / content
        directory.mkdir(parents=True)
        (directory / "teacher.private.json").write_text(json.dumps(audit), encoding="utf-8")
    ledger_path = tmp_path / "native_ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    paths["native_ledger"] = ledger_path
    # The existing normalizer's pixel/hash checks have independent tests; this
    # test verifies immutable source joins and exact 34-window export plumbing.
    def fake_normalizer(record, workspace):
        proposal = record["local_model_proposals"][0]
        if proposal.get("synthetic_no_summary"):
            return None
        windows = []
        for original in proposal["window_decisions"]:
            provenance = {**proposal["provenance"], "actual_snapshot_verified": True, "prompt_sha256": "e" * 64,
                "serving_attestation": {**identity, "model_files_verified": True}}
            target = {"frame_id": original["frame_ids"][0], "fact_id": "CASH_VISIBLE", "state": "present",
                "supporting_frame_ids": original["frame_ids"], "supervision_mask": True, "provenance": provenance}
            windows.append({**original, "window_id": str(original["second_index"]), "model_output_complete": True,
                "provenance": provenance, "frame_fact_targets": [target],
                "support_reviews": [{**target, "teacher_state": "present", "supports_state": True}]})
        return {"content_sha256": record["content_hash"], "source_pts_identity_checked": True,
            "windows": windows, "human_reviewed": False,
            "split": "train", "provenance": proposal["provenance"]}
    monkeypatch.setattr("annotations.machine.summarize_record", fake_normalizer)
    config = {key: str(path) for key, path in paths.items()}
    config.update({"expected_" + key + "_sha256": file_sha256(path) for key, path in paths.items()})
    config.update(raw_teacher_root=str(raw_root), expected_teacher=identity,
                  output=str(tmp_path / "new-only.jsonl"), proof_output=str(tmp_path / "proof.json"))
    return config


def test_seventeen_terminal_audits_export_without_modifying_original_bytes(tmp_path, monkeypatch):
    config = export_fixture(tmp_path, monkeypatch)
    audits = list(Path(config["raw_teacher_root"]).glob("train/*/teacher.private.json"))
    before = {str(path): file_sha256(path) for path in audits}
    proof = export_coverage(config)
    assert proof["records"] == 17 and proof["windows"] == proof["attempted_teacher_calls"] == 34
    assert proof["raw_audits_modified"] is False
    assert before == {str(path): file_sha256(path) for path in audits}
    assert len(Path(config["output"]).read_text().splitlines()) == 17


@pytest.mark.parametrize("normalizer_none", [False, True])
def test_fully_invalid_source_is_counted_and_valid_sources_are_exported(tmp_path, monkeypatch, normalizer_none):
    config = export_fixture(tmp_path, monkeypatch)
    path = next(Path(config["raw_teacher_root"]).glob("train/*/teacher.private.json"))
    row = json.loads(path.read_text())
    row["window_decisions"] = []
    row["invalid_teacher_windows"] = [{"state": "invalid_teacher_output"}] * 2
    row["synthetic_no_summary"] = normalizer_none
    path.write_text(json.dumps(row), encoding="utf-8")
    before = file_sha256(path)
    result = export_coverage(config)
    assert result["records"] == 16 and result["windows"] == 32
    assert result["planned_sources"] == result["attempted_source_receipts"] == 17
    assert result["attempted_teacher_calls"] == 34 and result["invalid_or_incomplete_call_count"] == 2
    assert result["excluded_source_count"] == 1
    proof = json.loads(Path(config["proof_output"]).read_text())
    assert len(proof["raw_audit_receipts"]) == 17 and len(proof["excluded_sources"]) == 1
    assert file_sha256(path) == before
    assert row["asset"]["content_hash"] not in Path(config["output"]).read_text()


def test_all_invalid_fails_with_complete_failure_proof_and_no_empty_training_export(tmp_path, monkeypatch):
    config = export_fixture(tmp_path, monkeypatch)
    for path in Path(config["raw_teacher_root"]).glob("train/*/teacher.private.json"):
        row = json.loads(path.read_text())
        row["window_decisions"] = []
        path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(TrainingGateError, match="no_eligible_targets"):
        export_coverage(config)
    assert not Path(config["output"]).exists()
    proof = json.loads(Path(config["proof_output"]).read_text())
    assert proof["status"] == "no_eligible_targets"
    assert proof["records"] == proof["windows"] == 0
    assert proof["attempted_source_receipts"] == proof["excluded_source_count"] == 17
    assert proof["attempted_teacher_calls"] == proof["invalid_or_incomplete_call_count"] == 34


@pytest.mark.parametrize("mutation", ["foreign_content", "foreign_window", "teacher", "call_count", "missing_audit"])
def test_foreign_or_unbounded_exports_fail_without_outputs(tmp_path, monkeypatch, mutation):
    config = export_fixture(tmp_path, monkeypatch)
    path = next(Path(config["raw_teacher_root"]).glob("train/*/teacher.private.json"))
    row = json.loads(path.read_text())
    if mutation == "foreign_content":
        row["asset"]["content_hash"] = "c" * 64
    elif mutation == "foreign_window":
        row["window_decisions"][0]["second_index"] = -1
    elif mutation == "teacher":
        row["provenance"]["model_revision"] = "c" * 40
    elif mutation == "call_count":
        row["coverage"]["attempted_model_requests"] = 3
    else:
        path.unlink()
    if mutation != "missing_audit":
        path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(TrainingGateError):
        export_coverage(config)
    assert not Path(config["output"]).exists()
