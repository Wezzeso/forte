"""Content sensitivity, safe output and explicit unknown provenance."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forte.config import WorkerSettings
from forte.decision.structured import DecisionConfig
from forte.model.adapter import LocalModelConfig
import forte.reproducibility as reproducibility


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@pytest.fixture
def context(tmp_path, monkeypatch):
    secret = "private-sentinel-never-export"
    files = {}
    for label in ("model", "rules", "taxonomy"):
        path = tmp_path / f"{secret}-{label}.json"
        path.write_text(json.dumps({"synthetic": label}), encoding="utf-8")
        files[label] = path
    code = tmp_path / "synthetic-code.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(reproducibility, "_installed_code_files", lambda: ([("forte/stub.py", code)], []))
    settings = WorkerSettings(
        database_url=f"postgresql://{secret}@private/db", data_dir=tmp_path / secret,
        model_base_url=f"http://{secret}.invalid/v1", model_name=secret,
        model_api_key=secret, worker_id=secret, heartbeat_file=tmp_path / secret,
        rules_path=files["rules"], taxonomy_path=files["taxonomy"], model_manifest_path=files["model"],
    )
    provider = LocalModelConfig(media_root=tmp_path / secret, api_key=secret)
    catalog = SimpleNamespace(payload={"synthetic_rules": ["source-a"], "private_metadata": secret})
    return settings, provider, catalog, DecisionConfig(), files, code, secret


def build(context, **changes):
    settings, provider, catalog, decision, *_ = context
    return reproducibility.build_reproducibility_bundle(
        changes.get("settings", settings), changes.get("catalog", catalog),
        changes.get("decision", decision), changes.get("provider", provider))


def test_same_bound_inputs_produce_same_bundle_without_paths_or_secrets(context):
    first, second = build(context), build(context)
    assert first == second
    assert first["bindings_complete"] and first["missing_bindings"] == []
    encoded = json.dumps(first)
    assert context[-1] not in encoded
    assert str(context[4]["model"].parent) not in encoded
    assert "postgresql://" not in encoded and "http://" not in encoded
    assert first["provider"]["generation"] == {"temperature": 0.0, "seed": 20261017, "max_tokens": 4096}
    assert not first["claim_limits"]["repeatability_measured"]


def test_bundle_digest_covers_entire_returned_snapshot(context):
    result = build(context)
    checksum = result.pop("bundle_sha256")
    assert checksum == canonical_sha(result)


def test_loaded_catalog_is_hashed_instead_of_rereading_catalog_file(context):
    first = build(context)
    context[4]["rules"].write_text('{"different_disk_catalog":true}', encoding="utf-8")
    disk_changed = build(context)
    assert first["loaded_catalog_sha256"] == disk_changed["loaded_catalog_sha256"]
    assert first["config_files"]["rules_file"] != disk_changed["config_files"]["rules_file"]
    context[2].payload["synthetic_rules"] = ["source-b"]
    loaded_changed = build(context)
    assert disk_changed["loaded_catalog_sha256"] != loaded_changed["loaded_catalog_sha256"]
    assert disk_changed["bundle_sha256"] != loaded_changed["bundle_sha256"]


def test_manifest_and_installed_source_content_changes_invalidate_bundle(context):
    first = build(context)
    context[4]["model"].write_text('{"synthetic":"changed-model"}', encoding="utf-8")
    model_changed = build(context)
    assert first["config_files"]["model_manifest"] != model_changed["config_files"]["model_manifest"]
    context[5].write_text("VALUE = 2\n", encoding="utf-8")
    code_changed = build(context)
    assert model_changed["code"]["source_sha256"] != code_changed["code"]["source_sha256"]
    assert model_changed["bundle_sha256"] != code_changed["bundle_sha256"]


def test_sampling_generation_and_threshold_changes_are_bound_without_threshold_names(context):
    first = build(context)
    assert build(context, settings=replace(context[0], sample_fps=3.0))["bundle_sha256"] != first["bundle_sha256"]
    assert build(context, provider=replace(context[1], seed=42))["bundle_sha256"] != first["bundle_sha256"]
    decision = vars(SimpleNamespace(**{field: getattr(context[3], field) for field in
                                      ("config_version",) + reproducibility._DECISION_FIELDS}))
    decision["thresholds"] = {context[-1]: {"minimum_score": .8}}
    threshold_a = build(context, decision=decision)
    decision["thresholds"][context[-1]]["minimum_score"] = .9
    threshold_b = build(context, decision=decision)
    assert threshold_a["bundle_sha256"] != threshold_b["bundle_sha256"]
    assert context[-1] not in json.dumps(threshold_b)


@pytest.mark.parametrize("provider", [None, SimpleNamespace(seed=1, max_tokens=9, api_key="private")])
def test_injected_or_missing_provider_never_inherits_production_generation(context, provider):
    result = build(context, provider=provider)
    assert result["provider"] == {"kind": "unknown_or_injected", "generation": None, "limits": None}
    assert not result["bindings_complete"]
    assert "provider.actual_configuration_unknown" in result["missing_bindings"]


def test_unavailable_or_nonfinite_bindings_are_explicit_without_exception_text(context):
    context[4]["model"].unlink()
    context[2].payload = {"synthetic": float("nan")}
    result = build(context)
    assert result["loaded_catalog_sha256"] is None
    assert not result["bindings_complete"]
    assert "catalog.loaded_payload_unavailable" in result["missing_bindings"]
    assert "config.model_manifest_unavailable" in result["missing_bindings"]
    assert context[-1] not in json.dumps(result, allow_nan=False)


def test_missing_code_is_not_reported_as_a_complete_source_binding(context, monkeypatch):
    monkeypatch.setattr(reproducibility, "_installed_code_files", lambda: ([], ["code.package_source_unavailable"]))
    result = build(context)
    assert result["code"]["source_sha256"] is None
    assert result["code"]["source_file_count"] == 0
    assert not result["bindings_complete"]
    assert "code.source_set_empty" in result["missing_bindings"]


def test_generic_config_binding_does_not_read_media_or_workbook_paths(context, monkeypatch):
    media = context[4]["model"].with_suffix(".mov")
    media.write_bytes(b"synthetic media must never be opened by this helper")
    original = reproducibility._stable_file_sha256
    read_paths = []

    def track(path):
        read_paths.append(Path(path))
        return original(path)

    monkeypatch.setattr(reproducibility, "_stable_file_sha256", track)
    result = build(context, settings=replace(context[0], model_manifest_path=media))
    assert media not in read_paths
    assert "config.model_manifest_unavailable" in result["missing_bindings"]


def test_circular_catalog_payload_becomes_static_missing_binding(context):
    context[2].payload["cycle"] = context[2].payload
    result = build(context)
    assert result["loaded_catalog_sha256"] is None
    assert "catalog.loaded_payload_unavailable" in result["missing_bindings"]
    assert context[-1] not in json.dumps(result)


def test_actual_source_inventory_is_confined_and_covers_explicit_modules(monkeypatch):
    monkeypatch.undo()
    files, problems = reproducibility._installed_code_files()
    assert not problems
    names = {name for name, _ in files}
    assert "forte/reproducibility.py" in names
    assert "rules/catalog.py" in names and "serving/scheduler.py" in names
    assert all(path.suffix == ".py" and not path.is_symlink() for _, path in files)
    assert not any("/private/" in name or "/labels/" in name for name in names)
