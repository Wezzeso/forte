"""The frozen quality-family binder accepts only the proven timeout delta."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from deploy.qwen38_bind_runtime import BindError, bind_quality_family, canonical, sha256
from serving.qwen38_source_lineage import verify_source_lineage
from serving.tests.qwen38_adapter_fixtures import archive_fixture


@pytest.fixture
def family_timeout_fixture(tmp_path):
    current = Path("src/forte/pipeline.py").read_bytes()
    added = next(line for line in current.splitlines(keepends=True)
        if line.strip() == b'request_timeout_sec=config["limits"]["request_timeout_seconds"],')
    old = current.replace(added, b"")
    original, runtime = tmp_path/"original.tar", tmp_path/"runtime.tar"
    structured = b"# generated unchanged prompt implementation\n"
    archive_fixture(original, {"src/forte/pipeline.py": old, "src/forte/decision/structured.py": structured})
    archive_fixture(runtime, {"src/forte/pipeline.py": current, "src/forte/decision/structured.py": structured})
    lineage = verify_source_lineage(training_archive=original, training_sha256=sha256(original),
        recorded_training_sha256=sha256(original), runtime_archive=runtime, runtime_sha256=sha256(runtime))
    code = tmp_path/"code"
    for name, raw in {"src/forte/pipeline.py": current, "src/forte/decision/structured.py": structured,
        "rules/catalog.json": b"generated rule bytes", "annotations/taxonomy.v1.json": b"generated taxonomy bytes"}.items():
        target = code/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    config = {"limits": {"max_images": 16, "max_context_tokens": 32768, "max_pixels_per_image": 262144},
        "generation": {"max_new_tokens": 4096, "seed": 20261017, "do_sample": False, "temperature": 0, "num_beams": 1},
        "source_archive": {"sha256": sha256(runtime)}}
    recipe = {"worker_settings": {"window_seconds": 8, "stride_seconds": 4, "sample_fps": 2,
        "max_frames_per_window": 16, "staged_screening": False, "max_concurrent_clips": 1,
        "max_concurrent_jobs": 1, "max_inflight_per_job": 1, "max_refinement_passes": 8, "job_timeout_seconds": 285},
        "context_tokens": 32768, "generation": config["generation"],
        "rule_selection": {"enabled_rules": 4, "optional_expanded_rules_selected": False, "visible_fact_axes": 19},
        "prompt_implementation_pins": {"src/forte/pipeline.py": hashlib.sha256(old).hexdigest(),
            "src/forte/decision/structured.py": hashlib.sha256(structured).hexdigest()},
        "rules_sha256": sha256(code/"rules/catalog.json"),
        "taxonomy_sha256": sha256(code/"annotations/taxonomy.v1.json"), "source_archive_sha256": sha256(original)}
    family = tmp_path/"family.json"
    family.write_text(json.dumps({"schema_version": "forte.qwen38-quality-family.v1", "recipe": recipe,
        "recipe_sha256": canonical(recipe)}), encoding="utf-8")
    return family, config, code, lineage


def test_frozen_family_discloses_exact_pins_and_audited_timeout_separately(family_timeout_fixture):
    path, config, code, lineage = family_timeout_fixture
    proof = bind_quality_family(path, sha256(path), config, code, lineage_proof=lineage)
    assert proof["same_family_prompt_and_pipeline_bytes_verified"] is False
    assert proof["family_prompt_pipeline_verified_with_audited_timeout"] is True
    assert proof["prompt_implementation_pin_count"] == 2
    assert proof["prompt_implementation_byte_identical_count"] == 1
    assert proof["audited_prompt_implementation_delta_count"] == 1
    assert proof["audited_prompt_implementation_deltas"] == [{"path": "src/forte/pipeline.py",
        "proof": lineage["candidate_http_timeout_delta"]}]


@pytest.mark.parametrize("change", ["missing", "old_hash", "new_hash", "runtime_hash", "not_verified",
    "wrong_keyword", "wrong_value", "other_prompt_change", "self_reported_config"])
def test_family_exception_requires_independent_lineage_and_exact_old_new_bytes(family_timeout_fixture, change):
    path, config, code, original_lineage = family_timeout_fixture
    lineage = deepcopy(original_lineage)
    if change == "missing": lineage = None
    elif change == "old_hash": lineage["candidate_http_timeout_delta"]["old_source_sha256"] = "0"*64
    elif change == "new_hash": lineage["candidate_http_timeout_delta"]["new_source_sha256"] = "0"*64
    elif change == "runtime_hash": lineage["runtime_source_archive_sha256"] = "0"*64
    elif change == "not_verified": lineage["verified"] = False
    elif change == "wrong_keyword": lineage["candidate_http_timeout_delta"]["added_keyword"] = "max_tokens"
    elif change == "wrong_value": lineage["candidate_http_timeout_delta"]["added_value"] = "240"
    elif change == "other_prompt_change": (code/"src/forte/decision/structured.py").write_bytes(b"unreviewed")
    else:
        config["training_source_lineage"] = original_lineage
        lineage = None
    with pytest.raises(BindError, match="prompt_bytes_changed"):
        bind_quality_family(path, sha256(path), config, code, lineage_proof=lineage)
