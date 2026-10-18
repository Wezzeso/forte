"""Frozen public source bytes and generated family; no private population/model."""
from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from deploy.qwen38_bind_runtime import BindError, bind_quality_family, canonical, sha256
from evaluation.qwen38_quality_equivalence import common_paths
from qa.integration.qwen38_timeout_lineage_test import ORIGINAL, ORIGINAL_SHA, revised_pipeline
from serving.qwen38_source_lineage import verify_source_lineage

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def family_binding(tmp_path):
    assert sha256(ORIGINAL) == ORIGINAL_SHA
    with tarfile.open(ORIGINAL) as bundle:
        original = {member.name: bundle.extractfile(member).read() for member in bundle}
    files = dict(original)
    files["src/forte/pipeline.py"] = revised_pipeline().encode()
    for name in ("deploy/qwen38_bind_runtime.py", "serving/qwen38_identity.py",
        "serving/qwen38_adapter_contract.py", "serving/qwen38_source_lineage.py",
        "evaluation/qwen38_quality_equivalence.py"):
        files[name] = (ROOT / name).read_bytes()
    code = tmp_path / "generated-installed-source"
    runtime = tmp_path / "generated-reviewed-serving.tar"
    with tarfile.open(runtime, "w") as bundle:
        for name, raw in sorted(files.items()):
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            bundle.addfile(member, BytesIO(raw))
    original_inventory = {name: hashlib.sha256(raw).hexdigest() for name,raw in original.items()}
    protected = common_paths(original_inventory)
    assert len(protected) == 88
    config = {"limits": {"max_images":16, "max_context_tokens":32768, "max_pixels_per_image":262144},
        "generation": {"max_new_tokens":4096, "seed":20261017, "do_sample":False, "temperature":0, "num_beams":1},
        "source_archive": {"sha256":sha256(runtime)}}
    recipe = {"worker_settings": {"window_seconds":8, "stride_seconds":4, "sample_fps":2,
        "max_frames_per_window":16, "staged_screening":False, "max_concurrent_clips":1,
        "max_concurrent_jobs":1, "max_inflight_per_job":1, "max_refinement_passes":8,
        "job_timeout_seconds":285}, "context_tokens":32768, "generation":deepcopy(config["generation"]),
        "rule_selection": {"enabled_rules":4, "optional_expanded_rules_selected":False, "visible_fact_axes":19},
        "prompt_implementation_pins": {name:original_inventory[name] for name in sorted(protected)},
        "rules_sha256":original_inventory["rules/catalog.json"],
        "taxonomy_sha256":original_inventory["annotations/taxonomy.v1.json"],
        "source_archive_sha256":ORIGINAL_SHA}
    family = tmp_path / "generated-family.json"
    family.write_text(json.dumps({"schema_version":"forte.qwen38-quality-family.v1", "recipe":recipe,
                                 "recipe_sha256":canonical(recipe)}))
    lineage = verify_source_lineage(runtime_archive=runtime, runtime_sha256=sha256(runtime),
        training_archive=ORIGINAL, training_sha256=ORIGINAL_SHA, recorded_training_sha256=ORIGINAL_SHA)
    return family, config, code, lineage


def test_genuine_public_source_family_binds_with_eighty_seven_exact_and_one_audited_pin(family_binding):
    family, config, code, lineage = family_binding
    before = sha256(family)
    result = bind_quality_family(family, before, config, code, lineage_proof=lineage)
    assert result["family_source_archive_sha256"] == ORIGINAL_SHA
    assert result["same_family_prompt_and_pipeline_bytes_verified"] is False
    assert result["family_prompt_pipeline_verified_with_audited_timeout"] is True
    assert result["prompt_implementation_pin_count"] == 88
    assert result["prompt_implementation_byte_identical_count"] == 87
    assert result["audited_prompt_implementation_delta_count"] == 1
    assert result["audited_prompt_implementation_deltas"][0]["proof"] == lineage["candidate_http_timeout_delta"]
    assert sha256(family) == before and result["family_rewritten"] is False


@pytest.mark.parametrize("mutation", ["missing", "wrong_old", "wrong_new", "wrong_runtime", "unverified_lineage",
    "unverified_delta", "other_ast_changed", "other_bytes_changed", "baseline_changed", "wrong_keyword", "wrong_value"])
def test_family_timeout_exception_requires_the_verified_matching_lineage_proof(family_binding, mutation):
    family, config, code, lineage = family_binding
    proof = deepcopy(lineage)
    delta = proof["candidate_http_timeout_delta"]
    if mutation == "missing": proof = None
    elif mutation == "wrong_old": delta["old_source_sha256"] = "0" * 64
    elif mutation == "wrong_new": delta["new_source_sha256"] = "0" * 64
    elif mutation == "wrong_runtime": proof["runtime_source_archive_sha256"] = "0" * 64
    elif mutation == "unverified_lineage": proof["verified"] = False
    elif mutation == "unverified_delta": delta["verified"] = False
    elif mutation == "other_ast_changed": delta["other_ast_identical"] = False
    elif mutation == "other_bytes_changed": delta["other_source_bytes_identical"] = False
    elif mutation == "baseline_changed": delta["baseline_default_unchanged"] = False
    elif mutation == "wrong_keyword": delta["added_keyword"] = "max_tokens"
    else: delta["added_value"] = "240"
    with pytest.raises(BindError):
        bind_quality_family(family, sha256(family), config, code, lineage_proof=proof)


@pytest.mark.parametrize("name", ["src/forte/decision/structured.py", "rules/catalog.json", "annotations/taxonomy.v1.json"])
def test_reviewed_timeout_cannot_cover_another_actual_prompt_rule_or_taxonomy_change(family_binding, name):
    family, config, code, lineage = family_binding
    path = code / name
    path.write_bytes(path.read_bytes() + b"\nUNREVIEWED\n")
    with pytest.raises(BindError):
        bind_quality_family(family, sha256(family), config, code, lineage_proof=lineage)
