"""Independent generated CPU proof of the sole candidate-timeout AST delta."""
from io import BytesIO
from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from forte.pipeline import _provider_configuration
from serving.qwen38_identity import CandidateError, file_sha256
from serving.qwen38_source_lineage import SourceLineageError, verify_source_lineage
from test_candidate_pipeline_config import candidate
from test_qwen38_quality_family import fixtures as quality_family_fixture

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "tmp/qwen38-ml-support-grammar-final-20261017T151420Z.tar.gz"
ORIGINAL_SHA = "91263a59cb2e73af48b46166b427d507ce7c3fa05106a46402acec950c20048d"
PIPELINE = "src/forte/pipeline.py"


def original_pipeline():
    assert file_sha256(ORIGINAL) == ORIGINAL_SHA
    with tarfile.open(ORIGINAL) as bundle:
        return bundle.extractfile(PIPELINE).read().decode()


def revised_pipeline():
    original = original_pipeline()
    newline = "\r\n" if "\r\n" in original else "\n"
    needle = 'values.update(expected_revision=identity["model_revision"],' + newline
    assert original.count(needle) == 1
    return original.replace(needle, needle + '                      request_timeout_sec=config["limits"]["request_timeout_seconds"],' + newline)


def archive(path, pipeline, *, overrides=None, remove=()):
    files = {"training/architectures.py": b"original architecture",
        "training/data.py": b"original masks", "rules/catalog.json": b"original rules",
        "annotations/taxonomy.v1.json": b"original taxonomy", PIPELINE: pipeline.encode(),
        "serving/qwen38_grammar.py": b"original grammar",
        "evaluation/qwen38_pipeline.py": b"original evaluation"}
    files.update(overrides or {})
    for name in remove:
        files.pop(name)
    with tarfile.open(path, "w") as bundle:
        for name, raw in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            bundle.addfile(item, BytesIO(raw))


def verify_pair(tmp_path, changed, *, overrides=None, remove=()):
    old, new = tmp_path / "training.tar", tmp_path / "serving.tar"
    archive(old, original_pipeline())
    archive(new, changed, overrides=overrides, remove=remove)
    return verify_source_lineage(training_archive=old, training_sha256=file_sha256(old),
        recorded_training_sha256=file_sha256(old), runtime_archive=new, runtime_sha256=file_sha256(new))


def test_exact_candidate_timeout_addition_is_accepted_without_relabeling_training(tmp_path):
    result = verify_pair(tmp_path, revised_pipeline())
    assert result["verified"] is True and result["same_archive"] is False
    assert PIPELINE in result["audited_runtime_deltas"]
    assert result["run_bytes_changed"] is False and result["adapter_bytes_changed"] is False
    assert result["protected_files_verified"] == 5
    assert result["protected_files_byte_identical"] == 4
    assert result["protected_files_with_audited_timeout_delta"] == 1
    assert result["candidate_http_timeout_delta"]["added_keyword"] == "request_timeout_sec"
    assert result["candidate_http_timeout_delta"]["other_ast_identical"] is True
    assert result["candidate_http_timeout_delta"]["other_source_bytes_identical"] is True


@pytest.mark.parametrize("mutation", ["constant_timeout", "wrong_limit", "extra_generation_change",
    "extra_statement", "removed_statement", "new_import", "outside_candidate_branch",
    "second_timeout", "baseline_default", "comment_only", "whitespace_only"])
def test_timeout_exception_cannot_authorize_any_other_pipeline_ast_change(tmp_path, mutation):
    changed = revised_pipeline()
    keyword = 'request_timeout_sec=config["limits"]["request_timeout_seconds"]'
    if mutation == "constant_timeout": changed = changed.replace(keyword, "request_timeout_sec=240")
    elif mutation == "wrong_limit": changed = changed.replace(keyword, 'request_timeout_sec=config["limits"]["max_images"]')
    elif mutation == "extra_generation_change": changed = changed.replace('max_tokens=config["generation"]["max_new_tokens"]', 'max_tokens=8192')
    elif mutation == "extra_statement": changed += "\nEXTRA_UNREVIEWED = True\n"
    elif mutation == "removed_statement": changed = changed.replace('        config = load_json(path)\r\n', "")
    elif mutation == "new_import": changed = "import random\n" + changed
    elif mutation == "outside_candidate_branch":
        changed = original_pipeline().replace('result.update(model_revision=identity["model_revision"],',
            'result.update(' + keyword + ', model_revision=identity["model_revision"],')
    elif mutation == "second_timeout": changed = changed.replace('    return LocalModelConfig(**values)',
        '    values.update(' + keyword + ')\n    return LocalModelConfig(**values)')
    elif mutation == "comment_only": changed += "\n# unrelated source change\n"
    elif mutation == "whitespace_only": changed = changed.replace("class AnalysisPipeline:", "class AnalysisPipeline :")
    else: changed = changed.replace('"max_frames": settings.max_frames_per_window, "max_tokens": 4096,',
        '"max_frames": settings.max_frames_per_window, "max_tokens": 4096, "request_timeout_sec": 240,')
    ast.parse(changed)
    with pytest.raises(SourceLineageError):
        verify_pair(tmp_path, changed)


@pytest.mark.parametrize("name", ["training/data.py", "rules/catalog.json", "annotations/taxonomy.v1.json",
    "serving/qwen38_grammar.py", "evaluation/qwen38_pipeline.py", "training/extra.py", "src/forte/extra.py"])
def test_exact_timeout_exception_keeps_every_other_source_member_protected(tmp_path, name):
    with pytest.raises(SourceLineageError):
        verify_pair(tmp_path, revised_pipeline(), overrides={name: b"unreviewed"})


@pytest.mark.parametrize("name", [PIPELINE, "training/data.py", "evaluation/qwen38_pipeline.py"])
def test_deleted_source_member_cannot_hide_behind_timeout_exception(tmp_path, name):
    with pytest.raises(SourceLineageError):
        verify_pair(tmp_path, revised_pipeline(), remove=(name,))


@pytest.mark.parametrize("timeout", [120, 240, 600])
def test_candidate_uses_validated_config_timeout_not_implicit_sixty_seconds(candidate, timeout):
    settings, config, path = candidate
    config["limits"]["request_timeout_seconds"] = timeout
    path.write_text(json.dumps(config))
    result = _provider_configuration(settings)
    assert result.request_timeout_sec == timeout
    baseline = _provider_configuration(replace(settings, model_candidate_launch_config_path=None))
    assert baseline.request_timeout_sec == 60


@pytest.mark.parametrize("timeout", [0, False, 601, "240"])
def test_bad_candidate_timeout_cannot_bypass_validated_config(candidate, timeout):
    settings, config, path = candidate
    config["limits"]["request_timeout_seconds"] = timeout
    path.write_text(json.dumps(config))
    with pytest.raises(CandidateError):
        _provider_configuration(settings)


def quality_proof(tmp_path, *, overrides=None):
    from evaluation.io import digest
    from evaluation.qwen38_quality_equivalence import prove
    old, new = tmp_path / "old-quality.tar", tmp_path / "new-quality.tar"
    archive(old, original_pipeline())
    archive(new, revised_pipeline(), overrides=overrides)
    family, _, _ = quality_family_fixture()
    recipe = family["recipe"]
    recipe.update(source_archive_sha256=file_sha256(old),
        rules_sha256=hashlib.sha256(b"original rules").hexdigest(),
        taxonomy_sha256=hashlib.sha256(b"original taxonomy").hexdigest(),
        prompt_implementation_pins={PIPELINE: hashlib.sha256(original_pipeline().encode()).hexdigest()})
    recipe["prompt_source_sha256"] = digest(recipe["prompt_implementation_pins"])
    family["recipe_sha256"] = digest(recipe)
    result = prove(family, old, new, disclosure="Candidate HTTP timeout forwarding only; other source bytes unchanged.",
                   allow_candidate_timeout_forwarding=True)
    return family, result


def test_quality_proof_discloses_one_common_delta_and_preserves_old_family(tmp_path):
    from evaluation.io import digest
    from evaluation.qwen38_quality_equivalence import validate_equivalence
    family, receipt = quality_proof(tmp_path)
    before = digest(family)
    assert receipt["protected_common_file_count"] == 4
    assert receipt["byte_identical_common_file_count"] == 3
    assert receipt["audited_common_source_delta_count"] == 1
    assert receipt["all_common_files_byte_identical"] is False
    validate_equivalence(family, receipt, receipt["actual_serving_source_archive_sha256"])
    assert digest(family) == before


@pytest.mark.parametrize("name", ["src/forte/new_prompt.py", "rules/new.json", "annotations/new.py"])
def test_quality_timeout_exception_cannot_hide_new_common_member(tmp_path, name):
    with pytest.raises(ValueError):
        quality_proof(tmp_path, overrides={name: b"unreviewed new source"})


def test_quality_timeout_delta_cannot_omit_honest_common_source_counts(tmp_path):
    from evaluation.io import digest
    from evaluation.qwen38_quality_equivalence import validate_equivalence
    family, receipt = quality_proof(tmp_path)
    for field in ("byte_identical_common_file_sha256", "protected_common_file_count",
        "byte_identical_common_file_count", "audited_common_source_delta_count", "all_common_files_byte_identical"):
        receipt.pop(field)
    receipt["receipt_sha256"] = digest({key:value for key,value in receipt.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError):
        validate_equivalence(family, receipt, receipt["actual_serving_source_archive_sha256"])
