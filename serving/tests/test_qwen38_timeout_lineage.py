"""Only the reviewed candidate HTTP timeout insertion may differ in pipeline."""

import hashlib
from pathlib import Path

import pytest

from serving.qwen38_identity import file_sha256
from serving.qwen38_source_lineage import (SourceLineageError,
    verify_candidate_timeout_forwarding, verify_source_lineage)
from serving.tests.qwen38_adapter_fixtures import archive_fixture


ADDITION = b'                      request_timeout_sec=config["limits"]["request_timeout_seconds"],'


def sources():
    updated = Path("src/forte/pipeline.py").read_bytes()
    assert updated.count(ADDITION) == 1
    line = next(line for line in updated.splitlines(keepends=True) if line.strip() == ADDITION.strip())
    return updated.replace(line, b""), updated


def test_only_audited_candidate_keyword_is_removed_and_entire_module_is_preserved():
    old, new = sources()
    proof = verify_candidate_timeout_forwarding(old, new)
    assert proof["verified"] and proof["candidate_branch_only"] and proof["other_ast_identical"]
    assert proof["other_source_bytes_identical"] and proof["baseline_default_unchanged"]
    assert proof["old_source_sha256"] == hashlib.sha256(old).hexdigest()
    assert proof["new_source_sha256"] == hashlib.sha256(new).hexdigest()


@pytest.mark.parametrize("mutation", ["wrong_value", "wrong_key", "second_keyword", "changed_other_keyword",
    "other_statement", "other_comment", "other_whitespace", "already_present", "no_addition"])
def test_timeout_audit_rejects_any_other_ast_or_source_change(mutation):
    old, new = sources()
    if mutation == "wrong_value": new = new.replace(b'request_timeout_sec=config["limits"]["request_timeout_seconds"]', b'request_timeout_sec=240')
    elif mutation == "wrong_key": new = new.replace(b'request_timeout_sec=config["limits"]["request_timeout_seconds"]', b'request_timeout_sec=config["limits"]["max_images"]')
    elif mutation == "second_keyword": new = new.replace(ADDITION, ADDITION + ADDITION)
    elif mutation == "changed_other_keyword": new = new.replace(b'max_tokens=config["generation"]["max_new_tokens"]', b'max_tokens=4096')
    elif mutation == "other_statement": new += b'\nimport os as unreviewed_import\n'
    elif mutation == "other_comment": new += b'\n# an unreviewed comment is still a byte change\n'
    elif mutation == "other_whitespace": new += b'\n'
    elif mutation == "already_present": old = new
    else: new = old
    with pytest.raises(SourceLineageError): verify_candidate_timeout_forwarding(old, new)


def test_actual_archive_member_bytes_prove_pipeline_timeout_delta(tmp_path):
    old, new = sources()
    original, runtime = tmp_path/"original.tar", tmp_path/"runtime.tar"
    archive_fixture(original, {"src/forte/pipeline.py": old})
    archive_fixture(runtime, {"src/forte/pipeline.py": new})
    proof = verify_source_lineage(training_archive=original, training_sha256=file_sha256(original),
        recorded_training_sha256=file_sha256(original), runtime_archive=runtime, runtime_sha256=file_sha256(runtime))
    assert proof["protected_files_verified"] == 5
    assert proof["protected_files_byte_identical"] == 4
    assert proof["protected_files_with_audited_timeout_delta"] == 1
    assert proof["candidate_http_timeout_delta"] == verify_candidate_timeout_forwarding(old, new)


def test_another_common_source_change_is_rejected_beside_a_valid_timeout(tmp_path):
    old, new = sources()
    original, runtime = tmp_path/"original.tar", tmp_path/"runtime.tar"
    archive_fixture(original, {"src/forte/pipeline.py": old})
    archive_fixture(runtime, {"src/forte/pipeline.py": new, "src/forte/model/adapter.py": b"unreviewed"})
    with pytest.raises(SourceLineageError, match="unaudited_source_difference"):
        verify_source_lineage(training_archive=original, training_sha256=file_sha256(original),
            recorded_training_sha256=file_sha256(original), runtime_archive=runtime, runtime_sha256=file_sha256(runtime))
