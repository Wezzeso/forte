"""The PEFT activation correction cannot authorize other source differences."""

import hashlib
from pathlib import Path

import pytest

from serving.qwen38_identity import file_sha256
from serving.qwen38_source_lineage import (SourceLineageError,
    verify_candidate_inference_mode, verify_source_lineage)
from serving.tests.qwen38_adapter_fixtures import archive_fixture
from serving.tests.test_qwen38_timeout_lineage import sources as timeout_sources


ADDITION = b", inference_mode=True"


def sources():
    updated = Path("serving/qwen38_service.py").read_bytes()
    assert updated.count(b'self.model.set_adapter("candidate", inference_mode=True)') == 1
    return updated.replace(b'self.model.set_adapter("candidate", inference_mode=True)',
                           b'self.model.set_adapter("candidate")'), updated


def test_exact_adapter_activation_delta_preserves_every_other_byte_and_ast():
    old, new = sources()
    proof = verify_candidate_inference_mode(old, new)
    assert proof["verified"] and proof["candidate_runtime_load_only"] and proof["serving_only"]
    assert proof["other_ast_identical"] and proof["other_source_bytes_identical"]
    assert proof["added_keyword"] == "inference_mode" and proof["added_value"] == "True"
    assert proof["old_source_sha256"] == "3b97e3ea1c30d086da0b1a0286cd8a84fbfb173ec0e88c314a0fa70a5af191c8"
    assert proof["new_source_sha256"] == hashlib.sha256(new).hexdigest()


@pytest.mark.parametrize("mutation", ["false", "integer", "other_name", "other_adapter", "extra_keyword",
    "other_guard", "other_comment", "whitespace", "split_keyword", "already_present", "no_addition",
    "second_activation", "other_method"])
def test_activation_exception_rejects_other_value_scope_or_source(mutation):
    old, new = sources()
    if mutation == "false": new = new.replace(ADDITION, b", inference_mode=False")
    elif mutation == "integer": new = new.replace(ADDITION, b", inference_mode=1")
    elif mutation == "other_name": new = new.replace(ADDITION, b", is_trainable=False")
    elif mutation == "other_adapter": new = new.replace(b'set_adapter("candidate",', b'set_adapter("other",')
    elif mutation == "extra_keyword": new = new.replace(ADDITION, ADDITION + b", other=True")
    elif mutation == "other_guard": new = new.replace(b"all(not parameter.requires_grad", b"all(parameter.requires_grad")
    elif mutation == "other_comment": new += b"\n# unreviewed\n"
    elif mutation == "whitespace": new = new.replace(ADDITION, b", inference_mode = True")
    elif mutation == "split_keyword": new = new.replace(ADDITION, b",\n                                    inference_mode=True")
    elif mutation == "already_present": old = new
    elif mutation == "no_addition": new = old
    elif mutation == "second_activation": new = new.replace(b"        self.model.eval()", b'        self.model.set_adapter("candidate")\n        self.model.eval()')
    elif mutation == "other_method": new = new.replace(b"    def load(self)", b"    def other(self)")
    with pytest.raises(SourceLineageError): verify_candidate_inference_mode(old, new)


def test_paired_real_source_exceptions_have_separate_honest_counts(tmp_path):
    old_service, new_service = sources()
    old_pipeline, new_pipeline = timeout_sources()
    original, runtime = tmp_path/"training.tar", tmp_path/"runtime.tar"
    archive_fixture(original, {"src/forte/pipeline.py": old_pipeline, "serving/qwen38_service.py": old_service})
    archive_fixture(runtime, {"src/forte/pipeline.py": new_pipeline, "serving/qwen38_service.py": new_service})
    proof = verify_source_lineage(training_archive=original, training_sha256=file_sha256(original),
        recorded_training_sha256=file_sha256(original), runtime_archive=runtime, runtime_sha256=file_sha256(runtime))
    assert proof["protected_files_verified"] == 5
    assert proof["protected_files_byte_identical"] == 4
    assert proof["protected_files_with_audited_timeout_delta"] == 1
    assert proof["serving_files_with_audited_inference_mode_delta"] == 1
    assert proof["candidate_inference_mode_delta"] == verify_candidate_inference_mode(old_service, new_service)
    assert proof["run_bytes_changed"] is False and proof["adapter_bytes_changed"] is False


def test_service_file_creation_and_unreviewed_change_are_rejected(tmp_path):
    old, new = sources()
    original, runtime = tmp_path/"training.tar", tmp_path/"runtime.tar"
    for initial, changed in [(None, new), (old, new + b"\n# extra\n"), (old, None)]:
        archive_fixture(original, {} if initial is None else {"serving/qwen38_service.py": initial})
        archive_fixture(runtime, {} if changed is None else {"serving/qwen38_service.py": changed})
        with pytest.raises(SourceLineageError):
            verify_source_lineage(training_archive=original, training_sha256=file_sha256(original),
                recorded_training_sha256=file_sha256(original), runtime_archive=runtime, runtime_sha256=file_sha256(runtime))
