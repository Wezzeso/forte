"""Genuine public predecessor bytes; generated archive-only pairing checks."""
from copy import deepcopy
from io import BytesIO
import hashlib
import tarfile
from pathlib import Path

import pytest

from evaluation.io import digest
from evaluation.qwen38_quality_equivalence import prove, comparable_runtime_profiles
from serving.qwen38_identity import file_sha256
from serving.qwen38_source_lineage import SourceLineageError, verify_candidate_inference_mode
from test_qwen38_quality_family import fixtures as family_fixture

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "tmp/qwen38-ml-support-grammar-final-20261017T151420Z.tar.gz"
DA8 = ROOT / "tmp/qwen38-ml-timeout-final-20261017T164509Z.tar.gz"
DA8_SHA = "da8a4b838003783ab6d201efc1e3b0eff54bbaa10aaa30db7cabe3cbecc98b28"
SERVICE = "serving/qwen38_service.py"


def public_sources():
    assert file_sha256(DA8) == DA8_SHA
    with tarfile.open(DA8) as bundle:
        return {row.name: bundle.extractfile(row).read() for row in bundle}


def test_actual_predecessor_service_is_restored_by_removing_exact_keyword():
    old, new = public_sources()[SERVICE], (ROOT / SERVICE).read_bytes()
    proof = verify_candidate_inference_mode(old, new)
    assert proof["old_source_sha256"] == "3b97e3ea1c30d086da0b1a0286cd8a84fbfb173ec0e88c314a0fa70a5af191c8"
    assert proof["new_source_sha256"] == "513460f34644edd9876db3294ec611477c0cb5c44f6b0a8ebbb69eb759585d7f"
    assert new.replace(b", inference_mode=True", b"") == old
    assert proof["other_ast_identical"] and proof["other_source_bytes_identical"]


@pytest.mark.parametrize("mutation", [b", inference_mode=1", b", inference_mode=False",
    b", inference_mode = True", b", inference_mode=True, other=True"])
def test_genuine_source_pair_rejects_other_activation_bytes(mutation):
    old, new = public_sources()[SERVICE], (ROOT / SERVICE).read_bytes()
    with pytest.raises(SourceLineageError):
        verify_candidate_inference_mode(old, new.replace(b", inference_mode=True", mutation))


def paired(tmp_path, *, other_change=None):
    assert file_sha256(ORIGINAL) == "91263a59cb2e73af48b46166b427d507ce7c3fa05106a46402acec950c20048d"
    original = {}
    with tarfile.open(ORIGINAL) as bundle:
        for row in bundle:
            original[row.name] = bundle.extractfile(row).read()
    family, _, _ = family_fixture()
    recipe = family["recipe"]
    recipe.update(source_archive_sha256=file_sha256(ORIGINAL),
        rules_sha256=hashlib.sha256(original["rules/catalog.json"]).hexdigest(),
        taxonomy_sha256=hashlib.sha256(original["annotations/taxonomy.v1.json"]).hexdigest(),
        prompt_implementation_pins={"src/forte/pipeline.py": hashlib.sha256(original["src/forte/pipeline.py"]).hexdigest()})
    recipe["prompt_source_sha256"] = digest(recipe["prompt_implementation_pins"])
    family["recipe_sha256"] = digest(recipe)
    files = public_sources()
    for name in (SERVICE, "serving/qwen38_source_lineage.py", "evaluation/qwen38_quality_equivalence.py",
                 "serving/tests/test_qwen38_inference_lineage.py"):
        files[name] = (ROOT / name).read_bytes()
    if other_change == "changed": files["serving/qwen38_grammar.py"] += b"\n# unreviewed\n"
    elif other_change == "added": files["serving/unreviewed.py"] = b"# extra source\n"
    elif other_change == "removed": files.pop("serving/qwen38_grammar.py")
    runtime = tmp_path / "generated-reviewed-runtime.tar"
    with tarfile.open(runtime, "w") as bundle:
        for name, raw in sorted(files.items()):
            row = tarfile.TarInfo(name)
            row.size = len(raw)
            bundle.addfile(row, BytesIO(raw))
    before = digest(family)
    old_receipt = prove(family, ORIGINAL, DA8, disclosure="Preserved historical timeout release.",
                         allow_candidate_timeout_forwarding=True)
    receipt = prove(family, ORIGINAL, runtime, disclosure="Explicit adapter inference activation only; no math-source changes.",
        allow_candidate_timeout_forwarding=True, serving_predecessor_archive=DA8, serving_predecessor_sha256=DA8_SHA,
        training_source_archive=ORIGINAL, training_source_sha256=file_sha256(ORIGINAL))
    assert digest(family) == before
    previous = {"runtime_bindings": {"source_archive_sha256": DA8_SHA}, "source_equivalence": old_receipt,
        "comparison_runtime_profile": {"source_archive_sha256": DA8_SHA, "generation": {"seed":20261017},
            "limits": {"request_timeout_seconds":240}, "quantization": {"format":"nf4"}}}
    newer = deepcopy(previous)
    newer["runtime_bindings"]["source_archive_sha256"] = file_sha256(runtime)
    newer["comparison_runtime_profile"]["source_archive_sha256"] = file_sha256(runtime)
    newer["source_equivalence"] = receipt
    return previous, newer, receipt


def test_genuine_predecessor_pair_preserves_profiles_and_honest_common_counts(tmp_path):
    previous, newer, receipt = paired(tmp_path)
    before = digest([previous, newer])
    result = comparable_runtime_profiles(previous, newer)
    assert result["exact_profile_identity"] is False
    assert result["audited_serving_control_variant"] is True
    assert result["pure_model_only_causal_claim"] is False
    assert (receipt["protected_common_file_count"], receipt["byte_identical_common_file_count"],
            receipt["audited_common_source_delta_count"], receipt["audited_serving_source_delta_count"]) == (88,87,1,1)
    assert digest([previous, newer]) == before
    newer["source_equivalence"].pop("audited_serving_source_deltas")
    with pytest.raises(ValueError):
        comparable_runtime_profiles(previous, newer)


@pytest.mark.parametrize("other_change", ["changed", "added", "removed"])
def test_adapter_activation_pair_cannot_authorize_other_generation_source_changes(tmp_path, other_change):
    with pytest.raises(ValueError):
        previous, newer, _ = paired(tmp_path, other_change=other_change)
        comparable_runtime_profiles(previous, newer)
