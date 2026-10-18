"""Generated CPU metadata checks; no uploads, model calls or Docker actions."""
from copy import deepcopy
from io import BytesIO
import tarfile

import pytest

from benchmarks.performance import qwen38_acceptance as driver
from benchmarks.performance import qwen38_acceptance_rebind as rebind
from benchmarks.performance.repeatability import RepeatabilityError, digest, file_digest, write_new


def archive(tmp_path, entries):
    path = tmp_path / "generated.tar"
    with tarfile.open(path, "w") as target:
        for name, data in entries:
            item = tarfile.TarInfo(name)
            item.size = len(data)
            target.addfile(item, BytesIO(data))
    return path


def test_actual_archive_requires_matching_bytes_and_safe_members(tmp_path):
    path = archive(tmp_path, [("src/forte/generated.py", b"generated = True\n")])
    pins = rebind.archive_pins(path, file_digest(path))
    assert pins == {"src/forte/generated.py": driver.hashlib.sha256(b"generated = True\n").hexdigest()}
    with pytest.raises(RepeatabilityError, match="actual_source_archive"):
        rebind.archive_pins(path, "a" * 64)


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "C:/absolute.py", "bad\\member.py"])
def test_source_archive_path_redirection_rejected(tmp_path, name):
    path = archive(tmp_path, [(name, b"generated = True\n")])
    with pytest.raises(RepeatabilityError, match="member_invalid"):
        rebind.archive_pins(path, file_digest(path))


def test_actual_import_origin_must_match_worker_archive(tmp_path, monkeypatch):
    generated = tmp_path / "module.py"
    generated.write_bytes(b"generated = True\n")
    monkeypatch.setattr(rebind.importlib.util, "find_spec", lambda module: type("Spec", (), {"origin": str(generated)})())
    mapping = {"generated.module": "src/forte/generated.py"}
    pins = {"src/forte/generated.py": file_digest(generated)}
    assert rebind.actual_modules(mapping, pins)["src/forte/generated.py"]["sha256"] == file_digest(generated)
    with pytest.raises(RepeatabilityError, match="differs_from_archive"):
        rebind.actual_modules(mapping, {"src/forte/generated.py": "a" * 64})


def test_common_prompt_rule_and_sampling_changes_block_old_family():
    pins = {"src/forte/pipeline.py": "a" * 64, "src/forte/video/sampling.py": "b" * 64}
    recipe = {"prompt_implementation_pins": pins, "rules_sha256": "c" * 64, "taxonomy_sha256": "d" * 64}
    family = {"schema_version": "forte.qwen38-quality-family.v1", "recipe": recipe, "recipe_sha256": digest(recipe)}
    actual = {name: {"sha256": value} for name, value in pins.items()}
    model = {**pins, "rules/catalog.json": "c" * 64, "annotations/taxonomy.v1.json": "d" * 64}
    rebind.require_family_equivalence(family, actual, model, rules_sha256="c" * 64, taxonomy_sha256="d" * 64)
    changed = deepcopy(actual)
    changed["src/forte/video/sampling.py"]["sha256"] = "e" * 64
    with pytest.raises(RepeatabilityError, match="common_source_changed"):
        rebind.require_family_equivalence(family, changed, model, rules_sha256="c" * 64, taxonomy_sha256="d" * 64)
    with pytest.raises(RepeatabilityError, match="rules_or_taxonomy"):
        rebind.require_family_equivalence(family, actual, model, rules_sha256="e" * 64, taxonomy_sha256="d" * 64)


def test_pending_context_creates_no_new_bound_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "PRIVATE", tmp_path)
    def pending(**kwargs):
        raise RepeatabilityError("runtime_selection_pending")
    monkeypatch.setattr(rebind, "capture", pending)
    output = tmp_path / "new-rebind"
    with pytest.raises(RepeatabilityError, match="runtime_selection_pending"):
        rebind.rebind(output=output, population_path=tmp_path / "population.json")
    assert not output.exists()


def test_only_independently_validated_candidate_timeout_delta_can_match_actual_worker(monkeypatch):
    import evaluation.qwen38_quality_equivalence as checker
    calls = []
    monkeypatch.setattr(checker, "validate_equivalence", lambda *args: calls.append(args))
    pins = {"src/forte/pipeline.py": "a"*64, "src/forte/video/sampling.py": "b"*64}
    recipe = {"prompt_implementation_pins": pins, "rules_sha256": "c"*64, "taxonomy_sha256": "d"*64}
    family = {"schema_version": "forte.qwen38-quality-family.v1", "recipe": recipe, "recipe_sha256": digest(recipe)}
    model = {**pins, "src/forte/pipeline.py": "e"*64, "rules/catalog.json": "c"*64, "annotations/taxonomy.v1.json": "d"*64}
    actual = {name: {"sha256": model[name]} for name in pins}
    receipt = {"actual_serving_source_archive_sha256": "f"*64, "actual_serving_archive_files": model,
        "audited_common_source_deltas": [{"path": "src/forte/pipeline.py"}]}
    rebind.require_family_equivalence(family, actual, model, rules_sha256="c"*64, taxonomy_sha256="d"*64, equivalence=receipt)
    assert calls == [(family, receipt, "f"*64)]
    actual["src/forte/pipeline.py"]["sha256"] = "0"*64
    with pytest.raises(RepeatabilityError, match="common_source_changed"):
        rebind.require_family_equivalence(family, actual, model, rules_sha256="c"*64, taxonomy_sha256="d"*64, equivalence=receipt)
    receipt["audited_common_source_deltas"].append({"path": "src/forte/video/sampling.py"})
    with pytest.raises(RepeatabilityError, match="unexpected_audited"):
        rebind.require_family_equivalence(family, actual, model, rules_sha256="c"*64, taxonomy_sha256="d"*64, equivalence=receipt)
