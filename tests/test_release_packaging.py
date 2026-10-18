"""Synthetic checks that deployment artifacts exclude private material."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from deploy.package_code import allowed, selected
from deploy.package_policy import capture_stable, public_file, verify_package, write_package


def file_at(root: Path, name: str, content: bytes = b"public source\n") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.mark.parametrize("name", [
    "deploy/secrets/api_keys", "deploy/Secrets/postgres_password", "deploy/.env",
    "deploy/.env.production", "deploy/session.local.env", "deploy/runtime.env",
    "configs/selection.json", "configs/trial.labels.json", "configs/raw_outputs.json",
    "configs/raw-output.json", "configs/machine_annotations.json", "configs/x.split.json",
    "configs/user_coarse_anchors.train.json", "configs/observations.json",
    "serving/private/a.py", "serving/Private/a.py", "configs/backups/a.json",
    "configs/splits/a.json", "configs/selections/a.json", "brand/reference-captures/a.png",
    "web/qa/screenshots/a.png", "web/test-results/results.json", "deploy/id_ed25519",
    "deploy/id_rsa.pub", "deploy/key.pem", "deploy/key.key", "deploy/source.tar.gz",
    "configs/backup.json", "configs/a.sqlite", "configs/a.db", "configs/weights.gguf",
    "configs/model.safetensors", "configs/a.jsonl", "configs/a.npy", "configs/a.parquet",
    "web/public/secret.MOV", "web/public/secret.MP4", "web/public/secret.webm",
    "configs/serving/qwen38/public-snapshot/tokenizer.json", "configs/source-frame.jpg",
    "configs/serving/qwen38/speed-research-v031/upstream.py",
    "configs/serving/qwen38/speed-research-v031/findings.json",
    "configs/credentials.json", "deploy/password.txt", "configs/access_token.txt", "deploy/api_key.json",
    "deploy/unknown.credentials", "deploy/token.dump",
    "configs/candidate.runtime.bound.json", "configs/candidate.runtime.local.json",
    "deploy/credentials/session.json", "deploy/keys/token.json",
])
def test_private_paths_are_excluded(tmp_path: Path, name: str):
    assert not allowed(file_at(tmp_path, name), tmp_path)


@pytest.mark.parametrize("name", [
    "deploy/.env.example", "deploy/ml.env.example", "deploy/Dockerfile.model",
    "serving/qwen38_service.py", "src/forte/observations.py", "scripts/split_gate.py",
    "training/qwen38_silver.py", "training/configs/qwen38-silver.example.json",
    "configs/serving/qwen38/public-candidate.manifest.json", "rules/sources/manifest.json",
    "rules/sources/nbk_231_current.txt", "brand/assets/logo.png", "web/public/logo.svg",
    "src/forte/evidence/assembly.py", "src/forte/observations/catalog.py",
])
def test_public_source_and_placeholders_are_allowed(tmp_path: Path, name: str):
    assert public_file(file_at(tmp_path, name), tmp_path)


def test_links_and_linked_ancestors_are_rejected(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = file_at(outside, "token.json")
    root = tmp_path / "source"
    root.mkdir()
    try:
        (root / "linked.json").symlink_to(secret)
        (root / "linked-dir").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this host")
    assert not public_file(root / "linked.json", root)
    assert not public_file(root / "linked-dir/token.json", root)
    assert not public_file(secret, root)


def test_capture_rejects_a_source_changing_between_reads(tmp_path: Path):
    source = file_at(tmp_path, "source.py", b"a\n")
    calls = 0

    def changing_selector(root):
        nonlocal calls
        calls += 1
        source.write_bytes(str(calls).encode())
        return [source]

    with pytest.raises(RuntimeError, match="public_source_changed_during_capture"):
        capture_stable(tmp_path, changing_selector)


def test_capture_rejects_tree_membership_changes(tmp_path: Path):
    first = file_at(tmp_path, "one.py")
    second = file_at(tmp_path, "two.py")
    calls = 0

    def changing_selector(root):
        nonlocal calls
        calls += 1
        return [first] if calls % 2 else [first, second]

    with pytest.raises(RuntimeError, match="public_source_changed_during_capture"):
        capture_stable(tmp_path, changing_selector)


def test_archives_are_deterministic_immutable_and_hash_bound(tmp_path: Path):
    contents = {"src/public.py": b"print('synthetic')\n", "README.md": b"public\n"}
    first = tmp_path / "one.tar.gz"
    second = tmp_path / "two.tar.gz"
    receipt = write_package(contents, first, schema="synthetic.v1")
    again = write_package(dict(reversed(list(contents.items()))), second, schema="synthetic.v1")
    assert receipt["sha256"] == again["sha256"]
    assert first.read_bytes() == second.read_bytes()
    assert receipt["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert verify_package(first) == {"sha256": receipt["sha256"], "file_count": 2, "verified": True}
    with tarfile.open(fileobj=io.BytesIO(first.read_bytes()), mode="r:gz") as archive:
        assert set(archive.getnames()) == set(contents)
        for member in archive:
            assert member.isfile() and member.mtime == 0 and member.uid == 0 and member.gid == 0
            raw = archive.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest() == receipt["files"][member.name]
    assert json.loads(first.with_suffix(".gz.manifest.json").read_text())["files"] == receipt["files"]
    with pytest.raises(FileExistsError, match="public_package_output_already_exists"):
        write_package(contents, first, schema="synthetic.v1")


@pytest.mark.parametrize("name", ["../escape", "/absolute.py", "x/../escape", "C:/escape.py", "x\\escape.py", "x//escape.py"])
def test_archive_rejects_unsafe_member_names(tmp_path: Path, name: str):
    with pytest.raises(ValueError, match="public_package_member_name_invalid"):
        write_package({name: b"synthetic"}, tmp_path / "bad.tar.gz", schema="synthetic.v1")


def test_verifier_rejects_changed_archive_and_member_binding(tmp_path: Path):
    output = tmp_path / "source.tar.gz"
    write_package({"README.md": b"public"}, output, schema="synthetic.v1")
    sidecar = output.with_suffix(".gz.manifest.json")
    receipt = json.loads(sidecar.read_text())
    receipt["files"]["README.md"] = "0" * 64
    sidecar.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="member_hash_mismatch"):
        verify_package(output)
    output.write_bytes(output.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive_hash_mismatch"):
        verify_package(output)


def test_missing_required_source_fails_and_excludes_are_relative(tmp_path: Path):
    with pytest.raises(ValueError, match="required_public_deployment_source"):
        selected(tmp_path)
    with pytest.raises(ValueError, match="repository_relative_posix"):
        selected(tmp_path, exclude_files=("../secrets",))


def test_real_selections_keep_policy_and_no_private_artifacts():
    from deploy.package_ml_code import selected as ml_selected
    root = Path(__file__).resolve().parents[1]
    for selector in (selected, ml_selected):
        paths = selector(root)
        names = {path.relative_to(root).as_posix() for path in paths}
        assert "deploy/package_policy.py" in names
        assert "src/forte/evidence/assembly.py" in names
        assert all(public_file(path, root) for path in paths)
        assert not any("public-snapshot/" in name or "private/" in name
                       or "speed-research-v031/" in name for name in names)
    cpu = {path.relative_to(root).as_posix() for path in selected(root, cpu_update=True)}
    assert not any(name.startswith(("serving/", "configs/serving/", "models/")) for name in cpu)


def test_ml_package_contains_runtime_postprocessing_dependencies():
    from deploy.package_ml_code import regulatory_dependencies, selected as ml_selected
    root = Path(__file__).resolve().parents[1]
    names = {path.relative_to(root).as_posix() for path in ml_selected(root)}
    expected = {"annotations/machine.py", "annotations/machine.schema.json", "annotations/proposals.py",
                "annotations/store.py", "annotations/rebase.py", "annotations/catalog.py",
                "annotations/taxonomy.v1.json", "training/coverage_export.py", "training/silver_combine.py",
                "deploy/ml_source_smoke.py", "training/source-revisions/qwen38-silver.phase3.json"}
    assert expected <= names
    dependencies = regulatory_dependencies(root)
    assert len(dependencies) == 4
    assert {path.relative_to(root).as_posix() for path in dependencies} <= names
    optional_fla = {"deploy/Dockerfile.qwen38-fla", "deploy/Dockerfile.qwen38-fla.dockerignore",
                    "deploy/requirements-qwen38-fla.lock", "deploy/qwen38-fla-runtime-lock.json",
                    "deploy/qwen38_fla_runtime_check.py"}
    assert optional_fla <= names


def test_regulatory_dependencies_fail_closed_on_missing_tampered_or_escaping_files(tmp_path: Path):
    from deploy.package_ml_code import regulatory_dependencies
    raw = b"synthetic official text"
    archive = file_at(tmp_path, "rules/sources/official.txt", raw)
    reference = {"path": "sources/official.txt", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    catalog = {"sources": [{"archives": [reference], "text_archive_path": reference["path"]}]}
    path = file_at(tmp_path, "rules/catalog.json", json.dumps(catalog).encode())
    assert regulatory_dependencies(tmp_path) == [archive]
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pin_mismatch"):
        regulatory_dependencies(tmp_path)
    archive.unlink()
    with pytest.raises(ValueError, match="missing_or_private"):
        regulatory_dependencies(tmp_path)
    reference["path"] = "../private/source.txt"
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="path_invalid"):
        regulatory_dependencies(tmp_path)


def test_generated_ml_dependency_smoke_runs_without_private_inputs_or_models():
    from deploy.ml_source_smoke import smoke
    result = smoke()
    assert result["status"] == "passed" and result["catalog_archives_checksum_verified"] == 4
    assert result["generated_coverage_export_records"] == result["generated_combined_records"] == 17
    assert result["generated_original_jpegs_copied"] == 17 and result["model_calls"] == 0
    assert result["generated_attested_exact_frame_targets"] == result["generated_window_receipts"] == 17
    assert result["generated_one_invalid_source_retained_records"] == 16
    assert result["generated_all_invalid_export_blocked"] is True
    assert result["private_inputs_read"] is False and result["generated_outputs_removed"] is True
