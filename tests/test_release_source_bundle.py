"""Source ZIP entrypoint and verified-profile bindings, using synthetic trees."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import zipfile

import pytest

from qa.integration import build_release_bundle as release


def put(root: Path, name: str, content: bytes = b"synthetic public source\n") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def names(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in release.files_for_bundle(root)}


def test_release_includes_only_audited_tools_and_workflow_from_new_roots(tmp_path):
    wanted = {*release.PUBLIC_TOOL_FILES, *release.PUBLIC_WORKFLOW_FILES}
    for name in wanted:
        put(tmp_path, name)
    forbidden = {
        "tools/probe.py", "tools/results.json", "tools/labels.json", "tools/session.json",
        "tools/private/fast_eval.py", "tools/.env", "tools/secrets/api_keys", "tools/model.pt",
        ".github/credentials.json", ".github/state.json", ".github/.env",
        ".github/workflows/credentials.yml", ".github/workflows/other.yml",
        ".github/workflows/private/fast-regression.yml", ".github/actions/local-state.json",
    }
    for name in forbidden:
        put(tmp_path, name, b"synthetic private fixture; must not ship")
    assert names(tmp_path) == wanted


@pytest.mark.parametrize("name", [
    "src/forte/private/state.json", "configs/private/fast_config.json",
    "training/runs/checkpoint.pt", "training/weights/model.safetensors",
    "benchmarks/private/record.json", "evaluation/private/report.json", "deploy/secrets/api_keys",
    "deploy/.env", "deploy/key.pem", "benchmarks/quality/input-selection-review.json",
])
def test_existing_private_exclusions_stay_denied(tmp_path, name):
    put(tmp_path, name)
    assert not names(tmp_path)


def test_new_entrypoints_reject_symlinks_and_linked_ancestors(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside"
    secret = put(outside, "fast_eval.py", b"synthetic secret fixture")
    try:
        (root / "tools").symlink_to(outside, target_is_directory=True)
        (root / ".github/workflows").mkdir(parents=True)
        (root / release.PUBLIC_WORKFLOW_FILES[0]).symlink_to(secret)
    except OSError:
        pytest.skip("symlinks unavailable on this host")
    assert not names(root)


def test_source_zip_binds_verified_profile_active_fast_source_and_specialist_dependencies(tmp_path):
    root = tmp_path / "source"
    config = {"use_verify": True, "require_corroboration": True, "specialists": {"enabled": False}}
    verified = json.dumps(config, indent=2).encode()
    expected = {
        release.VERIFIED_FAST_CONFIGURATION: verified,
        "src/forte/fast/analyzer.py": b"# synthetic active analyzer\n",
        "src/forte/fast/policy.py": b"# synthetic policy\n",
        "src/forte/fast/specialists.py": b"# synthetic detector runtime\n",
        "deploy/compose.fast.yaml": b"services: {}\n",
        "deploy/compose.fastlive.yaml": b"services: {}\n",
        "deploy/compose.specialists.yaml": b"services: {}\n",
        "deploy/requirements-specialist-runtime.txt": b"synthetic==1\n",
        "deploy/Dockerfile.specialists": b"# synthetic\n",
        "configs/fast-specialists.example.json": b'{"specialists":{"enabled":false}}\n',
        "training/specialists/requirements.txt": b"synthetic==1\n",
        release.FAST_MEASUREMENT_REFERENCES[0]: b'{"summary":{"n":40}}\n',
        **{name: b"# synthetic public CLI\n" for name in release.PUBLIC_TOOL_FILES},
        **{name: b"name: synthetic\n" for name in release.PUBLIC_WORKFLOW_FILES},
    }
    for name, raw in expected.items():
        put(root, name, raw)
    put(root, "benchmarks/private/labels.json", b"synthetic private fixture")
    put(root, ".github/credentials.json", b"synthetic private fixture")
    output = tmp_path / "synthetic-release.zip"
    manifest = release.build(root, output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(expected)
        for name, raw in expected.items():
            assert archive.read(name) == raw
            assert manifest["configuration_sha256"][name] == sha256(raw).hexdigest()
    binding = manifest["fast_profile_binding"]
    assert binding["verified_configuration"] == {
        "path": release.VERIFIED_FAST_CONFIGURATION, "file_sha256": sha256(verified).hexdigest(),
        "canonical_sha256": sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                                separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
    }
    assert set(binding["packaged_fast_source_sha256"]) == {name for name in expected if name.startswith("src/forte/fast/")}
    assert set(binding["measurement_references_sha256"]) == {release.FAST_MEASUREMENT_REFERENCES[0]}
    assert manifest["artifact_establishes_public_deployment"] is False
    assert manifest["private_media_included"] is False and manifest["weights_included"] is False
    assert json.loads(output.with_suffix(".manifest.json").read_text()) == manifest


def test_empty_tree_does_not_invent_verified_configuration(tmp_path):
    root = tmp_path / "source"
    put(root, "README.md")
    report = release.build(root, tmp_path / "synthetic.zip")
    assert report["fast_profile_binding"]["verified_configuration"] is None


def test_real_new_roots_match_audited_allowlist_and_have_no_known_secret_signatures():
    root = Path(__file__).resolve().parents[1]
    selected = {name for name in names(root) if name.startswith(("tools/", ".github/"))}
    assert selected == {*release.PUBLIC_TOOL_FILES, *release.PUBLIC_WORKFLOW_FILES}
    signature = re.compile(r"sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")
    for name in selected:
        if signature.search((root / name).read_text(encoding="utf-8")):
            pytest.fail(f"Possible credential signature in audited public entrypoint: {name}")
