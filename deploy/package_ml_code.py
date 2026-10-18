"""Package public executable ML source; mount private inputs and weights separately."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.package_policy import (DENIED_PARTS, PUBLIC_CODE_ARTIFACT_PARTS, capture_stable as _capture_stable,
                                   public_file, write_package)


def regulatory_dependencies(root: Path) -> list[Path]:
    """Select only the public source bytes explicitly pinned by the rule catalog."""
    rules = root / "rules"
    catalog = json.loads((rules / "catalog.json").read_text(encoding="utf-8"))
    files = []
    for source in catalog["sources"]:
        archived = set()
        for entry in source["archives"]:
            relative = PurePosixPath(entry["path"])
            if (relative.is_absolute() or relative.as_posix() != entry["path"]
                    or ".." in relative.parts or relative.parts[:1] != ("sources",)):
                raise ValueError("public_regulatory_archive_path_invalid")
            path = rules.joinpath(*relative.parts)
            if not public_file(path, root):
                raise ValueError("public_regulatory_archive_missing_or_private")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != entry["bytes"]:
                raise ValueError("public_regulatory_archive_pin_mismatch")
            files.append(path)
            archived.add(relative.as_posix())
        if source["text_archive_path"] not in archived:
            raise ValueError("public_regulatory_text_archive_unpinned")
    return sorted(set(files))


def selected(root: Path) -> list[Path]:
    files = []
    for directory in ("serving", "src/forte", "training", "evaluation"):
        for parent, directories, names in os.walk(root / directory, followlinks=False):
            directories[:] = [name for name in directories
                              if (name.lower() not in DENIED_PARTS or (
                                  Path(parent).relative_to(root).parts[:2] == ("src", "forte")
                                  and name.lower() in PUBLIC_CODE_ARTIFACT_PARTS))
                              and not (Path(parent).relative_to(root).as_posix() == "training" and name == "grounding")
                              and not (Path(parent) / name).is_symlink()
                              and not ((Path(parent) / name).is_junction()
                                       if hasattr(Path(parent) / name, "is_junction") else False)]
            files.extend(Path(parent) / name for name in names if name.endswith(".py")
                         and (Path(parent) / name).relative_to(root).as_posix() != "serving/grounding_pilot.py"
                         and public_file(Path(parent) / name, root))
    for pattern in ("configs/serving/encoders/*.manifest.json", "training/temporal/*.json",
                    "training/configs/*.example.json", "configs/serving/qwen38/*.manifest.json",
                    "configs/serving/qwen38/*.example.json"):
        files.extend(path for path in root.glob(pattern) if public_file(path, root))
    names = (
        "training/runtime-lock.json", "training/requirements-addons.lock",
        "annotations/__init__.py", "annotations/catalog.py", "annotations/taxonomy.v1.json",
        "annotations/machine.py", "annotations/machine.schema.json",
        "annotations/proposals.py", "annotations/rebase.py", "annotations/store.py",
        "rules/__init__.py", "rules/catalog.py", "rules/catalog.json",
        "configs/decision/default.json", "models/manifest.json",
        "benchmarks/performance/fresh_head_repeatability.py", "benchmarks/performance/repeatability.py",
        "scripts/audit_data.py", "scripts/build_reviewed_split.py",
        "scripts/exposure_registry.py", "scripts/split_gate.py", "scripts/build_automated_split.py",
        "deploy/feature_runtime_check.py", "deploy/feature-runtime-lock.json",
        "deploy/requirements-features.lock", "deploy/Dockerfile.features",
        "deploy/Dockerfile.features.dockerignore", "deploy/compose.ml.yaml", "deploy/ml.env.example",
        "deploy/stage_training_sources.py", "deploy/package_ml_code.py", "deploy/package_policy.py",
        "deploy/ml_source_smoke.py", "training/source-revisions/qwen38-silver.phase3.json",
    )
    files.extend(root / name for name in names)
    files.extend(regulatory_dependencies(root))
    # The candidate image is separate from the historical feature runtime. Keep
    # its public build definition/locks with the source once they are present.
    candidate_names = (
        "deploy/Dockerfile.qwen38-training", "deploy/Dockerfile.qwen38-training.dockerignore",
        "deploy/requirements-qwen38.lock", "deploy/qwen38-training-runtime-lock.json",
        "deploy/qwen38_runtime_check.py",
        "deploy/compose.qwen38-candidate.yaml",
        "deploy/compose.qwen38-candidate-operational.yaml", "deploy/qwen38_bind_runtime.py",
        "deploy/qwen38_acceptance_host.py",
        # Optional FLA runtime provenance only; inclusion does not select this
        # runtime or establish GPU kernel/model compatibility.
        "deploy/Dockerfile.qwen38-fla", "deploy/Dockerfile.qwen38-fla.dockerignore",
        "deploy/requirements-qwen38-fla.lock", "deploy/qwen38-fla-runtime-lock.json",
        "deploy/qwen38_fla_runtime_check.py",
    )
    files.extend(root / name for name in candidate_names if (root / name).exists())
    if any(not public_file(path, root) for path in files):
        raise ValueError("required_public_ml_source_missing_or_excluded")
    return sorted(set(files))


def source_snapshot(root: Path) -> tuple[list[Path], dict[str, tuple[int, int]]]:
    files = selected(root)
    return files, {path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
                   for path in files}


def capture_stable(root: Path, attempts: int = 3) -> dict[str, bytes]:
    return _capture_stable(root, selected, attempts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/forte-ml-code.tar.gz"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = write_package(capture_stable(root), args.output, schema="forte.ml-code-package.v2",
                           metadata={"unfinished_grounding_pilot_included": False,
                                     "training_performed": False, "git_commit": None})
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))


if __name__ == "__main__":
    main()
