"""Package public deployment source with a stable, immutable hash receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.package_policy import (DENIED_PARTS, DENIED_SUFFIXES, capture_stable,
                                   public_file, write_package)


def allowed(path: Path, root: Path) -> bool:
    if not public_file(path, root):
        return False
    relative = path.relative_to(root)
    return relative.parts[:2] != ("web", "dist")


def selected(root: Path, *, cpu_update: bool = False, exclude_files: tuple[str, ...] = ()) -> list[Path]:
    required = ("pyproject.toml", "uv.lock", "README.md", ".dockerignore",
                "annotations/__init__.py", "annotations/catalog.py", "annotations/taxonomy.v1.json",
                "models/manifest.json")
    files = [root / name for name in required]
    for directory in ("src", "rules", "configs", "serving", "web", "brand", "deploy", "tools"):
        files.extend(path for path in (root / directory).rglob("*") if allowed(path, root))
    excluded = set(exclude_files)
    if any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name for name in excluded):
        raise ValueError("exclude_file_must_be_repository_relative_posix")
    files = [path for path in sorted(set(files)) if path.relative_to(root).as_posix() not in excluded]
    if cpu_update:
        gpu_roots = ("serving/", "configs/serving/", "src/forte/model/", "models/")
        gpu_files = {"deploy/Dockerfile.model", "deploy/compose.gpu.yaml"}
        files = [path for path in files if not path.relative_to(root).as_posix().startswith(gpu_roots)
                 and path.relative_to(root).as_posix() not in gpu_files]
    if not files or any(not allowed(path, root) for path in files):
        raise ValueError("required_public_deployment_source_missing_or_excluded")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/forte-deploy.tar.gz"))
    parser.add_argument("--cpu-update", action="store_true", help="Preserve separately managed model/serving source")
    parser.add_argument("--exclude-file", action="append", default=[], help="Omit an optional repository-relative file")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    captured = capture_stable(root, lambda base: selected(base, cpu_update=args.cpu_update,
                                                        exclude_files=tuple(args.exclude_file)))
    result = write_package(captured, args.output, schema="forte.deployment-source-package.v2",
                           metadata={"private_media_included": False, "cpu_update": args.cpu_update,
                                     "public_model_payload_included": False,
                                     "deployment_performed": False, "git_commit": None})
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))


if __name__ == "__main__":
    main()
