"""Build a source/operator bundle from explicit code/public-doc directories.

Organizer inputs, labels, local annotation/evaluation records, secrets, model
weights and runtime state have no inclusion path. This does not package a
running GPU service or establish a successful deployment.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile


ROOT_FILES = {
    "README.md", "pyproject.toml", "uv.lock", ".gitignore", ".dockerignore",
    "FORTE_20_AGENT_EXECUTION_PLAN.md",
}
ROOT_DIRECTORIES = {
    "src", "contracts", "configs", "rules", "serving", "web", "frontend",
    "tests", "infra", "deploy", "training", "evaluation", "benchmarks", "brand", "docs", "qa", "scripts", "models",
}
# These roots also contain local records in development. Keep their public
# entry points explicit instead of recursively admitting either directory.
PUBLIC_TOOL_FILES = (
    "tools/fast_eval.py", "tools/fast_validation.py", "tools/fast_load_test.py",
    "tools/fast_host_preflight.py",
    "tools/run_public_training_host.py", "tools/receive_public_training_bundle.py",
    "tools/run_public_evaluation_host.py",
    "tools/preserve_public_training_run.py",
    "tools/stage_public_candidates_host.py",
)
PUBLIC_WORKFLOW_FILES = (".github/workflows/fast-regression.yml",)
VERIFIED_FAST_CONFIGURATION = "benchmarks/quality/accuracy-upgrade-verified-config-20261018.json"
FAST_MEASUREMENT_REFERENCES = (
    "benchmarks/quality/accuracy-upgrade-baseline-20261018.json",
    "benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json",
    "benchmarks/quality/accuracy-upgrade-stage-20261018.json",
    "benchmarks/quality/accuracy-upgrade-public-load-20261018.json",
    "benchmarks/quality/accuracy-upgrade-rollout-20261018.json",
)
DENIED_DIRECTORIES = {
    "private", "runs", "weights", "secrets", "node_modules", "__pycache__",
    ".pytest_cache", ".venv", ".git", ".cache", "uploads", "results", "work",
}
DENIED_SUFFIXES = {
    ".mov", ".mp4", ".mkv", ".avi", ".webm", ".xlsx", ".xls", ".csv",
    ".tsv", ".jsonl", ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
    ".pem", ".key", ".log", ".pyc", ".pyo", ".db", ".sqlite", ".sqlite3",
    ".zip", ".7z", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".whl",
    ".exe", ".dll", ".pyd", ".so", ".dylib",
}
CONFIGURATION_FILES = (
    "pyproject.toml", "uv.lock", "contracts/version.json", "models/manifest.json",
    "configs/serving/baseline.json", "configs/serving/image-lock.json",
    "rules/catalog.json", "annotations/taxonomy.v1.json", "brand/tokens.json",
    "web/package.json", "web/package-lock.json", "deploy/compose.yaml",
    "deploy/compose.gpu.yaml", "deploy/Dockerfile", "deploy/Dockerfile.model",
    "deploy/Caddyfile", "deploy/dokploy-ingress.yaml", "src/forte/config.py",
    "src/forte/pipeline.py", "src/forte/worker.py",
    "deploy/compose.fast.yaml", "deploy/compose.fastlive.yaml",
    "deploy/compose.specialists.yaml", "deploy/bootstrap_fast_host.sh",
    "deploy/Dockerfile.fastpatch", "deploy/Dockerfile.specialists",
    "deploy/requirements-specialist-runtime.txt", "configs/fast-specialists.example.json",
    "configs/training/specialists.example.json", "configs/training/specialists-evaluate.example.json",
    "configs/training/specialist-lora.example.json", "configs/training/specialist-temporal.example.json",
    "configs/training/specialists-public-train.example.json", "training/configs/public-crop-lora.example.json",
    "training/requirements-specialists.txt", "training/requirements-specialist-hands.txt",
    "training/specialists/requirements.txt", VERIFIED_FAST_CONFIGURATION,
    *FAST_MEASUREMENT_REFERENCES, *PUBLIC_TOOL_FILES, *PUBLIC_WORKFLOW_FILES,
)


def _regular_owned_file(path: Path, root: Path) -> bool:
    """Reject linked files/ancestors, including Windows directory junctions."""
    try:
        if not path.is_file():
            return False
        relative = path.relative_to(root)
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
                return False
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def files_for_bundle(root: Path):
    for name in sorted(ROOT_FILES):
        path = root / name
        if _regular_owned_file(path, root):
            yield path
    for name in sorted(ROOT_DIRECTORIES):
        directory = root / name
        if (not directory.is_dir() or directory.is_symlink()
                or (hasattr(directory, "is_junction") and directory.is_junction())):
            continue
        # Prune private/runtime trees before traversal rather than visiting
        # every checkpoint, cache or annotation record to reject it later.
        candidates = []
        for current, subdirectories, filenames in directory.walk(top_down=True, follow_symlinks=False):
            subdirectories[:] = sorted(child for child in subdirectories
                if child.lower() not in DENIED_DIRECTORIES
                and (current / child).relative_to(root).parts[:2] != ("infra", "quotes")
                and not (current / child).is_symlink()
                and not (hasattr(current / child, "is_junction") and (current / child).is_junction()))
            candidates.extend(current / filename for filename in filenames)
        for path in sorted(candidates):
            relative = path.relative_to(root)
            if (any(part.lower() in DENIED_DIRECTORIES for part in relative.parts)
                    or relative.parts[:2] == ("infra", "quotes")
                    or path.suffix.lower() in DENIED_SUFFIXES
                    or (path.name.lower().startswith(".env") and path.name.lower() != ".env.example")
                    or path.name.lower() in {"api_keys", "api_key", "db_password", "credentials.json"}
                    or path.name == "resources.json"
                    or path.name == "spending-policy.yaml"
                    or path.name in {"brev-quotes-20261017.json", "stop-extension-proposal-20261017.json"}
                    or path.name in {"input-selection-review.json", "input-selection-aggregate.json"}
                    or not _regular_owned_file(path, root)):
                continue
            # Public annotation definitions are needed at runtime; local
            # annotation records are intentionally outside the archive.
            yield path
    for name in (*PUBLIC_TOOL_FILES, *PUBLIC_WORKFLOW_FILES):
        path = root / name
        if _regular_owned_file(path, root):
            yield path
    annotations = root / "annotations"
    for path in sorted(annotations.glob("*.py")):
        if _regular_owned_file(path, root):
            yield path
    for name in ("taxonomy.v1.json", "schema.v1.json", "machine.schema.json", "README.md", "REMOTE_REVIEW.md", "FETCH_REVIEW.md", "REVIEW_FORMS.md", "AUTOMATED_FIRST_PASS.md"):
        path = annotations / name
        if _regular_owned_file(path, root):
            yield path
    for path in sorted((annotations / "ui").glob("*")):
        if _regular_owned_file(path, root) and path.suffix in {".html", ".css", ".js", ".mjs"}:
            yield path
    # Include only the generic data-audit methodology, never its records.
    for name in ("data/manifests/README.md", "data/splits/README.md"):
        path = root / name
        if _regular_owned_file(path, root):
            yield path


def build(root: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = list(files_for_bundle(root))
    captured_configuration: dict[str, str] = {}
    verified_config_contents: bytes | None = None
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in entries:
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            contents = path.read_bytes()
            bundle.writestr(info, contents)
            if name in CONFIGURATION_FILES or name.startswith("src/forte/fast/"):
                captured_configuration[name] = sha256(contents).hexdigest()
            if name == VERIFIED_FAST_CONFIGURATION:
                verified_config_contents = contents
    payload = destination.read_bytes()
    report = {
        "status": "source_operator_bundle_only",
        "file_count": len(entries),
        "archive_bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
        "private_media_included": False,
        "weights_included": False,
        "artifact_establishes_public_deployment": False,
        "configuration_sha256": dict(sorted(captured_configuration.items())),
        "fast_profile_binding": {
            "verified_configuration": ({
                "path": VERIFIED_FAST_CONFIGURATION,
                "file_sha256": captured_configuration[VERIFIED_FAST_CONFIGURATION],
                "canonical_sha256": sha256(json.dumps(
                    json.loads(verified_config_contents), sort_keys=True, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            } if verified_config_contents is not None else None),
            "packaged_fast_source_sha256": {name: value for name, value in sorted(captured_configuration.items())
                                            if name.startswith("src/forte/fast/")},
            "measurement_references_sha256": {name: captured_configuration[name] for name in FAST_MEASUREMENT_REFERENCES
                                             if name in captured_configuration},
            "specialist_configuration_scope": "example_config_binding_only; no_model_weights_or_private_directory_contents",
            "scope": "exact_source_and_config_bytes_only; referenced_measurements_do_not_establish_bundle_deployment",
        },
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("dist/forte-release-source.zip"))
    args = parser.parse_args()
    report = build(args.root.resolve(), args.output.resolve())
    print(json.dumps({key: value for key, value in report.items() if key != "configuration_sha256"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
