"""Shared source-only archive policy; private inputs travel separately.

Path filtering is a defence in depth, not de-identification. Package callers
select explicit public source roots and never select private artifact roots.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from typing import Callable


DENIED_PARTS = frozenset({
    ".git", ".venv", "node_modules", "__pycache__", "private", "secrets",
    "reference-captures", "videos", "uploads", "results", "evidence", "frames",
    "labels", "observations", "raw", "raw_outputs", "raw-outputs", "backups",
    "backup", "splits", "selections", "checkpoints", "weights", "runs", "cache",
    "public-snapshot", "upstream-source", "speed-research-v031", "qa", "screenshots", "test-results",
    "keys", "credentials", "sessions", ".cache",
})
DENIED_SUFFIXES = frozenset({
    ".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg",
    ".wav", ".mp3", ".flac", ".ogg", ".xlsx", ".xls", ".csv", ".tsv",
    ".jsonl", ".ndjson", ".safetensors", ".bin", ".pt", ".pth", ".gguf",
    ".onnx", ".npy", ".npz", ".pkl", ".pickle", ".parquet", ".arrow",
    ".key", ".pem", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db", ".sql",
    ".pyc", ".tar", ".gz", ".zip", ".7z", ".bak", ".backup", ".old",
    ".docx", ".pdf",
})
SOURCE_SUFFIXES = frozenset({
    ".py", ".json", ".txt", ".md", ".yaml", ".yml", ".toml", ".lock",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".css", ".scss", ".html", ".svg",
    ".jinja", ".j2", ".gitignore", ".dockerignore", ".gitattributes", ".example",
    ".ttf", ".woff", ".woff2", ".otf", ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".sh", ".ps1", ".bat",
})
PUBLIC_CODE_ARTIFACT_PARTS = frozenset({"evidence", "frames", "observations", "results", "cache"})
_PRIVATE_ARTIFACT = re.compile(
    r"(?:^|[._-])(?:selection|selections|split|splits|labels|observations|"
    r"raw[-_]outputs?|machine[-_]annotations?|workbook[-_]rows|review[-_]queue|"
    r"provided[-_]binary[-_]labels|user[-_]coarse[-_]anchors|model[-_]exposure|"
    r"backup|backups|api[-_]?keys?|credentials|password|access[-_]token|"
    r"id[-_](?:rsa|ed25519))(?:[._-]|$)", re.IGNORECASE)


def public_file(path: Path, root: Path) -> bool:
    """Reject links, traversals, private artifacts and non-example environments."""
    root = root.resolve()
    try:
        relative = path.relative_to(root)
        if not relative.parts or any(part in {".", ".."} for part in relative.parts):
            return False
        source_code = relative.parts[:2] == ("src", "forte") and path.suffix.lower() == ".py"
        if any(part.lower() in DENIED_PARTS and not (
                source_code and part.lower() in PUBLIC_CODE_ARTIFACT_PARTS) for part in relative.parts):
            return False
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
                return False
        path.resolve().relative_to(root)
    except (ValueError, OSError, RuntimeError):
        return False
    name = path.name.lower()
    code = path.suffix.lower() in {".py", ".js", ".mjs", ".ts", ".tsx", ".jsx"}
    if path.suffix.lower() in DENIED_SUFFIXES or (not code and _PRIVATE_ARTIFACT.search(name)):
        return False
    if name.startswith(("id_rsa", "id_ed25519")):
        return False
    if path.suffix.lower() not in SOURCE_SUFFIXES and not (
            name.startswith("dockerfile") or name in {
                "caddyfile", "license", "notice", ".gitignore", ".dockerignore", ".gitattributes"}):
        return False
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        public_assets = (("brand", "assets"), ("web", "public"), ("web", "src", "assets"))
        if not any(relative.parts[:len(prefix)] == prefix for prefix in public_assets):
            return False
    if name.startswith(".env") and name != ".env.example":
        return False
    if name.endswith((".local.env", ".env")) or name in {"api_keys", "postgres_password"}:
        return False
    if not code and name.endswith((".bound.json", ".local.json")):
        return False
    return path.is_file()


def capture_stable(root: Path, select: Callable[[Path], list[Path]], attempts: int = 3) -> dict[str, bytes]:
    """Double-read the selected tree and reject concurrent edits or membership changes."""
    def snapshot() -> tuple[list[Path], dict[str, tuple[int, int]]]:
        paths = select(root)
        for path in paths:
            if not public_file(path, root):
                raise ValueError("nonpublic_or_missing_source_path")
        return paths, {path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
                       for path in paths}

    for _ in range(attempts):
        paths, before = snapshot()
        captured = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
        _, after = snapshot()
        if before != after:
            continue
        if any(path.read_bytes() != captured[path.relative_to(root).as_posix()] for path in paths):
            continue
        _, final = snapshot()
        if before == final:
            return dict(sorted(captured.items()))
    raise RuntimeError("public_source_changed_during_capture")


def write_package(captured: dict[str, bytes], output: Path, *, schema: str,
                  metadata: dict | None = None) -> dict:
    """Create an immutable deterministic gzip/tar and bind every member's bytes."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("public_package_output_already_exists")
    if not captured or any(not safe_member_name(name) for name in captured):
        raise ValueError("public_package_member_name_invalid")
    bindings = {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(captured.items())}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".source-package-", delete=False) as staging:
            temporary = Path(staging.name)
            with gzip.GzipFile(filename="", fileobj=staging, mode="wb", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name, raw in sorted(captured.items()):
                        member = tarfile.TarInfo(name)
                        member.size, member.mode, member.mtime = len(raw), 0o644, 0
                        archive.addfile(member, io.BytesIO(raw))
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    try:
        with output.open("xb") as target:
            target.write(temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)
    result = {"schema_version": schema, "file_count": len(captured),
              "archive_bytes": output.stat().st_size,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "stable_source_capture_verified": True, "private_data_included": False,
              "files": bindings, **(metadata or {})}
    with sidecar.open("x", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (isinstance(name, str) and bool(name) and not path.is_absolute()
            and "\\" not in name and ":" not in name and "\x00" not in name
            and path.as_posix() == name and all(part not in {".", ".."} for part in path.parts))


def verify_package(output: Path) -> dict:
    """Independently bind tar member bytes to its sidecar without extracting."""
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    receipt = json.loads(sidecar.read_text(encoding="utf-8"))
    raw = output.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt.get("sha256") or len(raw) != receipt.get("archive_bytes"):
        raise ValueError("public_package_archive_hash_mismatch")
    expected, observed = receipt.get("files"), {}
    if not isinstance(expected, dict):
        raise ValueError("public_package_manifest_invalid")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            if not safe_member_name(member.name) or not member.isfile() or member.name in observed:
                raise ValueError("public_package_member_invalid")
            observed[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    if observed != expected or len(observed) != receipt.get("file_count"):
        raise ValueError("public_package_member_hash_mismatch")
    return {"sha256": receipt["sha256"], "file_count": len(observed), "verified": True}
