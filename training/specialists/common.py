"""Small shared guards; failures expose codes rather than private paths/labels."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from training.guards import (HEX64, TrainingGateError, canonical_sha256,
                             file_sha256, load_json, require)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX64.fullmatch(value))


def number(value: Any, minimum: float = 0, maximum: float = float("inf")) -> bool:
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and minimum <= value <= maximum)


def relative_file(root: Path, value: Any) -> Path:
    require(isinstance(value, str) and bool(value) and "://" not in value, "specialist_path_invalid")
    candidate = (root / value).resolve()
    require(candidate.is_relative_to(root.resolve()), "specialist_path_outside_root")
    require(candidate.is_file(), "specialist_file_missing")
    return candidate


def pinned_json(path: Path, expected: str) -> dict:
    require(sha(expected), "specialist_expected_hash_required")
    require(path.is_file() and file_sha256(path) == expected, "specialist_input_hash_mismatch")
    value = load_json(path)
    require(isinstance(value, dict), "specialist_json_object_required")
    return value


def cli_error(exc: Exception) -> int:
    # Never print third-party exception strings, which may contain private data.
    code = str(exc) if isinstance(exc, TrainingGateError) else "specialist_execution_failed"
    print(json.dumps({"ok": False, "error": code}))
    return 2
