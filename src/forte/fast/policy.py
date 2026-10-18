"""Bind a video-type promotion to the actual analyzer decision configuration."""
from __future__ import annotations

import hashlib
import json


def analyzer_policy_sha256(config: dict) -> str:
    from forte.fast.analyzer import DEFAULT_CONFIG

    normalized = {**DEFAULT_CONFIG, **config}
    specialists = dict(normalized.get("specialists") or {})
    # Checkpoint bytes are separately bound; the class gate is replaced by the
    # video gate after acceptance. Paths must not create a promotion cycle.
    specialists.pop("checkpoint", None)
    specialists.pop("gate_path", None)
    normalized["specialists"] = specialists
    raw = json.dumps({"schema": "forte.fast.policy.v1", "config": normalized},
                     sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
