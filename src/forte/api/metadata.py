"""Narrow, deterministic UI projection of a completed local worker audit.

The audit is private: only reviewed catalog strings and fixed limitation
translations may leave this module. Model notes, descriptions, paths, raw
predictions, support IDs, scores and audit provenance are never returned.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from forte.shared.schemas import CONTRACT_VERSION, VerifiedEvent

from .config import Settings

MAX_PRIVATE_AUDIT_BYTES = 16 * 1024 * 1024
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_LIMIT_TEXT = {
    "unclear_visual_context": "В отдельных фрагментах видимость или контекст недостаточны для уверенной проверки.",
    "unclear_visual_context_irrelevant_to_refuted_rules": "Часть визуальных признаков осталась неясной; необходимые условия рассматриваемых нарушений опровергнуты другими подтвержденными признаками в том же фрагменте.",
    "required_visual_facts_not_established": "Для части возможных событий не подтверждены все необходимые визуальные признаки.",
    "longer_procedure_context_required": "Для проверки отдельных действий требуется более полный контекст процедуры.",
    "no_verified_applicable_rule": "Для части наблюдений не установлена применимая проверенная норма; они не включены в подтвержденные события.",
    "unresolved_coarse_candidate": "Часть предварительных наблюдений осталась неподтвержденной и требует ручной проверки.",
    "unresolved_refined_candidate": "Часть наблюдений после дополнительной проверки осталась неподтвержденной.",
    "candidate_requires_explicit_confirmation_or_refutation": "Для части предварительных наблюдений требуется явное подтверждение или опровержение человеком.",
    "evidence_support_exceeds_frame_budget": "Для части возможных событий недостаточно допустимого набора доказательных кадров.",
    "missing_source_anchor": "Для части возможных событий не установлен надежный исходный кадр.",
    "invalid_or_conflicting_visual_support": "Часть визуальных признаков противоречива или недостаточно подтверждена.",
    "conflicting_candidate_id": "Некоторые предварительные наблюдения противоречат друг другу и требуют ручной проверки.",
    "candidate_outside_analyzed_context": "Часть предварительных наблюдений выходит за пределы подтвержденного контекста.",
    "candidate_clip_mismatch": "Для части предварительных наблюдений не подтверждена связь с проверенным фрагментом.",
    "unknown_event_type": "Некоторые предварительные наблюдения не относятся к проверяемым типам событий.",
    "missing_affirmative_visual_support": "Для части возможных событий отсутствуют необходимые утвердительные визуальные признаки.",
    "temporal_conditions_not_encoded": "Для части возможных событий временные условия нормы не подтверждены.",
    "exceptions_not_resolved": "Для части возможных событий не исключены предусмотренные нормой исключения.",
    "omission_anchor:last_supported_frame_at_procedure_end": "Для предполагаемого пропуска действия указан последний подтвержденный кадр соответствующего этапа процедуры.",
}


class _ProjectionModel(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class _Coverage(_ProjectionModel):
    complete: Annotated[bool, Field(strict=True)]
    policy: Annotated[str, Field(max_length=128)] = ""


class _Evidence(_ProjectionModel):
    event_count: Annotated[int, Field(strict=True, ge=0, le=256)]


class _Audit(_ProjectionModel):
    schema_version: Literal["forte.local-proposal.v1"]
    phase: Literal["done"]
    publication_ready: Annotated[bool, Field(strict=True)]
    analysis_version: str
    catalog_version: Annotated[str, Field(min_length=1, max_length=256)]
    job_id: Annotated[str, Field(min_length=1, max_length=128)]
    attempt: Annotated[int, Field(strict=True, ge=1)]
    coverage: _Coverage
    evidence: _Evidence
    events: Annotated[list[VerifiedEvent], Field(max_length=256)]
    limitations: Annotated[list[Annotated[str, Field(max_length=2048)]], Field(max_length=256)] = Field(default_factory=list)


def _get(record: Any, key: str, default: Any = None) -> Any:
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


@lru_cache(maxsize=4)
def _catalog_at_version(path: str, modified_ns: int, size: int):
    # Only operator-configured catalog paths reach here. Its own loader checks
    # archived source checksums, official hosts and reviewed fact bindings.
    from rules.catalog import load_catalog

    return load_catalog(path, verify_archives=True)


def _read_audit(path: Path, root: Path) -> dict[str, Any] | None:
    resolved = path.resolve(strict=True)
    # Reject symlink/junction redirection even to another path inside the data
    # root. The only permitted filename is constructed by this module.
    if not resolved.is_relative_to(root) or resolved != path.absolute():
        return None
    with resolved.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > MAX_PRIVATE_AUDIT_BYTES:
            return None
        raw = stream.read(MAX_PRIVATE_AUDIT_BYTES + 1)
    if len(raw) > MAX_PRIVATE_AUDIT_BYTES:
        return None
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else None


def _limits(values: list[str], approved: set[str]) -> list[str]:
    # A free-form visibility note can be present in VerifiedEvent.limitations.
    # It cannot be echoed, translated or interpreted as a reviewed observation.
    return list(dict.fromkeys(
        _LIMIT_TEXT[value] if value in _LIMIT_TEXT else value
        for value in values if value in _LIMIT_TEXT or value in approved
    ))


def completed_ui_metadata(config: Settings, record: Any) -> dict[str, Any]:
    """Return safe enrichment, or empty enrichment for any unusable audit.

    A final audit is written from the same ordered event tuple passed to
    evidence assembly. Its fenced attempt and final evidence event count bind
    that order to the completed job without rereading its large base64 payload.
    """
    empty: dict[str, Any] = {"limitations": [], "events": []}
    if _get(record, "status") != "done":
        return empty
    job_id, attempt = _get(record, "job_id"), _get(record, "attempt")
    if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
        return empty
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        return empty
    root = config.data_dir.resolve()
    try:
        from forte.fast.service import ui_metadata as _fast_ui_metadata
        fast = _fast_ui_metadata(root, job_id, attempt)
        if fast is not None:
            return fast
    except Exception:
        pass
    path = root / "work" / job_id / f"attempt-{attempt}" / "analysis.private.json"
    try:
        payload = _read_audit(path, root)
        if payload is None:
            return empty
        audit = _Audit.model_validate(payload)
        if (audit.job_id != job_id or audit.attempt != attempt
                or audit.analysis_version != CONTRACT_VERSION or not audit.publication_ready
                or not audit.coverage.complete or audit.evidence.event_count != len(audit.events)):
            return empty
        if len({event.event_id for event in audit.events}) != len(audit.events):
            return empty
        if "verified_events" in payload and payload["verified_events"] != payload["events"]:
            return empty
        info = config.rules_path.stat()
        if info.st_size > 1024 * 1024:
            return empty
        catalog = _catalog_at_version(str(config.rules_path), info.st_mtime_ns, info.st_size)
        if catalog.version != audit.catalog_version:
            return empty
        approved_list = catalog.payload["approved_limitations_ru"]
        approved = set(approved_list)
        global_limits = _limits(audit.limitations, approved)
        # These catalog phrases describe the scope of every result. The third
        # statement refers to a factual sign and is only applicable to positives.
        global_limits.extend(approved_list if audit.events else approved_list[:2])
        if audit.coverage.policy == "overlapping_sparse_windows":
            global_limits.append("Кадры анализируются с временным шагом; короткие действия между ними могут остаться незамеченными.")
        events = []
        for event in audit.events:
            rule = catalog.get_rule(event.rule_id, require_enabled=True)
            binding = catalog.bindings[rule.rule_id]
            if binding["event_type_id"] != event.event_type_id:
                return empty
            states = {fact.fact_id: fact for fact in event.observed_facts}
            required = binding["required_fact_states"]
            if len(states) != len(event.observed_facts) or any(
                fact_id not in states or states[fact_id].state != state or not states[fact_id].supporting_frame_ids
                for fact_id, state in required.items()
            ):
                return empty
            selected = binding.get("description_fact_ids", list(required))
            phrases = [catalog.fact_description(fact_id, required[fact_id]) for fact_id in selected]
            if urlparse(rule.source_url).scheme != "https":
                return empty
            rule_reference = {"title": rule.title, "url": rule.source_url}
            if rule.clause:
                rule_reference["clause"] = rule.clause
            event_limits = _limits(event.limitations + audit.limitations, approved)
            global_limits.extend(event_limits)
            events.append({
                "rule": rule_reference,
                "observed_facts": list(dict.fromkeys(phrases)),
                "limitations": event_limits,
            })
        return {"limitations": list(dict.fromkeys(global_limits)), "events": events}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, ValidationError):
        # Metadata is optional. An absent/stale/corrupt audit never changes the
        # canonical immutable result and never becomes a public diagnostic.
        return empty
