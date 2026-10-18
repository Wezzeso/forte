"""Deterministic temporal grouping and fail-closed visual rule application."""

from .builder import (
    EventBuildConfig,
    EventBuildResult,
    RefinementRequest,
    UnresolvedInterval,
    RuleRelevance,
    analyze_rule_relevance,
    build_events,
)

__all__ = [
    "EventBuildConfig", "EventBuildResult", "RefinementRequest",
    "UnresolvedInterval", "RuleRelevance", "analyze_rule_relevance", "build_events",
]
