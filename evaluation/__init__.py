"""Local, failure-inclusive evaluation. Reports contain aggregate diagnostics only."""

from .audit import audit_split, records_digest
from .metrics import evaluate, match_events
from .selection import select_candidate

__all__ = ["audit_split", "records_digest", "evaluate", "match_events", "select_candidate"]
