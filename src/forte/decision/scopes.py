"""Source-bound state transitions; no observation or absence is inferred here."""

from collections.abc import Collection, Mapping
from math import isfinite

from forte.shared.schemas import FactDecision


def candidate_local_transition(
    previous: FactDecision, local: FactDecision, *, start_sec: float, end_sec: float,
    frame_times: Mapping[str, float], candidate_frame_ids: Collection[str],
) -> bool:
    """Allow an explicit opposite fact only in a nonoverlapping source scope.

    All local supports must be cited inside the candidate. The entire previous
    support span must be disjoint, including frames sharing a boundary PTS. An
    unclear or unsupported fact cannot manufacture a decisive observation.
    """
    if (previous.fact_id != local.fact_id
            or {previous.state, local.state} != {"present", "absent"}
            or not previous.supporting_frame_ids or not local.supporting_frame_ids
            or not isfinite(start_sec) or not isfinite(end_sec) or start_sec > end_sec):
        return False
    before, inside = set(previous.supporting_frame_ids), set(local.supporting_frame_ids)
    if (len(before) != len(previous.supporting_frame_ids) or len(inside) != len(local.supporting_frame_ids)
            or before & inside or not inside <= set(candidate_frame_ids)
            or not (before | inside) <= frame_times.keys()):
        return False
    previous_times = [frame_times[key] for key in before]
    return (all(isfinite(frame_times[key]) and start_sec <= frame_times[key] <= end_sec for key in inside)
            and all(isfinite(value) for value in previous_times)
            and (max(previous_times) < start_sec or min(previous_times) > end_sec))
