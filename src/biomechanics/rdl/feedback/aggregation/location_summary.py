
from __future__ import annotations

from collections import Counter
from typing import Any

from src.biomechanics.rdl.feedback.evidence_normalizer.constants import PHASE_BOTTOM, PHASE_CONCENTRIC, PHASE_ECCENTRIC, PHASE_FULL_REP, PHASE_UNKNOWN

from .schema import AggregatedIssue

_LOCATION_ORDER = [
    "eccentric_start",
    "eccentric_early",
    "eccentric_mid",
    "eccentric_mid_late",
    "eccentric_late",
    "eccentric_late_bottom",
    "eccentric_range",
    "bottom",
    "concentric_early",
    "concentric_mid",
    "concentric_late",
    "concentric_late_lockout",
    "lockout",
    "full_rep",
    "unknown",
]


def merge_locations(location_labels: list[str]) -> list[str]:
    uniq = []
    seen = set()
    for raw in location_labels:
        loc = str(raw or "unknown")
        if loc in seen:
            continue
        seen.add(loc)
        uniq.append(loc)
    rank = {name: i for i, name in enumerate(_LOCATION_ORDER)}
    return sorted(uniq, key=lambda x: rank.get(x, len(rank)))


def dominant_phase(phases: list[str]) -> str:
    clean = [str(p or PHASE_UNKNOWN) for p in phases if str(p or "").strip()]
    if not clean:
        return PHASE_UNKNOWN
    uniq = set(clean)
    if len(uniq) == 1:
        return clean[0]
    if uniq.issubset({PHASE_ECCENTRIC, PHASE_BOTTOM}):
        return PHASE_ECCENTRIC
    if uniq.issubset({PHASE_BOTTOM, PHASE_CONCENTRIC}):
        return PHASE_CONCENTRIC
    if len(uniq) > 2:
        return PHASE_FULL_REP
    counts = Counter(clean)
    return counts.most_common(1)[0][0]


def location_summary_from_issue(issue: AggregatedIssue) -> dict[str, Any]:
    return {
        "dominant_phase": dominant_phase(issue.phases),
        "location_labels": merge_locations(issue.location_labels),
        "anchors": list(issue.anchors),
        "frames": list(issue.frames),
    }
