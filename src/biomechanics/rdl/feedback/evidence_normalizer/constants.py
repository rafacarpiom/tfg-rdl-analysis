
from __future__ import annotations

PHASE_ECCENTRIC = "eccentric"
PHASE_BOTTOM = "bottom"
PHASE_CONCENTRIC = "concentric"
PHASE_LOCKOUT = "lockout"
PHASE_FULL_REP = "full_rep"
PHASE_UNKNOWN = "unknown"

SEVERITY_RANK: dict[str, int] = {
    "none": 0,
    "posible": 1,
    "leve": 2,
    "media": 3,
    "grave": 4,
}

KNOWN_PHASES: tuple[str, ...] = (
    PHASE_ECCENTRIC,
    PHASE_BOTTOM,
    PHASE_CONCENTRIC,
    PHASE_LOCKOUT,
    PHASE_FULL_REP,
    PHASE_UNKNOWN,
)
