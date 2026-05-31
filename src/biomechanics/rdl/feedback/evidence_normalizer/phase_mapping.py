
from __future__ import annotations

from collections import Counter

from .constants import PHASE_BOTTOM, PHASE_CONCENTRIC, PHASE_ECCENTRIC, PHASE_FULL_REP, PHASE_LOCKOUT, PHASE_UNKNOWN


def phase_from_anchor(anchor: str) -> str:
    a = str(anchor or "").strip().lower()
    if a in {"ecc_0", "ecc_25", "ecc_50", "ecc_75"}:
        return PHASE_ECCENTRIC
    if a in {"ecc_100", "bottom", "con_0"}:
        return PHASE_BOTTOM
    if a in {"con_25", "con_50", "con_75"}:
        return PHASE_CONCENTRIC
    if a in {"con_100", "end"}:
        return PHASE_LOCKOUT
    return PHASE_UNKNOWN


def phase_from_anchors(anchors: list[str]) -> str:
    phases = [phase_from_anchor(a) for a in anchors if a]
    phases = [p for p in phases if p != PHASE_UNKNOWN]
    if not phases:
        return PHASE_UNKNOWN
    uniq = set(phases)
    if len(uniq) == 1:
        return phases[0]
    if uniq.issubset({PHASE_ECCENTRIC, PHASE_BOTTOM}):
        return PHASE_ECCENTRIC
    if uniq.issubset({PHASE_BOTTOM, PHASE_CONCENTRIC}):
        return PHASE_CONCENTRIC
    return PHASE_FULL_REP


def location_label_from_anchors(anchors: list[str]) -> str:
    clean = [str(a).strip().lower() for a in anchors if str(a).strip()]
    if not clean:
        return "unknown"
    uniq = set(clean)
    if len(uniq) == 1:
        only = next(iter(uniq))
        single_map = {
            "ecc_0": "eccentric_start",
            "ecc_25": "eccentric_early",
            "ecc_50": "eccentric_mid",
            "ecc_75": "eccentric_late",
            "ecc_100": "bottom",
            "bottom": "bottom",
            "con_0": "bottom",
            "con_25": "concentric_early",
            "con_50": "concentric_mid",
            "con_75": "concentric_late",
            "con_100": "lockout",
            "end": "lockout",
        }
        mapped = single_map.get(only)
        if mapped:
            return mapped
    if uniq.issubset({"ecc_0", "ecc_25"}) and {"ecc_0", "ecc_25"} & uniq:
        return "eccentric_early"
    if uniq.issubset({"ecc_25", "ecc_50"}) and {"ecc_25", "ecc_50"} & uniq:
        return "eccentric_mid"
    if uniq.issubset({"ecc_50", "ecc_75"}) and {"ecc_50", "ecc_75"} & uniq:
        return "eccentric_mid_late"
    if uniq.issubset({"ecc_75", "ecc_100", "bottom"}) and ("ecc_75" in uniq) and (("ecc_100" in uniq) or ("bottom" in uniq)):
        return "eccentric_late_bottom"
    if ("ecc_0" in uniq) and ("bottom" in uniq):
        return "eccentric_range"
    if uniq.issubset({"con_25", "con_50"}) and {"con_25", "con_50"} & uniq:
        return "concentric_mid"
    if uniq.issubset({"con_75", "con_100"}) and ("con_75" in uniq) and ("con_100" in uniq):
        return "concentric_late_lockout"
    if len(clean) == 1 and clean[0] in {"con_100", "end"}:
        return "lockout"
    phase = phase_from_anchors(clean)
    if phase == PHASE_ECCENTRIC:
        if any(a in {"ecc_100", "bottom"} for a in clean):
            if any(a in {"ecc_75", "ecc_50"} for a in clean):
                return "eccentric_late_bottom"
            return "eccentric_range"
        if set(clean).issubset({"ecc_25", "ecc_50"}):
            return "eccentric_mid"
        if set(clean).issubset({"ecc_75", "ecc_50"}):
            return "eccentric_late"
        if "ecc_0" in clean:
            return "eccentric_range"
        return "eccentric"
    if phase == PHASE_BOTTOM:
        return "bottom"
    if phase == PHASE_CONCENTRIC:
        if any(a in {"con_75", "con_100", "end"} for a in clean):
            return "concentric_late"
        return "concentric"
    if phase == PHASE_LOCKOUT:
        return "lockout"
    # Reserva determinista por la fase más frecuente.
    counts = Counter(phase_from_anchor(a) for a in clean)
    top = counts.most_common(1)[0][0] if counts else PHASE_UNKNOWN
    return top if top != PHASE_UNKNOWN else "unknown"
