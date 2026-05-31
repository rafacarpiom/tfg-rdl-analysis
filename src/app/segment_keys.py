
from __future__ import annotations

CANONICAL_SEGMENT_IDS: tuple[str, ...] = (
    "ecc_0_to_ecc_25",
    "ecc_25_to_ecc_50",
    "ecc_50_to_ecc_75",
    "ecc_75_to_ecc_100",
)

CANONICAL_SEGMENT_LABELS: dict[str, str] = {
    "ecc_0_to_ecc_25": "0% → 25%",
    "ecc_25_to_ecc_50": "25% → 50%",
    "ecc_50_to_ecc_75": "50% → 75%",
    "ecc_75_to_ecc_100": "75% → 100%",
}

_ANCHOR_TO_CANONICAL: dict[str, str] = {
    "ecc_0": "ecc_0_to_ecc_25",
    "ecc_25": "ecc_0_to_ecc_25",
    "ecc_50": "ecc_25_to_ecc_50",
    "ecc_75": "ecc_50_to_ecc_75",
    "ecc_100": "ecc_75_to_ecc_100",
}

_ALIASES: dict[str, str] = {
    "ecc_75_to_bottom": "ecc_75_to_ecc_100",
    "ecc 0 to ecc 25": "ecc_0_to_ecc_25",
    "ecc 25 to ecc 50": "ecc_25_to_ecc_50",
    "ecc 50 to ecc 75": "ecc_50_to_ecc_75",
    "ecc 75 to bottom": "ecc_75_to_ecc_100",
}


def normalize_segment_key(segment_or_anchor: str) -> str:
    raw = str(segment_or_anchor).strip().replace(" ", "_")
    if raw in CANONICAL_SEGMENT_IDS:
        return raw
    if raw in _ANCHOR_TO_CANONICAL:
        return _ANCHOR_TO_CANONICAL[raw]
    if raw in _ALIASES:
        return _ALIASES[raw]
    return raw


def to_canonical_segment(segment_or_anchor: str) -> str | None:
    key = normalize_segment_key(segment_or_anchor)
    if key in CANONICAL_SEGMENT_IDS:
        return key
    return None
