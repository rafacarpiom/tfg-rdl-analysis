
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = _jsonable(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return _jsonable_scalar(value)


@dataclass
class EvidenceItem:
    # rep_index: índice interno 0-based dentro de detector_result["rep_results"].
    # user_rep_order: número humano 1-based de repetición.
    # user_rep_raw_index: índice raw/original de segmentación (si existe).
    detector: str
    error_code: str
    rep_index: int | None
    user_rep_raw_index: int | None
    user_rep_order: int | None
    phase: str
    anchors: list[str]
    frames: list[int]
    severity: str
    score: float
    detected: bool
    # segment: identificador de segmento ("ecc_25", "ecc_50", ...) o None si la evidencia es solo por rep.
    segment: str | None = None
    confidence: float | None = None
    location_label: str = "unknown"
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvidenceResult:
    exercise: str
    evidence_items: list[EvidenceItem]
    summary: dict[str, Any]
    warnings: list[str]


def evidence_item_to_dict(item: EvidenceItem) -> dict[str, Any]:
    return {
        "detector": str(item.detector),
        "error_code": str(item.error_code),
        "rep_index": item.rep_index,
        "user_rep_raw_index": item.user_rep_raw_index,
        "user_rep_order": item.user_rep_order,
        "segment": item.segment,
        "phase": str(item.phase),
        "anchors": [str(a) for a in item.anchors],
        "frames": [int(f) for f in item.frames],
        "severity": str(item.severity),
        "score": float(item.score),
        "detected": bool(item.detected),
        "confidence": None if item.confidence is None else float(item.confidence),
        "location_label": str(item.location_label),
        "summary_metrics": _jsonable(item.summary_metrics),
        "source": _jsonable(item.source),
        "warnings": [str(w) for w in item.warnings],
    }


def evidence_result_to_dict(result: EvidenceResult) -> dict[str, Any]:
    return {
        "exercise": str(result.exercise),
        "evidence_items": [evidence_item_to_dict(item) for item in result.evidence_items],
        "summary": _jsonable(result.summary),
        "warnings": [str(w) for w in result.warnings],
    }
