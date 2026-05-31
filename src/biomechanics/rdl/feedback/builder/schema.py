
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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return _jsonable_scalar(value)


@dataclass
class FeedbackItem:
    priority: str
    bucket: str
    error_code: str
    detector: str
    severity: str
    title: str
    where: str
    what_happens: str
    why_it_matters: str
    how_to_fix: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class RepFeedback:
    user_rep_order: int
    max_severity: str
    primary_errors: list[FeedbackItem] = field(default_factory=list)
    secondary_errors: list[FeedbackItem] = field(default_factory=list)
    observations: list[FeedbackItem] = field(default_factory=list)


# Alias retrocompatible usado por imports del __init__.py del paquete.
@dataclass
class PerRepFeedbackSummary:
    user_rep_order: int
    max_severity: str
    issues: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FeedbackReport:
    exercise: str
    status: str
    summary: dict[str, Any]
    headline: str
    rep_feedback: list[RepFeedback]
    plain_text: str = ""
    warnings: list[str] = field(default_factory=list)


def feedback_item_to_dict(item: FeedbackItem) -> dict[str, Any]:
    return {
        "priority": str(item.priority),
        "bucket": str(item.bucket),
        "error_code": str(item.error_code),
        "detector": str(item.detector),
        "severity": str(item.severity),
        "title": str(item.title),
        "where": str(item.where),
        "what_happens": str(item.what_happens),
        "why_it_matters": str(item.why_it_matters),
        "how_to_fix": str(item.how_to_fix),
        "warnings": [str(w) for w in item.warnings],
    }


def rep_feedback_to_dict(rep: RepFeedback) -> dict[str, Any]:
    return {
        "user_rep_order": int(rep.user_rep_order),
        "max_severity": str(rep.max_severity),
        "primary_errors": [feedback_item_to_dict(i) for i in rep.primary_errors],
        "secondary_errors": [feedback_item_to_dict(i) for i in rep.secondary_errors],
        "observations": [feedback_item_to_dict(i) for i in rep.observations],
    }


def feedback_report_to_dict(report: FeedbackReport) -> dict[str, Any]:
    return {
        "exercise": str(report.exercise),
        "status": str(report.status),
        "summary": _jsonable(report.summary),
        "headline": str(report.headline),
        "rep_feedback": [rep_feedback_to_dict(r) for r in report.rep_feedback],
        "plain_text": str(report.plain_text),
        "warnings": [str(w) for w in report.warnings],
    }
