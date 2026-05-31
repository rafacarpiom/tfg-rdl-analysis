
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
class AggregatedIssue:
    error_code: str
    detector: str
    severity: str
    priority_bucket: str
    reps: list[int]
    rep_raw_indices: list[int]
    phases: list[str]
    location_labels: list[str]
    anchors: list[str]
    frames: list[int]
    max_score: float
    mean_score: float
    max_confidence: float | None
    mean_confidence: float | None
    count: int
    affected_rep_count: int
    summary_metrics: dict[str, Any]
    evidence_refs: list[dict[str, Any]]
    warnings: list[str]


@dataclass
class RepAggregation:
    user_rep_order: int
    user_rep_raw_index: int | None
    issues: list[AggregatedIssue]
    phases: dict[str, list[AggregatedIssue]]
    max_severity: str
    num_issues: int


@dataclass
class PhaseAggregation:
    phase: str
    issues: list[AggregatedIssue]
    max_severity: str
    num_issues: int


@dataclass
class FeedbackAggregationResult:
    exercise: str
    status: str
    issues: list[AggregatedIssue]
    primary_focus: list[AggregatedIssue]
    secondary_focus: list[AggregatedIssue]
    observations: list[AggregatedIssue]
    by_rep: list[RepAggregation]
    by_phase: list[PhaseAggregation]
    summary: dict[str, Any]
    warnings: list[str]


def aggregated_issue_to_dict(issue: AggregatedIssue) -> dict[str, Any]:
    return {
        "error_code": issue.error_code,
        "detector": issue.detector,
        "severity": issue.severity,
        "priority_bucket": issue.priority_bucket,
        "reps": [int(x) for x in issue.reps],
        "rep_raw_indices": [int(x) for x in issue.rep_raw_indices],
        "phases": [str(x) for x in issue.phases],
        "location_labels": [str(x) for x in issue.location_labels],
        "anchors": [str(x) for x in issue.anchors],
        "frames": [int(x) for x in issue.frames],
        "max_score": float(issue.max_score),
        "mean_score": float(issue.mean_score),
        "max_confidence": None if issue.max_confidence is None else float(issue.max_confidence),
        "mean_confidence": None if issue.mean_confidence is None else float(issue.mean_confidence),
        "count": int(issue.count),
        "affected_rep_count": int(issue.affected_rep_count),
        "summary_metrics": _jsonable(issue.summary_metrics),
        "evidence_refs": _jsonable(issue.evidence_refs),
        "warnings": [str(w) for w in issue.warnings],
    }


def rep_aggregation_to_dict(rep: RepAggregation) -> dict[str, Any]:
    return {
        "user_rep_order": int(rep.user_rep_order),
        "user_rep_raw_index": None if rep.user_rep_raw_index is None else int(rep.user_rep_raw_index),
        "issues": [aggregated_issue_to_dict(i) for i in rep.issues],
        "phases": {str(k): [aggregated_issue_to_dict(i) for i in v] for k, v in rep.phases.items()},
        "max_severity": rep.max_severity,
        "num_issues": int(rep.num_issues),
    }


def phase_aggregation_to_dict(phase: PhaseAggregation) -> dict[str, Any]:
    return {
        "phase": phase.phase,
        "issues": [aggregated_issue_to_dict(i) for i in phase.issues],
        "max_severity": phase.max_severity,
        "num_issues": int(phase.num_issues),
    }


def aggregation_result_to_dict(result: FeedbackAggregationResult) -> dict[str, Any]:
    return {
        "exercise": result.exercise,
        "status": result.status,
        "issues": [aggregated_issue_to_dict(i) for i in result.issues],
        "primary_focus": [aggregated_issue_to_dict(i) for i in result.primary_focus],
        "secondary_focus": [aggregated_issue_to_dict(i) for i in result.secondary_focus],
        "observations": [aggregated_issue_to_dict(i) for i in result.observations],
        "by_rep": [rep_aggregation_to_dict(r) for r in result.by_rep],
        "by_phase": [phase_aggregation_to_dict(p) for p in result.by_phase],
        "summary": _jsonable(result.summary),
        "warnings": [str(w) for w in result.warnings],
    }
