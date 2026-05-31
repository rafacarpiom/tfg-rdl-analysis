
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.biomechanics.rdl.feedback.evidence_normalizer.constants import PHASE_UNKNOWN

from .aggregation import aggregate_evidence_items
from .schema import (
    AggregatedIssue,
    FeedbackAggregationResult,
    PhaseAggregation,
    RepAggregation,
    aggregation_result_to_dict,
)
from .severity_grouping import max_severity, split_priority_buckets


def _issue_sort_key(issue: AggregatedIssue) -> tuple[int, int, float, float]:
    sev_rank = {"none": 0, "posible": 1, "leve": 2, "media": 3, "grave": 4}
    return (
        -sev_rank.get(issue.severity, 0),
        -issue.affected_rep_count,
        -issue.max_score,
        -issue.mean_score,
    )


def _force_observation_bucket_for_possible_spine(
    primary: list[AggregatedIssue],
    secondary: list[AggregatedIssue],
    observations: list[AggregatedIssue],
) -> tuple[list[AggregatedIssue], list[AggregatedIssue], list[AggregatedIssue]]:
    demoted: list[AggregatedIssue] = []
    kept_primary: list[AggregatedIssue] = []
    kept_secondary: list[AggregatedIssue] = []

    for issue in primary:
        if issue.error_code == "spine_flexion_possible":
            demoted.append(issue)
        else:
            kept_primary.append(issue)
    for issue in secondary:
        if issue.error_code == "spine_flexion_possible":
            demoted.append(issue)
        else:
            kept_secondary.append(issue)

    merged_observations = list(observations)
    for issue in demoted:
        if issue not in merged_observations:
            merged_observations.append(issue)

    return (
        sorted(kept_primary, key=_issue_sort_key),
        sorted(kept_secondary, key=_issue_sort_key),
        sorted(merged_observations, key=_issue_sort_key),
    )


def aggregate_rdl_feedback_evidence(
    *,
    feedback_evidence: dict[str, Any],
    analysis_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ = analysis_context
    evidence_items_raw = feedback_evidence.get("evidence_items", []) if isinstance(feedback_evidence, dict) else []
    evidence_items = [i for i in evidence_items_raw if isinstance(i, dict) and bool(i.get("detected", False))]
    issues = aggregate_evidence_items(evidence_items)
    issues = sorted(issues, key=_issue_sort_key)

    primary, secondary, observations = split_priority_buckets(issues)
    primary, secondary, observations = _force_observation_bucket_for_possible_spine(
        primary, secondary, observations
    )

    for issue in primary:
        issue.priority_bucket = "primary"
    for issue in secondary:
        issue.priority_bucket = "secondary"
    for issue in observations:
        issue.priority_bucket = "observation"

    issue_by_error_detector = {(i.error_code, i.detector): i for i in issues}

    rep_to_issues: dict[int, list[AggregatedIssue]] = defaultdict(list)
    rep_to_raw: dict[int, int | None] = {}
    rep_to_phases: dict[int, dict[str, list[AggregatedIssue]]] = defaultdict(lambda: defaultdict(list))
    for item in evidence_items:
        rep_order = item.get("user_rep_order")
        if not isinstance(rep_order, int):
            continue
        key = (str(item.get("error_code", "")), str(item.get("detector", "")))
        issue = issue_by_error_detector.get(key)
        if issue is None:
            continue
        if issue not in rep_to_issues[rep_order]:
            rep_to_issues[rep_order].append(issue)
        raw_idx = item.get("user_rep_raw_index")
        if isinstance(raw_idx, int):
            rep_to_raw[rep_order] = raw_idx
        phase = str(item.get("phase") or PHASE_UNKNOWN)
        if issue not in rep_to_phases[rep_order][phase]:
            rep_to_phases[rep_order][phase].append(issue)

    by_rep: list[RepAggregation] = []
    for rep_order in sorted(rep_to_issues.keys()):
        rep_issues = sorted(rep_to_issues[rep_order], key=_issue_sort_key)
        phases_map = {phase: sorted(phase_issues, key=_issue_sort_key) for phase, phase_issues in rep_to_phases[rep_order].items()}
        by_rep.append(
            RepAggregation(
                user_rep_order=rep_order,
                user_rep_raw_index=rep_to_raw.get(rep_order),
                issues=rep_issues,
                phases=phases_map,
                max_severity=max_severity([i.severity for i in rep_issues]),
                num_issues=len(rep_issues),
            )
        )

    phase_to_issues: dict[str, list[AggregatedIssue]] = defaultdict(list)
    for issue in issues:
        for phase in issue.phases:
            if issue not in phase_to_issues[phase]:
                phase_to_issues[phase].append(issue)
    by_phase = [
        PhaseAggregation(
            phase=phase,
            issues=sorted(phase_issues, key=_issue_sort_key),
            max_severity=max_severity([i.severity for i in phase_issues]),
            num_issues=len(phase_issues),
        )
        for phase, phase_issues in sorted(phase_to_issues.items(), key=lambda kv: kv[0])
    ]

    all_warnings = sorted(
        {
            *[str(w) for w in (feedback_evidence.get("warnings", []) if isinstance(feedback_evidence, dict) else [])],
            *[str(w) for issue in issues for w in issue.warnings],
        }
    )
    summary = {
        "num_evidence_items": len(evidence_items),
        "num_issues": len(issues),
        "num_primary": len(primary),
        "num_secondary": len(secondary),
        "num_observations": len(observations),
        "max_severity": max_severity([i.severity for i in issues]),
        "primary_error_codes": [i.error_code for i in primary],
        "secondary_error_codes": [i.error_code for i in secondary],
        "observation_error_codes": [i.error_code for i in observations],
        "affected_reps": sorted({rep for issue in issues for rep in issue.reps}),
        "affected_phases": sorted({phase for issue in issues for phase in issue.phases}),
        "warnings_count": len(all_warnings),
    }

    result = FeedbackAggregationResult(
        exercise=str(feedback_evidence.get("exercise", "RDL")) if isinstance(feedback_evidence, dict) else "RDL",
        status="issues_detected" if issues else "no_relevant_issues",
        issues=issues,
        primary_focus=primary,
        secondary_focus=secondary,
        observations=observations,
        by_rep=by_rep,
        by_phase=by_phase,
        summary=summary,
        warnings=all_warnings,
    )
    return aggregation_result_to_dict(result)
