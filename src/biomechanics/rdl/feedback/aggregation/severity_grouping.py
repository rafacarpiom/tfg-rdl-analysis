
from __future__ import annotations

from .schema import AggregatedIssue
from src.biomechanics.rdl.feedback.evidence_normalizer.constants import SEVERITY_RANK

# Señal de columna incierta: siempre como observación, no foco primario/secundario.
_INFERRED_SPINE_ERROR_CODES = frozenset({"spine_flexion_possible"})


def severity_rank(severity: str) -> int:
    return SEVERITY_RANK.get(str(severity or "none").lower(), 0)


def max_severity(severities: list[str]) -> str:
    best = "none"
    for sev in severities:
        if severity_rank(sev) > severity_rank(best):
            best = str(sev or "none").lower()
    return best


def _sort_issues(issues: list[AggregatedIssue]) -> list[AggregatedIssue]:
    return sorted(
        issues,
        key=lambda i: (
            -severity_rank(i.severity),
            -int(i.affected_rep_count),
            -float(i.max_score),
            -float(i.mean_score),
        ),
    )


def _demote_inferred_and_posible_to_observations(
    primary: list[AggregatedIssue],
    secondary: list[AggregatedIssue],
    observations: list[AggregatedIssue],
) -> tuple[list[AggregatedIssue], list[AggregatedIssue], list[AggregatedIssue]]:
    demoted: list[AggregatedIssue] = []
    kept_primary: list[AggregatedIssue] = []
    kept_secondary: list[AggregatedIssue] = []
    for issue in primary:
        if issue.error_code in _INFERRED_SPINE_ERROR_CODES or issue.severity == "posible":
            demoted.append(issue)
        else:
            kept_primary.append(issue)
    for issue in secondary:
        if issue.error_code in _INFERRED_SPINE_ERROR_CODES or issue.severity == "posible":
            demoted.append(issue)
        else:
            kept_secondary.append(issue)
    merged_observations = list(observations)
    for issue in demoted:
        if issue not in merged_observations:
            merged_observations.append(issue)
    return kept_primary, kept_secondary, merged_observations


def split_priority_buckets(
    issues: list[AggregatedIssue],
) -> tuple[list[AggregatedIssue], list[AggregatedIssue], list[AggregatedIssue]]:
    if not issues:
        return [], [], []
    m = max_severity([i.severity for i in issues])
    if m == "grave":
        primary = [i for i in issues if i.severity == "grave"]
        secondary = [i for i in issues if i.severity == "media"]
        observations = [i for i in issues if i.severity in {"leve", "posible"}]
    elif m == "media":
        primary = [i for i in issues if i.severity == "media"]
        secondary = [i for i in issues if i.severity == "leve"]
        observations = [i for i in issues if i.severity == "posible"]
    elif m == "leve":
        primary = [i for i in issues if i.severity == "leve"]
        secondary = []
        observations = [i for i in issues if i.severity == "posible"]
    elif m == "posible":
        primary = []
        secondary = []
        observations = [i for i in issues if i.severity == "posible"]
    else:
        primary, secondary, observations = [], [], []
    primary, secondary, observations = _demote_inferred_and_posible_to_observations(
        primary, secondary, observations
    )
    return _sort_issues(primary), _sort_issues(secondary), _sort_issues(observations)
