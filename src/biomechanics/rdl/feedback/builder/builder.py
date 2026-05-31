
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.biomechanics.rdl.feedback.aggregation.severity_grouping import severity_rank

from .labels import build_where_text
from .messages import get_message
from .recommendations import get_recommendation
from .renderer import render_plain_text_report
from .schema import FeedbackItem, FeedbackReport, RepFeedback, feedback_item_to_dict, feedback_report_to_dict, rep_feedback_to_dict
from src.biomechanics.rdl.feedback.segment_keys import CANONICAL_SEGMENT_IDS, to_canonical_segment

from .segment_filter import filter_evidences_by_segment, select_rep_errors_by_severity

_SEVERITY_LEVELS = ("grave", "media", "leve", "posible", "none")

_REP_LEVEL_ERROR_CODES = frozenset(
    {
        "lockout",
        "short_rom",
        "asymmetry_arms",
        "asymmetry_legs",
    }
)

_BLOCKER_LABELS: dict[str, str] = {
    "neck_movement": "movimiento cervical",
    "knee_dominant": "dominancia de rodilla",
}


def _sev(v: Any) -> str:
    s = str(v or "none").lower()
    return s if s in _SEVERITY_LEVELS else "none"


def _max_severity(values: list[str]) -> str:
    if not values:
        return "none"
    return max((_sev(v) for v in values), key=severity_rank)


def _error_code(issue: dict[str, Any]) -> str:
    return str(issue.get("error_code", "")).lower()


def _is_spine_observation(issue: dict[str, Any]) -> bool:
    return _error_code(issue) == "spine_flexion_possible"


def _uniq_str(items: list[Any]) -> list[str]:
    out: list[str] = []
    for x in items:
        s = str(x)
        if s and s not in out:
            out.append(s)
    return out


def _safe_float(v: Any) -> float | None:
    try:
        out = float(v)
    except Exception:
        return None
    return out


def _entries_from_evidence(feedback_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    items = feedback_evidence.get("evidence_items") if isinstance(feedback_evidence, dict) else None
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rep_order = item.get("user_rep_order")
        if not isinstance(rep_order, int) or rep_order <= 0:
            continue
        severity = _sev(item.get("severity"))
        if severity == "none":
            continue
        out.append(
            {
                "rep_order": rep_order,
                "error_code": str(item.get("error_code", "")),
                "detector": str(item.get("detector", "")),
                "severity": severity,
                "score": _safe_float(item.get("score")),
                "confidence": _safe_float(item.get("confidence")),
                "location_label": str(item.get("location_label", "")),
                "phase": str(item.get("phase", "")),
                "segment": item.get("segment"),
                "anchors": [str(a) for a in item.get("anchors", [])] if isinstance(item.get("anchors"), list) else [],
                "warnings": [str(w) for w in item.get("warnings", [])] if isinstance(item.get("warnings"), list) else [],
                "summary_metrics": item.get("summary_metrics") if isinstance(item.get("summary_metrics"), dict) else {},
            }
        )
    return out


def _entries_from_aggregation(feedback_aggregation: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_rep = feedback_aggregation.get("by_rep") if isinstance(feedback_aggregation, dict) else None
    if isinstance(by_rep, list):
        for rep_row in by_rep:
            if not isinstance(rep_row, dict):
                continue
            rep_order = rep_row.get("user_rep_order")
            if not isinstance(rep_order, int) or rep_order <= 0:
                continue
            issues = rep_row.get("issues")
            if not isinstance(issues, list):
                continue
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                severity = _sev(issue.get("severity"))
                if severity == "none":
                    continue
                refs = issue.get("evidence_refs")
                used_ref = False
                if isinstance(refs, list):
                    for ref in refs:
                        if not isinstance(ref, dict):
                            continue
                        if int(ref.get("user_rep_order", -1)) != rep_order:
                            continue
                        used_ref = True
                        out.append(
                            {
                                "rep_order": rep_order,
                                "error_code": str(issue.get("error_code", "")),
                                "detector": str(issue.get("detector", "")),
                                "severity": _sev(ref.get("severity", severity)),
                                "score": _safe_float(ref.get("score", issue.get("max_score"))),
                                "confidence": _safe_float(issue.get("max_confidence")),
                                "location_label": "",
                                "phase": str(ref.get("phase", "")),
                                "anchors": [str(a) for a in ref.get("anchors", [])] if isinstance(ref.get("anchors"), list) else [],
                                "warnings": [str(w) for w in issue.get("warnings", [])] if isinstance(issue.get("warnings"), list) else [],
                                "summary_metrics": issue.get("summary_metrics") if isinstance(issue.get("summary_metrics"), dict) else {},
                            }
                        )
                if used_ref:
                    continue
                out.append(
                    {
                        "rep_order": rep_order,
                        "error_code": str(issue.get("error_code", "")),
                        "detector": str(issue.get("detector", "")),
                        "severity": severity,
                        "score": _safe_float(issue.get("max_score", issue.get("score"))),
                        "confidence": _safe_float(issue.get("max_confidence")),
                        "location_label": (
                            str(issue.get("location_labels", [""])[0])
                            if isinstance(issue.get("location_labels"), list) and issue.get("location_labels")
                            else ""
                        ),
                        "phase": (
                            str(issue.get("phases", [""])[0])
                            if isinstance(issue.get("phases"), list) and issue.get("phases")
                            else ""
                        ),
                        "anchors": [str(a) for a in issue.get("anchors", [])] if isinstance(issue.get("anchors"), list) else [],
                        "warnings": [str(w) for w in issue.get("warnings", [])] if isinstance(issue.get("warnings"), list) else [],
                        "summary_metrics": issue.get("summary_metrics") if isinstance(issue.get("summary_metrics"), dict) else {},
                    }
                )
    return out


def _merge_rep_error_group(rep_order: int, entries: list[dict[str, Any]]) -> dict[str, Any]:
    base = entries[0]
    worst_sev = _max_severity([e.get("severity", "none") for e in entries])
    worst_entries = [e for e in entries if _sev(e.get("severity")) == worst_sev]
    all_scores = [v for v in (_safe_float(e.get("score")) for e in entries) if v is not None]
    all_conf = [v for v in (_safe_float(e.get("confidence")) for e in entries) if v is not None]
    labels = _uniq_str([e.get("location_label", "") for e in worst_entries if str(e.get("location_label", ""))])
    phases = _uniq_str([e.get("phase", "") for e in worst_entries if str(e.get("phase", ""))])
    anchors = _uniq_str([a for e in worst_entries for a in e.get("anchors", [])])
    warnings = _uniq_str([w for e in entries for w in e.get("warnings", [])])
    summary_metrics: dict[str, Any] = {}
    for e in worst_entries:
        metrics = e.get("summary_metrics")
        if isinstance(metrics, dict):
            summary_metrics.update(metrics)
    return {
        "rep_order": rep_order,
        "error_code": str(base.get("error_code", "")),
        "detector": str(base.get("detector", "")),
        "severity": worst_sev,
        "max_score": max(all_scores) if all_scores else 0.0,
        "max_confidence": max(all_conf) if all_conf else None,
        "location_labels": labels,
        "phases": phases,
        "anchors": anchors,
        "summary_metrics": summary_metrics,
        "warnings": warnings,
    }


def _group_issues_by_rep_and_error(
    feedback_aggregation: dict[str, Any],
    feedback_evidence: dict[str, Any] | None,
) -> dict[int, list[dict[str, Any]]]:
    entries = _entries_from_evidence(feedback_evidence or {})
    if not entries:
        entries = _entries_from_aggregation(feedback_aggregation or {})
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        rep_order = e.get("rep_order")
        error_code = str(e.get("error_code", ""))
        if not isinstance(rep_order, int) or rep_order <= 0 or not error_code:
            continue
        grouped[(rep_order, error_code)].append(e)
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (rep_order, _), rows in grouped.items():
        out[rep_order].append(_merge_rep_error_group(rep_order, rows))
    return {k: v for k, v in out.items()}


def _bucket_by_rep(normal_issues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    max_sev = _max_severity([str(i.get("severity", "none")) for i in normal_issues])
    if max_sev == "grave":
        primary = [i for i in normal_issues if _sev(i.get("severity")) == "grave"]
        secondary = [i for i in normal_issues if _sev(i.get("severity")) == "media"]
    elif max_sev == "media":
        primary = [i for i in normal_issues if _sev(i.get("severity")) == "media"]
        secondary = [i for i in normal_issues if _sev(i.get("severity")) == "posible"]
    elif max_sev == "leve":
        primary = [i for i in normal_issues if _sev(i.get("severity")) == "leve"]
        secondary = [i for i in normal_issues if _sev(i.get("severity")) == "posible"]
    elif max_sev == "posible":
        primary = [i for i in normal_issues if _sev(i.get("severity")) == "posible"]
        secondary = []
    else:
        primary, secondary = [], []
    return primary, secondary, max_sev


def _build_feedback_item(issue: dict[str, Any], *, priority: str, bucket: str) -> FeedbackItem:
    msg = get_message(str(issue.get("error_code", "")))
    return FeedbackItem(
        priority=priority,
        bucket=bucket,
        error_code=str(issue.get("error_code", "")),
        detector=str(issue.get("detector", "")),
        severity=_sev(issue.get("severity", "none")),
        title=msg["title"],
        where=build_where_text(issue, include_reps=False),
        what_happens=msg["what_happens"],
        why_it_matters=msg["why_it_matters"],
        how_to_fix=get_recommendation(str(issue.get("error_code", "")), issue),
        warnings=[str(w) for w in issue.get("warnings", [])] if isinstance(issue.get("warnings"), list) else [],
    )


def _build_headline(max_sev: str) -> str:
    if max_sev == "grave":
        return "Se han detectado errores graves en una o más repeticiones."
    if max_sev == "media":
        return "Se han detectado errores moderados en una o más repeticiones."
    if max_sev == "leve":
        return "Se han detectado errores leves en una o más repeticiones."
    return "No se han detectado errores técnicos relevantes en las repeticiones analizadas."


def build_rdl_feedback_report(
    *,
    feedback_aggregation: dict[str, Any],
    feedback_evidence: dict[str, Any] | None = None,
    analysis_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ = analysis_context
    agg = feedback_aggregation if isinstance(feedback_aggregation, dict) else {}

    raw_evidences = _entries_from_evidence(feedback_evidence or {})
    if not raw_evidences:
        raw_evidences = _entries_from_aggregation(agg)

    filtered = filter_evidences_by_segment(raw_evidences)
    by_rep_filtered = filtered["by_rep"]
    by_segment_filtered = filtered["by_segment"]
    rep_level_only = filtered["rep_level_only"]

    rep_issues_map: dict[int, list[dict[str, Any]]] = {}

    for rep, items in by_rep_filtered.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            error_code = str(item.get("error_code", ""))
            if error_code:
                grouped[error_code].append(item)

        merged = []
        for error_code, items_group in grouped.items():
            merged.append(_merge_rep_error_group(rep, items_group))

        rep_issues_map[rep] = select_rep_errors_by_severity(merged)

    rep_feedback: list[RepFeedback] = []
    global_normal_severities: list[str] = []

    for rep_order in sorted(rep_issues_map.keys()):
        issues = [i for i in rep_issues_map[rep_order] if isinstance(i, dict)]
        normal_issues = [i for i in issues if not _is_spine_observation(i)]
        observation_issues = [
            i for i in issues if _is_spine_observation(i) and _sev(i.get("severity")) != "none"
        ]
        primary_raw, secondary_raw, rep_max_sev = _bucket_by_rep(normal_issues)
        global_normal_severities.extend([_sev(i.get("severity")) for i in normal_issues])
        sort_key = lambda i: (severity_rank(_sev(i.get("severity"))), float(i.get("max_score", 0.0)))
        primary_raw = sorted(primary_raw, key=sort_key, reverse=True)
        secondary_raw = sorted(secondary_raw, key=sort_key, reverse=True)
        observation_issues = sorted(observation_issues, key=sort_key, reverse=True)
        rep_feedback.append(
            RepFeedback(
                user_rep_order=rep_order,
                max_severity=rep_max_sev,
                primary_errors=[_build_feedback_item(i, priority="primary", bucket="primary_errors") for i in primary_raw],
                secondary_errors=[_build_feedback_item(i, priority="secondary", bucket="secondary_errors") for i in secondary_raw],
                observations=[_build_feedback_item(i, priority="observation", bucket="observations") for i in observation_issues],
            )
        )

    rep_feedback = [
        r
        for r in rep_feedback
        if r.primary_errors
        or r.secondary_errors
        or r.observations
    ]
    affected_reps = [r.user_rep_order for r in rep_feedback]
    global_max_sev = _max_severity(global_normal_severities)
    status = "issues_detected" if affected_reps else "no_relevant_issues"
    summary = {
        "overall_status": status,
        "num_reps_with_issues": len(affected_reps),
        "max_severity": global_max_sev,
        "affected_reps": affected_reps,
    }
    warnings = sorted({*[str(w) for w in agg.get("warnings", [])]}) if isinstance(agg.get("warnings"), list) else []
    report = FeedbackReport(
        exercise=str(agg.get("exercise", "RDL")),
        status=status,
        summary=summary,
        headline=_build_headline(global_max_sev),
        rep_feedback=rep_feedback,
        plain_text="",
        warnings=warnings,
    )
    out = feedback_report_to_dict(report)
    out["rep_feedback"] = [rep_feedback_to_dict(rep) for rep in rep_feedback]
    out["plain_text"] = render_plain_text_report(out)
    out["by_segment"] = _build_segment_view(
        by_segment_filtered,
        rep_level_only,
        raw_evidences,
    )
    out["by_serie"] = _build_serie_view(rep_feedback, global_max_sev)
    return out


def _tramo_item_from_entry(entry: dict[str, Any], *, kind: str = "error") -> dict[str, Any]:
    error_code = str(entry.get("error_code", ""))
    msg = get_message(error_code)
    metrics = entry.get("summary_metrics") if isinstance(entry.get("summary_metrics"), dict) else {}
    display_severity = _sev(entry.get("severity"))
    if error_code == "spine_flexion_possible":
        torso_sev = metrics.get("torso_low_severity")
        if torso_sev:
            display_severity = _sev(torso_sev)
        kind = "observation"
    return {
        "kind": kind,
        "error_code": error_code,
        "detector": str(entry.get("detector", "")),
        "severity": display_severity,
        "score": entry.get("score", 0.0),
        "title": msg["title"],
        "where": build_where_text(entry, include_reps=False),
        "what_happens": msg["what_happens"],
        "why_it_matters": msg["why_it_matters"],
        "how_to_fix": get_recommendation(error_code, entry),
    }


def _spine_observations_by_segment(
    rep: int,
    segment_entries: dict[str, list[dict[str, Any]]],
    raw_evidences: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {seg_id: [] for seg_id in CANONICAL_SEGMENT_IDS}
    for entry in raw_evidences:
        if entry.get("rep_order") != rep:
            continue
        if str(entry.get("error_code", "")) != "spine_flexion_possible":
            continue
        segment = entry.get("segment")
        canonical = to_canonical_segment(str(segment)) if segment else None
        if canonical is None:
            continue
        already = any(
            isinstance(x, dict) and str(x.get("error_code", "")) == "spine_flexion_possible"
            for x in segment_entries.get(canonical, [])
        )
        if already:
            continue
        metrics = entry.get("summary_metrics") if isinstance(entry.get("summary_metrics"), dict) else {}
        blockers = metrics.get("blocking_errors")
        if not isinstance(blockers, list):
            blockers = []
        if not blockers:
            for seg_item in segment_entries.get(canonical, []):
                if not isinstance(seg_item, dict):
                    continue
                det = str(seg_item.get("detector", ""))
                if det == "neck_movement" and "neck_movement" not in blockers:
                    blockers.append("neck_movement")
                if det == "knee_dominant" and "knee_dominant" not in blockers:
                    blockers.append("knee_dominant")
        if not blockers:
            continue
        obs = _tramo_item_from_entry(entry, kind="observation")
        blocker_text = ", ".join(_BLOCKER_LABELS.get(b, b) for b in blockers)
        obs["what_happens"] = (
            f"{obs['what_happens']} En este tramo hay {blocker_text} activo, por eso no se confirma como error de tronco."
        )
        out[canonical].append(obs)
    return out


def _build_segment_view(
    by_segment_filtered: dict[int, dict[str, list[dict[str, Any]]]],
    rep_level_only: dict[int, list[dict[str, Any]]],
    raw_evidences: list[dict[str, Any]],
) -> dict[str, Any]:
    segments_data: dict[str, Any] = {}
    all_reps = set(by_segment_filtered.keys()) | set(rep_level_only.keys())

    for rep in sorted(all_reps):
        rep_segments: dict[str, list[dict[str, Any]]] = {seg_id: [] for seg_id in CANONICAL_SEGMENT_IDS}
        segments = by_segment_filtered.get(rep, {})
        if isinstance(segments, dict):
            for segment, items in segments.items():
                if not isinstance(items, list):
                    continue
                canonical = to_canonical_segment(str(segment))
                if canonical is None:
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    code = str(item.get("error_code", ""))
                    kind = "observation" if code == "spine_flexion_possible" else "error"
                    rep_segments[canonical].append(_tramo_item_from_entry(item, kind=kind))

        raw_segment_entries = segments if isinstance(segments, dict) else {}
        segment_observations = _spine_observations_by_segment(rep, raw_segment_entries, raw_evidences)
        for seg_id in CANONICAL_SEGMENT_IDS:
            for obs in segment_observations.get(seg_id, []):
                rep_segments[seg_id].append(obs)

        rep_level_items: list[dict[str, Any]] = []
        for entry in rep_level_only.get(rep, []):
            if not isinstance(entry, dict):
                continue
            code = str(entry.get("error_code", ""))
            if code in _REP_LEVEL_ERROR_CODES:
                rep_level_items.append(_tramo_item_from_entry(entry, kind="error"))

        has_segment_content = any(rep_segments[seg_id] for seg_id in CANONICAL_SEGMENT_IDS)
        if not has_segment_content and not rep_level_items:
            continue

        segments_data[f"rep_{rep}"] = {
            "segments": rep_segments,
            "rep_level": rep_level_items,
        }

    return segments_data


def _build_serie_view(rep_feedback: list[RepFeedback], global_max_sev: str) -> dict[str, Any]:
    total_reps = len(rep_feedback)
    reps_with_graves = len([r for r in rep_feedback if r.max_severity == "grave"])
    reps_with_medias = len([r for r in rep_feedback if r.max_severity == "media"])
    reps_with_leves = len([r for r in rep_feedback if r.max_severity == "leve"])

    all_errors: dict[str, dict[str, Any]] = {}

    for rep in rep_feedback:
        for item in rep.primary_errors + rep.secondary_errors:
            error_code = item.error_code
            if error_code not in all_errors:
                all_errors[error_code] = {
                    "error_code": error_code,
                    "detector": item.detector,
                    "title": item.title,
                    "max_severity": item.severity,
                    "affected_reps": [],
                }
            all_errors[error_code]["affected_reps"].append(rep.user_rep_order)
            if severity_rank(item.severity) > severity_rank(all_errors[error_code]["max_severity"]):
                all_errors[error_code]["max_severity"] = item.severity

    sorted_errors = sorted(
        all_errors.values(),
        key=lambda x: (-severity_rank(x["max_severity"]), -len(x["affected_reps"])),
    )

    return {
        "total_reps_analyzed": total_reps,
        "reps_with_graves": reps_with_graves,
        "reps_with_medias": reps_with_medias,
        "reps_with_leves": reps_with_leves,
        "global_max_severity": global_max_sev,
        "common_errors": sorted_errors[:5],
    }
