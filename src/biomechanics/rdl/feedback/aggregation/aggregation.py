
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .location_summary import merge_locations
from .schema import AggregatedIssue
from .severity_grouping import max_severity

_SUMMARY_KEYS = {
    "failed_anchors",
    "failed_segments",
    "neck_direction",
    "subtype",
    "rom_norm",
    "max_knee",
    "max_deficit",
    "error_lockout_deg",
    "max_torso_drop_vs_ideal",
    "possible_segments",
    "blocking_errors",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _ordered_unique_str(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _ordered_unique_int(values: list[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        i = _safe_int(v)
        if i is None or i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def aggregate_evidence_items(
    evidence_items: list[dict[str, Any]],
) -> list[AggregatedIssue]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("error_code", "")), str(item.get("detector", "")))
        groups[key].append(item)

    issues: list[AggregatedIssue] = []
    for (error_code, detector), items in groups.items():
        severities = [str(i.get("severity", "none")).lower() for i in items]
        scores = [_safe_float(i.get("score")) for i in items]
        confidences = [_safe_float(i.get("confidence")) for i in items if i.get("confidence") is not None]
        reps = sorted(_ordered_unique_int([i.get("user_rep_order") for i in items]))
        rep_raw_indices = sorted(_ordered_unique_int([i.get("user_rep_raw_index") for i in items]))
        phases = sorted(_ordered_unique_str([i.get("phase", "unknown") for i in items]))
        location_labels = merge_locations(_ordered_unique_str([i.get("location_label", "unknown") for i in items]))
        anchors = _ordered_unique_str([a for i in items for a in (i.get("anchors") or [])])
        frames = sorted(_ordered_unique_int([f for i in items for f in (i.get("frames") or [])]))

        summary_metrics: dict[str, Any] = {
            "items": len(items),
            "affected_rep_count": len(reps),
            "detectors": [detector],
            "error_code": error_code,
            "max_score": max(scores) if scores else 0.0,
            "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
            "locations": location_labels,
        }
        for key in _SUMMARY_KEYS:
            collected = [i.get("summary_metrics", {}).get(key) for i in items if isinstance(i.get("summary_metrics"), dict) and i.get("summary_metrics", {}).get(key) is not None]
            if not collected:
                continue
            summary_metrics[key] = collected[0] if len(collected) == 1 else collected

        evidence_refs = [
            {
                "detector": i.get("detector"),
                "error_code": i.get("error_code"),
                "user_rep_order": i.get("user_rep_order"),
                "phase": i.get("phase"),
                "severity": i.get("severity"),
                "score": i.get("score"),
                "anchors": i.get("anchors"),
                "frames": i.get("frames"),
                "source": i.get("source"),
            }
            for i in items
        ]
        warnings = sorted({str(w) for i in items for w in (i.get("warnings") or [])})

        priority_bucket = "observation"
        if error_code == "spine_flexion_possible" or max_severity(severities) == "posible":
            priority_bucket = "observation"

        issues.append(
            AggregatedIssue(
                error_code=error_code,
                detector=detector,
                severity=max_severity(severities),
                priority_bucket=priority_bucket,
                reps=reps,
                rep_raw_indices=rep_raw_indices,
                phases=phases,
                location_labels=location_labels,
                anchors=anchors,
                frames=frames,
                max_score=max(scores) if scores else 0.0,
                mean_score=(sum(scores) / len(scores)) if scores else 0.0,
                max_confidence=max(confidences) if confidences else None,
                mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
                count=len(items),
                affected_rep_count=len(reps),
                summary_metrics=summary_metrics,
                evidence_refs=evidence_refs,
                warnings=warnings,
            )
        )
    return issues
