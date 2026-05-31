
from __future__ import annotations

from typing import Any

import numpy as np

from src.biomechanics.rdl.detectors.knee_dominant.metrics import KNEE_DOMINANT_ANCHORS, compute_knee_dominant_metrics
from src.biomechanics.rdl.detectors.knee_dominant.rules import classify_knee_anchor, detect_knee_dominant

_SEV_RANK = {"none": 0, "leve": 1, "media": 2, "grave": 3}


def _max_severity(a: str, b: str) -> str:
    return b if _SEV_RANK.get(b, 0) > _SEV_RANK.get(a, 0) else a


def detect_knee_dominant_error(analysis_context: dict) -> dict:
    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    paired_reps = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else []
    if not isinstance(paired_reps, list):
        paired_reps = []

    rep_results: list[dict[str, Any]] = []
    global_warnings: list[str] = []
    global_severity = "none"
    global_score = 0.0
    num_detected = 0

    for rep_idx, paired_rep in enumerate(paired_reps):
        if not isinstance(paired_rep, dict):
            global_warnings.append(f"INVALID_PAIRED_REP:{rep_idx}")
            continue
        anchors = paired_rep.get("anchors") if isinstance(paired_rep.get("anchors"), dict) else {}
        rep_warnings: list[str] = []
        metrics_by_anchor: dict[str, Any] = {}

        for anchor in KNEE_DOMINANT_ANCHORS:
            pair = anchors.get(anchor) if isinstance(anchors, dict) else None
            if not isinstance(pair, dict) or not bool(pair.get("valid", False)):
                rep_warnings.append(f"ANCHOR_INVALID_OR_MISSING:{anchor}")
                continue
            ideal_kps = pair.get("ideal_kps_normalized")
            user_kps = pair.get("user_kps_normalized")
            if ideal_kps is None:
                rep_warnings.append(f"IDEAL_NORMALIZED_MISSING_FOR_KNEE_DOMINANT:{anchor}")
                continue
            if user_kps is None:
                rep_warnings.append(f"USER_NORMALIZED_MISSING_FOR_KNEE_DOMINANT:{anchor}")
                continue
            try:
                metrics = compute_knee_dominant_metrics(
                    ideal_kps_normalized=np.asarray(ideal_kps, dtype=np.float64),
                    user_kps_normalized=np.asarray(user_kps, dtype=np.float64),
                    anchor=anchor,
                )
                metrics_by_anchor[anchor] = metrics
            except Exception as exc:
                rep_warnings.append(f"ANCHOR_METRICS_FAILED:{anchor}:{exc}")

        verdict = detect_knee_dominant(metrics_by_anchor)
        anchor_metrics_out: dict[str, Any] = {}
        for anchor, m in metrics_by_anchor.items():
            anchor_metrics_out[anchor] = {
                "hip_ideal": m.hip_ideal,
                "hip_user": m.hip_user,
                "delta_hip": m.delta_hip,
                "knee_ideal": m.knee_ideal,
                "knee_user": m.knee_user,
                "delta_knee": m.delta_knee,
            }
        anchor_rulings_out: dict[str, Any] = {}
        failed_anchors: list[str] = []
        for anchor, r in verdict.per_anchor.items():
            if r.failed:
                failed_anchors.append(anchor)
            local_sev = classify_knee_anchor(r.delta_knee)
            if r.failed and local_sev == "none":
                local_sev = "leve"
            anchor_rulings_out[anchor] = {
                "failed": r.failed,
                "severity": local_sev,
                "delta_knee": r.delta_knee,
                "delta_hip": r.delta_hip,
                "reject_reason": r.reject_reason,
                "trace": list(r.trace),
            }

        rep_result = {
            "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
            "user_rep_order": int(rep_idx + 1),
            "detected": bool(verdict.detected),
            "severity": verdict.severity,
            "score": verdict.magnitude,
            "confidence": verdict.confidence,
            "phase": verdict.phase,
            "magnitude": verdict.magnitude,
            "mean_knee": verdict.mean_knee,
            "max_knee": verdict.max_knee,
            "n_failed": verdict.n_failed,
            "failed_anchors": failed_anchors,
            "anchor_metrics": anchor_metrics_out,
            "anchor_rulings": anchor_rulings_out,
            "warnings": sorted(set(rep_warnings)),
        }
        rep_results.append(rep_result)
        if verdict.detected:
            num_detected += 1
            global_severity = _max_severity(global_severity, verdict.severity)
            global_score = max(global_score, float(verdict.magnitude))

    return {
        "detector": "knee_dominant",
        "detected": num_detected > 0,
        "severity": global_severity if num_detected > 0 else "none",
        "score": global_score if num_detected > 0 else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_detected,
        "rep_results": rep_results,
        "warnings": sorted(set(global_warnings)),
    }
