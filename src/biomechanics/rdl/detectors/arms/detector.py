
from __future__ import annotations

from typing import Any

from src.biomechanics.rdl.detectors.arms.metrics import (
    BENT_ARMS_ANCHORS,
    compute_bent_arms_anchor_metrics,
)
from src.biomechanics.rdl.detectors.arms.rules import detect_bent_arms_from_metrics


def _severity_rank(severity: str) -> int:
    rank = {"none": 0, "leve": 1, "media": 2, "grave": 3}
    return rank.get(severity, 0)


def detect_bent_arms(
    analysis_context: dict,
) -> dict:
    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    paired_reps = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else []
    if not isinstance(paired_reps, list):
        paired_reps = []

    rep_results: list[dict[str, Any]] = []
    global_warnings: list[str] = []
    max_detected_confidence = 0.0
    global_severity = "none"

    for rep_idx, paired_rep in enumerate(paired_reps):
        if not isinstance(paired_rep, dict):
            global_warnings.append(f"paired_repetition[{rep_idx}] invalida")
            continue

        user_rep_raw_index = int(paired_rep.get("user_rep_raw_index", -1))
        user_rep_order = int(rep_idx + 1)
        anchors = paired_rep.get("anchors") if isinstance(paired_rep.get("anchors"), dict) else {}
        rep_warnings: list[str] = []
        metrics_by_anchor: dict[str, Any] = {}

        for anchor in BENT_ARMS_ANCHORS:
            pair = anchors.get(anchor) if isinstance(anchors, dict) else None
            if not isinstance(pair, dict):
                rep_warnings.append(f"{anchor}: pair no disponible")
                continue
            if not bool(pair.get("valid", False)):
                rep_warnings.append(f"{anchor}: pair invalido")
                continue

            user_kps = pair.get("user_kps_clean")
            if user_kps is None:
                user_kps = pair.get("user_kps_normalized")
            if user_kps is None:
                rep_warnings.append(f"{anchor}: user_kps ausente")
                continue
            metrics_by_anchor[anchor] = compute_bent_arms_anchor_metrics(user_kps, anchor)

        verdict = detect_bent_arms_from_metrics(metrics_by_anchor)

        anchor_metrics = {
            anchor: {"angle_elbow": metrics.angle_elbow}
            for anchor, metrics in verdict.per_anchor.items()
        }
        anchor_rulings = {
            anchor: {
                "severity": ruling.severity,
                "failed": ruling.failed,
                "grave": ruling.grave,
                "trace": list(ruling.trace),
            }
            for anchor, ruling in verdict.per_anchor.items()
        }

        rep_result = {
            "user_rep_raw_index": user_rep_raw_index,
            "user_rep_order": user_rep_order,
            "detected": verdict.detected,
            "severity": verdict.severity,
            "score": verdict.confidence,
            "magnitude": verdict.magnitude,
            "n_failed": verdict.n_failed,
            "failed_anchors": list(verdict.failed_anchors),
            "anchor_metrics": anchor_metrics,
            "anchor_rulings": anchor_rulings,
            "warnings": rep_warnings,
        }
        rep_results.append(rep_result)

        if verdict.detected:
            max_detected_confidence = max(max_detected_confidence, verdict.confidence)
            if _severity_rank(verdict.severity) > _severity_rank(global_severity):
                global_severity = verdict.severity

    num_reps_detected = sum(1 for rep in rep_results if bool(rep.get("detected")))
    detected = num_reps_detected > 0

    return {
        "detector": "bent_arms",
        "detected": detected,
        "severity": global_severity if detected else "none",
        "score": max_detected_confidence if detected else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_reps_detected,
        "rep_results": rep_results,
        "warnings": global_warnings,
    }
