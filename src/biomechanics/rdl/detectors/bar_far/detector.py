
from __future__ import annotations

from typing import Any

import numpy as np

from src.biomechanics.rdl.detectors.bar_far.metrics import (
    BAR_FAR_ANCHORS,
    compute_bar_far_anchor_metrics,
)
from src.biomechanics.rdl.detectors.bar_far.rules import classify_bar_far_anchor

BAR_FAR_MIN_FAILED_ANCHORS = 2
_SEVERITY_RANK = {"none": 0, "leve": 1, "media": 2, "grave": 3}


def _max_severity(current: str, candidate: str) -> str:
    return candidate if _SEVERITY_RANK.get(candidate, 0) > _SEVERITY_RANK.get(current, 0) else current


def detect_bar_far(analysis_context: dict) -> dict:
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
        user_rep_raw_index = int(paired_rep.get("user_rep_raw_index", -1))
        user_rep_order = int(rep_idx + 1)
        anchor_results: dict[str, dict[str, Any]] = {}
        rep_warnings: list[str] = []
        rep_severity = "none"
        rep_score = 0.0
        failed_anchors: list[str] = []
        rep_confidence = "normal"

        for anchor in BAR_FAR_ANCHORS:
            pair = anchors.get(anchor) if isinstance(anchors, dict) else None
            if not isinstance(pair, dict) or not bool(pair.get("valid", False)):
                anchor_results[anchor] = {
                    "anchor": anchor,
                    "valid": False,
                    "user_frame": pair.get("user_frame") if isinstance(pair, dict) else None,
                    "ideal_frame": pair.get("ideal_frame") if isinstance(pair, dict) else None,
                    "metrics": {},
                    "verdict": {},
                    "warnings": [f"ANCHOR_INVALID_OR_MISSING:{anchor}"],
                }
                continue

            ideal_kps = pair.get("ideal_kps_normalized")
            user_kps = pair.get("user_kps_normalized")
            anchor_warn: list[str] = []
            if ideal_kps is None:
                warn = f"IDEAL_NORMALIZED_MISSING_FOR_BAR_FAR:{anchor}"
                anchor_warn.append(warn)
                rep_warnings.append(warn)
                anchor_results[anchor] = {
                    "anchor": anchor,
                    "valid": False,
                    "user_frame": pair.get("user_frame"),
                    "ideal_frame": pair.get("ideal_frame"),
                    "metrics": {},
                    "verdict": {},
                    "warnings": anchor_warn,
                }
                continue
            if user_kps is None:
                user_kps = pair.get("user_kps_clean")
                if user_kps is not None:
                    warn = f"USER_NORMALIZED_MISSING_FOR_BAR_FAR:{anchor}"
                    anchor_warn.append(warn)
                    rep_warnings.append(warn)
            if user_kps is None:
                warn = f"USER_NORMALIZED_MISSING_FOR_BAR_FAR:{anchor}"
                anchor_warn.append(warn)
                rep_warnings.append(warn)
                anchor_results[anchor] = {
                    "anchor": anchor,
                    "valid": False,
                    "user_frame": pair.get("user_frame"),
                    "ideal_frame": pair.get("ideal_frame"),
                    "metrics": {},
                    "verdict": {},
                    "warnings": anchor_warn,
                }
                continue

            try:
                metrics = compute_bar_far_anchor_metrics(
                    ideal_kps_normalized=np.asarray(ideal_kps, dtype=np.float64),
                    user_kps_normalized=np.asarray(user_kps, dtype=np.float64),
                    anchor=anchor,
                )
                verdict = classify_bar_far_anchor(metrics)
            except Exception as exc:
                warn = f"ANCHOR_COMPUTE_FAILED:{anchor}:{exc}"
                anchor_warn.append(warn)
                rep_warnings.append(warn)
                anchor_results[anchor] = {
                    "anchor": anchor,
                    "valid": False,
                    "user_frame": pair.get("user_frame"),
                    "ideal_frame": pair.get("ideal_frame"),
                    "metrics": {},
                    "verdict": {},
                    "warnings": anchor_warn,
                }
                continue

            metrics_dict = {
                "wrist_error_px": metrics.wrist_error_px,
                "torso_length": metrics.torso_length,
                "wrist_error_norm": metrics.wrist_error_norm,
                "wrist_error_x_px": metrics.wrist_error_x_px,
                "wrist_error_x_norm": metrics.wrist_error_x_norm,
                "wrist_error_y_px": metrics.wrist_error_y_px,
                "wrist_error_y_norm": metrics.wrist_error_y_norm,
                "delta_x_wrist": metrics.delta_x_wrist,
                "arm_dir_delta": metrics.arm_dir_delta,
                "elbow_angle_ideal": metrics.elbow_angle_ideal,
                "elbow_angle_user": metrics.elbow_angle_user,
                "elbow_angle_delta": metrics.elbow_angle_delta,
            }
            verdict_dict = {
                "base_severity": verdict.base_severity,
                "severity": verdict.severity,
                "confidence": verdict.confidence,
                "applied_rules": list(verdict.applied_rules),
                "trace": list(verdict.trace),
            }
            anchor_results[anchor] = {
                "anchor": anchor,
                "valid": True,
                "user_frame": pair.get("user_frame"),
                "ideal_frame": pair.get("ideal_frame"),
                "metrics": metrics_dict,
                "verdict": verdict_dict,
                "warnings": anchor_warn,
            }

            if verdict.severity != "none":
                failed_anchors.append(anchor)
                rep_severity = _max_severity(rep_severity, verdict.severity)
                rep_score = max(rep_score, float(metrics.wrist_error_x_norm) if isinstance(metrics.wrist_error_x_norm, (int, float)) else 0.0)
                if verdict.confidence == "alta":
                    rep_confidence = "alta"

        n_failed = len(failed_anchors)
        rep_detected = n_failed >= BAR_FAR_MIN_FAILED_ANCHORS
        if not rep_detected:
            rep_severity = "none"
            rep_score = 0.0
            rep_confidence = "normal"

        rep_result = {
            "user_rep_raw_index": user_rep_raw_index,
            "user_rep_order": user_rep_order,
            "detected": rep_detected,
            "severity": rep_severity,
            "score": rep_score,
            "confidence": rep_confidence,
            "n_failed": n_failed,
            "failed_anchors": failed_anchors,
            "anchor_results": anchor_results,
            "warnings": rep_warnings,
        }
        rep_results.append(rep_result)
        if rep_detected:
            num_detected += 1
            global_severity = _max_severity(global_severity, rep_severity)
            global_score = max(global_score, rep_score)

    return {
        "detector": "bar_far",
        "detected": num_detected > 0,
        "severity": global_severity if num_detected > 0 else "none",
        "score": global_score if num_detected > 0 else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_detected,
        "rep_results": rep_results,
        "warnings": sorted(set(global_warnings)),
    }
