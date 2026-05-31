
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.biomechanics.rdl.detectors.neck_movement.metrics import ORDERED_ANCHORS, compute_neck_movement_segments
from src.biomechanics.rdl.detectors.neck_movement.rules import detect_neck_movement

_SEV_RANK = {"none": 0, "leve": 1, "media": 2, "grave": 3}


def _safe_deg(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return math.degrees(v) if math.isfinite(v) else float("nan")


def _severity_max(current: str, candidate: str) -> str:
    return candidate if _SEV_RANK.get(candidate, 0) > _SEV_RANK.get(current, 0) else current


def _extract_score_from_pose(pose_clean: dict[str, Any], frame: int | None) -> np.ndarray | None:
    if not isinstance(frame, int):
        return None
    kps_score = pose_clean.get("kps_score_clean")
    if not isinstance(kps_score, np.ndarray) or kps_score.ndim != 2:
        return None
    if not (0 <= frame < kps_score.shape[0]):
        return None
    return np.asarray(kps_score[frame], dtype=np.float64)


def detect_neck_movement_error(analysis_context: dict) -> dict:
    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    paired_reps = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else []
    if not isinstance(paired_reps, list):
        paired_reps = []

    user_pose_clean = (analysis_context.get("user") or {}).get("pose_clean") if isinstance((analysis_context.get("user") or {}), dict) else {}
    ref_pose_clean = (analysis_context.get("reference") or {}).get("pose_clean") if isinstance((analysis_context.get("reference") or {}), dict) else {}

    rep_results: list[dict[str, Any]] = []
    detector_warnings: list[str] = []
    global_severity = "none"
    global_score = 0.0
    num_detected = 0

    for rep_idx, paired_rep in enumerate(paired_reps):
        if not isinstance(paired_rep, dict):
            detector_warnings.append(f"INVALID_PAIRED_REP:{rep_idx}")
            continue

        anchors = paired_rep.get("anchors") if isinstance(paired_rep.get("anchors"), dict) else {}
        rep_warnings: list[str] = []
        user_kps_by_anchor: dict[str, np.ndarray] = {}
        ideal_kps_by_anchor: dict[str, np.ndarray] = {}
        user_scores_by_anchor: dict[str, np.ndarray] = {}
        ideal_scores_by_anchor: dict[str, np.ndarray] = {}
        user_frame_by_anchor: dict[str, int | None] = {}
        ideal_frame_by_anchor: dict[str, int | None] = {}

        for anchor in ORDERED_ANCHORS:
            pair = anchors.get(anchor)
            if not isinstance(pair, dict) or not bool(pair.get("valid", False)):
                rep_warnings.append(f"MISSING_NECK_ANCHOR:{anchor}")
                continue
            user_kps = pair.get("user_kps_clean")
            ideal_kps = pair.get("ideal_kps_clean")
            if user_kps is None or ideal_kps is None:
                rep_warnings.append(f"MISSING_NECK_CLEAN_KPS:{anchor}")
                continue
            user_arr = np.asarray(user_kps, dtype=np.float64)
            ideal_arr = np.asarray(ideal_kps, dtype=np.float64)
            if user_arr.shape != (17, 2) or ideal_arr.shape != (17, 2):
                rep_warnings.append(f"INVALID_NECK_KPS_SHAPE:{anchor}")
                continue

            user_frame = pair.get("user_frame") if isinstance(pair.get("user_frame"), int) else None
            ideal_frame = pair.get("ideal_frame") if isinstance(pair.get("ideal_frame"), int) else None
            user_frame_by_anchor[anchor] = user_frame
            ideal_frame_by_anchor[anchor] = ideal_frame
            user_kps_by_anchor[anchor] = user_arr
            ideal_kps_by_anchor[anchor] = ideal_arr

            user_score = pair.get("user_kps_score_clean")
            if user_score is None:
                user_score = pair.get("user_scores")
            if isinstance(user_score, np.ndarray):
                user_scores_by_anchor[anchor] = np.asarray(user_score, dtype=np.float64)
            else:
                fallback_user = _extract_score_from_pose(user_pose_clean if isinstance(user_pose_clean, dict) else {}, user_frame)
                if isinstance(fallback_user, np.ndarray):
                    user_scores_by_anchor[anchor] = fallback_user

            ideal_score = pair.get("ideal_kps_score_clean")
            if ideal_score is None:
                ideal_score = pair.get("ideal_scores")
            if isinstance(ideal_score, np.ndarray):
                ideal_scores_by_anchor[anchor] = np.asarray(ideal_score, dtype=np.float64)
            else:
                fallback_ideal = _extract_score_from_pose(ref_pose_clean if isinstance(ref_pose_clean, dict) else {}, ideal_frame)
                if isinstance(fallback_ideal, np.ndarray):
                    ideal_scores_by_anchor[anchor] = fallback_ideal

        if not user_kps_by_anchor or not ideal_kps_by_anchor:
            rep_results.append(
                {
                    "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
                    "user_rep_order": int(rep_idx + 1),
                    "detected": False,
                    "severity": "none",
                    "score": 0.0,
                    "magnitude_rad": 0.0,
                    "magnitude_deg": 0.0,
                    "confidence": 0.0,
                    "phase": "ecc",
                    "n_failed": 0,
                    "failed_anchors": [],
                    "failed_segments": [],
                    "subtype": "inconclusive",
                    "neck_direction": "unclear",
                    "mean_signed_excess_rad": float("nan"),
                    "mean_signed_excess_deg": float("nan"),
                    "anchor_results": {},
                    "segment_results": {},
                    "warnings": rep_warnings,
                }
            )
            continue

        try:
            segment_metrics = compute_neck_movement_segments(
                user_kps_by_anchor,
                ideal_kps_by_anchor,
                user_scores_by_anchor=user_scores_by_anchor,
                ideal_scores_by_anchor=ideal_scores_by_anchor,
            )
            verdict = detect_neck_movement(segment_metrics)
        except Exception as exc:
            rep_warnings.append(f"NECK_MOVEMENT_RULES_FAILED:{exc}")
            rep_results.append(
                {
                    "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
                    "user_rep_order": int(rep_idx + 1),
                    "detected": False,
                    "severity": "none",
                    "score": 0.0,
                    "magnitude_rad": 0.0,
                    "magnitude_deg": 0.0,
                    "confidence": 0.0,
                    "phase": "ecc",
                    "n_failed": 0,
                    "failed_anchors": [],
                    "failed_segments": [],
                    "subtype": "inconclusive",
                    "neck_direction": "unclear",
                    "mean_signed_excess_rad": float("nan"),
                    "mean_signed_excess_deg": float("nan"),
                    "anchor_results": {},
                    "segment_results": {},
                    "warnings": rep_warnings,
                }
            )
            continue

        anchor_results: dict[str, dict[str, Any]] = {}
        for anchor, ruling in verdict.per_anchor.items():
            anchor_results[anchor] = {
                "failed": bool(ruling.failed),
                "severity": str(ruling.severity),
                "neck_direction": str(ruling.neck_direction),
                "subtype": str(ruling.subtype),
                "B_rad": ruling.classifier_B_value,
                "B_deg": _safe_deg(ruling.classifier_B_value),
                "drift_from_start_rad": ruling.classifier_A_value,
                "drift_from_start_deg": _safe_deg(ruling.classifier_A_value),
                "status": str(ruling.status),
                "confidence": str(ruling.confidence),
                "user_frame": user_frame_by_anchor.get(anchor),
                "ideal_frame": ideal_frame_by_anchor.get(anchor),
                "trace": list(ruling.trace),
            }

        segment_results: dict[str, dict[str, Any]] = {}
        for segment, ruling in verdict.per_segment.items():
            segment_results[segment] = {
                "anchor": ruling.anchor,
                "failed": bool(ruling.failed),
                "severity": str(ruling.severity),
                "neck_direction": str(ruling.neck_direction),
                "subtype": str(ruling.subtype),
                "classifier_A_deg": _safe_deg(ruling.classifier_A_value),
                "classifier_B_deg": _safe_deg(ruling.classifier_B_value),
                "theta_head_start_deg": _safe_deg(ruling.theta_head_start),
                "theta_head_end_deg": _safe_deg(ruling.theta_head_end),
                "theta_torso_start_deg": _safe_deg(ruling.theta_torso_start),
                "theta_torso_end_deg": _safe_deg(ruling.theta_torso_end),
                "neck_relative_start_deg": _safe_deg(ruling.neck_relative_start),
                "neck_relative_end_deg": _safe_deg(ruling.neck_relative_end),
                "selected_face_axis_start": ruling.selected_face_axis_start,
                "selected_face_axis_end": ruling.selected_face_axis_end,
                "selected_face_keypoints_start": ruling.selected_face_keypoints_start,
                "selected_face_keypoints_end": ruling.selected_face_keypoints_end,
                "nose_confidence_start": ruling.nose_confidence_start,
                "nose_confidence_end": ruling.nose_confidence_end,
                "face_ref_confidence_start": ruling.face_ref_confidence_start,
                "face_ref_confidence_end": ruling.face_ref_confidence_end,
                "reject_reason": ruling.reject_reason,
                "trace": list(ruling.trace),
            }

        magnitude_deg = _safe_deg(verdict.magnitude)
        rep_result = {
            "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
            "user_rep_order": int(rep_idx + 1),
            "detected": verdict.detected,
            "severity": verdict.severity,
            "score": magnitude_deg if math.isfinite(magnitude_deg) else 0.0,
            "magnitude_rad": verdict.magnitude,
            "magnitude_deg": magnitude_deg,
            "confidence": verdict.confidence,
            "phase": verdict.phase,
            "n_failed": verdict.n_failed,
            "failed_anchors": list(verdict.failed_anchors),
            "failed_segments": list(verdict.failed_segments),
            "subtype": verdict.subtype,
            "neck_direction": verdict.neck_direction,
            "mean_signed_excess_rad": verdict.mean_signed_excess,
            "mean_signed_excess_deg": _safe_deg(verdict.mean_signed_excess),
            "anchor_results": anchor_results,
            "segment_results": segment_results,
            "warnings": rep_warnings,
        }
        rep_results.append(rep_result)

        if verdict.detected:
            num_detected += 1
            global_severity = _severity_max(global_severity, verdict.severity)
            if math.isfinite(magnitude_deg):
                global_score = max(global_score, magnitude_deg)

    return {
        "detector": "neck_movement",
        "detected": num_detected > 0,
        "severity": global_severity if num_detected > 0 else "none",
        "score": global_score if num_detected > 0 else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_detected,
        "rep_results": rep_results,
        "warnings": sorted(set(detector_warnings)),
    }
