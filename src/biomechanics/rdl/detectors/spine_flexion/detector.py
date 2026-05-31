
from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import (
    SPINE_ANCHORS,
    align_ideal_to_user_torso_for_spine_geometry,
    compute_spine_anchor_geometry,
    geometry_by_segment,
    shoulder_drop_from_top_norm,
)
from .rules import detect_spine_flexion

_SEV_RANK = {"none": 0, "leve": 1, "media": 2, "grave": 3}


def _max_severity(a: str, b: str) -> str:
    return b if _SEV_RANK.get(b, 0) > _SEV_RANK.get(a, 0) else a


def _find_rep_result(detector_result: dict | None, user_rep_order: int, user_rep_raw_index: int) -> dict | None:
    if not isinstance(detector_result, dict):
        return None
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return None
    for rep in rep_results:
        if isinstance(rep, dict) and int(rep.get("user_rep_order", -1)) == int(user_rep_order):
            return rep
    for rep in rep_results:
        if isinstance(rep, dict) and int(rep.get("user_rep_raw_index", -1)) == int(user_rep_raw_index):
            return rep
    return None


def _pick_kps(pair: dict[str, Any], warnings: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    user_kps = pair.get("user_kps_clean")
    if user_kps is None:
        user_kps = pair.get("user_kps_normalized")
        if user_kps is not None:
            warnings.append("USER_CLEAN_MISSING_USING_NORMALIZED_FOR_SPINE_FLEXION")
    ideal_kps = pair.get("ideal_kps_clean")
    if ideal_kps is None:
        ideal_kps = pair.get("ideal_kps_normalized")
        if ideal_kps is not None:
            warnings.append("IDEAL_CLEAN_MISSING_USING_NORMALIZED_FOR_SPINE_FLEXION")
    if user_kps is None or ideal_kps is None:
        return None, None
    user_arr = np.asarray(user_kps, dtype=np.float64)
    ideal_arr = np.asarray(ideal_kps, dtype=np.float64)
    if user_arr.shape != (17, 2) or ideal_arr.shape != (17, 2):
        return None, None
    return user_arr, ideal_arr


def detect_spine_flexion_error(
    analysis_context: dict,
    detector_results: dict[str, Any],
) -> dict:
    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    paired_reps = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else []
    if not isinstance(paired_reps, list):
        paired_reps = []

    hip_global = detector_results.get("hip_hinge") if isinstance(detector_results, dict) else None
    knee_global = detector_results.get("knee_dominant") if isinstance(detector_results, dict) else None
    neck_global = detector_results.get("neck_movement") if isinstance(detector_results, dict) else None
    detector_warnings: list[str] = []
    if not isinstance(hip_global, dict):
        detector_warnings.append("SPINE_DEPENDENCY_MISSING:hip_hinge")
    if not isinstance(knee_global, dict):
        detector_warnings.append("SPINE_DEPENDENCY_MISSING:knee_dominant")
    if not isinstance(neck_global, dict):
        detector_warnings.append("SPINE_DEPENDENCY_MISSING:neck_movement")

    rep_results: list[dict[str, Any]] = []
    global_severity = "none"
    global_score = 0.0
    num_detected = 0

    for idx, paired_rep in enumerate(paired_reps):
        if not isinstance(paired_rep, dict):
            detector_warnings.append(f"INVALID_PAIRED_REP:{idx}")
            continue
        user_rep_order = int(idx + 1)
        user_rep_raw_index = int(paired_rep.get("user_rep_raw_index", -1))
        rep_warnings: list[str] = []
        anchors = paired_rep.get("anchors") if isinstance(paired_rep.get("anchors"), dict) else {}

        hip_rep = _find_rep_result(hip_global, user_rep_order, user_rep_raw_index)
        knee_rep = _find_rep_result(knee_global, user_rep_order, user_rep_raw_index)
        neck_rep = _find_rep_result(neck_global, user_rep_order, user_rep_raw_index)

        user_kps_top: np.ndarray | None = None
        ideal_kps_top: np.ndarray | None = None
        top_pair = anchors.get("ecc_0") if isinstance(anchors, dict) else None
        if isinstance(top_pair, dict) and bool(top_pair.get("valid", False)):
            user_kps_top, ideal_kps_top = _pick_kps(top_pair, rep_warnings)

        by_anchor: dict[str, dict[str, Any]] = {}
        for anchor in SPINE_ANCHORS:
            pair = anchors.get(anchor) if isinstance(anchors, dict) else None
            if not isinstance(pair, dict) or not bool(pair.get("valid", False)):
                by_anchor[anchor] = {
                    "anchor": anchor,
                    "segment": None,
                    "status": "inconclusive",
                    "reason": "missing_anchor",
                    "user_frame": pair.get("user_frame") if isinstance(pair, dict) else None,
                    "ideal_frame": pair.get("ideal_frame") if isinstance(pair, dict) else None,
                }
                continue

            user_kps, ideal_kps = _pick_kps(pair, rep_warnings)
            if user_kps is None or ideal_kps is None:
                by_anchor[anchor] = {
                    "anchor": anchor,
                    "segment": None,
                    "status": "inconclusive",
                    "reason": "missing_clean_or_normalized_keypoints",
                    "user_frame": pair.get("user_frame"),
                    "ideal_frame": pair.get("ideal_frame"),
                }
                continue
            try:
                ideal_aligned = align_ideal_to_user_torso_for_spine_geometry(ideal_kps=ideal_kps, user_kps=user_kps)
                item = compute_spine_anchor_geometry(
                    user_kps=user_kps,
                    ideal_aligned_kps=ideal_aligned,
                    anchor=anchor,
                    user_frame=pair.get("user_frame") if isinstance(pair.get("user_frame"), int) else None,
                    ideal_frame=pair.get("ideal_frame") if isinstance(pair.get("ideal_frame"), int) else None,
                )
                if (
                    item.get("status") == "ok"
                    and user_kps_top is not None
                    and ideal_kps_top is not None
                ):
                    scale = item.get("body_scale")
                    item["user_shoulder_drop_from_top_norm"] = shoulder_drop_from_top_norm(
                        user_kps, user_kps_top, scale=float(scale) if isinstance(scale, (int, float)) else None
                    )
                    item["ideal_shoulder_drop_from_top_norm"] = shoulder_drop_from_top_norm(
                        ideal_kps, ideal_kps_top, scale=float(scale) if isinstance(scale, (int, float)) else None
                    )
                hip_metrics = None
                if isinstance(hip_rep, dict):
                    am = hip_rep.get("anchor_metrics")
                    if isinstance(am, dict):
                        hip_metrics = am.get(anchor)
                if isinstance(hip_metrics, dict):
                    user_h = hip_metrics.get("user")
                    ideal_h = hip_metrics.get("ideal")
                    if isinstance(user_h, dict) and isinstance(ideal_h, dict):
                        uhb = user_h.get("hip_back_norm")
                        ihb = ideal_h.get("hip_back_norm")
                        if isinstance(uhb, (int, float)) and np.isfinite(float(uhb)):
                            item["user_hip_back_norm"] = float(uhb)
                        if isinstance(ihb, (int, float)) and np.isfinite(float(ihb)):
                            item["ideal_hip_back_norm"] = float(ihb)
                by_anchor[anchor] = item
            except Exception as exc:
                by_anchor[anchor] = {
                    "anchor": anchor,
                    "segment": None,
                    "status": "inconclusive",
                    "reason": f"geometry_exception:{exc}",
                    "user_frame": pair.get("user_frame"),
                    "ideal_frame": pair.get("ideal_frame"),
                }

        spine_geometry = geometry_by_segment(by_anchor)
        spine_geometry["_by_anchor"] = by_anchor
        verdict = detect_spine_flexion(
            hip_result=hip_rep,
            knee_result=knee_rep,
            neck_result=neck_rep,
            spine_geometry=spine_geometry,
        )

        per_segment: dict[str, dict[str, Any]] = {}
        triggered_norms: list[float] = []
        for segment, ruling in verdict.per_segment.items():
            if ruling.triggered and isinstance(ruling.torso_low_norm, float):
                triggered_norms.append(float(ruling.torso_low_norm))
            per_segment[segment] = {
                "triggered": bool(ruling.triggered),
                "possible": bool(ruling.possible),
                "severity": str(ruling.severity),
                "reason": str(ruling.reason),
                "anchor": str(ruling.anchor),
                "torso_low_failed": bool(ruling.torso_low_failed),
                "torso_low_severity": str(ruling.torso_low_severity),
                "torso_low_norm": ruling.torso_low_norm,
                "torso_low_px": ruling.torso_low_px,
                "hip_hinge_failed": bool(ruling.hip_hinge_failed),
                "knee_dominant_failed": bool(ruling.knee_dominant_failed),
                "neck_movement_failed": bool(ruling.neck_movement_failed),
                "neck_direction": str(ruling.neck_direction),
                "geometry_status": str(ruling.geometry_status),
                "trace": list(ruling.trace),
            }

        if not verdict.detected:
            score = 0.0
        elif triggered_norms:
            score = float(max(triggered_norms))
        else:
            score = 1.0

        rep_result = {
            "user_rep_raw_index": user_rep_raw_index,
            "user_rep_order": user_rep_order,
            "detected": bool(verdict.detected),
            "severity": str(verdict.severity),
            "score": score,
            "method": str(verdict.method),
            "n_segments_triggered": int(verdict.n_segments_triggered),
            "triggered_segments": list(verdict.triggered_segments),
            "possible_segments": list(verdict.possible_segments),
            "per_segment": per_segment,
            "per_anchor": dict(verdict.per_anchor),
            "spine_geometry_by_anchor": by_anchor,
            "dependency_results": {
                "hip_hinge_available": isinstance(hip_rep, dict),
                "knee_dominant_available": isinstance(knee_rep, dict),
                "neck_movement_available": isinstance(neck_rep, dict),
            },
            "trace": list(verdict.trace),
            "warnings": sorted(set(rep_warnings)),
        }
        rep_results.append(rep_result)
        if verdict.detected:
            num_detected += 1
            global_severity = _max_severity(global_severity, str(verdict.severity))
            global_score = max(global_score, float(score))

    return {
        "detector": "spine_flexion",
        "detected": num_detected > 0,
        "severity": global_severity if num_detected > 0 else "none",
        "score": global_score if num_detected > 0 else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_detected,
        "rep_results": rep_results,
        "warnings": sorted(set(detector_warnings)),
    }
