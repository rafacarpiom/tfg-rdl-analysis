
from __future__ import annotations

from typing import Any

import numpy as np

from src.biomechanics.rdl.detectors.hip_hinge.metrics import HIP_HINGE_ANCHORS, compute_hip_back_metrics_for_anchor
from src.biomechanics.rdl.detectors.hip_hinge.rules import detect_hip_hinge_from_trajectory

_SEV_RANK = {"none": 0, "leve": 1, "media": 2, "grave": 3}


def _max_severity(a: str, b: str) -> str:
    return b if _SEV_RANK.get(b, 0) > _SEV_RANK.get(a, 0) else a


def _get_user_clean_sequence(analysis_context: dict) -> np.ndarray | None:
    user = analysis_context.get("user") if isinstance(analysis_context, dict) else {}
    pose_clean = user.get("pose_clean") if isinstance(user, dict) and isinstance(user.get("pose_clean"), dict) else {}
    if "kps_xy_clean" in pose_clean:
        return np.asarray(pose_clean["kps_xy_clean"], dtype=np.float64)
    if "kps_xy" in pose_clean:
        return np.asarray(pose_clean["kps_xy"], dtype=np.float64)
    return None


def _get_ideal_clean_sequence(analysis_context: dict) -> np.ndarray | None:
    reference = analysis_context.get("reference") if isinstance(analysis_context, dict) else {}
    ref_pose = reference.get("pose_clean") if isinstance(reference, dict) and isinstance(reference.get("pose_clean"), dict) else {}
    if "kps_xy_clean" in ref_pose:
        return np.asarray(ref_pose["kps_xy_clean"], dtype=np.float64)
    if "kps_xy" in ref_pose:
        return np.asarray(ref_pose["kps_xy"], dtype=np.float64)
    norm = reference.get("normalization") if isinstance(reference, dict) and isinstance(reference.get("normalization"), dict) else {}
    if "kps_xy_clean" in norm and norm.get("kps_xy_clean") is not None:
        return np.asarray(norm["kps_xy_clean"], dtype=np.float64)
    return None


def detect_hip_hinge(analysis_context: dict) -> dict:
    global_warnings: list[str] = []
    user_kps_xy = _get_user_clean_sequence(analysis_context)
    if user_kps_xy is None:
        return {
            "detector": "hip_hinge",
            "detected": False,
            "severity": "none",
            "score": 0.0,
            "num_reps_analyzed": 0,
            "num_reps_detected": 0,
            "rep_results": [],
            "warnings": ["USER_CLEAN_MISSING_FOR_HIP_HINGE"],
        }
    ideal_kps_xy = _get_ideal_clean_sequence(analysis_context)
    if ideal_kps_xy is None:
        global_warnings.append("IDEAL_CLEAN_MISSING_FOR_HIP_HINGE")

    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    paired_reps = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else []
    if not isinstance(paired_reps, list):
        paired_reps = []

    rep_results: list[dict[str, Any]] = []
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
        user_start_frame = None
        ideal_start_frame = None
        ecc0 = anchors.get("ecc_0") if isinstance(anchors, dict) else None
        if isinstance(ecc0, dict) and bool(ecc0.get("valid", False)):
            if isinstance(ecc0.get("user_frame"), int):
                user_start_frame = int(ecc0["user_frame"])
            if isinstance(ecc0.get("ideal_frame"), int):
                ideal_start_frame = int(ecc0["ideal_frame"])
        if user_start_frame is None:
            rep_warnings.append("USER_START_FRAME_MISSING:ecc_0")
        if ideal_start_frame is None:
            rep_warnings.append("IDEAL_START_FRAME_MISSING:ecc_0")

        for anchor in HIP_HINGE_ANCHORS:
            pair = anchors.get(anchor) if isinstance(anchors, dict) else None
            if not isinstance(pair, dict) or not bool(pair.get("valid", False)):
                rep_warnings.append(f"ANCHOR_INVALID_OR_MISSING:{anchor}")
                continue
            uf = pair.get("user_frame")
            inf = pair.get("ideal_frame")
            if not isinstance(uf, int) or not isinstance(inf, int):
                rep_warnings.append(f"ANCHOR_FRAMES_MISSING:{anchor}")
                continue
            if user_start_frame is None or ideal_start_frame is None:
                continue

            ideal_seq = ideal_kps_xy
            local_ideal_start = ideal_start_frame
            local_ideal_anchor = int(inf)
            if ideal_seq is None and pair.get("ideal_kps_clean") is not None:
                ideal_seq = np.asarray([pair["ideal_kps_clean"]], dtype=np.float64)
                local_ideal_anchor = 0
                local_ideal_start = 0
            if ideal_seq is None:
                rep_warnings.append("IDEAL_CLEAN_MISSING_FOR_HIP_HINGE")
                continue
            try:
                metrics = compute_hip_back_metrics_for_anchor(
                    user_kps_xy=user_kps_xy,
                    ideal_kps_xy=ideal_seq,
                    user_start_frame=int(user_start_frame),
                    ideal_start_frame=int(local_ideal_start),
                    user_anchor_frame=int(uf),
                    ideal_anchor_frame=int(local_ideal_anchor),
                    anchor=anchor,
                )
                metrics_by_anchor[anchor] = metrics
            except Exception as exc:
                rep_warnings.append(f"ANCHOR_METRICS_FAILED:{anchor}:{exc}")

        verdict = detect_hip_hinge_from_trajectory(metrics_by_anchor)
        anchor_metrics_out: dict[str, Any] = {}
        for anchor, m in metrics_by_anchor.items():
            anchor_metrics_out[anchor] = {
                "delta_hip_back": m.delta_hip_back,
                "user": {
                    "hip_x_start": m.user.hip_x_start,
                    "hip_x_anchor": m.user.hip_x_anchor,
                    "hip_back_px": m.user.hip_back_px,
                    "torso_length": m.user.torso_length,
                    "hip_back_norm": m.user.hip_back_norm,
                },
                "ideal": {
                    "hip_x_start": m.ideal.hip_x_start,
                    "hip_x_anchor": m.ideal.hip_x_anchor,
                    "hip_back_px": m.ideal.hip_back_px,
                    "torso_length": m.ideal.torso_length,
                    "hip_back_norm": m.ideal.hip_back_norm,
                },
            }
        anchor_rulings_out: dict[str, Any] = {}
        for anchor, r in verdict.per_anchor.items():
            anchor_rulings_out[anchor] = {
                "failed": r.failed,
                "delta_hip_back": r.delta_hip_back,
                "trace": list(r.trace),
            }
        failed_anchors = [a for a, r in verdict.per_anchor.items() if r.failed]
        rep_result = {
            "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
            "user_rep_order": int(rep_idx + 1),
            "detected": bool(verdict.detected),
            "severity": verdict.severity,
            "score": verdict.magnitude,
            "confidence": verdict.confidence,
            "dominant_phase": verdict.dominant_phase,
            "magnitude": verdict.magnitude,
            "mean_deficit": verdict.mean_deficit,
            "max_deficit": verdict.max_deficit,
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
        "detector": "hip_hinge",
        "detected": num_detected > 0,
        "severity": global_severity if num_detected > 0 else "none",
        "score": global_score if num_detected > 0 else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_detected,
        "rep_results": rep_results,
        "warnings": sorted(set(global_warnings)),
    }
