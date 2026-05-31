
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.biomechanics.rdl.detectors.rom.metrics import compute_rom_metrics
from src.biomechanics.rdl.detectors.rom.rules import detect_short_rom

_SEV_RANK = {"none": 0, "leve": 1, "media": 2, "grave": 3}


def _safe_deg(v: Any) -> float:
    try:
        f = float(v)
    except Exception:
        return float("nan")
    return math.degrees(f) if math.isfinite(f) else float("nan")


def _extract_clean_kps(pose_clean: dict[str, Any], frame: int | None) -> np.ndarray | None:
    if not isinstance(frame, int):
        return None
    kps = pose_clean.get("kps_xy_clean")
    if not isinstance(kps, np.ndarray) or kps.ndim != 3:
        return None
    if not (0 <= frame < kps.shape[0]):
        return None
    out = np.asarray(kps[frame], dtype=np.float64)
    return out if out.shape == (17, 2) else None


def _max_sev(a: str, b: str) -> str:
    return b if _SEV_RANK.get(b, 0) > _SEV_RANK.get(a, 0) else a


def detect_short_rom_error(analysis_context: dict) -> dict:
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

    for idx, rep in enumerate(paired_reps):
        if not isinstance(rep, dict):
            detector_warnings.append(f"INVALID_PAIRED_REP:{idx}")
            continue
        anchors = rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
        start_pair = anchors.get("ecc_0") if isinstance(anchors.get("ecc_0"), dict) else None
        bottom_pair = anchors.get("bottom") if isinstance(anchors.get("bottom"), dict) else None
        rep_warnings: list[str] = []

        if not isinstance(start_pair, dict) or not bool(start_pair.get("valid", False)):
            rep_warnings.append("ROM_REQUIRED_ANCHOR_MISSING:ecc_0")
        if not isinstance(bottom_pair, dict) or not bool(bottom_pair.get("valid", False)):
            rep_warnings.append("ROM_REQUIRED_ANCHOR_MISSING:bottom")
        if rep_warnings:
            rep_results.append(
                {
                    "user_rep_raw_index": int(rep.get("user_rep_raw_index", -1)),
                    "user_rep_order": int(idx + 1),
                    "detected": False,
                    "severity": "none",
                    "score": 0.0,
                    "magnitude": 0.0,
                    "confidence": 0.0,
                    "used_ideal": False,
                    "anchors": {"start": "ecc_0", "bottom": "bottom"},
                    "frames": {},
                    "metrics": {},
                    "trace": [],
                    "warnings": rep_warnings,
                }
            )
            continue

        user_start_frame = start_pair.get("user_frame") if isinstance(start_pair.get("user_frame"), int) else None
        user_bottom_frame = bottom_pair.get("user_frame") if isinstance(bottom_pair.get("user_frame"), int) else None
        ideal_start_frame = start_pair.get("ideal_frame") if isinstance(start_pair.get("ideal_frame"), int) else None
        ideal_bottom_frame = bottom_pair.get("ideal_frame") if isinstance(bottom_pair.get("ideal_frame"), int) else None

        user_start_kps = start_pair.get("user_kps_clean")
        user_bottom_kps = bottom_pair.get("user_kps_clean")
        if user_start_kps is None:
            user_start_kps = _extract_clean_kps(user_pose_clean if isinstance(user_pose_clean, dict) else {}, user_start_frame)
        if user_bottom_kps is None:
            user_bottom_kps = _extract_clean_kps(user_pose_clean if isinstance(user_pose_clean, dict) else {}, user_bottom_frame)
        if user_start_kps is None or user_bottom_kps is None:
            rep_warnings.append("USER_CLEAN_MISSING_FOR_ROM")
            rep_results.append(
                {
                    "user_rep_raw_index": int(rep.get("user_rep_raw_index", -1)),
                    "user_rep_order": int(idx + 1),
                    "detected": False,
                    "severity": "none",
                    "score": 0.0,
                    "magnitude": 0.0,
                    "confidence": 0.0,
                    "used_ideal": False,
                    "anchors": {"start": "ecc_0", "bottom": "bottom"},
                    "frames": {
                        "user_start": user_start_frame,
                        "user_bottom": user_bottom_frame,
                        "ideal_start": ideal_start_frame,
                        "ideal_bottom": ideal_bottom_frame,
                    },
                    "metrics": {},
                    "trace": [],
                    "warnings": rep_warnings,
                }
            )
            continue

        ideal_start_kps = start_pair.get("ideal_kps_clean")
        ideal_bottom_kps = bottom_pair.get("ideal_kps_clean")
        if ideal_start_kps is None:
            ideal_start_kps = _extract_clean_kps(ref_pose_clean if isinstance(ref_pose_clean, dict) else {}, ideal_start_frame)
        if ideal_bottom_kps is None:
            ideal_bottom_kps = _extract_clean_kps(ref_pose_clean if isinstance(ref_pose_clean, dict) else {}, ideal_bottom_frame)
        if ideal_start_kps is None or ideal_bottom_kps is None:
            ideal_start_kps = start_pair.get("ideal_kps_normalized")
            ideal_bottom_kps = bottom_pair.get("ideal_kps_normalized")
            if ideal_start_kps is not None and ideal_bottom_kps is not None:
                rep_warnings.append("IDEAL_CLEAN_MISSING_USING_NORMALIZED_FOR_ROM")
            else:
                ideal_start_kps = None
                ideal_bottom_kps = None

        try:
            metrics = compute_rom_metrics(
                user_kps_start=np.asarray(user_start_kps, dtype=np.float64),
                user_kps_bottom=np.asarray(user_bottom_kps, dtype=np.float64),
                ideal_kps_start=np.asarray(ideal_start_kps, dtype=np.float64) if ideal_start_kps is not None else None,
                ideal_kps_bottom=np.asarray(ideal_bottom_kps, dtype=np.float64) if ideal_bottom_kps is not None else None,
            )
            verdict = detect_short_rom(metrics)
        except Exception as exc:
            rep_warnings.append(f"ROM_METRICS_FAILED:{exc}")
            rep_results.append(
                {
                    "user_rep_raw_index": int(rep.get("user_rep_raw_index", -1)),
                    "user_rep_order": int(idx + 1),
                    "detected": False,
                    "severity": "none",
                    "score": 0.0,
                    "magnitude": 0.0,
                    "confidence": 0.0,
                    "used_ideal": False,
                    "anchors": {"start": "ecc_0", "bottom": "bottom"},
                    "frames": {
                        "user_start": user_start_frame,
                        "user_bottom": user_bottom_frame,
                        "ideal_start": ideal_start_frame,
                        "ideal_bottom": ideal_bottom_frame,
                    },
                    "metrics": {},
                    "trace": [],
                    "warnings": rep_warnings,
                }
            )
            continue

        score = (1.0 - verdict.rom_norm) if verdict.used_ideal and math.isfinite(verdict.rom_norm) else float(verdict.magnitude if math.isfinite(verdict.magnitude) else 0.0)
        score = max(0.0, float(score))
        rep_result = {
            "user_rep_raw_index": int(rep.get("user_rep_raw_index", -1)),
            "user_rep_order": int(idx + 1),
            "detected": verdict.detected,
            "severity": verdict.severity,
            "score": score,
            "magnitude": verdict.magnitude,
            "confidence": verdict.confidence,
            "used_ideal": verdict.used_ideal,
            "anchors": {"start": "ecc_0", "bottom": "bottom"},
            "frames": {
                "user_start": user_start_frame,
                "user_bottom": user_bottom_frame,
                "ideal_start": ideal_start_frame,
                "ideal_bottom": ideal_bottom_frame,
            },
            "metrics": {
                "theta_start_rad": metrics.theta_start,
                "theta_bottom_rad": metrics.theta_bottom,
                "theta_start_deg": _safe_deg(metrics.theta_start),
                "theta_bottom_deg": _safe_deg(metrics.theta_bottom),
                "rom_user_rad": metrics.rom_user,
                "rom_user_deg": _safe_deg(metrics.rom_user),
                "rom_user_abs_rad": metrics.rom_user_abs,
                "rom_user_abs_deg": _safe_deg(metrics.rom_user_abs),
                "rom_ideal_abs_rad": metrics.rom_ideal_abs,
                "rom_ideal_abs_deg": _safe_deg(metrics.rom_ideal_abs),
                "rom_norm": metrics.rom_norm,
            },
            "trace": list(verdict.trace),
            "warnings": rep_warnings,
        }
        rep_results.append(rep_result)

        if verdict.detected:
            num_detected += 1
            global_severity = _max_sev(global_severity, verdict.severity)
            global_score = max(global_score, score)

    return {
        "detector": "rom",
        "detected": num_detected > 0,
        "severity": global_severity if num_detected > 0 else "none",
        "score": global_score if num_detected > 0 else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_detected,
        "rep_results": rep_results,
        "warnings": sorted(set(detector_warnings)),
    }
