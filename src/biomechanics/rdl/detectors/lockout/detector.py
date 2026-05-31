
from __future__ import annotations

from typing import Any

import math
import numpy as np

from src.biomechanics.rdl.detectors.lockout.metrics import LOCKOUT_ANCHOR_CANDIDATES, compute_lockout_metrics
from src.biomechanics.rdl.detectors.lockout.rules import detect_no_lockout

_SEV_RANK = {"none": 0, "leve": 1, "media": 2, "grave": 3}


def _max_severity(a: str, b: str) -> str:
    return b if _SEV_RANK.get(b, 0) > _SEV_RANK.get(a, 0) else a


def _pick_end_anchor(anchors: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    for name in LOCKOUT_ANCHOR_CANDIDATES:
        pair = anchors.get(name)
        if isinstance(pair, dict) and bool(pair.get("valid", False)):
            return name, pair
    return None, None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def detect_no_lockout_error(analysis_context: dict) -> dict:
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
        anchor_name, pair = _pick_end_anchor(anchors)
        if anchor_name is None or not isinstance(pair, dict):
            rep_warnings.append("LOCKOUT_END_ANCHOR_MISSING")
            rep_results.append(
                {
                    "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
                    "user_rep_order": int(rep_idx + 1),
                    "detected": False,
                    "severity": "none",
                    "score": 0.0,
                    "magnitude": 0.0,
                    "confidence": 0.0,
                    "anchor": None,
                    "user_frame": None,
                    "ideal_frame": None,
                    "metrics": {},
                    "trace": [],
                    "warnings": rep_warnings,
                }
            )
            continue

        ideal_kps = pair.get("ideal_kps_normalized")
        user_kps = pair.get("user_kps_normalized")
        if ideal_kps is None:
            rep_warnings.append("IDEAL_NORMALIZED_MISSING_FOR_LOCKOUT")
        if user_kps is None:
            rep_warnings.append("USER_NORMALIZED_MISSING_FOR_LOCKOUT")
        if ideal_kps is None or user_kps is None:
            rep_results.append(
                {
                    "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
                    "user_rep_order": int(rep_idx + 1),
                    "detected": False,
                    "severity": "none",
                    "score": 0.0,
                    "magnitude": 0.0,
                    "confidence": 0.0,
                    "anchor": anchor_name,
                    "user_frame": pair.get("user_frame"),
                    "ideal_frame": pair.get("ideal_frame"),
                    "metrics": {},
                    "trace": [],
                    "warnings": rep_warnings,
                }
            )
            continue

        try:
            metrics = compute_lockout_metrics(
                user_kps_end=np.asarray(user_kps, dtype=np.float64),
                ideal_kps_end=np.asarray(ideal_kps, dtype=np.float64),
            )
            verdict = detect_no_lockout(metrics)
        except Exception as exc:
            rep_warnings.append(f"LOCKOUT_METRICS_FAILED:{exc}")
            rep_results.append(
                {
                    "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
                    "user_rep_order": int(rep_idx + 1),
                    "detected": False,
                    "severity": "none",
                    "score": 0.0,
                    "magnitude": 0.0,
                    "confidence": 0.0,
                    "anchor": anchor_name,
                    "user_frame": pair.get("user_frame"),
                    "ideal_frame": pair.get("ideal_frame"),
                    "metrics": {},
                    "trace": [],
                    "warnings": rep_warnings,
                }
            )
            continue

        rep_result = {
            "user_rep_raw_index": int(paired_rep.get("user_rep_raw_index", -1)),
            "user_rep_order": int(rep_idx + 1),
            "detected": verdict.detected,
            "severity": verdict.severity,
            "score": verdict.magnitude,
            "magnitude": verdict.magnitude,
            "confidence": verdict.confidence,
            "anchor": anchor_name,
            "user_frame": pair.get("user_frame"),
            "ideal_frame": pair.get("ideal_frame"),
            "metrics": {
                "theta_end_user": metrics.theta_end_user,
                "theta_end_ideal": metrics.theta_end_ideal,
                "error_lockout": metrics.error_lockout,
                "theta_end_user_deg": verdict.theta_end_user_deg,
                "theta_end_ideal_deg": verdict.theta_end_ideal_deg,
                "error_lockout_deg": verdict.error_lockout_deg,
            },
            "trace": list(verdict.trace),
            "warnings": rep_warnings,
        }
        rep_results.append(rep_result)
        if verdict.detected:
            num_detected += 1
            global_severity = _max_severity(global_severity, verdict.severity)
            global_score = max(global_score, _safe_float(verdict.magnitude))

    return {
        "detector": "lockout",
        "detected": num_detected > 0,
        "severity": global_severity if num_detected > 0 else "none",
        "score": global_score if num_detected > 0 else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_detected,
        "rep_results": rep_results,
        "warnings": sorted(set(global_warnings)),
    }
