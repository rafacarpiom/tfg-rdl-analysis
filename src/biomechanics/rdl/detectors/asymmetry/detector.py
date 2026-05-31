
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.biomechanics.rdl.detectors.asymmetry.metrics import (
    ARM_JOINTS,
    ARM_PRIMARY_JOINTS,
    ARM_SECONDARY_JOINTS,
    LEG_JOINTS,
    LEG_PRIMARY_JOINTS,
    LEG_SECONDARY_JOINTS,
    frame_asymmetry,
)
from src.biomechanics.rdl.detectors.asymmetry.rules import (
    CONSISTENCY_THR,
    NORM_THR,
    RATIO_THR,
    Severity,
    classify_group,
    frame_severity,
    max_severity,
    stability_label,
)

_SEV_RANK = {"none": 0, "posible": 1, "leve": 2, "media": 3, "grave": 4}
ARM_SECONDARY_REINFORCEMENT_THR_MEDIA = 0.35
ARM_SECONDARY_REINFORCEMENT_THR_GRAVE = 0.50
LEG_GRAVE_RESCUE_MAGNITUDE_THR = 0.50
LEG_GRAVE_RESCUE_ANKLE_MAX_THR = 0.60
LEG_GRAVE_RESCUE_KNEE_MAX_THR = 0.30
LEG_GRAVE_RESCUE_RATIO_THR = 0.50
LEG_GRAVE_RESCUE_REINFORCEMENT_THR = 0.70


@dataclass
class JointStats:
    joint: str
    severity: Severity
    mean_fwd_norm: float
    max_fwd_norm: float
    std_fwd_norm: float
    n_frames: int
    dominant_side: str
    consistency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "mean_fwd_norm": _safe_round(self.mean_fwd_norm),
            "max_fwd_norm": _safe_round(self.max_fwd_norm),
            "std_fwd_norm": _safe_round(self.std_fwd_norm),
            "n_frames": self.n_frames,
            "dominant_side": self.dominant_side,
            "consistency": _safe_round(self.consistency),
        }


@dataclass
class PhaseStats:
    phase: str
    ratio: float
    dominant_side: str
    consistency: float
    side_imbalance: float
    mean_fwd_norm: float
    max_fwd_norm: float
    std_fwd_norm: float
    secondary_reinforcement: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": _safe_round(self.ratio),
            "dominant_side": self.dominant_side,
            "consistency": _safe_round(self.consistency),
            "side_imbalance": _safe_round(self.side_imbalance),
            "mean_fwd_norm": _safe_round(self.mean_fwd_norm),
            "max_fwd_norm": _safe_round(self.max_fwd_norm),
            "std_fwd_norm": _safe_round(self.std_fwd_norm),
            "secondary_reinforcement": _safe_round(self.secondary_reinforcement),
        }


@dataclass
class GroupAsymmetryResult:
    severity: Severity
    dominant_phase: str
    dominant_side: str
    consistency: float
    side_imbalance: float
    stability: str
    confidence: str
    magnitude: float
    phase_ratios: dict[str, float]
    phase_stats: dict[str, PhaseStats]
    joint_breakdown: dict[str, JointStats]
    secondary_reinforcement: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "dominant_phase": self.dominant_phase,
            "dominant_side": self.dominant_side,
            "consistency": _safe_round(self.consistency),
            "side_imbalance": _safe_round(self.side_imbalance),
            "stability": self.stability,
            "confidence": self.confidence,
            "magnitude": _safe_round(self.magnitude),
            "phase_ratios": self.phase_ratios,
            "phase_stats": {k: v.to_dict() for k, v in self.phase_stats.items()},
            "joint_breakdown": {k: v.to_dict() for k, v in self.joint_breakdown.items()},
            "secondary_reinforcement": _safe_round(self.secondary_reinforcement),
        }


def _safe_round(v: float, n: int = 4) -> float | None:
    return round(v, n) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _phase_ranges(rep: dict) -> dict[str, tuple[int, int]]:
    phases: dict[str, tuple[int, int]] = {}
    if "ecc_start" in rep and "ecc_end" in rep:
        phases["ecc"] = (int(rep["ecc_start"]), int(rep["ecc_end"]))
    elif "max_start" in rep and "min_bottom" in rep:
        phases["ecc"] = (int(rep["max_start"]), int(rep["min_bottom"]))

    if "con_start" in rep and "con_end" in rep:
        phases["con"] = (int(rep["con_start"]), int(rep["con_end"]))
    elif "min_bottom" in rep and "max_end" in rep:
        phases["con"] = (int(rep["min_bottom"]), int(rep["max_end"]))
    return phases


def _confidence_label(high: int, low: int) -> str:
    total = high + low
    if total == 0:
        return "low"
    ratio_low = low / total
    if ratio_low > 0.5:
        return "low"
    if ratio_low > 0.2:
        return "mixed"
    return "high"


def _sign_to_side(sign_count_l: int, sign_count_r: int) -> tuple[str, float]:
    total = sign_count_l + sign_count_r
    if total == 0:
        return "none", 0.0
    if sign_count_l >= sign_count_r:
        return "L", sign_count_l / total
    return "R", sign_count_r / total


def _aggregate_phase(primary_joints: tuple[str, ...], secondary_joints: tuple[str, ...], frames: list[dict[str, Any]]) -> tuple[PhaseStats, int, dict[str, tuple[int, int]], dict[str, list[tuple[float, float, bool]]], int, int]:
    primary_fwd: list[float] = []
    asym_frames = 0
    valid_frames = 0
    sign_l = 0
    sign_r = 0
    worst_idx = -1
    worst_score = -1.0
    high_conf = 0
    low_conf = 0
    co_asym = 0
    primary_asym_count = 0
    joint_trace: dict[str, list[tuple[float, float, bool]]] = {}
    joint_sign_counts: dict[str, tuple[int, int]] = {}

    for frame_data in frames:
        primary_vals: list[float] = []
        primary_signs: list[float] = []
        primary_confident_all = True
        primary_any_asym = False

        for joint in primary_joints:
            a = frame_data.get(joint)
            if a is None or not math.isfinite(a.forward_diff_norm):
                continue
            primary_vals.append(a.forward_diff_norm)
            primary_signs.append(a.signed_diff)
            if not a.confident:
                primary_confident_all = False
            if frame_severity(a.forward_diff_norm, a.confident) != "none":
                primary_any_asym = True
            joint_trace.setdefault(joint, []).append((a.forward_diff_norm, a.signed_diff, a.confident))

        secondary_any_asym = False
        for joint in secondary_joints:
            a = frame_data.get(joint)
            if a is None or not math.isfinite(a.forward_diff_norm):
                continue
            if frame_severity(a.forward_diff_norm, a.confident) != "none":
                secondary_any_asym = True
            joint_trace.setdefault(joint, []).append((a.forward_diff_norm, a.signed_diff, a.confident))

        if not primary_vals:
            continue
        valid_frames += 1
        mean_primary = float(np.mean(primary_vals))
        max_primary = float(np.max(primary_vals))
        primary_fwd.append(mean_primary)

        if primary_any_asym:
            asym_frames += 1
            primary_asym_count += 1
            mean_signed = float(np.mean(primary_signs))
            if mean_signed > 0:
                sign_l += 1
            elif mean_signed < 0:
                sign_r += 1
            if secondary_any_asym:
                co_asym += 1

        if primary_confident_all:
            high_conf += 1
        else:
            low_conf += 1

        if max_primary > worst_score:
            worst_score = max_primary
            worst_idx = int(frame_data.get("__frame_idx", -1))

    for joint, entries in joint_trace.items():
        n_l = sum(1 for fv, sv, conf in entries if fv > NORM_THR["leve"] and conf and sv > 0)
        n_r = sum(1 for fv, sv, conf in entries if fv > NORM_THR["leve"] and conf and sv < 0)
        joint_sign_counts[joint] = (n_l, n_r)

    ratio = (asym_frames / valid_frames) if valid_frames > 0 else 0.0
    dominant_side, consistency = _sign_to_side(sign_l, sign_r)
    side_imbalance = (abs(sign_l - sign_r) / valid_frames) if valid_frames > 0 else 0.0
    mean_fwd = float(np.mean(primary_fwd)) if primary_fwd else float("nan")
    max_fwd = float(np.max(primary_fwd)) if primary_fwd else float("nan")
    asym_primary = [v for v in primary_fwd if v > NORM_THR["leve"]]
    std_fwd = float(np.std(asym_primary)) if len(asym_primary) >= 3 else 0.0
    reinforcement = (co_asym / primary_asym_count) if primary_asym_count > 0 else 0.0
    stats = PhaseStats(
        phase="",
        ratio=ratio,
        dominant_side=dominant_side,
        consistency=consistency,
        side_imbalance=side_imbalance,
        mean_fwd_norm=mean_fwd,
        max_fwd_norm=max_fwd,
        std_fwd_norm=std_fwd,
        secondary_reinforcement=reinforcement,
    )
    return stats, worst_idx, joint_sign_counts, joint_trace, high_conf, low_conf


def _aggregate_group(primary_joints: tuple[str, ...], secondary_joints: tuple[str, ...], all_joints: tuple[str, ...], phase_frame_data: dict[str, list[dict[str, Any]]]) -> tuple[GroupAsymmetryResult, dict[str, int]]:
    phase_stats_map: dict[str, PhaseStats] = {}
    phase_ratios: dict[str, float] = {}
    worst_frame_per_phase: dict[str, int] = {}
    total_high_conf = 0
    total_low_conf = 0
    merged_trace: dict[str, list[tuple[float, float, bool]]] = {j: [] for j in all_joints}
    merged_signs: dict[str, list[int]] = {j: [0, 0] for j in all_joints}

    for phase, frames in phase_frame_data.items():
        stats, worst_idx, sign_counts, trace, hc, lc = _aggregate_phase(primary_joints, secondary_joints, frames)
        stats.phase = phase
        phase_stats_map[phase] = stats
        phase_ratios[phase] = round(stats.ratio, 4)
        if worst_idx >= 0:
            worst_frame_per_phase[phase] = worst_idx
        total_high_conf += hc
        total_low_conf += lc
        for joint, entries in trace.items():
            merged_trace.setdefault(joint, []).extend(entries)
            n_l, n_r = sign_counts.get(joint, (0, 0))
            cur = merged_signs.setdefault(joint, [0, 0])
            cur[0] += n_l
            cur[1] += n_r

    dominant_phase = max(phase_ratios, key=lambda p: phase_ratios[p]) if phase_ratios else ""
    dom_stats = phase_stats_map.get(
        dominant_phase,
        PhaseStats("", 0.0, "none", 0.0, 0.0, float("nan"), float("nan"), float("nan"), 0.0),
    )
    multi_above_media = sum(1 for s in phase_stats_map.values() if s.ratio > RATIO_THR["media"]) >= 2
    confidence = _confidence_label(total_high_conf, total_low_conf)
    mean_m = dom_stats.mean_fwd_norm if math.isfinite(dom_stats.mean_fwd_norm) else 0.0
    max_m = dom_stats.max_fwd_norm if math.isfinite(dom_stats.max_fwd_norm) else 0.0
    magnitude_blend = 0.7 * mean_m + 0.3 * max_m if max_m > 0.0 else mean_m

    classification = classify_group(
        magnitude=magnitude_blend,
        ratio=dom_stats.ratio,
        consistency=dom_stats.consistency,
        side_imbalance=dom_stats.side_imbalance,
        stability_std=dom_stats.std_fwd_norm if math.isfinite(dom_stats.std_fwd_norm) else 0.0,
        dominant_side=dom_stats.dominant_side,
        dominant_phase=dominant_phase,
        confidence_label=confidence,
        multi_phase_above_media=multi_above_media,
    )

    joint_breakdown: dict[str, JointStats] = {}
    for joint in all_joints:
        entries = merged_trace.get(joint, [])
        if entries:
            vals = np.array([e[0] for e in entries], dtype=float)
            mean_v = float(np.nanmean(vals))
            max_v = float(np.nanmax(vals))
            std_v = float(np.nanstd(vals))
            sev: Severity = "none"
            for fv, sv, conf in entries:
                sev = max_severity(sev, frame_severity(fv, conf))
            n_l, n_r = merged_signs.get(joint, [0, 0])
            d_side, cons = _sign_to_side(n_l, n_r)
        else:
            mean_v = max_v = std_v = float("nan")
            sev = "none"
            d_side, cons = "none", 0.0
        if joint in secondary_joints and sev in ("media", "grave"):
            primary_mean = dom_stats.mean_fwd_norm
            if not (math.isfinite(primary_mean) and primary_mean > NORM_THR["media"]):
                sev = "leve"
        joint_breakdown[joint] = JointStats(joint, sev, mean_v, max_v, std_v, len(entries), d_side, cons)

    return (
        GroupAsymmetryResult(
            severity=classification.severity,
            dominant_phase=classification.dominant_phase,
            dominant_side=classification.dominant_side,
            consistency=classification.consistency,
            side_imbalance=classification.side_imbalance,
            stability=stability_label(classification.stability_std),
            confidence=classification.confidence_label,
            magnitude=classification.magnitude,
            phase_ratios=phase_ratios,
            phase_stats=phase_stats_map,
            joint_breakdown=joint_breakdown,
            secondary_reinforcement=dom_stats.secondary_reinforcement,
        ),
        worst_frame_per_phase,
    )


def _severity_max(a: str, b: str) -> str:
    return a if _SEV_RANK.get(a, 0) >= _SEV_RANK.get(b, 0) else b


def _apply_arm_conservatism(result: GroupAsymmetryResult) -> tuple[GroupAsymmetryResult, bool, str]:
    original = str(result.severity)
    reinforcement = float(result.secondary_reinforcement or 0.0)
    downgraded = False
    reason = "none"
    new_sev = original

    if original == "grave":
        if reinforcement < ARM_SECONDARY_REINFORCEMENT_THR_MEDIA:
            new_sev = "leve"
            downgraded = True
            reason = "wrist_only_weak_elbow_reinforcement"
        elif reinforcement < ARM_SECONDARY_REINFORCEMENT_THR_GRAVE:
            new_sev = "media"
            downgraded = True
            reason = "grave_blocked_by_elbow_reinforcement"
    elif original == "media" and reinforcement < ARM_SECONDARY_REINFORCEMENT_THR_MEDIA:
        new_sev = "leve"
        downgraded = True
        reason = "media_blocked_by_elbow_reinforcement"

    if not downgraded:
        return result, False, reason
    return GroupAsymmetryResult(
        severity=new_sev,
        dominant_phase=result.dominant_phase,
        dominant_side=result.dominant_side,
        consistency=result.consistency,
        side_imbalance=result.side_imbalance,
        stability=result.stability,
        confidence=result.confidence,
        magnitude=result.magnitude,
        phase_ratios=result.phase_ratios,
        phase_stats=result.phase_stats,
        joint_breakdown=result.joint_breakdown,
        secondary_reinforcement=result.secondary_reinforcement,
    ), True, reason


def _joint_mean_max(result: GroupAsymmetryResult, joint: str) -> tuple[float | None, float | None]:
    js = result.joint_breakdown.get(joint) if isinstance(result.joint_breakdown, dict) else None
    if js is None:
        return None, None
    mean_v = getattr(js, "mean_fwd_norm", None)
    max_v = getattr(js, "max_fwd_norm", None)
    mean_out = float(mean_v) if isinstance(mean_v, (int, float)) and math.isfinite(float(mean_v)) else None
    max_out = float(max_v) if isinstance(max_v, (int, float)) and math.isfinite(float(max_v)) else None
    return mean_out, max_out


def _dominant_ratio(result: GroupAsymmetryResult) -> float:
    phase = str(result.dominant_phase or "")
    ratios = result.phase_ratios if isinstance(result.phase_ratios, dict) else {}
    try:
        r = float(ratios.get(phase, 0.0))
    except Exception:
        r = 0.0
    return r if math.isfinite(r) else 0.0


def _apply_arm_perspective_filter(
    result: GroupAsymmetryResult,
) -> tuple[GroupAsymmetryResult, bool, str, dict[str, float | None]]:
    wrist_mean, wrist_max = _joint_mean_max(result, "wrist")
    elbow_mean, elbow_max = _joint_mean_max(result, "elbow")
    ratio = _dominant_ratio(result)
    reinforcement = float(result.secondary_reinforcement or 0.0)
    consistency = float(result.consistency or 0.0)

    applied = False
    reason = "none"
    new_sev = str(result.severity)

    # Gates media/grave: codo acompaña + refuerzo fuerte + ratio + consistencia.
    if new_sev in {"media", "grave"}:
        elbow_ok_for_media = (elbow_mean is not None and elbow_mean > NORM_THR["leve"]) or (
            elbow_max is not None and elbow_max > NORM_THR["media"]
        )
        if not elbow_ok_for_media:
            new_sev = "leve"
            applied = True
            reason = "blocked_media_grave_by_elbow_not_accompanying"
        elif reinforcement < 0.50 or ratio <= RATIO_THR["media"] or consistency < CONSISTENCY_THR:
            new_sev = "leve"
            applied = True
            reason = "blocked_media_grave_by_weak_ratio_consistency_or_reinforcement"

        if str(result.severity) == "grave" and applied:
            # Codo débil para media → degradar más.
            elbow_ok_for_grave = elbow_max is not None and elbow_max > NORM_THR["media"]
            if not elbow_ok_for_grave:
                new_sev = "leve"
                reason = "blocked_grave_by_elbow_below_media"

    # Gate leve: solo muñeca leve sin codo → preferir none/posible.
    if new_sev == "leve":
        elbow_support = elbow_mean is not None and elbow_mean > NORM_THR["leve"]
        wrist_support = wrist_mean is not None and wrist_mean > NORM_THR["leve"]
        if wrist_support and not elbow_support:
            # Confianza alta pero codo quieto → ruido de perspectiva.
            if str(result.confidence) == "high":
                new_sev = "none"
                applied = True
                reason = "wrist_only_high_confidence_perspective_noise"
            else:
                new_sev = "posible"
                applied = True
                reason = "wrist_only_low_mixed_confidence"

    if not applied:
        return result, False, reason, {
            "wrist_mean_fwd_norm": wrist_mean,
            "wrist_max_fwd_norm": wrist_max,
            "elbow_mean_fwd_norm": elbow_mean,
            "elbow_max_fwd_norm": elbow_max,
        }

    updated = GroupAsymmetryResult(
        severity=new_sev,  # type: ignore[arg-type]
        dominant_phase=result.dominant_phase,
        dominant_side=result.dominant_side,
        consistency=result.consistency,
        side_imbalance=result.side_imbalance,
        stability=result.stability,
        confidence=result.confidence,
        magnitude=result.magnitude,
        phase_ratios=result.phase_ratios,
        phase_stats=result.phase_stats,
        joint_breakdown=result.joint_breakdown,
        secondary_reinforcement=result.secondary_reinforcement,
    )
    return updated, True, reason, {
        "wrist_mean_fwd_norm": wrist_mean,
        "wrist_max_fwd_norm": wrist_max,
        "elbow_mean_fwd_norm": elbow_mean,
        "elbow_max_fwd_norm": elbow_max,
    }


def _apply_leg_grave_rescue_or_no_stability_demote(
    legs_result: GroupAsymmetryResult,
) -> tuple[GroupAsymmetryResult, bool, str, str]:
    ankle_mean, ankle_max = _joint_mean_max(legs_result, "ankle")
    knee_mean, knee_max = _joint_mean_max(legs_result, "knee")
    _ = ankle_mean, knee_mean
    ratio = _dominant_ratio(legs_result)
    reinforcement = float(legs_result.secondary_reinforcement or 0.0)
    consistency = float(legs_result.consistency or 0.0)
    magnitude = float(legs_result.magnitude or 0.0)

    severity_before = str(legs_result.severity)
    rescue = False
    reason = "none"

    extreme_by_max = (
        (ankle_max is not None and ankle_max > LEG_GRAVE_RESCUE_ANKLE_MAX_THR)
        and (knee_max is not None and knee_max > LEG_GRAVE_RESCUE_KNEE_MAX_THR)
        and ratio > LEG_GRAVE_RESCUE_RATIO_THR
    )
    extreme_by_magnitude = (
        magnitude > LEG_GRAVE_RESCUE_MAGNITUDE_THR
        and ratio > RATIO_THR["media"]
        and reinforcement >= LEG_GRAVE_RESCUE_REINFORCEMENT_THR
        and consistency >= CONSISTENCY_THR
    )

    if str(legs_result.stability) == "low" and (extreme_by_max or extreme_by_magnitude):
        rescue = True
        reason = "extreme_leg_asymmetry_rescue_ignore_stability"

    if not rescue:
        return legs_result, False, reason, severity_before

    updated = GroupAsymmetryResult(
        severity="grave",  # type: ignore[arg-type]
        dominant_phase=legs_result.dominant_phase,
        dominant_side=legs_result.dominant_side,
        consistency=legs_result.consistency,
        side_imbalance=legs_result.side_imbalance,
        stability=legs_result.stability,
        confidence=legs_result.confidence,
        magnitude=legs_result.magnitude,
        phase_ratios=legs_result.phase_ratios,
        phase_stats=legs_result.phase_stats,
        joint_breakdown=legs_result.joint_breakdown,
        secondary_reinforcement=legs_result.secondary_reinforcement,
    )
    return updated, True, reason, severity_before


def _apply_leg_mean_guard(
    legs_result: GroupAsymmetryResult,
) -> tuple[GroupAsymmetryResult, bool, str]:
    dom_phase = str(legs_result.dominant_phase or "")
    stats = (legs_result.phase_stats or {}).get(dom_phase)
    mean_dom = float(getattr(stats, "mean_fwd_norm", 0.0)) if stats is not None else 0.0
    if not math.isfinite(mean_dom):
        mean_dom = 0.0

    sev = str(legs_result.severity)
    if sev not in {"media", "grave"}:
        return legs_result, False, "none"
    if mean_dom > NORM_THR["media"]:
        return legs_result, False, "none"

    new_sev = "leve" if sev == "media" else "media"
    updated = GroupAsymmetryResult(
        severity=new_sev,  # type: ignore[arg-type]
        dominant_phase=legs_result.dominant_phase,
        dominant_side=legs_result.dominant_side,
        consistency=legs_result.consistency,
        side_imbalance=legs_result.side_imbalance,
        stability=legs_result.stability,
        confidence=legs_result.confidence,
        magnitude=legs_result.magnitude,
        phase_ratios=legs_result.phase_ratios,
        phase_stats=legs_result.phase_stats,
        joint_breakdown=legs_result.joint_breakdown,
        secondary_reinforcement=legs_result.secondary_reinforcement,
    )
    return updated, True, "dominant_phase_mean_below_media_threshold"


def _legs_grave_block_reasons(legs_result: GroupAsymmetryResult) -> list[str]:
    out: list[str] = []
    if float(legs_result.magnitude or 0.0) <= NORM_THR["grave"]:
        out.append("magnitude_below_grave")
    dom_phase = str(legs_result.dominant_phase or "")
    ratio = float((legs_result.phase_ratios or {}).get(dom_phase, 0.0))
    if ratio <= RATIO_THR["grave"]:
        out.append("ratio_below_grave")
    if float(legs_result.consistency or 0.0) < CONSISTENCY_THR:
        out.append("consistency_below_grave")
    if float(legs_result.side_imbalance or 0.0) < 0.60:
        out.append("side_imbalance_below_grave")
    if str(legs_result.stability or "") == "low":
        out.append("stability_demoted")
    return out


def detect_asymmetry(
    analysis_context: dict,
    *,
    thr_conf: float = 0.3,
) -> dict:
    warnings: list[str] = []
    user = analysis_context.get("user") if isinstance(analysis_context, dict) else {}
    pose_clean = user.get("pose_clean") if isinstance(user, dict) and isinstance(user.get("pose_clean"), dict) else {}
    segmentation = user.get("segmentation_result") if isinstance(user, dict) and isinstance(user.get("segmentation_result"), dict) else {}
    if "kps_xy_clean" in pose_clean:
        kps_xy = np.asarray(pose_clean["kps_xy_clean"], dtype=np.float64)
    elif "kps_xy" in pose_clean:
        kps_xy = np.asarray(pose_clean["kps_xy"], dtype=np.float64)
        warnings.append("POSE_CLEAN_FALLBACK_TO_KPS_XY")
    else:
        return {
            "detector": "asymmetry",
            "detected": False,
            "severity": "none",
            "score": 0.0,
            "num_reps_analyzed": 0,
            "num_reps_detected": 0,
            "rep_results": [],
            "warnings": ["POSE_KEYPOINTS_MISSING_IN_ANALYSIS_CONTEXT"],
        }
    if "kps_score_clean" in pose_clean:
        kps_score = np.asarray(pose_clean["kps_score_clean"], dtype=np.float64)
    elif "kps_score" in pose_clean:
        kps_score = np.asarray(pose_clean["kps_score"], dtype=np.float64)
        warnings.append("POSE_SCORE_FALLBACK_TO_KPS_SCORE")
    else:
        return {
            "detector": "asymmetry",
            "detected": False,
            "severity": "none",
            "score": 0.0,
            "num_reps_analyzed": 0,
            "num_reps_detected": 0,
            "rep_results": [],
            "warnings": warnings + ["POSE_SCORES_MISSING_IN_ANALYSIS_CONTEXT"],
        }
    if kps_xy.ndim != 3 or kps_xy.shape[1:] != (17, 2) or kps_score.ndim != 2 or kps_score.shape[0] != kps_xy.shape[0]:
        return {
            "detector": "asymmetry",
            "detected": False,
            "severity": "none",
            "score": 0.0,
            "num_reps_analyzed": 0,
            "num_reps_detected": 0,
            "rep_results": [],
            "warnings": warnings + ["POSE_SHAPES_INVALID"],
        }

    reps = segmentation.get("reps") if isinstance(segmentation, dict) else []
    if not isinstance(reps, list):
        reps = []
    rep_results: list[dict[str, Any]] = []
    global_severity = "none"
    max_score = 0.0
    num_detected = 0
    for rep_idx, rep in enumerate(reps):
        if not isinstance(rep, dict):
            continue
        if rep.get("anchor_valid", True) is not True:
            continue
        phases = _phase_ranges(rep)
        phase_frame_data: dict[str, list[dict[str, Any]]] = {}
        rep_warn: list[str] = []
        for phase, (start, end) in phases.items():
            frames: list[dict[str, Any]] = []
            if start > end:
                start, end = end, start
            for fi in range(start, end + 1):
                if not (0 <= fi < kps_xy.shape[0]):
                    continue
                fa = frame_asymmetry(kps_xy[fi], kps_score[fi], thr_conf=thr_conf)
                fa["__frame_idx"] = fi  # type: ignore[index]
                frames.append(fa)
            if not frames:
                rep_warn.append(f"NO_FRAMES_IN_PHASE:{phase}")
            phase_frame_data[phase] = frames
        if not phase_frame_data:
            rep_warn.append("NO_PHASES_AVAILABLE")
        arms_result_raw, arms_worst = _aggregate_group(ARM_PRIMARY_JOINTS, ARM_SECONDARY_JOINTS, ARM_JOINTS, phase_frame_data)
        arms_after_cons, arms_conservatism_applied, arms_conservatism_reason = _apply_arm_conservatism(arms_result_raw)
        arms_result, arm_pf_applied, arm_pf_reason, arm_joint_stats = _apply_arm_perspective_filter(arms_after_cons)

        legs_result_raw, legs_worst = _aggregate_group(LEG_PRIMARY_JOINTS, LEG_SECONDARY_JOINTS, LEG_JOINTS, phase_frame_data)
        legs_after_mean_guard, leg_mean_guard_applied, leg_mean_guard_reason = _apply_leg_mean_guard(legs_result_raw)
        legs_result, leg_rescue_applied, leg_rescue_reason, legs_severity_before_rescue = _apply_leg_grave_rescue_or_no_stability_demote(legs_after_mean_guard)
        rep_severity = _severity_max(arms_result.severity, legs_result.severity)
        rep_detected = (arms_result.severity != "none") or (legs_result.severity != "none")
        rep_score = float(max(arms_result.magnitude, legs_result.magnitude))
        legs_grave_blocked_by: list[str] = []
        if legs_result.severity != "grave":
            legs_grave_blocked_by = _legs_grave_block_reasons(legs_result)
        else:
            # Rescate a grave → quitar marcadores de degradación por estabilidad.
            legs_grave_blocked_by = [x for x in _legs_grave_block_reasons(legs_result) if x != "stability_demoted"]
        rep_out = {
            "user_rep_raw_index": int(rep_idx),
            "user_rep_order": int(rep_idx + 1),
            "detected": rep_detected,
            "severity": rep_severity,
            "score": rep_score,
            "arms": {
                **arms_result.to_dict(),
                "severity_before_group_adjustment": str(arms_result_raw.severity),
                "secondary_reinforcement": _safe_round(arms_result.secondary_reinforcement),
                "arm_conservatism_applied": bool(arms_conservatism_applied),
                "arm_conservatism_reason": str(arms_conservatism_reason),
                "arm_perspective_filter_applied": bool(arm_pf_applied),
                "arm_perspective_filter_reason": str(arm_pf_reason),
                **arm_joint_stats,
            },
            "legs": {
                **legs_result.to_dict(),
                "severity_before_group_adjustment": str(legs_result_raw.severity),
                "leg_mean_guard_applied": bool(leg_mean_guard_applied),
                "leg_mean_guard_reason": str(leg_mean_guard_reason),
                "severity_before_stability_or_rescue": str(legs_severity_before_rescue),
                "secondary_reinforcement": _safe_round(legs_result.secondary_reinforcement),
                "leg_grave_rescue_applied": bool(leg_rescue_applied),
                "leg_grave_rescue_reason": str(leg_rescue_reason),
                "grave_blocked_by": legs_grave_blocked_by,
            },
            "worst_frames": {"arms": arms_worst, "legs": legs_worst},
            "warnings": rep_warn,
        }
        rep_results.append(rep_out)
        if rep_detected:
            num_detected += 1
            global_severity = _severity_max(global_severity, rep_severity)
            max_score = max(max_score, rep_score)

    return {
        "detector": "asymmetry",
        "detected": num_detected > 0,
        "severity": global_severity if num_detected > 0 else "none",
        "score": max_score if num_detected > 0 else 0.0,
        "num_reps_analyzed": len(rep_results),
        "num_reps_detected": num_detected,
        "rep_results": rep_results,
        "warnings": warnings,
    }
