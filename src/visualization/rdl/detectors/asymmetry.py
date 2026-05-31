
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.biomechanics.rdl.detectors.asymmetry import (
    ARM_JOINTS,
    BILATERAL_PAIRS,
    LEG_JOINTS,
    frame_asymmetry,
)
from src.visualization.rdl.debug.common import ensure_dir, save_json

_FULL_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12),
)
_SEV_BG = {"none": "#ecf0f1", "posible": "#fff8e1", "leve": "#fff3bf", "media": "#ffe0b2", "grave": "#ffcdd2"}
_SEV_COLOR = {"none": "#95a5a6", "posible": "#f39c12", "leve": "#f1c40f", "media": "#e67e22", "grave": "#e74c3c"}
_SEV_LEGEND = "colors: none=gray, posible=yellow, leve=yellow, media=orange, grave=red"
_SEV_RANK = {"none": 0, "posible": 1, "leve": 2, "media": 3, "grave": 4}


def _sev_max(a: str, b: str) -> str:
    return a if _SEV_RANK.get(str(a), 0) >= _SEV_RANK.get(str(b), 0) else b


def _get_pose_arrays(analysis_context: dict) -> tuple[np.ndarray, np.ndarray]:
    user = analysis_context.get("user") if isinstance(analysis_context, dict) else {}
    pose = user.get("pose_clean") if isinstance(user, dict) and isinstance(user.get("pose_clean"), dict) else {}
    if "kps_xy_clean" in pose:
        kps_xy = np.asarray(pose["kps_xy_clean"], dtype=np.float64)
    else:
        kps_xy = np.asarray(pose["kps_xy"], dtype=np.float64)
    if "kps_score_clean" in pose:
        kps_score = np.asarray(pose["kps_score_clean"], dtype=np.float64)
    else:
        kps_score = np.asarray(pose["kps_score"], dtype=np.float64)
    return kps_xy, kps_score


def _setup_axis(ax: Any, kps: np.ndarray) -> None:
    valid = kps[np.isfinite(kps).all(axis=1)]
    if valid.size == 0:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    xmin, ymin = valid.min(axis=0)
    xmax, ymax = valid.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    pad = span * 0.2
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    ax.set_xlim(cx - span / 2.0 - pad, cx + span / 2.0 + pad)
    ax.set_ylim(cy + span / 2.0 + pad, cy - span / 2.0 - pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_skeleton(ax: Any, kps: np.ndarray, asym_map: dict[str, Any]) -> None:
    for i, j in _FULL_SKELETON_EDGES:
        if np.isfinite(kps[i]).all() and np.isfinite(kps[j]).all():
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]], color="#9e9e9e", alpha=0.4, linewidth=1.0)
    for idx in range(17):
        if not np.isfinite(kps[idx]).all():
            continue
        color = "#bdbdbd"
        if idx in {5, 7, 9, 11, 13, 15}:
            color = "#4c78a8"
        if idx in {6, 8, 10, 12, 14, 16}:
            color = "#e15759"
        ax.scatter([kps[idx, 0]], [kps[idx, 1]], color=color, s=28, zorder=3)
    for joint, (li, ri) in BILATERAL_PAIRS.items():
        a = asym_map.get(joint)
        if a is None or not math.isfinite(a.forward_diff_norm):
            continue
        lx, ly = float(kps[li, 0]), float(kps[li, 1])
        rx, ry = float(kps[ri, 0]), float(kps[ri, 1])
        if not (math.isfinite(lx) and math.isfinite(ly) and math.isfinite(rx) and math.isfinite(ry)):
            continue
        ax.annotate("", xy=(max(lx, rx), (ly + ry) / 2), xytext=(min(lx, rx), (ly + ry) / 2), arrowprops={"arrowstyle": "->", "color": "#455a64", "lw": 1.5})
        ax.text((lx + rx) / 2, (ly + ry) / 2 - 5, f"{joint}:{a.forward_diff_norm:.2f}", fontsize=6, ha="center", color="#263238")
    _setup_axis(ax, kps)


def _draw_joint_bars(ax: Any, rep: dict) -> None:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    arms_adj = str(((rep.get("arms") or {}) if isinstance(rep.get("arms"), dict) else {}).get("severity", "none"))
    legs_adj = str(((rep.get("legs") or {}) if isinstance(rep.get("legs"), dict) else {}).get("severity", "none"))
    for group_name, joints in (("arms", ARM_JOINTS), ("legs", LEG_JOINTS)):
        breakdown = rep.get(group_name, {}).get("joint_breakdown", {})
        for joint in joints:
            entry = breakdown.get(joint, {})
            mean = entry.get("mean_fwd_norm")
            sev_raw = str(entry.get("severity", "none"))
            adj = arms_adj if group_name == "arms" else legs_adj
            sev_display = sev_raw
            label = f"{group_name}:{joint} ({sev_raw})"
            # Grupo degradado/rescatado → mostrar raw→ajustado.
            if _SEV_RANK.get(sev_raw, 0) > _SEV_RANK.get(adj, 0):
                sev_display = adj
                label = f"{group_name}:{joint} (raw {sev_raw}→{adj})"
            elif _SEV_RANK.get(adj, 0) > _SEV_RANK.get(sev_raw, 0):
                # Raro: grupo rescatado por encima de severidad articular; mostrarlo.
                label = f"{group_name}:{joint} (joint {sev_raw}, group {adj})"
                sev_display = _sev_max(sev_raw, adj)
            labels.append(label)
            values.append(float(mean) if isinstance(mean, (int, float)) and math.isfinite(float(mean)) else 0.0)
            colors.append(_SEV_COLOR.get(sev_display, _SEV_COLOR["none"]))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.85, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("mean fwd_diff_norm", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Magnitud por articulacion (A=arms, L=legs)", fontsize=9)
    ax.text(0.01, -0.18, _SEV_LEGEND, transform=ax.transAxes, fontsize=7, ha="left", va="top")


def _draw_text(ax: Any, rep: dict) -> None:
    ax.set_axis_off()
    arms = rep.get("arms", {})
    legs = rep.get("legs", {})
    arms_down = str(arms.get("arm_conservatism_reason", "none"))
    arms_down_msg = ""
    if bool(arms.get("arm_conservatism_applied", False)):
        arms_down_msg = f"arms downgraded: {arms_down}"
    arms_pf_msg = ""
    if bool(arms.get("arm_perspective_filter_applied", False)):
        arms_pf_msg = f"arms perspective_filter: {arms.get('arm_perspective_filter_reason', 'n/a')}"
    legs_rescue_msg = ""
    if bool(legs.get("leg_grave_rescue_applied", False)):
        legs_rescue_msg = f"legs rescued to grave: {legs.get('leg_grave_rescue_reason', 'n/a')}"
    lines = [
        f"rep={rep.get('user_rep_order')}",
        f"detected={rep.get('detected')} global_severity={rep.get('severity')}",
        "",
        f"arms: sev={arms.get('severity')} (pre={arms.get('severity_before_group_adjustment')}) "
        f"mag={arms.get('magnitude')} phase={arms.get('dominant_phase')} side={arms.get('dominant_side')} "
        f"reinforcement={arms.get('secondary_reinforcement')}",
        (
            f"arms wrist_mean={arms.get('wrist_mean_fwd_norm')} wrist_max={arms.get('wrist_max_fwd_norm')} "
            f"elbow_mean={arms.get('elbow_mean_fwd_norm')} elbow_max={arms.get('elbow_max_fwd_norm')}"
        ),
        f"legs: sev={legs.get('severity')} (pre={legs.get('severity_before_group_adjustment')}) "
        f"mag={legs.get('magnitude')} phase={legs.get('dominant_phase')} side={legs.get('dominant_side')} "
        f"reinforcement={legs.get('secondary_reinforcement')}",
        (f"legs grave_blocked_by={legs.get('grave_blocked_by')}" if legs.get("grave_blocked_by") else ""),
        arms_down_msg,
        arms_pf_msg,
        legs_rescue_msg,
        "",
        f"worst_frames={rep.get('worst_frames', {})}",
    ]
    ax.text(0.02, 0.98, "\n".join([x for x in lines if str(x)]), va="top", ha="left", fontsize=8, family="monospace")


def export_asymmetry_debug(
    *,
    analysis_context: dict,
    asymmetry_result: dict,
    output_dir: str | Path,
) -> dict:
    out_dir = ensure_dir(output_dir)
    kps_xy, kps_score = _get_pose_arrays(analysis_context)
    rep_results = asymmetry_result.get("rep_results") if isinstance(asymmetry_result, dict) else []
    if not isinstance(rep_results, list):
        rep_results = []
    warnings: list[str] = []
    files: list[str] = []
    compact_reps: list[dict[str, Any]] = []

    for rep in rep_results:
        if not isinstance(rep, dict):
            continue
        rep_order = int(rep.get("user_rep_order", 0))
        rep_raw = int(rep.get("user_rep_raw_index", -1))
        if rep_order <= 0:
            rep_order = rep_raw + 1 if rep_raw >= 0 else 1
        worst = rep.get("worst_frames", {}) if isinstance(rep.get("worst_frames"), dict) else {}
        ecc_frame = int(worst.get("arms", {}).get("ecc", -1)) if isinstance(worst.get("arms"), dict) else -1
        con_frame = int(worst.get("arms", {}).get("con", -1)) if isinstance(worst.get("arms"), dict) else -1
        if isinstance(worst.get("legs"), dict):
            ecc_frame = max(ecc_frame, int(worst["legs"].get("ecc", -1)))
            con_frame = max(con_frame, int(worst["legs"].get("con", -1)))

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        rep_severity = str(rep.get("severity", "none"))
        arms_sev = str((rep.get("arms") or {}).get("severity", "none")) if isinstance(rep.get("arms"), dict) else "none"
        legs_sev = str((rep.get("legs") or {}).get("severity", "none")) if isinstance(rep.get("legs"), dict) else "none"
        for ax, phase, fi in ((axes[0, 0], "ECC", ecc_frame), (axes[0, 1], "CON", con_frame)):
            ax.set_facecolor(_SEV_BG.get(rep_severity, "#ecf0f1"))
            if fi < 0 or fi >= kps_xy.shape[0]:
                ax.set_axis_off()
                ax.set_title(f"{phase} no frame")
                continue
            asym_map = frame_asymmetry(kps_xy[fi], kps_score[fi], thr_conf=0.3)
            _draw_skeleton(ax, kps_xy[fi], asym_map)
            ax.set_title(f"{phase} frame={fi} | global:{rep_severity} arms:{arms_sev} legs:{legs_sev}")
        _draw_joint_bars(axes[1, 0], rep)
        _draw_text(axes[1, 1], rep)
        fig.tight_layout()
        png_name = f"rep_{rep_order}_asymmetry.png"
        fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(png_name)

        arms = rep.get("arms", {}) if isinstance(rep.get("arms"), dict) else {}
        legs = rep.get("legs", {}) if isinstance(rep.get("legs"), dict) else {}
        compact_reps.append(
            {
                "user_rep_order": rep.get("user_rep_order"),
                "user_rep_raw_index": rep.get("user_rep_raw_index"),
                "detected": rep.get("detected"),
                "severity": rep.get("severity"),
                "arms": {
                    "severity": arms.get("severity"),
                    "severity_before_group_adjustment": arms.get("severity_before_group_adjustment"),
                    "magnitude": arms.get("magnitude"),
                    "dominant_phase": arms.get("dominant_phase"),
                    "dominant_side": arms.get("dominant_side"),
                    "secondary_reinforcement": arms.get("secondary_reinforcement"),
                    "arm_conservatism_applied": arms.get("arm_conservatism_applied"),
                    "arm_conservatism_reason": arms.get("arm_conservatism_reason"),
                },
                "legs": {
                    "severity": legs.get("severity"),
                    "severity_before_group_adjustment": legs.get("severity_before_group_adjustment"),
                    "magnitude": legs.get("magnitude"),
                    "dominant_phase": legs.get("dominant_phase"),
                    "dominant_side": legs.get("dominant_side"),
                    "secondary_reinforcement": legs.get("secondary_reinforcement"),
                    "grave_blocked_by": legs.get("grave_blocked_by", []),
                },
                "worst_frames": rep.get("worst_frames", {}),
                "warnings": rep.get("warnings", []),
                "png": png_name,
            }
        )

    summary = {
        "detector": asymmetry_result.get("detector", "asymmetry"),
        "detected": asymmetry_result.get("detected", False),
        "severity": asymmetry_result.get("severity", "none"),
        "score": asymmetry_result.get("score", 0.0),
        "num_reps_analyzed": asymmetry_result.get("num_reps_analyzed", 0),
        "num_reps_detected": asymmetry_result.get("num_reps_detected", 0),
        "rep_results": compact_reps,
        "warnings": list(asymmetry_result.get("warnings", [])) + warnings,
    }
    save_json(out_dir / "asymmetry_summary.json", summary)
    return {"output_dir": str(out_dir), "files": ["asymmetry_summary.json", *files], "warnings": summary["warnings"]}
