
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.biomechanics.rdl.detectors.neck_movement import ORDERED_ANCHORS, compute_neck_pose_state
from src.visualization.rdl.debug.common import ensure_dir, save_json

KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_SHOULDER = 6
KP_HIP = 12
KP_KNEE = 14
_CHAIN = ((KP_SHOULDER, KP_HIP), (KP_HIP, KP_KNEE))
_SEV_BG = {
    "none": "#ecf0f1",
    "leve": "#fff3bf",
    "media": "#ffe0b2",
    "grave": "#ffcdd2",
}


def _get_rep_pairs(analysis_context: dict, rep_order: int, rep_raw: int) -> dict[str, dict]:
    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    paired = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else []
    if not isinstance(paired, list):
        return {}
    for rep in paired:
        if isinstance(rep, dict) and int(rep.get("user_rep_order", -1)) == int(rep_order):
            return rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
    for rep in paired:
        if isinstance(rep, dict) and int(rep.get("user_rep_raw_index", -1)) == int(rep_raw):
            return rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
    rep_index = int(rep_order - 1) if int(rep_order) > 0 else -1
    for rep in paired:
        if isinstance(rep, dict) and int(rep.get("rep_index", -1)) == rep_index:
            return rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
    if rep_index >= 0 and rep_index < len(paired):
        rep = paired[rep_index]
        if isinstance(rep, dict):
            return rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
    return {}


def _draw_chain(ax: Any, kps: np.ndarray, color: str, label: str) -> None:
    for i, j in _CHAIN:
        if np.isfinite(kps[i]).all() and np.isfinite(kps[j]).all():
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]], color=color, linewidth=1.6, alpha=0.9, label=label if i == KP_SHOULDER and j == KP_HIP else None)
    ax.scatter(kps[[KP_SHOULDER, KP_HIP, KP_KNEE], 0], kps[[KP_SHOULDER, KP_HIP, KP_KNEE], 1], s=16, color=color, zorder=3)


def _is_finite_kp(kps: np.ndarray, idx: int) -> bool:
    return bool(np.isfinite(np.asarray(kps[idx], dtype=np.float64)).all())


def _torso_aligned_ideal_for_display(user_kps: np.ndarray, ideal_kps: np.ndarray) -> np.ndarray:
    user_arr = np.asarray(user_kps, dtype=np.float64)
    ideal_arr = np.asarray(ideal_kps, dtype=np.float64)
    if not (_is_finite_kp(user_arr, KP_HIP) and _is_finite_kp(user_arr, KP_SHOULDER)):
        return ideal_arr
    if not (_is_finite_kp(ideal_arr, KP_HIP) and _is_finite_kp(ideal_arr, KP_SHOULDER)):
        return ideal_arr
    user_torso = float(np.linalg.norm(user_arr[KP_SHOULDER] - user_arr[KP_HIP]))
    ideal_torso = float(np.linalg.norm(ideal_arr[KP_SHOULDER] - ideal_arr[KP_HIP]))
    if user_torso <= 1e-9 or ideal_torso <= 1e-9:
        return ideal_arr
    scale = user_torso / ideal_torso
    return (ideal_arr - ideal_arr[KP_HIP]) * scale + user_arr[KP_HIP]


def _draw_face_and_torso(ax: Any, kps: np.ndarray, state: Any, color: str) -> tuple[str, str]:
    if _is_finite_kp(kps, KP_HIP) and _is_finite_kp(kps, KP_SHOULDER):
        ax.plot([kps[KP_HIP, 0], kps[KP_SHOULDER, 0]], [kps[KP_HIP, 1], kps[KP_SHOULDER, 1]], color=color, linewidth=2.4, alpha=0.95)
        ax.scatter([kps[KP_HIP, 0], kps[KP_SHOULDER, 0]], [kps[KP_HIP, 1], kps[KP_SHOULDER, 1]], s=30, color=color, zorder=4)
    else:
        return "none", "missing_torso_keypoints"

    face_axis = str(getattr(state, "selected_face_axis", "none"))
    face_pair = tuple(getattr(state, "selected_face_keypoints", (-1, -1)))
    ref_idx = int(face_pair[0]) if len(face_pair) > 0 else -1
    nose_idx = int(face_pair[1]) if len(face_pair) > 1 else -1
    if ref_idx < 0 or nose_idx != KP_NOSE:
        return face_axis, "missing_face_keypoints"
    if not (_is_finite_kp(kps, ref_idx) and _is_finite_kp(kps, KP_NOSE)):
        return face_axis, "missing_face_keypoints"
    if _is_finite_kp(kps, ref_idx) and _is_finite_kp(kps, KP_SHOULDER):
        ax.plot(
            [kps[ref_idx, 0], kps[KP_SHOULDER, 0]],
            [kps[ref_idx, 1], kps[KP_SHOULDER, 1]],
            color=color,
            linewidth=1.9,
            linestyle="-.",
            alpha=0.96,
        )
    ax.plot([kps[ref_idx, 0], kps[KP_NOSE, 0]], [kps[ref_idx, 1], kps[KP_NOSE, 1]], color=color, linewidth=2.2, linestyle="--", alpha=0.98)
    ax.scatter([kps[ref_idx, 0], kps[KP_NOSE, 0]], [kps[ref_idx, 1], kps[KP_NOSE, 1]], s=36, color=color, zorder=5, edgecolors="white", linewidths=0.6)
    return face_axis, "ok"


def _setup_axis(ax: Any, user_kps: np.ndarray, ideal_kps: np.ndarray) -> None:
    pts = np.vstack([user_kps, ideal_kps])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    pad = span * 0.25
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    ax.set_xlim(cx - span * 0.5 - pad, cx + span * 0.5 + pad)
    ax.set_ylim(cy + span * 0.5 + pad, cy - span * 0.5 - pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)
    ax.set_xticks([])
    ax.set_yticks([])


def export_neck_movement_debug(*, analysis_context: dict, neck_movement_result: dict, output_dir: str | Path) -> dict:
    out_dir = ensure_dir(output_dir)
    rep_results = neck_movement_result.get("rep_results") if isinstance(neck_movement_result, dict) else []
    if not isinstance(rep_results, list):
        rep_results = []

    files: list[str] = []
    compact_reps: list[dict[str, Any]] = []
    for rep in rep_results:
        if not isinstance(rep, dict):
            continue
        rep_order = int(rep.get("user_rep_order", 0))
        rep_raw = int(rep.get("user_rep_raw_index", -1))
        if rep_order <= 0:
            rep_order = rep_raw + 1 if rep_raw >= 0 else 1
        anchors = _get_rep_pairs(analysis_context, rep_order, rep_raw)
        anchor_results = rep.get("anchor_results") if isinstance(rep.get("anchor_results"), dict) else {}

        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(3, 3, width_ratios=[1.1, 1.1, 0.9], height_ratios=[1.0, 1.0, 0.8], wspace=0.25, hspace=0.25)
        anchor_axes = {
            "ecc_0": fig.add_subplot(gs[0, 0]),
            "ecc_25": fig.add_subplot(gs[0, 1]),
            "ecc_50": fig.add_subplot(gs[1, 0]),
            "ecc_75": fig.add_subplot(gs[1, 1]),
            "ecc_100": fig.add_subplot(gs[2, 0]),
        }
        ax_bar = fig.add_subplot(gs[2, 1])
        ax_text = fig.add_subplot(gs[:, 2])
        ax_text.set_axis_off()

        b_values: list[float] = []
        for anchor in ORDERED_ANCHORS:
            ax = anchor_axes.get(anchor)
            if ax is None:
                continue
            pair = anchors.get(anchor) if isinstance(anchors, dict) else None
            ares = anchor_results.get(anchor) if isinstance(anchor_results, dict) else {}
            user_kps = (pair or {}).get("user_kps_clean")
            ideal_kps = (pair or {}).get("ideal_kps_clean")
            user_frame = (ares or {}).get("user_frame")
            ideal_frame = (ares or {}).get("ideal_frame")
            b_deg = (ares or {}).get("B_deg")
            drift_deg = (ares or {}).get("drift_from_start_deg")
            direction = (ares or {}).get("neck_direction", "neutral")
            sev = (ares or {}).get("severity", "none")
            conf = (ares or {}).get("confidence", "unknown")
            status = (ares or {}).get("status", "inconclusive")
            if isinstance(b_deg, (int, float)) and np.isfinite(float(b_deg)):
                b_values.append(float(b_deg))

            panel_level = str(sev)
            if str(status) != "ok":
                panel_level = "inconclusive"
                ax.set_facecolor("#eeeeee")
            else:
                ax.set_facecolor(_SEV_BG.get(str(sev), "#ecf0f1"))
            ax.set_title(
                f"{anchor}\nu:{user_frame} i:{ideal_frame}\n"
                f"B={b_deg} drift={drift_deg}\n"
                f"level:{panel_level} dir:{direction} conf:{conf}",
                fontsize=8,
            )
            if isinstance(user_kps, np.ndarray) and isinstance(ideal_kps, np.ndarray) and user_kps.shape == (17, 2) and ideal_kps.shape == (17, 2):
                user_arr = np.asarray(user_kps, dtype=np.float64)
                ideal_arr_raw = np.asarray(ideal_kps, dtype=np.float64)
                ideal_arr = _torso_aligned_ideal_for_display(user_arr, ideal_arr_raw)
                _draw_chain(ax, ideal_arr, "#2ca02c", "ideal torso-aligned")
                _draw_chain(ax, user_arr, "#d62728", "user")
                try:
                    user_state = compute_neck_pose_state(user_arr, anchor)
                    ideal_state = compute_neck_pose_state(ideal_arr_raw, anchor)
                    user_face, user_draw_status = _draw_face_and_torso(ax, user_arr, user_state, "#d62728")
                    ideal_face, ideal_draw_status = _draw_face_and_torso(ax, ideal_arr, ideal_state, "#2ca02c")
                    if user_draw_status != "ok" or ideal_draw_status != "ok":
                        status = "missing_face_keypoints" if "face" in (user_draw_status + ideal_draw_status) else "missing_torso_keypoints"
                        ax.set_facecolor("#eeeeee")
                    ax.text(0.03, 0.03, f"user face:{user_face}\nstatus:{status}", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
                    ax.text(0.97, 0.03, f"ideal face:{ideal_face}\n(u:red i:green)", transform=ax.transAxes, fontsize=7, ha="right", va="bottom")
                except Exception:
                    pass
                _setup_axis(ax, user_arr, ideal_arr)
                ax.legend(loc="lower right", fontsize=7, frameon=True)
            else:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

        x = np.arange(len(ORDERED_ANCHORS))
        y = [anchor_results.get(a, {}).get("B_deg", np.nan) if isinstance(anchor_results, dict) else np.nan for a in ORDERED_ANCHORS]
        ax_bar.bar(x, y, color="#1f77b4", alpha=0.85)
        for thr, col in ((15.0, "#f1c40f"), (25.0, "#e67e22"), (40.0, "#c0392b")):
            ax_bar.axhline(thr, color=col, linestyle="--", linewidth=1.0)
            ax_bar.axhline(-thr, color=col, linestyle="--", linewidth=1.0)
        ax_bar.set_xticks(x, ORDERED_ANCHORS, rotation=20)
        ax_bar.set_title("B por anchor")
        ax_bar.grid(alpha=0.15, axis="y")

        text_lines = [
            f"detected={rep.get('detected', False)}",
            f"severity(level)={rep.get('severity', 'none')}",
            f"subtype={rep.get('subtype', 'none')}",
            f"neck_direction={rep.get('neck_direction', 'neutral')}",
            f"confidence={rep.get('confidence', 0.0)}",
            f"magnitude_deg={rep.get('magnitude_deg', 0.0)}",
            f"failed_anchors={rep.get('failed_anchors', [])}",
            f"failed_segments={rep.get('failed_segments', [])}",
            "",
            "levels: none < leve < media < grave",
            "direction: down/up/neutral/mixed",
        ]
        ax_text.text(0.02, 0.98, "\n".join(text_lines), ha="left", va="top", fontsize=9, family="monospace")

        fig.suptitle(f"neck_movement rep={rep_order} severity={rep.get('severity')} score={rep.get('score')}", fontsize=11)
        fig.tight_layout()
        png_name = f"rep_{rep_order}_neck_movement.png"
        fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(png_name)

        compact_anchors: dict[str, Any] = {}
        for anchor in ORDERED_ANCHORS:
            ares = anchor_results.get(anchor) if isinstance(anchor_results, dict) else {}
            pair = anchors.get(anchor) if isinstance(anchors, dict) else None
            user_kps = (pair or {}).get("user_kps_clean")
            selected_face_axis = None
            if isinstance(user_kps, np.ndarray) and user_kps.shape == (17, 2):
                try:
                    selected_face_axis = compute_neck_pose_state(np.asarray(user_kps, dtype=np.float64), anchor).selected_face_axis
                except Exception:
                    selected_face_axis = None
            compact_anchors[anchor] = {
                "user_frame": (ares or {}).get("user_frame"),
                "ideal_frame": (ares or {}).get("ideal_frame"),
                "B_deg": (ares or {}).get("B_deg"),
                "drift_from_start_deg": (ares or {}).get("drift_from_start_deg"),
                "direction": (ares or {}).get("neck_direction", "neutral"),
                "severity": (ares or {}).get("severity", "none"),
                "status": (ares or {}).get("status", "inconclusive"),
                "confidence": (ares or {}).get("confidence", "unknown"),
                "selected_face_axis": selected_face_axis,
                "warnings": [],
            }
        compact_reps.append(
            {
                "user_rep_order": rep.get("user_rep_order"),
                "user_rep_raw_index": rep.get("user_rep_raw_index"),
                "detected": rep.get("detected"),
                "severity": rep.get("severity"),
                "confidence": rep.get("confidence"),
                "subtype": rep.get("subtype"),
                "neck_direction": rep.get("neck_direction"),
                "magnitude_deg": rep.get("magnitude_deg"),
                "failed_anchors": rep.get("failed_anchors", []),
                "failed_segments": rep.get("failed_segments", []),
                "anchors": compact_anchors,
                "warnings": rep.get("warnings", []),
                "png": png_name,
            }
        )

    summary = {
        "detector": neck_movement_result.get("detector", "neck_movement"),
        "detected": neck_movement_result.get("detected", False),
        "severity": neck_movement_result.get("severity", "none"),
        "score": neck_movement_result.get("score", 0.0),
        "num_reps_analyzed": neck_movement_result.get("num_reps_analyzed", 0),
        "num_reps_detected": neck_movement_result.get("num_reps_detected", 0),
        "rep_results": compact_reps,
        "warnings": neck_movement_result.get("warnings", []),
    }
    save_json(out_dir / "neck_movement_summary.json", summary)
    return {"output_dir": str(out_dir), "files": ["neck_movement_summary.json", *files], "warnings": summary["warnings"]}
