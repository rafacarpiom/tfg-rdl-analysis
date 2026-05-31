
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.rdl.debug.common import ensure_dir, save_json

KP_SHOULDER = 6
KP_ELBOW = 8
KP_WRIST = 10
KP_L_HIP = 11
KP_HIP = 12
KP_KNEE = 14
KP_ANKLE = 16
_CHAIN = (
    (KP_SHOULDER, KP_ELBOW),
    (KP_ELBOW, KP_WRIST),
    (KP_SHOULDER, KP_HIP),
    (KP_HIP, KP_KNEE),
    (KP_KNEE, KP_ANKLE),
)
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
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]], color=color, linewidth=1.8, alpha=0.95, label=label if i == KP_SHOULDER and j == KP_ELBOW else None)
    idxs = [KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE]
    ax.scatter(kps[idxs, 0], kps[idxs, 1], s=18, color=color, zorder=3)


def _draw_torso(ax: Any, kps: np.ndarray, color: str, label: str) -> None:
    sh = np.asarray(kps[KP_SHOULDER], dtype=np.float64)
    lh = np.asarray(kps[KP_L_HIP], dtype=np.float64)
    rh = np.asarray(kps[KP_HIP], dtype=np.float64)
    if not np.isfinite(sh).all():
        return
    pelvis = (lh + rh) * 0.5 if np.isfinite(lh).all() and np.isfinite(rh).all() else rh if np.isfinite(rh).all() else lh
    if not np.isfinite(pelvis).all():
        return
    ax.plot([pelvis[0], sh[0]], [pelvis[1], sh[1]], color=color, linewidth=2.8, alpha=0.95)
    ax.text(float((pelvis[0] + sh[0]) * 0.5), float((pelvis[1] + sh[1]) * 0.5), label, fontsize=8, color=color, bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": color, "alpha": 0.85})


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
    ax.grid(alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])


def export_lockout_debug(*, analysis_context: dict, lockout_result: dict, output_dir: str | Path) -> dict:
    out_dir = ensure_dir(output_dir)
    rep_results = lockout_result.get("rep_results") if isinstance(lockout_result, dict) else []
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
        anchor_name = str(rep.get("anchor", "con_100"))
        anchors = _get_rep_pairs(analysis_context, rep_order, rep_raw)
        pair = anchors.get(anchor_name) if isinstance(anchors, dict) else None
        user_kps = (pair or {}).get("user_kps_normalized")
        ideal_kps = (pair or {}).get("ideal_kps_normalized")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), width_ratios=[1.15, 0.85])
        ax_pose, ax_text = axes
        ax_text.set_axis_off()
        severity = str(rep.get("severity", "none"))
        ax_pose.set_facecolor(_SEV_BG.get(severity, "#ecf0f1"))
        ax_text.set_facecolor(_SEV_BG.get(severity, "#ecf0f1"))

        if isinstance(user_kps, np.ndarray) and isinstance(ideal_kps, np.ndarray) and user_kps.shape == (17, 2) and ideal_kps.shape == (17, 2):
            _draw_chain(ax_pose, ideal_kps, "#2ca02c", "ideal")
            _draw_chain(ax_pose, user_kps, "#d62728", "user")
            _draw_torso(ax_pose, ideal_kps, "#0f766e", "torso ideal")
            _draw_torso(ax_pose, user_kps, "#7c3aed", "torso user")
            _setup_axis(ax_pose, user_kps, ideal_kps)
            ax_pose.legend(loc="lower right", fontsize=8, frameon=True)
        else:
            ax_pose.text(0.5, 0.5, "Anchor sin keypoints válidos", transform=ax_pose.transAxes, ha="center", va="center")
            ax_pose.set_xticks([])
            ax_pose.set_yticks([])
        ax_pose.set_title(f"Anchor final: {anchor_name} | level:{severity}", fontsize=10)

        metrics = rep.get("metrics") if isinstance(rep.get("metrics"), dict) else {}
        trace = rep.get("trace") if isinstance(rep.get("trace"), list) else []
        text_lines = [
            f"detected={rep.get('detected', False)}",
            f"severity(level)={severity}",
            f"confidence={rep.get('confidence', 0.0)}",
            f"magnitude={rep.get('magnitude', 0.0)}",
            "",
            f"theta_end_user_deg={metrics.get('theta_end_user_deg')}",
            f"theta_end_ideal_deg={metrics.get('theta_end_ideal_deg')}",
            f"error_lockout_deg={metrics.get('error_lockout_deg')}",
            "thresholds: leve>=5.0 media>=7.5 grave>10.0",
            "",
            "trace:",
            *[str(x) for x in trace],
        ]
        ax_text.text(0.02, 0.98, "\n".join(text_lines), ha="left", va="top", fontsize=9, family="monospace")
        fig.suptitle(f"lockout rep={rep_order} severity={rep.get('severity')} score={rep.get('score')}", fontsize=11)
        fig.tight_layout()
        png_name = f"rep_{rep_order}_lockout.png"
        fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(png_name)

        compact_reps.append(
            {
                "user_rep_order": rep.get("user_rep_order"),
                "user_rep_raw_index": rep.get("user_rep_raw_index"),
                "detected": rep.get("detected"),
                "severity": rep.get("severity"),
                "confidence": rep.get("confidence"),
                "magnitude": rep.get("magnitude"),
                "anchor": rep.get("anchor"),
                "user_frame": rep.get("user_frame"),
                "ideal_frame": rep.get("ideal_frame"),
                "theta_end_user_deg": metrics.get("theta_end_user_deg"),
                "theta_end_ideal_deg": metrics.get("theta_end_ideal_deg"),
                "error_lockout_deg": metrics.get("error_lockout_deg"),
                "warnings": rep.get("warnings", []),
                "png": png_name,
            }
        )

    summary = {
        "detector": lockout_result.get("detector", "lockout"),
        "detected": lockout_result.get("detected", False),
        "severity": lockout_result.get("severity", "none"),
        "score": lockout_result.get("score", 0.0),
        "num_reps_analyzed": lockout_result.get("num_reps_analyzed", 0),
        "num_reps_detected": lockout_result.get("num_reps_detected", 0),
        "rep_results": compact_reps,
        "warnings": lockout_result.get("warnings", []),
    }
    save_json(out_dir / "lockout_summary.json", summary)
    return {"output_dir": str(out_dir), "files": ["lockout_summary.json", *files], "warnings": summary["warnings"]}
