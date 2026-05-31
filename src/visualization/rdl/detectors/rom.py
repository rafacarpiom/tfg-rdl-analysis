
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.rdl.debug.common import ensure_dir, save_json

KP_SHOULDER = 6
KP_ELBOW = 8
KP_WRIST = 10
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
_SEV_BG = {"none": "#ecf0f1", "leve": "#fff3bf", "media": "#ffe0b2", "grave": "#ffcdd2"}


def _get_rep_pairs(analysis_context: dict, rep_order: int, rep_raw: int) -> dict[str, dict]:
    ap = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    reps = ap.get("paired_repetitions") if isinstance(ap, dict) else []
    if not isinstance(reps, list):
        return {}
    for rep in reps:
        if isinstance(rep, dict) and int(rep.get("user_rep_order", -1)) == int(rep_order):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    for rep in reps:
        if isinstance(rep, dict) and int(rep.get("user_rep_raw_index", -1)) == int(rep_raw):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    rep_index = int(rep_order - 1) if int(rep_order) > 0 else -1
    for rep in reps:
        if isinstance(rep, dict) and int(rep.get("rep_index", -1)) == rep_index:
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    if rep_index >= 0 and rep_index < len(reps):
        rep = reps[rep_index]
        if isinstance(rep, dict):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    return {}


def _draw_chain(ax: Any, kps: np.ndarray, color: str, label: str) -> None:
    for i, j in _CHAIN:
        if np.isfinite(kps[i]).all() and np.isfinite(kps[j]).all():
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]], color=color, linewidth=1.8, alpha=0.9, label=label if i == KP_SHOULDER and j == KP_ELBOW else None)
    ax.scatter(kps[[KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE], 0], kps[[KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE], 1], s=18, color=color, zorder=3)


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


def _draw_torso(ax: Any, kps: np.ndarray, color: str, label: str) -> float:
    if not (_is_finite_kp(kps, KP_HIP) and _is_finite_kp(kps, KP_SHOULDER)):
        return float("nan")
    hip = np.asarray(kps[KP_HIP], dtype=np.float64)
    shoulder = np.asarray(kps[KP_SHOULDER], dtype=np.float64)
    ax.plot([hip[0], shoulder[0]], [hip[1], shoulder[1]], color=color, linewidth=2.8, alpha=0.95)
    angle = float(np.degrees(np.arctan2(float(shoulder[1] - hip[1]), float(shoulder[0] - hip[0]))))
    ax.text(float((hip[0] + shoulder[0]) * 0.5), float((hip[1] + shoulder[1]) * 0.5), f"{label}:{angle:+.1f}deg", fontsize=8, color=color, bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": color, "alpha": 0.85})
    return angle


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


def export_rom_debug(*, analysis_context: dict, rom_result: dict, output_dir: str | Path) -> dict:
    out_dir = ensure_dir(output_dir)
    rep_results = rom_result.get("rep_results") if isinstance(rom_result, dict) else []
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
        start_pair = anchors.get("ecc_0") if isinstance(anchors, dict) else None
        bottom_pair = anchors.get("bottom") if isinstance(anchors, dict) else None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), width_ratios=[1.0, 1.0, 0.9])
        ax_start, ax_bottom, ax_text = axes
        ax_text.set_axis_off()
        severity = str(rep.get("severity", "none"))
        for ax in (ax_start, ax_bottom, ax_text):
            ax.set_facecolor(_SEV_BG.get(severity, "#ecf0f1"))

        # Panel inicio
        user_start = (start_pair or {}).get("user_kps_clean")
        ideal_start = (start_pair or {}).get("ideal_kps_clean")
        if user_start is None:
            user_start = (start_pair or {}).get("user_kps_normalized")
        if ideal_start is None:
            ideal_start = (start_pair or {}).get("ideal_kps_normalized")
        if isinstance(user_start, np.ndarray) and isinstance(ideal_start, np.ndarray) and user_start.shape == (17, 2) and ideal_start.shape == (17, 2):
            u = np.asarray(user_start, dtype=np.float64)
            i = _torso_aligned_ideal_for_display(u, np.asarray(ideal_start, dtype=np.float64))
            _draw_chain(ax_start, i, "#2ca02c", "ideal")
            _draw_chain(ax_start, u, "#d62728", "user")
            _draw_torso(ax_start, i, "#0f766e", "theta_start_i")
            _draw_torso(ax_start, u, "#7c3aed", "theta_start_u")
            _setup_axis(ax_start, u, i)
            ax_start.legend(loc="lower right", fontsize=8, frameon=True)
        else:
            ax_start.text(0.5, 0.5, "missing", transform=ax_start.transAxes, ha="center", va="center")
            ax_start.set_xticks([])
            ax_start.set_yticks([])
        ax_start.set_title(f"ecc_0 | u:{rep.get('frames', {}).get('user_start')} i:{rep.get('frames', {}).get('ideal_start')}", fontsize=9)

        # Panel fondo
        user_bottom = (bottom_pair or {}).get("user_kps_clean")
        ideal_bottom = (bottom_pair or {}).get("ideal_kps_clean")
        if user_bottom is None:
            user_bottom = (bottom_pair or {}).get("user_kps_normalized")
        if ideal_bottom is None:
            ideal_bottom = (bottom_pair or {}).get("ideal_kps_normalized")
        if isinstance(user_bottom, np.ndarray) and isinstance(ideal_bottom, np.ndarray) and user_bottom.shape == (17, 2) and ideal_bottom.shape == (17, 2):
            u = np.asarray(user_bottom, dtype=np.float64)
            i = _torso_aligned_ideal_for_display(u, np.asarray(ideal_bottom, dtype=np.float64))
            _draw_chain(ax_bottom, i, "#2ca02c", "ideal")
            _draw_chain(ax_bottom, u, "#d62728", "user")
            _draw_torso(ax_bottom, i, "#0f766e", "theta_bottom_i")
            _draw_torso(ax_bottom, u, "#7c3aed", "theta_bottom_u")
            _setup_axis(ax_bottom, u, i)
            ax_bottom.legend(loc="lower right", fontsize=8, frameon=True)
        else:
            ax_bottom.text(0.5, 0.5, "missing", transform=ax_bottom.transAxes, ha="center", va="center")
            ax_bottom.set_xticks([])
            ax_bottom.set_yticks([])
        ax_bottom.set_title(f"bottom | u:{rep.get('frames', {}).get('user_bottom')} i:{rep.get('frames', {}).get('ideal_bottom')}", fontsize=9)

        m = rep.get("metrics") if isinstance(rep.get("metrics"), dict) else {}
        text_lines = [
            f"detected={rep.get('detected', False)}",
            f"severity(level)={severity}",
            f"confidence={rep.get('confidence', 0.0)}",
            f"used_ideal={rep.get('used_ideal', False)}",
            f"score={rep.get('score', 0.0)}",
            "",
            f"ROM_user_deg={m.get('rom_user_abs_deg')}",
            f"ROM_ideal_deg={m.get('rom_ideal_abs_deg')}",
            f"ROM_norm={m.get('rom_norm')}",
            "",
            "thresholds relative:",
            "leve < 0.85",
            "media < 0.75",
            "grave < 0.65",
            "",
            "trace:",
            *[str(x) for x in (rep.get("trace") if isinstance(rep.get("trace"), list) else [])],
        ]
        ax_text.text(0.02, 0.98, "\n".join(text_lines), ha="left", va="top", fontsize=9, family="monospace")
        fig.suptitle(f"rom rep={rep_order} severity={rep.get('severity')} score={rep.get('score')}", fontsize=11)
        fig.tight_layout()
        png_name = f"rep_{rep_order}_rom.png"
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
                "used_ideal": rep.get("used_ideal"),
                "user_start_frame": (rep.get("frames") or {}).get("user_start"),
                "user_bottom_frame": (rep.get("frames") or {}).get("user_bottom"),
                "ideal_start_frame": (rep.get("frames") or {}).get("ideal_start"),
                "ideal_bottom_frame": (rep.get("frames") or {}).get("ideal_bottom"),
                "rom_user_deg": (m or {}).get("rom_user_abs_deg"),
                "rom_ideal_deg": (m or {}).get("rom_ideal_abs_deg"),
                "rom_norm": (m or {}).get("rom_norm"),
                "score": rep.get("score"),
                "warnings": rep.get("warnings", []),
                "png": png_name,
            }
        )

    summary = {
        "detector": rom_result.get("detector", "rom"),
        "detected": rom_result.get("detected", False),
        "severity": rom_result.get("severity", "none"),
        "score": rom_result.get("score", 0.0),
        "num_reps_analyzed": rom_result.get("num_reps_analyzed", 0),
        "num_reps_detected": rom_result.get("num_reps_detected", 0),
        "rep_results": compact_reps,
        "warnings": rom_result.get("warnings", []),
    }
    save_json(out_dir / "rom_summary.json", summary)
    return {"output_dir": str(out_dir), "files": ["rom_summary.json", *files], "warnings": summary["warnings"]}
