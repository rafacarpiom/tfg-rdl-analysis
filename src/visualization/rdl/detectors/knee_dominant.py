
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.biomechanics.rdl.detectors.knee_dominant import KNEE_DOMINANT_ANCHORS
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
_SEV_BG = {
    "none": "#ecf0f1",
    "leve": "#fff3bf",
    "media": "#ffe0b2",
    "grave": "#ffcdd2",
}
_SEV_EDGE = {
    "none": "#b0b8be",
    "leve": "#f59e0b",
    "media": "#f97316",
    "grave": "#dc2626",
}
_SEV_USER = {
    "none": "#d62728",
    "leve": "#d97706",
    "media": "#ea580c",
    "grave": "#dc2626",
}


def _resolve_anchor_status(ruling: dict[str, Any], metrics: dict[str, Any]) -> tuple[str, bool]:
    failed = bool((ruling or {}).get("failed", False))
    severity = str((ruling or {}).get("severity", "none")).lower()
    if severity not in {"none", "leve", "media", "grave"}:
        severity = "none"
    if failed and severity == "none":
        try:
            dk = float((metrics or {}).get("delta_knee"))
        except Exception:
            dk = float("nan")
        if np.isfinite(dk):
            if dk > 18.0:
                severity = "grave"
            elif dk > 14.0:
                severity = "media"
            else:
                severity = "leve"
        else:
            severity = "leve"
    return severity, failed


def _get_rep_pairs(analysis_context: dict, rep_order: int, rep_raw: int) -> dict[str, dict]:
    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    paired = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else []
    if not isinstance(paired, list):
        return {}
    for rep in paired:
        if isinstance(rep, dict) and int(rep.get("user_rep_order", -1)) == int(rep_order):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    for rep in paired:
        if isinstance(rep, dict) and int(rep.get("user_rep_raw_index", -1)) == int(rep_raw):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    rep_index = int(rep_order - 1) if int(rep_order) > 0 else -1
    for rep in paired:
        if isinstance(rep, dict) and int(rep.get("rep_index", -1)) == rep_index:
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    if rep_index >= 0 and rep_index < len(paired):
        rep = paired[rep_index]
        if isinstance(rep, dict):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    return {}


def _draw_chain(ax: Any, kps: np.ndarray, color: str, label: str) -> None:
    for i, j in _CHAIN:
        if np.isfinite(kps[i]).all() and np.isfinite(kps[j]).all():
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]], color=color, linewidth=1.8, alpha=0.9, label=label if i == KP_SHOULDER and j == KP_ELBOW else None)
    ax.scatter(kps[[KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE], 0], kps[[KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE], 1], s=18, color=color, zorder=3)


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
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
    ax.set_ylim(cy + span / 2 + pad, cy - span / 2 - pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])


def export_knee_dominant_debug(*, analysis_context: dict, knee_dominant_result: dict, output_dir: str | Path) -> dict:
    out_dir = ensure_dir(output_dir)
    rep_results = knee_dominant_result.get("rep_results") if isinstance(knee_dominant_result, dict) else []
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
        anchor_pairs = _get_rep_pairs(analysis_context, rep_order, rep_raw)
        anchor_metrics = rep.get("anchor_metrics") if isinstance(rep.get("anchor_metrics"), dict) else {}
        anchor_rulings = rep.get("anchor_rulings") if isinstance(rep.get("anchor_rulings"), dict) else {}

        fig, axes = plt.subplots(2, 4, figsize=(16, 9))
        flat = list(axes.flatten())
        for idx, anchor in enumerate(KNEE_DOMINANT_ANCHORS):
            ax = flat[idx]
            pair = anchor_pairs.get(anchor) if isinstance(anchor_pairs, dict) else None
            metrics = anchor_metrics.get(anchor) if isinstance(anchor_metrics, dict) else {}
            ruling = anchor_rulings.get(anchor) if isinstance(anchor_rulings, dict) else {}
            severity, failed = _resolve_anchor_status(ruling, metrics)
            ax.set_facecolor(_SEV_BG.get(severity, "#ecf0f1"))
            for spine in ax.spines.values():
                spine.set_color(_SEV_EDGE.get(severity, "#b0b8be"))
                spine.set_linewidth(2.0 if failed else 1.0)
            user_frame = (pair or {}).get("user_frame") if isinstance(pair, dict) else None
            ideal_frame = (pair or {}).get("ideal_frame") if isinstance(pair, dict) else None
            delta_knee = (metrics or {}).get("delta_knee")
            delta_hip = (metrics or {}).get("delta_hip")
            ax.set_title(f"{anchor}\nu:{user_frame} i:{ideal_frame}\nΔk={delta_knee} Δh={delta_hip}\nlevel:{severity} {'FALLO' if failed else 'OK'}", fontsize=8)

            user_kps = (pair or {}).get("user_kps_normalized")
            ideal_kps = (pair or {}).get("ideal_kps_normalized")
            if user_kps is None or ideal_kps is None:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            user_arr = np.asarray(user_kps, dtype=np.float64)
            ideal_arr = np.asarray(ideal_kps, dtype=np.float64)
            if user_arr.shape != (17, 2) or ideal_arr.shape != (17, 2):
                ax.text(0.5, 0.5, "invalid", transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            _draw_chain(ax, ideal_arr, "#2ca02c", "ideal")
            _draw_chain(ax, user_arr, _SEV_USER.get(severity, "#d62728"), "user")
            if np.isfinite(user_arr[KP_KNEE]).all():
                ax.scatter([user_arr[KP_KNEE, 0]], [user_arr[KP_KNEE, 1]], s=80, facecolors="none", edgecolors="#7c3aed", linewidths=2)
            if np.isfinite(user_arr[KP_HIP]).all():
                ax.scatter([user_arr[KP_HIP, 0]], [user_arr[KP_HIP, 1]], s=80, facecolors="none", edgecolors="#1f2937", linewidths=2)
            _setup_axis(ax, user_arr, ideal_arr)

        summary_ax = flat[6]
        summary_ax.set_axis_off()
        summary_ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"detected={rep.get('detected')} severity={rep.get('severity')} confidence={rep.get('confidence')}",
                    f"magnitude={rep.get('magnitude')} mean_knee={rep.get('mean_knee')} max_knee={rep.get('max_knee')}",
                    f"n_failed={rep.get('n_failed')} failed_anchors={rep.get('failed_anchors', [])}",
                ]
            ),
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
        )
        flat[7].set_axis_off()
        fig.suptitle(f"rep={rep_order} knee_dominant severity={rep.get('severity')} score={rep.get('score')}", fontsize=11)
        fig.tight_layout()
        png_name = f"rep_{rep_order}_knee_dominant.png"
        fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(png_name)

        compact_anchors: dict[str, Any] = {}
        for anchor in KNEE_DOMINANT_ANCHORS:
            pair = anchor_pairs.get(anchor) if isinstance(anchor_pairs, dict) else {}
            m = anchor_metrics.get(anchor) if isinstance(anchor_metrics, dict) else {}
            r = anchor_rulings.get(anchor) if isinstance(anchor_rulings, dict) else {}
            compact_anchors[anchor] = {
                "user_frame": (pair or {}).get("user_frame"),
                "ideal_frame": (pair or {}).get("ideal_frame"),
                "hip_user": (m or {}).get("hip_user"),
                "hip_ideal": (m or {}).get("hip_ideal"),
                "delta_hip": (m or {}).get("delta_hip"),
                "knee_user": (m or {}).get("knee_user"),
                "knee_ideal": (m or {}).get("knee_ideal"),
                "delta_knee": (m or {}).get("delta_knee"),
                "failed": bool((r or {}).get("failed", False)),
                "severity": str((r or {}).get("severity", "none")),
                "reject_reason": (r or {}).get("reject_reason", ""),
                "warnings": list((pair or {}).get("warnings", [])),
            }

        compact_reps.append(
            {
                "user_rep_order": rep.get("user_rep_order"),
                "user_rep_raw_index": rep.get("user_rep_raw_index"),
                "detected": rep.get("detected"),
                "severity": rep.get("severity"),
                "confidence": rep.get("confidence"),
                "magnitude": rep.get("magnitude"),
                "mean_knee": rep.get("mean_knee"),
                "max_knee": rep.get("max_knee"),
                "failed_anchors": rep.get("failed_anchors", []),
                "anchors": compact_anchors,
                "warnings": rep.get("warnings", []),
                "png": png_name,
            }
        )

    summary = {
        "detector": knee_dominant_result.get("detector", "knee_dominant"),
        "detected": knee_dominant_result.get("detected", False),
        "severity": knee_dominant_result.get("severity", "none"),
        "score": knee_dominant_result.get("score", 0.0),
        "num_reps_analyzed": knee_dominant_result.get("num_reps_analyzed", 0),
        "num_reps_detected": knee_dominant_result.get("num_reps_detected", 0),
        "rep_results": compact_reps,
        "warnings": knee_dominant_result.get("warnings", []),
    }
    save_json(out_dir / "knee_dominant_summary.json", summary)
    return {"output_dir": str(out_dir), "files": ["knee_dominant_summary.json", *files], "warnings": summary["warnings"]}
