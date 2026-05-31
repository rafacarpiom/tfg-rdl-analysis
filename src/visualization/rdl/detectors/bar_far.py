
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.biomechanics.rdl.detectors.bar_far import BAR_FAR_ANCHORS
from src.visualization.rdl.debug.common import ensure_dir, save_json

KP_SHOULDER = 6
KP_ELBOW = 8
KP_WRIST = 10
KP_HIP = 12
KP_KNEE = 14
KP_ANKLE = 16

_RIGHT_CHAIN = (
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


def _get_anchor_pairs(analysis_context: dict, rep_order: int, rep_raw: int) -> dict[str, dict]:
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
    for i, j in _RIGHT_CHAIN:
        if np.isfinite(kps[i]).all() and np.isfinite(kps[j]).all():
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]], color=color, linewidth=1.8, alpha=0.9, label=label if i == KP_SHOULDER and j == KP_ELBOW else None)
    ax.scatter(kps[[KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE], 0], kps[[KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE], 1], s=16, color=color, zorder=3)


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


def export_bar_far_debug(*, analysis_context: dict, bar_far_result: dict, output_dir: str | Path) -> dict:
    out_dir = ensure_dir(output_dir)
    rep_results = bar_far_result.get("rep_results") if isinstance(bar_far_result, dict) else []
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
        anchor_pairs = _get_anchor_pairs(analysis_context, rep_order, rep_raw)
        anchor_results = rep.get("anchor_results") if isinstance(rep.get("anchor_results"), dict) else {}

        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        flat_axes = list(axes.flatten())
        for idx, anchor in enumerate(BAR_FAR_ANCHORS):
            ax = flat_axes[idx]
            pair = anchor_pairs.get(anchor) if isinstance(anchor_pairs, dict) else None
            result_anchor = anchor_results.get(anchor) if isinstance(anchor_results, dict) else {}
            metrics = result_anchor.get("metrics") if isinstance(result_anchor, dict) else {}
            verdict = result_anchor.get("verdict") if isinstance(result_anchor, dict) else {}
            severity = str((verdict or {}).get("severity", "none"))
            ax.set_facecolor(_SEV_BG.get(severity, "#ecf0f1"))
            user_frame = (pair or {}).get("user_frame") if isinstance(pair, dict) else None
            ideal_frame = (pair or {}).get("ideal_frame") if isinstance(pair, dict) else None
            wxn = (metrics or {}).get("wrist_error_x_norm")
            dx = (metrics or {}).get("delta_x_wrist")
            ax.set_title(f"{anchor}\nu:{user_frame} i:{ideal_frame}\nwxn:{wxn} dx:{dx}\nlevel:{severity}", fontsize=8)

            user_kps = (pair or {}).get("user_kps_normalized")
            if user_kps is None:
                user_kps = (pair or {}).get("user_kps_clean")
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
            _draw_chain(ax, user_arr, "#d62728", "user")
            if np.isfinite(ideal_arr[KP_WRIST]).all() and np.isfinite(user_arr[KP_WRIST]).all():
                ax.annotate("", xy=(user_arr[KP_WRIST, 0], user_arr[KP_WRIST, 1]), xytext=(ideal_arr[KP_WRIST, 0], ideal_arr[KP_WRIST, 1]), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#9467bd"})
            _setup_axis(ax, user_arr, ideal_arr)

        for ax in flat_axes[len(BAR_FAR_ANCHORS):]:
            ax.set_axis_off()
        fig.suptitle(f"rep={rep_order} detected={rep.get('detected')} severity={rep.get('severity')} score={rep.get('score')}", fontsize=11)
        fig.tight_layout()
        png_name = f"rep_{rep_order}_bar_far.png"
        fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(png_name)

        compact_anchors: dict[str, Any] = {}
        for anchor in BAR_FAR_ANCHORS:
            ar = anchor_results.get(anchor) if isinstance(anchor_results, dict) else {}
            metrics = ar.get("metrics") if isinstance(ar, dict) else {}
            verdict = ar.get("verdict") if isinstance(ar, dict) else {}
            compact_anchors[anchor] = {
                "user_frame": ar.get("user_frame") if isinstance(ar, dict) else None,
                "ideal_frame": ar.get("ideal_frame") if isinstance(ar, dict) else None,
                "wrist_error_x_norm": (metrics or {}).get("wrist_error_x_norm"),
                "delta_x_wrist": (metrics or {}).get("delta_x_wrist"),
                "severity": (verdict or {}).get("severity", "none"),
                "confidence": (verdict or {}).get("confidence", "normal"),
                "applied_rules": list((verdict or {}).get("applied_rules", [])),
                "warnings": list(ar.get("warnings", [])) if isinstance(ar, dict) else [],
            }

        compact_reps.append(
            {
                "user_rep_order": rep.get("user_rep_order"),
                "user_rep_raw_index": rep.get("user_rep_raw_index"),
                "detected": rep.get("detected"),
                "severity": rep.get("severity"),
                "score": rep.get("score"),
                "confidence": rep.get("confidence"),
                "failed_anchors": rep.get("failed_anchors", []),
                "anchors": compact_anchors,
                "warnings": rep.get("warnings", []),
                "png": png_name,
            }
        )

    summary = {
        "detector": bar_far_result.get("detector", "bar_far"),
        "detected": bar_far_result.get("detected", False),
        "severity": bar_far_result.get("severity", "none"),
        "score": bar_far_result.get("score", 0.0),
        "num_reps_analyzed": bar_far_result.get("num_reps_analyzed", 0),
        "num_reps_detected": bar_far_result.get("num_reps_detected", 0),
        "rep_results": compact_reps,
        "warnings": list(bar_far_result.get("warnings", [])) + warnings,
    }
    save_json(out_dir / "bar_far_summary.json", summary)
    return {"output_dir": str(out_dir), "files": ["bar_far_summary.json", *files], "warnings": summary["warnings"]}
