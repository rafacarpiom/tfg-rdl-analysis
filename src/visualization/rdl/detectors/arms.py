
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.biomechanics.rdl.detectors.arms import BENT_ARMS_ANCHORS
from src.visualization.rdl.debug.common import ensure_dir, save_json

KP_SHOULDER = 6
KP_ELBOW = 8
KP_WRIST = 10
KP_HIP = 12
KP_KNEE = 14
KP_ANKLE = 16

RIGHT_CHAIN_EDGES: tuple[tuple[int, int], ...] = (
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


def _get_rep_anchor_pairs(analysis_context: dict, user_rep_order: int, user_rep_raw_index: int) -> dict[str, dict]:
    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    paired_reps = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else []
    if not isinstance(paired_reps, list):
        return {}

    for rep in paired_reps:
        if not isinstance(rep, dict):
            continue
        if int(rep.get("user_rep_order", -1)) == int(user_rep_order):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}

    for rep in paired_reps:
        if not isinstance(rep, dict):
            continue
        if int(rep.get("user_rep_raw_index", -1)) == int(user_rep_raw_index):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    rep_index = int(user_rep_order - 1) if int(user_rep_order) > 0 else -1
    for rep in paired_reps:
        if not isinstance(rep, dict):
            continue
        if int(rep.get("rep_index", -1)) == rep_index:
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    if rep_index >= 0 and rep_index < len(paired_reps):
        rep = paired_reps[rep_index]
        if isinstance(rep, dict):
            anchors = rep.get("anchors")
            return anchors if isinstance(anchors, dict) else {}
    return {}


def _get_user_kps(pair: dict[str, Any]) -> np.ndarray | None:
    user_kps = pair.get("user_kps_clean")
    if user_kps is None:
        user_kps = pair.get("user_kps_normalized")
    if user_kps is None:
        return None
    arr = np.asarray(user_kps, dtype=np.float64)
    if arr.shape != (17, 2):
        return None
    return arr


def _draw_right_chain(ax: Any, kps: np.ndarray) -> None:
    for i, j in RIGHT_CHAIN_EDGES:
        pi = kps[i]
        pj = kps[j]
        if np.all(np.isfinite(pi)) and np.all(np.isfinite(pj)):
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#4c78a8", linewidth=2.0)

    idxs = [KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE]
    pts = np.asarray([kps[i] for i in idxs], dtype=np.float64)
    valid = np.isfinite(pts).all(axis=1)
    if np.any(valid):
        ax.scatter(pts[valid, 0], pts[valid, 1], s=30, color="#4c78a8", zorder=3)
    for key_idx, color in ((KP_SHOULDER, "#2ca02c"), (KP_ELBOW, "#d62728"), (KP_WRIST, "#9467bd")):
        p = kps[key_idx]
        if np.all(np.isfinite(p)):
            ax.scatter([p[0]], [p[1]], s=60, color=color, zorder=4)


def _elbow_angle_deg(kps: np.ndarray) -> float:
    s = np.asarray(kps[KP_SHOULDER], dtype=np.float64)
    e = np.asarray(kps[KP_ELBOW], dtype=np.float64)
    w = np.asarray(kps[KP_WRIST], dtype=np.float64)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(e)) and np.all(np.isfinite(w))):
        return float("nan")
    a = s - e
    b = w - e
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    cos_t = float(np.dot(a, b) / (na * nb))
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(math.degrees(math.acos(cos_t)))


def _draw_elbow_angle(ax: Any, kps: np.ndarray) -> None:
    angle = _elbow_angle_deg(kps)
    txt = f"elbow={angle:.1f}deg" if math.isfinite(angle) else "elbow=nan"
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=8, ha="left", va="bottom")


def _setup_axis(ax: Any, kps: np.ndarray) -> None:
    valid = np.isfinite(kps).all(axis=1)
    if not np.any(valid):
        ax.set_xticks([])
        ax.set_yticks([])
        return
    pts = kps[valid]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    pad = 0.25 * span
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    ax.set_xlim(cx - span / 2.0 - pad, cx + span / 2.0 + pad)
    ax.set_ylim(cy + span / 2.0 + pad, cy - span / 2.0 - pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])


def _format_angle(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(v):
        return "nan"
    return f"{v:.1f}"


def _rep_png_name(user_rep_order: int) -> str:
    return f"rep_{int(user_rep_order)}_bent_arms.png"


def export_bent_arms_debug(
    *,
    analysis_context: dict,
    bent_arms_result: dict,
    output_dir: str | Path,
) -> dict:
    out_dir = ensure_dir(output_dir)
    rep_results = bent_arms_result.get("rep_results") if isinstance(bent_arms_result, dict) else []
    if not isinstance(rep_results, list):
        rep_results = []

    exported_files: list[str] = []
    rep_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []

    for rep in rep_results:
        if not isinstance(rep, dict):
            warnings.append("rep_result invalido")
            continue
        user_rep_order = int(rep.get("user_rep_order", -1))
        user_rep_raw_index = int(rep.get("user_rep_raw_index", -1))
        if user_rep_order <= 0:
            user_rep_order = user_rep_raw_index + 1 if user_rep_raw_index >= 0 else 1
        anchor_pairs = _get_rep_anchor_pairs(analysis_context, user_rep_order, user_rep_raw_index)
        anchor_metrics = rep.get("anchor_metrics") if isinstance(rep.get("anchor_metrics"), dict) else {}
        anchor_rulings = rep.get("anchor_rulings") if isinstance(rep.get("anchor_rulings"), dict) else {}

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        flat_axes = list(axes.flatten())

        for idx, anchor in enumerate(BENT_ARMS_ANCHORS):
            ax = flat_axes[idx]
            pair = anchor_pairs.get(anchor) if isinstance(anchor_pairs, dict) else None
            metric = anchor_metrics.get(anchor) if isinstance(anchor_metrics, dict) else {}
            ruling = anchor_rulings.get(anchor) if isinstance(anchor_rulings, dict) else {}
            angle_txt = _format_angle((metric or {}).get("angle_elbow"))
            severity = str((ruling or {}).get("severity", "none"))
            failed = bool((ruling or {}).get("failed", False))

            frame_txt = "n/a"
            if isinstance(pair, dict):
                frame_val = pair.get("user_frame")
                if isinstance(frame_val, (int, float)):
                    frame_txt = str(int(frame_val))

            title = f"{anchor} | frame={frame_txt}\nangle={angle_txt}deg | {severity.upper()}"
            title_color = "#b22222" if failed else "#222222"
            ax.set_title(title, fontsize=9, color=title_color)

            ax.set_facecolor(_SEV_BG.get(severity, "#ecf0f1"))

            if not isinstance(pair, dict) or not bool(pair.get("valid", False)):
                ax.text(0.5, 0.5, "missing/invalid", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            kps = _get_user_kps(pair)
            if kps is None:
                ax.text(0.5, 0.5, "missing/invalid", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            _draw_right_chain(ax, kps)
            _draw_elbow_angle(ax, kps)
            _setup_axis(ax, kps)

        summary_line = (
            f"rep={user_rep_order} detected={rep.get('detected', False)} "
            f"severity={rep.get('severity', 'none')} n_failed={rep.get('n_failed', 0)}"
        )
        fig.suptitle(summary_line, fontsize=12)
        fig.tight_layout()

        png_name = _rep_png_name(user_rep_order)
        png_path = out_dir / png_name
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        exported_files.append(png_name)

        compact_anchor_metrics = {}
        for anchor in BENT_ARMS_ANCHORS:
            m = anchor_metrics.get(anchor) if isinstance(anchor_metrics, dict) else {}
            r = anchor_rulings.get(anchor) if isinstance(anchor_rulings, dict) else {}
            compact_anchor_metrics[anchor] = {
                "angle_elbow": (m or {}).get("angle_elbow"),
                "ruling": {
                    "severity": (r or {}).get("severity", "none"),
                    "failed": bool((r or {}).get("failed", False)),
                    "grave": bool((r or {}).get("grave", False)),
                    "trace": list((r or {}).get("trace", [])),
                },
            }

        rep_summaries.append(
            {
                "user_rep_order": user_rep_order,
                "user_rep_raw_index": user_rep_raw_index,
                "detected": bool(rep.get("detected", False)),
                "severity": rep.get("severity", "none"),
                "magnitude": float(rep.get("magnitude", 0.0)),
                "failed_anchors": list(rep.get("failed_anchors", [])),
                "anchors": compact_anchor_metrics,
                "png": png_name,
            }
        )

    summary = {
        "detector": bent_arms_result.get("detector", "bent_arms"),
        "detected": bool(bent_arms_result.get("detected", False)),
        "severity": bent_arms_result.get("severity", "none"),
        "score": float(bent_arms_result.get("score", 0.0)),
        "num_reps_analyzed": int(bent_arms_result.get("num_reps_analyzed", 0)),
        "num_reps_detected": int(bent_arms_result.get("num_reps_detected", 0)),
        "rep_results": rep_summaries,
        "warnings": list(bent_arms_result.get("warnings", [])) + warnings,
    }
    summary_path = out_dir / "bent_arms_summary.json"
    save_json(summary_path, summary)
    exported_files.insert(0, summary_path.name)

    return {
        "output_dir": str(out_dir),
        "files": exported_files,
        "warnings": summary["warnings"],
    }
