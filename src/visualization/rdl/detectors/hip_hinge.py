
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.biomechanics.rdl.detectors.hip_hinge import HIP_HINGE_ANCHORS
from src.biomechanics.rdl.detectors.hip_hinge.rules import SEV_THR
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
_CHAIN_KPS = tuple(sorted({kp for pair in _CHAIN for kp in pair}))
_SEV_BG = {
    "none": "#f3f4f6",
    "leve": "#fff59d",
    "media": "#ffcc80",
    "grave": "#ef9a9a",
}
_SEV_EDGE = {
    "none": "#9ca3af",
    "leve": "#f59e0b",
    "media": "#f97316",
    "grave": "#dc2626",
}
_SEV_USER = {
    "none": "#16a34a",
    "leve": "#ca8a04",
    "media": "#ea580c",
    "grave": "#b91c1c",
}


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


def _draw_chain(ax: Any, kps: np.ndarray, color: str, label: str, linewidth: float = 1.8) -> None:
    for i, j in _CHAIN:
        if np.isfinite(kps[i]).all() and np.isfinite(kps[j]).all():
            ax.plot(
                [kps[i, 0], kps[j, 0]],
                [kps[i, 1], kps[j, 1]],
                color=color,
                linewidth=linewidth,
                alpha=0.95,
                label=label if i == KP_SHOULDER and j == KP_ELBOW else None,
            )
    ax.scatter(kps[[KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE], 0], kps[[KP_SHOULDER, KP_ELBOW, KP_WRIST, KP_HIP, KP_KNEE, KP_ANKLE], 1], s=18, color=color, zorder=3)


def _setup_axis(ax: Any, user_kps: np.ndarray, ideal_kps: np.ndarray) -> None:
    # Ajustar solo a la cadena dibujada para evitar esqueletos minúsculos
    # si keypoints ajenos están ruidosos/fuera de frame.
    pts = np.vstack([user_kps[list(_CHAIN_KPS)], ideal_kps[list(_CHAIN_KPS)]])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    pad = span * 0.15
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
    ax.set_ylim(cy + span / 2 + pad, cy - span / 2 - pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_summary(ax: Any, rep: dict) -> None:
    ax.set_axis_off()
    lines = [
        f"detected={rep.get('detected')} severity={rep.get('severity')}",
        f"confidence={rep.get('confidence')} dominant_phase={rep.get('dominant_phase')}",
        f"magnitude={rep.get('magnitude')} mean_deficit={rep.get('mean_deficit')} max_deficit={rep.get('max_deficit')}",
        f"n_failed={rep.get('n_failed')} failed_anchors={rep.get('failed_anchors', [])}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=8, family="monospace")


def _anchor_severity_from_delta(delta_hip_back: Any) -> str:
    try:
        d = float(delta_hip_back)
    except Exception:
        return "none"
    if not np.isfinite(d):
        return "none"
    if d > float(SEV_THR["grave"]):
        return "grave"
    if d > float(SEV_THR["media"]):
        return "media"
    if d > float(SEV_THR["leve"]):
        return "leve"
    return "none"


def _resolve_anchor_status(metrics: dict[str, Any], ruling: dict[str, Any]) -> tuple[Any, bool, str]:
    failed = bool((ruling or {}).get("failed", False))
    delta = (metrics or {}).get("delta_hip_back")
    if delta is None:
        delta = (ruling or {}).get("delta_hip_back")
    severity = _anchor_severity_from_delta(delta)
    # Coherencia defensiva: si la regla marca fallo, severidad no puede ser "none".
    if failed and severity == "none":
        severity = "leve"
    return delta, failed, severity


def export_hip_hinge_debug(*, analysis_context: dict, hip_hinge_result: dict, output_dir: str | Path) -> dict:
    out_dir = ensure_dir(output_dir)
    rep_results = hip_hinge_result.get("rep_results") if isinstance(hip_hinge_result, dict) else []
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

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        flat = list(axes.flatten())
        for idx, anchor in enumerate(HIP_HINGE_ANCHORS):
            ax = flat[idx]
            pair = anchor_pairs.get(anchor) if isinstance(anchor_pairs, dict) else None
            metrics = anchor_metrics.get(anchor) if isinstance(anchor_metrics, dict) else {}
            ruling = anchor_rulings.get(anchor) if isinstance(anchor_rulings, dict) else {}
            delta, failed, severity = _resolve_anchor_status(metrics, ruling)
            ax.set_facecolor(_SEV_BG.get(severity, "#ecf0f1"))
            ax.patch.set_edgecolor(_SEV_EDGE.get(severity, "#9ca3af"))
            ax.patch.set_linewidth(2.5 if failed else 1.5)

            user_frame = (pair or {}).get("user_frame") if isinstance(pair, dict) else None
            ideal_frame = (pair or {}).get("ideal_frame") if isinstance(pair, dict) else None
            ax.set_title(
                f"{anchor}\nu:{user_frame} i:{ideal_frame}\ndelta={delta}\nlevel:{severity} {'FALLO' if failed else 'OK'}",
                fontsize=9,
            )

            user_kps = (pair or {}).get("user_kps_clean")
            ideal_kps = (pair or {}).get("ideal_kps_clean")
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
            _draw_chain(ax, ideal_arr, "#1d4ed8", "ideal", linewidth=2.0)
            _draw_chain(ax, user_arr, _SEV_USER.get(severity, "#16a34a"), "user", linewidth=2.6 if failed else 2.2)
            if np.isfinite(user_arr[KP_HIP]).all():
                ax.scatter([user_arr[KP_HIP, 0]], [user_arr[KP_HIP, 1]], s=80, facecolors="none", edgecolors="#1f2937", linewidths=2)
            _setup_axis(ax, user_arr, ideal_arr)

        _draw_summary(flat[6], rep)
        flat[7].set_axis_off()
        flat[7].text(
            0.02,
            0.98,
            "Color user by error:\nnone=green\nleve=yellow\nmedia=orange\ngrave=red",
            ha="left",
            va="top",
            fontsize=9,
        )
        fig.suptitle(f"rep={rep_order} hip_hinge severity={rep.get('severity')} score={rep.get('score')}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        png_name = f"rep_{rep_order}_hip_hinge.png"
        fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(png_name)

        compact_anchors: dict[str, Any] = {}
        for anchor in HIP_HINGE_ANCHORS:
            pair = anchor_pairs.get(anchor) if isinstance(anchor_pairs, dict) else {}
            m = anchor_metrics.get(anchor) if isinstance(anchor_metrics, dict) else {}
            r = anchor_rulings.get(anchor) if isinstance(anchor_rulings, dict) else {}
            user_side = (m or {}).get("user", {}) if isinstance((m or {}).get("user", {}), dict) else {}
            ideal_side = (m or {}).get("ideal", {}) if isinstance((m or {}).get("ideal", {}), dict) else {}
            compact_anchors[anchor] = {
                "user_frame": (pair or {}).get("user_frame"),
                "ideal_frame": (pair or {}).get("ideal_frame"),
                "delta_hip_back": (m or {}).get("delta_hip_back"),
                "user_hip_back_norm": user_side.get("hip_back_norm"),
                "ideal_hip_back_norm": ideal_side.get("hip_back_norm"),
                "failed": bool((r or {}).get("failed", False)),
                "warnings": list((pair or {}).get("warnings", [])),
            }

        compact_reps.append(
            {
                "user_rep_order": rep.get("user_rep_order"),
                "user_rep_raw_index": rep.get("user_rep_raw_index"),
                "detected": rep.get("detected"),
                "severity": rep.get("severity"),
                "confidence": rep.get("confidence"),
                "dominant_phase": rep.get("dominant_phase"),
                "magnitude": rep.get("magnitude"),
                "failed_anchors": rep.get("failed_anchors", []),
                "anchors": compact_anchors,
                "warnings": rep.get("warnings", []),
                "png": png_name,
            }
        )

    summary = {
        "detector": hip_hinge_result.get("detector", "hip_hinge"),
        "detected": hip_hinge_result.get("detected", False),
        "severity": hip_hinge_result.get("severity", "none"),
        "score": hip_hinge_result.get("score", 0.0),
        "num_reps_analyzed": hip_hinge_result.get("num_reps_analyzed", 0),
        "num_reps_detected": hip_hinge_result.get("num_reps_detected", 0),
        "rep_results": compact_reps,
        "warnings": hip_hinge_result.get("warnings", []),
    }
    save_json(out_dir / "hip_hinge_summary.json", summary)
    return {"output_dir": str(out_dir), "files": ["hip_hinge_summary.json", *files], "warnings": summary["warnings"]}
