
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.biomechanics.rdl.detectors.spine_flexion.metrics import KP_RIGHT_ANKLE, KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_SHOULDER, SPINE_ANCHORS, TORSO_DROP_GRAVE, TORSO_DROP_LEVE, TORSO_DROP_MEDIA, align_ideal_to_user_torso_for_spine_geometry
from src.visualization.rdl.debug.common import ensure_dir, save_json

_CHAIN = ((KP_RIGHT_ANKLE, KP_RIGHT_KNEE), (KP_RIGHT_KNEE, KP_RIGHT_HIP), (KP_RIGHT_HIP, KP_RIGHT_SHOULDER))
_SEV_BG = {"none": "#ecf0f1", "leve": "#fff3bf", "media": "#ffe0b2", "grave": "#ffcdd2"}
_SEV_RANK = {"none": 0, "leve": 1, "media": 2, "grave": 3}


def _get_rep_pairs(analysis_context: dict, rep_order: int, rep_raw: int) -> dict[str, dict]:
    ap = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else {}
    reps = ap.get("paired_repetitions") if isinstance(ap, dict) else []
    if not isinstance(reps, list):
        return {}
    for rep in reps:
        if isinstance(rep, dict) and int(rep.get("user_rep_order", -1)) == int(rep_order):
            return rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
    for rep in reps:
        if isinstance(rep, dict) and int(rep.get("user_rep_raw_index", -1)) == int(rep_raw):
            return rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
    rep_index = int(rep_order - 1) if int(rep_order) > 0 else -1
    for rep in reps:
        if isinstance(rep, dict) and int(rep.get("rep_index", -1)) == rep_index:
            return rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
    if rep_index >= 0 and rep_index < len(reps):
        rep = reps[rep_index]
        if isinstance(rep, dict):
            return rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
    return {}


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except Exception:
        return None
    return f if np.isfinite(f) else None


def _finite_kp(kps: np.ndarray, idx: int) -> bool:
    return bool(np.isfinite(np.asarray(kps[idx], dtype=np.float64)).all())


def _draw_chain(ax: Any, kps: np.ndarray, color: str, label: str) -> None:
    first = True
    for i, j in _CHAIN:
        if _finite_kp(kps, i) and _finite_kp(kps, j):
            ax.plot([kps[i, 0], kps[j, 0]], [kps[i, 1], kps[j, 1]], color=color, linewidth=2.0, label=label if first else None)
            first = False
    for idx in (KP_RIGHT_ANKLE, KP_RIGHT_KNEE, KP_RIGHT_HIP, KP_RIGHT_SHOULDER):
        if _finite_kp(kps, idx):
            ax.scatter(kps[idx, 0], kps[idx, 1], color=color, s=20, zorder=3)


def _setup_axis(ax: Any, user: np.ndarray, ideal: np.ndarray) -> None:
    pts = np.vstack([user, ideal])
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
    ax.grid(alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def _anchor_level(anchor_state: dict[str, Any], segment_state: dict[str, Any]) -> str:
    sev_anchor = str(anchor_state.get("severity", "none"))
    if sev_anchor in _SEV_RANK:
        return sev_anchor
    sev_segment = str(segment_state.get("severity", "none"))
    if sev_segment in _SEV_RANK:
        return sev_segment
    return "none"


def _plot_anchor_panel(ax: Any, anchor: str, pair: dict[str, Any], anchor_state: dict[str, Any], segment_state: dict[str, Any]) -> None:
    user = pair.get("user_kps_clean")
    if user is None:
        user = pair.get("user_kps_normalized")
    ideal = pair.get("ideal_kps_clean")
    if ideal is None:
        ideal = pair.get("ideal_kps_normalized")
    if not (isinstance(user, np.ndarray) and isinstance(ideal, np.ndarray) and user.shape == (17, 2) and ideal.shape == (17, 2)):
        ax.text(0.5, 0.5, f"{anchor}\nmissing", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    user_arr = np.asarray(user, dtype=np.float64)
    try:
        ideal_aligned = align_ideal_to_user_torso_for_spine_geometry(np.asarray(ideal, dtype=np.float64), user_arr)
    except Exception:
        ideal_aligned = np.asarray(ideal, dtype=np.float64)

    _draw_chain(ax, ideal_aligned, "#16a085", "ideal aligned")
    _draw_chain(ax, user_arr, "#8e44ad", "user")
    if _finite_kp(user_arr, KP_RIGHT_SHOULDER) and _finite_kp(ideal_aligned, KP_RIGHT_SHOULDER):
        uy = float(user_arr[KP_RIGHT_SHOULDER, 1])
        iy = float(ideal_aligned[KP_RIGHT_SHOULDER, 1])
        ux = float(user_arr[KP_RIGHT_SHOULDER, 0])
        ix = float(ideal_aligned[KP_RIGHT_SHOULDER, 0])
        xh = max(ux, ix) + 10.0
        ax.axhline(iy, color="#16a085", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.annotate("", xy=(xh, uy), xytext=(xh, iy), arrowprops={"arrowstyle": "->", "color": "#34495e", "lw": 1.2})
    _setup_axis(ax, user_arr, ideal_aligned)
    ax.legend(loc="lower right", fontsize=7, frameon=True)

    text = [
        f"{anchor} | u:{pair.get('user_frame')} i:{pair.get('ideal_frame')}",
        f"drop_norm={anchor_state.get('torso_drop_vs_ideal')}",
        f"drop_px={anchor_state.get('torso_drop_px')}",
        f"hip={'failed' if segment_state.get('hip_hinge_failed') else 'ok'}",
        f"knee={'failed' if segment_state.get('knee_dominant_failed') else 'ok'}",
        f"neck={'failed' if segment_state.get('neck_movement_failed') else 'ok'}",
        f"spine={segment_state.get('severity', 'none')}",
        f"suppress={anchor_state.get('suppression_reason', 'none')}",
    ]
    ax.text(0.02, 0.98, "\n".join(text), transform=ax.transAxes, ha="left", va="top", fontsize=7.4, bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#b0b0b0", "alpha": 0.92})


def export_spine_flexion_debug(*, analysis_context: dict, spine_flexion_result: dict, output_dir: str | Path) -> dict:
    out_dir = ensure_dir(output_dir)
    rep_results = spine_flexion_result.get("rep_results") if isinstance(spine_flexion_result, dict) else []
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
        by_anchor = rep.get("spine_geometry_by_anchor") if isinstance(rep.get("spine_geometry_by_anchor"), dict) else {}
        per_anchor = rep.get("per_anchor") if isinstance(rep.get("per_anchor"), dict) else {}
        per_segment = rep.get("per_segment") if isinstance(rep.get("per_segment"), dict) else {}
        sev = str(rep.get("severity", "none"))

        fig = plt.figure(figsize=(18, 9.5))
        gs = fig.add_gridspec(3, 5, height_ratios=[2.0, 1.1, 1.0], hspace=0.30, wspace=0.22)
        for i, anchor in enumerate(SPINE_ANCHORS):
            ax = fig.add_subplot(gs[0, i])
            pair = anchors.get(anchor) if isinstance(anchors, dict) and isinstance(anchors.get(anchor), dict) else {}
            anchor_state = per_anchor.get(anchor) if isinstance(per_anchor.get(anchor), dict) else {}
            segment = by_anchor.get(anchor, {}).get("segment")
            segment_state = per_segment.get(segment) if isinstance(segment, str) and isinstance(per_segment.get(segment), dict) else {}
            anchor_level = _anchor_level(anchor_state, segment_state)
            ax.set_facecolor(_SEV_BG.get(anchor_level, "#ecf0f1"))
            _plot_anchor_panel(ax, anchor, pair, anchor_state, segment_state)
            ax.set_title(f"{anchor} level:{anchor_level}", fontsize=9)

        ax_bar = fig.add_subplot(gs[1, :4])
        values = []
        bar_colors: list[str] = []
        for anchor in SPINE_ANCHORS:
            item = per_anchor.get(anchor) if isinstance(per_anchor.get(anchor), dict) else {}
            values.append(_safe_float(item.get("torso_drop_vs_ideal")))
            bar_colors.append(_SEV_BG.get(str(item.get("severity", "none")), "#6c757d"))
        x = np.arange(len(SPINE_ANCHORS))
        ax_bar.bar(x, [v if isinstance(v, float) else np.nan for v in values], color=bar_colors, edgecolor="#6c757d")
        for thr, name, color in ((TORSO_DROP_LEVE, "leve", "#f1c40f"), (TORSO_DROP_MEDIA, "media", "#e67e22"), (TORSO_DROP_GRAVE, "grave", "#c0392b")):
            ax_bar.axhline(thr, linestyle="--", linewidth=1.1, color=color, label=f"{name} {thr:.2f}")
        ax_bar.axhline(0.0, color="#7f8c8d", linewidth=0.9)
        ax_bar.set_xticks(x, SPINE_ANCHORS)
        ax_bar.set_ylabel("torso_drop_vs_ideal")
        ax_bar.grid(alpha=0.18, axis="y")
        ax_bar.legend(loc="best", fontsize=8)

        ax_text = fig.add_subplot(gs[1:, 4])
        ax_text.set_facecolor(_SEV_BG.get(sev, "#ecf0f1"))
        ax_text.set_xticks([])
        ax_text.set_yticks([])
        ax_text.text(
            0.03,
            0.98,
            "\n".join(
                [
                    f"detected={rep.get('detected', False)}",
                    f"severity={rep.get('severity', 'none')}",
                    f"score={rep.get('score', 0.0)}",
                    f"method={rep.get('method', '')}",
                    f"triggered={rep.get('triggered_segments', [])}",
                    "",
                    "trace:",
                    *[str(x) for x in (rep.get("trace") if isinstance(rep.get("trace"), list) else [])[:8]],
                ]
            ),
            ha="left",
            va="top",
            fontsize=8.6,
            family="monospace",
        )
        fig.suptitle(f"spine_flexion rep={rep_order} level:{sev}", fontsize=12)
        fig.tight_layout()
        png_name = f"rep_{rep_order}_spine_flexion.png"
        fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(png_name)

        anchor_summary: dict[str, Any] = {}
        for anchor in SPINE_ANCHORS:
            pair = anchors.get(anchor) if isinstance(anchors, dict) and isinstance(anchors.get(anchor), dict) else {}
            item = per_anchor.get(anchor) if isinstance(per_anchor.get(anchor), dict) else {}
            geom = by_anchor.get(anchor) if isinstance(by_anchor.get(anchor), dict) else {}
            anchor_summary[anchor] = {
                "user_frame": pair.get("user_frame"),
                "ideal_frame": pair.get("ideal_frame"),
                "torso_drop_vs_ideal": item.get("torso_drop_vs_ideal"),
                "torso_drop_px": item.get("torso_drop_px"),
                "torso_low": item.get("torso_low"),
                "torso_low_severity": item.get("torso_low_severity"),
                "hip_hinge_failed": item.get("hip_hinge_failed"),
                "knee_dominant_failed": item.get("knee_dominant_failed"),
                "neck_movement_failed": item.get("neck_movement_failed"),
                "suppression_reason": item.get("suppression_reason"),
                "result": item.get("result"),
                "geometry_status": geom.get("status"),
            }
        compact_reps.append(
            {
                "user_rep_order": rep.get("user_rep_order"),
                "user_rep_raw_index": rep.get("user_rep_raw_index"),
                "detected": rep.get("detected"),
                "severity": rep.get("severity"),
                "score": rep.get("score"),
                "method": rep.get("method"),
                "triggered_segments": rep.get("triggered_segments", []),
                "anchors": anchor_summary,
                "warnings": rep.get("warnings", []),
                "png": png_name,
            }
        )

    summary = {
        "detector": spine_flexion_result.get("detector", "spine_flexion"),
        "detected": spine_flexion_result.get("detected", False),
        "severity": spine_flexion_result.get("severity", "none"),
        "score": spine_flexion_result.get("score", 0.0),
        "num_reps_analyzed": spine_flexion_result.get("num_reps_analyzed", 0),
        "num_reps_detected": spine_flexion_result.get("num_reps_detected", 0),
        "rep_results": compact_reps,
        "warnings": spine_flexion_result.get("warnings", []),
    }
    save_json(out_dir / "spine_flexion_summary.json", summary)
    txt_lines = [
        "spine_flexion summary",
        f"detected={summary['detected']}",
        f"severity={summary['severity']}",
        f"score={summary['score']}",
        f"num_reps_detected={summary['num_reps_detected']}/{summary['num_reps_analyzed']}",
        "",
    ]
    for rep in compact_reps:
        txt_lines.append(f"rep {rep.get('user_rep_order')}: detected={rep.get('detected')} severity={rep.get('severity')} score={rep.get('score')}")
    (out_dir / "spine_flexion_summary.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    return {"output_dir": str(out_dir), "files": ["spine_flexion_summary.json", "spine_flexion_summary.txt", *files], "warnings": summary["warnings"]}
