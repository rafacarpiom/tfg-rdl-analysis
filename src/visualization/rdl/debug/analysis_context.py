
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.visualization.rdl.debug.common import plot_skeleton, set_equal_xy

ANCHORS_TO_PLOT = ("ecc_0", "ecc_25", "ecc_50", "ecc_75", "bottom", "con_25", "con_50", "con_75", "con_100")


def export_analysis_context_debug(*, analysis_context: dict, output_path: str | Path) -> dict:
    context_meta = analysis_context.get("context_meta", {}) if isinstance(analysis_context.get("context_meta"), dict) else {}
    anchor_pairs = analysis_context.get("anchor_pairs", {}) if isinstance(analysis_context.get("anchor_pairs"), dict) else {}
    paired_repetitions = anchor_pairs.get("paired_repetitions", []) if isinstance(anchor_pairs.get("paired_repetitions"), list) else []
    warnings: list[str] = list(analysis_context.get("warnings", [])) if isinstance(analysis_context.get("warnings"), list) else []
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes_flat = axes.flatten()
    selected_rep = paired_repetitions[0] if paired_repetitions else None
    if selected_rep is None:
        for ax in axes_flat:
            ax.axis("off")
        fig.text(0.5, 0.5, "No paired repetitions available", ha="center", va="center", fontsize=16)
        fig.tight_layout()
        fig.savefig(Path(output_path), dpi=150)
        plt.close(fig)
        return {
            "reference_name": context_meta.get("reference_name"),
            "reference_dir": context_meta.get("reference_dir"),
            "ideal_valid_rep_index": context_meta.get("ideal_valid_rep_index"),
            "ideal_rep_raw_index": context_meta.get("ideal_rep_raw_index"),
            "num_user_reps": context_meta.get("num_user_reps", 0),
            "num_paired_repetitions": context_meta.get("num_paired_repetitions", 0),
            "selected_user_rep_raw_index": None,
            "selected_user_rep_order": None,
            "anchors": {},
            "warnings": warnings + ["NO_PAIRED_REPETITIONS_AVAILABLE"],
        }
    anchors_payload = selected_rep.get("anchors", {}) if isinstance(selected_rep.get("anchors"), dict) else {}
    summary_anchors: dict[str, Any] = {}
    for i, anchor_name in enumerate(ANCHORS_TO_PLOT):
        ax = axes_flat[i]
        pair = anchors_payload.get(anchor_name, {})
        is_valid = bool(pair.get("valid", False))
        pair_warnings = pair.get("warnings", []) if isinstance(pair.get("warnings"), list) else []
        summary_anchors[anchor_name] = {
            "valid": is_valid,
            "user_frame": pair.get("user_frame"),
            "ideal_frame": pair.get("ideal_frame"),
            "user_source": pair.get("user_source"),
            "ideal_source": pair.get("ideal_source"),
            "warnings": pair_warnings,
        }
        if not is_valid:
            ax.axis("off")
            ax.text(0.03, 0.97, f"{anchor_name}\nvalid=False\nwarnings={pair_warnings}", ha="left", va="top", fontsize=8)
            continue
        plot_skeleton(ax, pair.get("user_kps_normalized"), color="C0", label="user", alpha=0.95)
        plot_skeleton(ax, pair.get("ideal_kps_normalized"), color="C1", label="ideal", alpha=0.85)
        ax.axhline(0.0, color="gray", alpha=0.3, linewidth=1.0)
        ax.axvline(0.0, color="gray", alpha=0.3, linewidth=1.0)
        ax.set_title(f"{anchor_name} | u={pair.get('user_frame')} | i={pair.get('ideal_frame')}", fontsize=9)
        set_equal_xy(ax)
        ax.invert_yaxis()
        ax.legend(loc="best", fontsize=7)
    for i in range(len(ANCHORS_TO_PLOT), len(axes_flat)):
        axes_flat[i].axis("off")
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150)
    plt.close(fig)
    return {
        "reference_name": context_meta.get("reference_name"),
        "reference_dir": context_meta.get("reference_dir"),
        "ideal_valid_rep_index": context_meta.get("ideal_valid_rep_index"),
        "ideal_rep_raw_index": context_meta.get("ideal_rep_raw_index"),
        "num_user_reps": context_meta.get("num_user_reps", 0),
        "num_paired_repetitions": context_meta.get("num_paired_repetitions", 0),
        "selected_user_rep_raw_index": selected_rep.get("user_rep_raw_index"),
        "selected_user_rep_order": selected_rep.get("user_rep_order"),
        "anchors": summary_anchors,
        "warnings": warnings,
    }
