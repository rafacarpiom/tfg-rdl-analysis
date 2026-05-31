
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.rdl.debug.common import plot_skeleton, set_equal_xy


def _valid_reps(segmentation_result: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    reps = segmentation_result.get("reps", [])
    if not isinstance(reps, list):
        return []
    return [(idx, rep) for idx, rep in enumerate(reps) if isinstance(rep, dict) and rep.get("anchor_valid", True) is True]


def _resolve_anchor_frames_for_view(rep: dict[str, Any]) -> dict[str, int | None]:
    anchors = rep.get("anchors", {}) if isinstance(rep.get("anchors"), dict) else {}
    out: dict[str, int | None] = {}
    for name in ("ecc_0", "ecc_50", "bottom", "con_50", "con_100"):
        value = anchors.get(name)
        if isinstance(value, dict):
            value = value.get("frame")
        try:
            out[name] = int(value) if value is not None else None
        except (TypeError, ValueError):
            out[name] = None
    return out


def export_normalization_debug(*, pose_clean: dict, user_pose_sequence_normalization: dict, segmentation_result: dict, output_path: str | Path) -> dict:
    kps_norm = np.asarray(user_pose_sequence_normalization["kps_xy_normalized"], dtype=np.float64)
    mask_norm = np.asarray(user_pose_sequence_normalization["mask_valid_normalized"], dtype=bool)
    raw_scales = np.asarray(user_pose_sequence_normalization.get("raw_scales"), dtype=np.float64)
    norm_meta = user_pose_sequence_normalization.get("meta", {})
    video_id = str(segmentation_result.get("video_id", norm_meta.get("video_id", "unknown")))
    selected_rep_index: int | None = None
    warnings: list[str] = []
    reps = _valid_reps(segmentation_result)
    if reps:
        selected_rep_index, selected_rep = reps[0]
        selected_anchor_frames = _resolve_anchor_frames_for_view(selected_rep)
    else:
        selected_anchor_frames = {"ecc_0": None, "ecc_50": None, "bottom": None, "con_50": None, "con_100": None}
        warnings.append("NO_VALID_REPS_FOR_NORMALIZATION_DEBUG")
    valid_indices = np.where(mask_norm)[0].tolist()
    chosen_frames: list[tuple[str, int]] = []
    for name in ("ecc_0", "ecc_50", "bottom", "con_50", "con_100"):
        frame = selected_anchor_frames.get(name)
        if isinstance(frame, int) and 0 <= frame < kps_norm.shape[0] and bool(mask_norm[frame]):
            chosen_frames.append((name, frame))
    if len(chosen_frames) < 5:
        used = {f for _, f in chosen_frames}
        for frame in valid_indices:
            if frame in used:
                continue
            chosen_frames.append((f"valid_{len(chosen_frames)+1}", int(frame)))
            if len(chosen_frames) >= 5:
                break
    if not chosen_frames:
        warnings.append("NO_VALID_NORMALIZED_FRAMES_FOR_DEBUG")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.1])
    ax_scales = fig.add_subplot(gs[0, 0:2]); ax_mask = fig.add_subplot(gs[0, 2:4]); ax_text = fig.add_subplot(gs[0, 4])
    frame_axes = [fig.add_subplot(gs[1, i]) for i in range(5)]
    ax_scales.plot(raw_scales, color="C0", linewidth=1.5); ax_scales.set_title("Raw torso scale over time"); ax_scales.set_xlabel("Frame"); ax_scales.set_ylabel("Scale"); ax_scales.grid(alpha=0.2)
    ax_mask.plot(mask_norm.astype(int), color="C2", linewidth=1.5); ax_mask.set_title("Normalized valid-frame mask"); ax_mask.set_xlabel("Frame"); ax_mask.set_ylabel("Valid (0/1)"); ax_mask.set_ylim(-0.1, 1.1); ax_mask.grid(alpha=0.2)
    ax_text.axis("off")
    info = [f"video_id: {video_id}", f"num_frames: {kps_norm.shape[0]}", f"valid_frame_count: {int(mask_norm.sum())}", f"valid_frame_ratio: {float(mask_norm.mean()) if len(mask_norm) else 0.0:.3f}", f"normalization_method: {norm_meta.get('normalization_method', 'n/d')}", f"sequence_scale_mode: {norm_meta.get('sequence_scale_mode', 'n/d')}", f"warnings: {norm_meta.get('warnings', [])}"]
    ax_text.text(0.01, 0.99, "\n".join(info), va="top", ha="left", fontsize=9)
    for idx, ax in enumerate(frame_axes):
        if idx < len(chosen_frames):
            anchor, frame = chosen_frames[idx]
            pts = kps_norm[frame]
            plot_skeleton(ax, pts, color="C0")
            ax.axhline(0.0, color="gray", alpha=0.3, linewidth=1.0); ax.axvline(0.0, color="gray", alpha=0.3, linewidth=1.0)
            ax.set_title(f"{anchor} | f={frame}", fontsize=9); ax.invert_yaxis(); set_equal_xy(ax)
        else:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(Path(output_path), dpi=150); plt.close(fig)
    return {
        "video_id": video_id,
        "num_frames": int(kps_norm.shape[0]),
        "valid_frame_count": int(mask_norm.sum()),
        "valid_frame_ratio": float(mask_norm.mean()) if len(mask_norm) else 0.0,
        "normalization_method": norm_meta.get("normalization_method", "pelvis_torso_scale"),
        "sequence_scale_mode": norm_meta.get("sequence_scale_mode", "fixed_median"),
        "selected_rep_index": selected_rep_index,
        "selected_anchor_frames": selected_anchor_frames,
        "warnings": warnings + list(norm_meta.get("warnings", [])),
    }
