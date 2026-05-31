
from __future__ import annotations

import numpy as np

from src.pose_cleaning.constants import RIGHT_CHAIN
from src.pose_cleaning.quality import compute_right_torso_femur_ratio


def _mean_raw_clean_diff(kps_xy: np.ndarray, kps_xy_clean: np.ndarray) -> float:
    finite = (
        np.isfinite(kps_xy[:, RIGHT_CHAIN]).all(axis=2)
        & np.isfinite(kps_xy_clean[:, RIGHT_CHAIN]).all(axis=2)
    )
    if not np.any(finite):
        return float("nan")
    raw = kps_xy[:, RIGHT_CHAIN][finite]
    clean = kps_xy_clean[:, RIGHT_CHAIN][finite]
    diff = np.linalg.norm(raw - clean, axis=1)
    return float(np.mean(np.abs(diff)))


def _safe_percentile(values: np.ndarray, pct: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanpercentile(finite, pct))


def _gap_lengths(valid_frames: np.ndarray) -> list[int]:
    gaps: list[int] = []
    i = 0
    n = int(valid_frames.size)
    while i < n:
        if valid_frames[i]:
            i += 1
            continue
        start = i
        while i < n and not valid_frames[i]:
            i += 1
        gaps.append(i - start)
    return gaps


def build_cleaning_diagnostics(
    kps_xy: np.ndarray,
    kps_xy_clean: np.ndarray,
    mask_valid_right_chain: np.ndarray,
    mask_valid_frames: np.ndarray,
) -> dict[str, float | int | list[int]]:
    raw_chain_valid = np.isfinite(kps_xy[:, RIGHT_CHAIN]).all(axis=2)
    raw_frames_valid = np.all(raw_chain_valid, axis=1)
    clean_frames_valid = np.asarray(mask_valid_frames, dtype=bool)
    gaps = _gap_lengths(clean_frames_valid)
    removed = raw_frames_valid & ~clean_frames_valid
    return {
        "right_chain": list(RIGHT_CHAIN),
        "right_chain_valid_frame_pct": float(np.mean(clean_frames_valid)),
        "right_chain_removed_frame_pct": float(np.mean(removed)),
        "right_chain_valid_keypoint_pct": float(np.mean(mask_valid_right_chain)),
        "num_gaps": int(len(gaps)),
        "mean_gap_length": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap_length": int(max(gaps)) if gaps else 0,
        "mean_diff_raw_vs_clean_right_chain": _mean_raw_clean_diff(kps_xy, kps_xy_clean),
        "median_ratio": _safe_percentile(compute_right_torso_femur_ratio(kps_xy_clean), 50.0),
    }

