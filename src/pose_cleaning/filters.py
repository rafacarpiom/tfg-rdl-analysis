
from __future__ import annotations

import numpy as np

from src.pose_cleaning.constants import RIGHT_CHAIN


def _indices(keypoint_indices: tuple[int, ...] | None) -> tuple[int, ...]:
    return RIGHT_CHAIN if keypoint_indices is None else tuple(int(i) for i in keypoint_indices)


def filter_by_confidence(
    kps_xy: np.ndarray,
    kps_score: np.ndarray,
    *,
    score_thr: float = 0.4,
    keypoint_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(kps_xy, dtype=np.float64).copy()
    score = np.asarray(kps_score, dtype=np.float64)
    mask_valid = np.isfinite(xy).all(axis=2)
    for k in _indices(keypoint_indices):
        ok = (score[:, k] >= float(score_thr)) & mask_valid[:, k]
        mask_valid[:, k] = ok
        xy[~ok, k] = np.nan
    return xy, mask_valid


def remove_velocity_outliers(
    kps_xy: np.ndarray,
    factor: float = 2.0,
    keypoint_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    cleaned = np.asarray(kps_xy, dtype=np.float64).copy()
    T, K, _ = cleaned.shape
    _ = K
    for k in _indices(keypoint_indices):
        velocities: list[tuple[int, float]] = []
        for t in range(1, T):
            prev = cleaned[t - 1, k]
            cur = cleaned[t, k]
            if np.isfinite(prev).all() and np.isfinite(cur).all():
                velocities.append((t, float(np.linalg.norm(cur - prev))))
        valid_vel = np.asarray([v for _t, v in velocities if np.isfinite(v)], dtype=np.float64)
        if valid_vel.size == 0:
            continue
        median_vel = float(np.median(valid_vel))
        if not np.isfinite(median_vel) or median_vel <= 1e-6:
            continue
        threshold = float(factor) * median_vel
        for t, vel in velocities:
            if vel > threshold:
                cleaned[t, k] = np.nan
    return cleaned
