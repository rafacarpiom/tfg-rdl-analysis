
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _finite_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(valid.size)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        start = i
        while i < n and valid[i]:
            i += 1
        runs.append((start, i))
    return runs


def _adjust_window(length: int, requested: int, polyorder: int) -> int | None:
    if length <= polyorder + 1:
        return None
    window = min(int(requested), int(length))
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return None
    if window < 3:
        return None
    return window


def smooth_keypoints_savgol(
    kps_xy: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 2,
    min_valid_ratio: float = 0.75,
) -> np.ndarray:
    out = np.asarray(kps_xy, dtype=np.float64).copy()
    T, K, C = out.shape
    min_ratio = float(np.clip(min_valid_ratio, 0.0, 1.0))
    for k in range(K):
        for axis in range(C):
            seq = out[:, k, axis]
            valid = np.isfinite(seq)
            if T > 0 and (float(np.sum(valid)) / float(T)) < min_ratio:
                continue
            for start, end in _finite_runs(valid):
                length = end - start
                window = _adjust_window(length, window_length, polyorder)
                if window is None:
                    continue
                out[start:end, k, axis] = savgol_filter(
                    seq[start:end],
                    window_length=window,
                    polyorder=min(polyorder, window - 1),
                    mode="interp",
                )
    return out
