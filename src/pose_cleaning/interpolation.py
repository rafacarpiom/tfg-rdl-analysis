
from __future__ import annotations

import numpy as np


def interpolate_short_gaps(
    kps_xy: np.ndarray,
    *,
    max_gap: int = 5,
) -> np.ndarray:
    out = np.asarray(kps_xy, dtype=np.float64).copy()
    T, K, C = out.shape
    gap_limit = max(0, int(max_gap))

    for k in range(K):
        for axis in range(C):
            seq = out[:, k, axis]
            valid = np.isfinite(seq)
            i = 0
            while i < T:
                if valid[i]:
                    i += 1
                    continue
                start = i
                while i < T and not valid[i]:
                    i += 1
                end = i - 1
                gap_len = end - start + 1
                left = start - 1
                right = end + 1
                if (
                    gap_len <= gap_limit
                    and left >= 0
                    and right < T
                    and valid[left]
                    and valid[right]
                ):
                    seq[start : end + 1] = np.linspace(
                        seq[left],
                        seq[right],
                        num=gap_len + 2,
                    )[1:-1]
                    valid[start : end + 1] = True
    return out

