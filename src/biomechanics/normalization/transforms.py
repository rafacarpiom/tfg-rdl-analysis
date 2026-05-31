
from __future__ import annotations

import numpy as np

from src.biomechanics.normalization.validation import (
    validate_finite_point,
    validate_keypoints_2d,
    validate_reference_segment,
)


def translate_to_origin(
    kps: np.ndarray,
    origin_idx: int,
) -> tuple[np.ndarray, dict]:
    arr = validate_keypoints_2d(kps)
    idx = int(origin_idx)
    validate_finite_point(arr, idx, name="origin")
    origin = arr[idx]
    translated = arr - origin
    info = {
        "origin_idx": int(idx),
        "origin": [float(origin[0]), float(origin[1])],
    }
    return translated, info


def scale_by_segment_length(
    kps: np.ndarray,
    idx_a: int,
    idx_b: int,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    arr = validate_keypoints_2d(kps)
    ia = int(idx_a)
    ib = int(idx_b)
    length = validate_reference_segment(
        arr,
        ia,
        ib,
        eps=float(eps),
        name="scale_segment",
    )
    scaled = arr / float(length)
    info = {
        "scale_idx_a": int(ia),
        "scale_idx_b": int(ib),
        "scale_length": float(length),
    }
    return scaled, info

