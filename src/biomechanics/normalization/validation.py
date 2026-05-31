
from __future__ import annotations

import numpy as np


def validate_keypoints_2d(
    kps: np.ndarray,
    *,
    min_keypoints: int = 17,
) -> np.ndarray:
    arr = np.asarray(kps, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected keypoints with ndim=2, got ndim={arr.ndim}.")
    if arr.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N, 2), got shape={arr.shape}.")
    if arr.shape[0] < int(min_keypoints):
        raise ValueError(
            f"Expected at least {int(min_keypoints)} keypoints, got {arr.shape[0]}."
        )
    return arr


def validate_keypoints_sequence_2d(
    kps_xy: np.ndarray,
    *,
    expected_keypoints: int = 17,
) -> np.ndarray:
    arr = np.asarray(kps_xy, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected keypoint sequence with ndim=3, got ndim={arr.ndim}.")
    exp_kps = int(expected_keypoints)
    if arr.shape[1:] != (exp_kps, 2):
        raise ValueError(
            f"Expected keypoint sequence shape (T, {exp_kps}, 2), got shape={arr.shape}."
        )
    return arr


def validate_finite_point(
    kps: np.ndarray,
    idx: int,
    *,
    name: str = "point",
) -> None:
    arr = np.asarray(kps, dtype=np.float64)
    i = int(idx)
    if i < 0 or i >= arr.shape[0]:
        raise ValueError(f"{name} index out of range: idx={i}, valid=[0, {arr.shape[0] - 1}].")
    point = arr[i]
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} at idx={i} must be a finite 2D point.")


def is_valid_reference_segment(
    kps: np.ndarray,
    idx_a: int,
    idx_b: int,
    *,
    eps: float = 1e-6,
) -> bool:
    arr = np.asarray(kps, dtype=np.float64)
    ia = int(idx_a)
    ib = int(idx_b)
    if ia < 0 or ia >= arr.shape[0] or ib < 0 or ib >= arr.shape[0]:
        return False
    pa = arr[ia]
    pb = arr[ib]
    if pa.shape != (2,) or pb.shape != (2,) or not np.isfinite(pa).all() or not np.isfinite(pb).all():
        return False
    length = float(np.linalg.norm(pb - pa))
    return bool(length > float(eps))


def validate_reference_segment(
    kps: np.ndarray,
    idx_a: int,
    idx_b: int,
    *,
    eps: float = 1e-6,
    name: str = "reference_segment",
) -> float:
    arr = np.asarray(kps, dtype=np.float64)
    ia = int(idx_a)
    ib = int(idx_b)
    if ia < 0 or ia >= arr.shape[0] or ib < 0 or ib >= arr.shape[0]:
        raise ValueError(
            f"{name} indices out of range: idx_a={ia}, idx_b={ib}, valid=[0, {arr.shape[0] - 1}]."
        )
    validate_finite_point(arr, ia, name=f"{name}_a")
    validate_finite_point(arr, ib, name=f"{name}_b")
    length = float(np.linalg.norm(arr[ib] - arr[ia]))
    if length <= float(eps):
        raise ValueError(f"{name} length must be > {float(eps)}, got {length}.")
    return float(length)

