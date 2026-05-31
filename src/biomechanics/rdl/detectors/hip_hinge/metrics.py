
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KP_SHOULDER = 6
KP_HIP = 12
KP_KNEE = 14
KP_ANKLE = 16

HIP_HINGE_ANCHORS: tuple[str, ...] = (
    "ecc_0", "ecc_25", "ecc_50", "ecc_75", "ecc_100", "bottom",
)


@dataclass(frozen=True)
class HipBackSide:

    anchor: str
    hip_x_start: float
    hip_x_anchor: float
    hip_back_px: float       # hip_x_anchor − hip_x_start (signed)
    torso_length: float      # ||shoulder − hip|| at start frame (raw)
    hip_back_norm: float     # hip_back_px / torso_length


@dataclass(frozen=True)
class HipBackMetrics:
    anchor: str
    user: HipBackSide
    ideal: HipBackSide
    delta_hip_back: float


def _validate_sequence(kps_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(kps_xy, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (17, 2):
        raise ValueError(f"Expected kps_xy shape (T, 17, 2), got {arr.shape}.")
    return arr


def _check_frame(arr: np.ndarray, frame: int, label: str) -> int:
    if not isinstance(frame, int):
        raise ValueError(f"{label} must be int, got {type(frame).__name__}")
    if frame < 0 or frame >= arr.shape[0]:
        raise ValueError(f"{label} out of range: {frame}")
    return frame


def compute_hip_back_side_from_frames(
    kps_xy: np.ndarray,
    *,
    start_frame: int,
    anchor_frame: int,
    anchor: str,
) -> HipBackSide:
    arr = _validate_sequence(kps_xy)
    sf = _check_frame(arr, int(start_frame), "start_frame")
    af = _check_frame(arr, int(anchor_frame), "anchor_frame")
    hip_start = arr[sf, KP_HIP]
    hip_anchor = arr[af, KP_HIP]
    shoulder_start = arr[sf, KP_SHOULDER]
    if not (np.isfinite(hip_start).all() and np.isfinite(hip_anchor).all() and np.isfinite(shoulder_start).all()):
        raise ValueError(f"Non-finite keypoints for anchor {anchor}")
    hip_x_start = float(hip_start[0])
    hip_x_anchor = float(hip_anchor[0])
    hip_back_px = float(hip_x_anchor - hip_x_start)
    torso_length = float(np.linalg.norm(shoulder_start - hip_start))
    hip_back_norm = float(hip_back_px / torso_length) if torso_length > 1e-6 else float("nan")
    return HipBackSide(
        anchor=anchor,
        hip_x_start=hip_x_start,
        hip_x_anchor=hip_x_anchor,
        hip_back_px=hip_back_px,
        torso_length=torso_length,
        hip_back_norm=hip_back_norm,
    )


def compute_hip_back_metrics_for_anchor(
    *,
    user_kps_xy: np.ndarray,
    ideal_kps_xy: np.ndarray,
    user_start_frame: int,
    ideal_start_frame: int,
    user_anchor_frame: int,
    ideal_anchor_frame: int,
    anchor: str,
) -> HipBackMetrics:
    user = compute_hip_back_side_from_frames(
        user_kps_xy,
        start_frame=int(user_start_frame),
        anchor_frame=int(user_anchor_frame),
        anchor=anchor,
    )
    ideal = compute_hip_back_side_from_frames(
        ideal_kps_xy,
        start_frame=int(ideal_start_frame),
        anchor_frame=int(ideal_anchor_frame),
        anchor=anchor,
    )
    return HipBackMetrics(
        anchor=anchor,
        user=user,
        ideal=ideal,
        delta_hip_back=float(user.hip_back_norm - ideal.hip_back_norm),
    )
