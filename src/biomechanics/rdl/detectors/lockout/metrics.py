
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KP_SHOULDER = 6
KP_L_HIP    = 11
KP_HIP      = 12
LOCKOUT_ANCHOR_CANDIDATES: tuple[str, ...] = ("con_100", "end")


def _pelvis_midpoint(kps: np.ndarray) -> np.ndarray:
    left  = np.asarray(kps[KP_L_HIP], dtype=np.float64)
    right = np.asarray(kps[KP_HIP],   dtype=np.float64)
    if np.all(np.isfinite(left)) and np.all(np.isfinite(right)):
        return 0.5 * (left + right)
    if np.all(np.isfinite(right)):
        return right
    return left


def _torso_angle(kps: np.ndarray) -> float:
    shoulder = np.asarray(kps[KP_SHOULDER], dtype=np.float64)
    pelvis   = _pelvis_midpoint(kps)
    if not (np.all(np.isfinite(shoulder)) and np.all(np.isfinite(pelvis))):
        return float("nan")
    vx = shoulder[0] - pelvis[0]
    vy = shoulder[1] - pelvis[1]
    if vx == 0.0 and vy == 0.0:
        return float("nan")
    return float(math.atan2(vy, vx))


@dataclass(frozen=True)
class LockoutMetrics:
    theta_end_user: float      # radians (user final top)
    theta_end_ideal: float     # radians (ideal final top)
    error_lockout: float       # radians, directed: θ_end_user − θ_end_ideal


def compute_lockout_metrics(
    user_kps_end: np.ndarray,
    ideal_kps_end: np.ndarray,
) -> LockoutMetrics:
    user_arr = np.asarray(user_kps_end, dtype=np.float64)
    ideal_arr = np.asarray(ideal_kps_end, dtype=np.float64)
    if user_arr.shape != (17, 2) or ideal_arr.shape != (17, 2):
        raise ValueError(f"Expected (17,2) keypoints, got {user_arr.shape} and {ideal_arr.shape}")
    th_u = _torso_angle(user_arr)
    th_i = _torso_angle(ideal_arr)
    if math.isfinite(th_u) and math.isfinite(th_i):
        err = th_u - th_i
    else:
        err = float("nan")
    return LockoutMetrics(
        theta_end_user=th_u,
        theta_end_ideal=th_i,
        error_lockout=err,
    )
