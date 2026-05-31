
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KP_SHOULDER = 6
KP_L_HIP    = 11
KP_HIP      = 12


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


def _wrap_pi(x: float) -> float:
    if not math.isfinite(x):
        return x
    while x >  math.pi: x -= 2.0 * math.pi
    while x < -math.pi: x += 2.0 * math.pi
    return x


@dataclass(frozen=True)
class RomMetrics:
    theta_start:    float   # θ_torso at ecc_0 (radians)
    theta_bottom:   float   # θ_torso at bottom (radians)
    rom_user:       float   # signed ROM_user = θ_bottom − θ_start (radians)
    rom_user_abs:   float   # |ROM_user| (radians)
    rom_ideal_abs:  float   # |ROM_ideal| si existe, si no nan
    rom_norm:       float   # rom_user_abs / rom_ideal_abs (nan sin ideal)


def compute_rom_metrics(
    user_kps_start:  np.ndarray,
    user_kps_bottom: np.ndarray,
    *,
    ideal_kps_start:  np.ndarray | None = None,
    ideal_kps_bottom: np.ndarray | None = None,
) -> RomMetrics:
    user_start = np.asarray(user_kps_start, dtype=np.float64)
    user_bottom = np.asarray(user_kps_bottom, dtype=np.float64)
    if user_start.shape != (17, 2) or user_bottom.shape != (17, 2):
        raise ValueError(f"Expected user keypoints shape (17,2), got {user_start.shape} and {user_bottom.shape}")
    th_s = _torso_angle(user_start)
    th_b = _torso_angle(user_bottom)
    rom_user = _wrap_pi(th_b - th_s)
    rom_user_abs = abs(rom_user) if math.isfinite(rom_user) else float("nan")

    rom_ideal_abs = float("nan")
    rom_norm = float("nan")
    if ideal_kps_start is not None and ideal_kps_bottom is not None:
        ideal_start = np.asarray(ideal_kps_start, dtype=np.float64)
        ideal_bottom = np.asarray(ideal_kps_bottom, dtype=np.float64)
        if ideal_start.shape != (17, 2) or ideal_bottom.shape != (17, 2):
            raise ValueError(f"Expected ideal keypoints shape (17,2), got {ideal_start.shape} and {ideal_bottom.shape}")
        th_is = _torso_angle(ideal_start)
        th_ib = _torso_angle(ideal_bottom)
        rom_ideal = _wrap_pi(th_ib - th_is)
        if math.isfinite(rom_ideal) and abs(rom_ideal) > 1e-6:
            rom_ideal_abs = abs(rom_ideal)
            if math.isfinite(rom_user_abs):
                rom_norm = rom_user_abs / rom_ideal_abs

    return RomMetrics(
        theta_start=th_s,
        theta_bottom=th_b,
        rom_user=rom_user,
        rom_user_abs=rom_user_abs,
        rom_ideal_abs=rom_ideal_abs,
        rom_norm=rom_norm,
    )
