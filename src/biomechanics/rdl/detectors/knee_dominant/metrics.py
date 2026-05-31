
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KP_SHOULDER = 6
KP_HIP = 12
KP_KNEE = 14
KP_ANKLE = 16

KNEE_DOMINANT_ANCHORS: tuple[str, ...] = (
    "ecc_0", "ecc_25", "ecc_50", "ecc_75", "ecc_100", "bottom",
)


@dataclass(frozen=True)
class KneeDominantMetrics:

    anchor: str
    hip_ideal: float
    hip_user: float
    delta_hip: float
    knee_ideal: float
    knee_user: float
    delta_knee: float


# ── Auxiliares ──────────────────────────────────────────────────────────────────

def _interior_angle(pts: np.ndarray, a: int, v: int, b: int) -> float:
    pa = np.asarray(pts[a], dtype=np.float64)
    pv = np.asarray(pts[v], dtype=np.float64)
    pb = np.asarray(pts[b], dtype=np.float64)
    if not (np.isfinite(pa).all() and np.isfinite(pv).all() and np.isfinite(pb).all()):
        return float("nan")
    va = pa - pv
    vb = pb - pv
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(
        float(np.dot(va, vb)) / (na * nb), -1.0, 1.0,
    ))))


def _flexion_angle(pts: np.ndarray, a: int, v: int, b: int) -> float:
    interior = _interior_angle(pts, a, v, b)
    if not math.isfinite(interior):
        return float("nan")
    return 180.0 - interior


def _signed_delta(user_v: float, ideal_v: float) -> float:
    if not (math.isfinite(user_v) and math.isfinite(ideal_v)):
        return float("nan")
    return float(user_v - ideal_v)


# ── API pública ───────────────────────────────────────────────────────────────

def compute_knee_dominant_metrics(
    ideal_kps_normalized: np.ndarray,
    user_kps_normalized: np.ndarray,
    anchor: str,
) -> KneeDominantMetrics:
    ideal_arr = np.asarray(ideal_kps_normalized, dtype=np.float64)
    user_arr = np.asarray(user_kps_normalized, dtype=np.float64)
    if ideal_arr.shape != (17, 2) or user_arr.shape != (17, 2):
        raise ValueError(
            f"Expected (17, 2) keypoints, got {ideal_arr.shape} and {user_arr.shape}."
        )

    hip_ideal = _flexion_angle(ideal_arr, KP_SHOULDER, KP_HIP, KP_KNEE)
    hip_user = _flexion_angle(user_arr, KP_SHOULDER, KP_HIP, KP_KNEE)
    knee_ideal = _flexion_angle(ideal_arr, KP_HIP, KP_KNEE, KP_ANKLE)
    knee_user = _flexion_angle(user_arr, KP_HIP, KP_KNEE, KP_ANKLE)

    return KneeDominantMetrics(
        anchor=anchor,
        hip_ideal=hip_ideal,
        hip_user=hip_user,
        delta_hip=_signed_delta(hip_user, hip_ideal),
        knee_ideal=knee_ideal,
        knee_user=knee_user,
        delta_knee=_signed_delta(knee_user, knee_ideal),
    )
