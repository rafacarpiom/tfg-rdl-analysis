
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Keypoints COCO-17 lado derecho para esta herramienta.
KP_SHOULDER = 6
KP_ELBOW    = 8
KP_WRIST    = 10
KP_HIP      = 12


BAR_FAR_ANCHORS: tuple[str, ...] = (
    "ecc_0", "ecc_25", "ecc_50", "ecc_75", "ecc_100",
    "bottom",
    "con_0", "con_25", "con_50", "con_75", "con_100",
)

@dataclass(frozen=True)
class BarFarAnchorMetrics:

    anchor: str
    wrist_error_px: float
    torso_length: float
    wrist_error_norm: float
    wrist_error_x_px: float
    wrist_error_x_norm: float
    wrist_error_y_px: float
    wrist_error_y_norm: float
    delta_x_wrist: float
    arm_dir_delta: float
    elbow_angle_ideal: float
    elbow_angle_user: float
    elbow_angle_delta: float


# ── Auxiliares ───────────────────────────────────────────────────────────────────

def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


def _angle_at_vertex(pts: np.ndarray, a: int, v: int, b: int) -> float:
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
    cos_val = float(np.dot(va, vb)) / (na * nb)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    if not (np.isfinite(v1).all() and np.isfinite(v2).all()):
        return float("nan")
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_val = float(np.dot(v1, v2)) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# ── API pública ────────────────────────────────────────────────────────────────

def compute_bar_far_anchor_metrics(
    ideal_kps_normalized: np.ndarray,
    user_kps_normalized: np.ndarray,
    anchor: str,
) -> BarFarAnchorMetrics:
    ideal_arr = np.asarray(ideal_kps_normalized, dtype=np.float64)
    user_arr = np.asarray(user_kps_normalized, dtype=np.float64)
    if ideal_arr.shape != (17, 2) or user_arr.shape != (17, 2):
        raise ValueError(
            f"Expected (17, 2) keypoints, got {ideal_arr.shape} and {user_arr.shape}."
        )

    ideal_wrist = np.asarray(ideal_arr[KP_WRIST], dtype=np.float64)
    user_wrist = np.asarray(user_arr[KP_WRIST], dtype=np.float64)
    user_shoulder = np.asarray(user_arr[KP_SHOULDER], dtype=np.float64)
    user_hip = np.asarray(user_arr[KP_HIP], dtype=np.float64)

    if _finite(*ideal_wrist.tolist(), *user_wrist.tolist()):
        wrist_error_px = float(np.linalg.norm(user_wrist - ideal_wrist))
        delta_x_wrist = float(user_wrist[0] - ideal_wrist[0])
        wrist_error_x_px = float(abs(delta_x_wrist))
        wrist_error_y_px = float(abs(user_wrist[1] - ideal_wrist[1]))
    else:
        wrist_error_px = float("nan")
        delta_x_wrist = float("nan")
        wrist_error_x_px = float("nan")
        wrist_error_y_px = float("nan")

    if _finite(*user_shoulder.tolist(), *user_hip.tolist()):
        torso_length = float(np.linalg.norm(user_shoulder - user_hip))
    else:
        torso_length = float("nan")

    if _finite(wrist_error_px, torso_length) and torso_length > 1e-6:
        wrist_error_norm = wrist_error_px / torso_length
    else:
        wrist_error_norm = float("nan")

    if _finite(wrist_error_x_px, torso_length) and torso_length > 1e-6:
        wrist_error_x_norm = wrist_error_x_px / torso_length
    else:
        wrist_error_x_norm = float("nan")

    if _finite(wrist_error_y_px, torso_length) and torso_length > 1e-6:
        wrist_error_y_norm = wrist_error_y_px / torso_length
    else:
        wrist_error_y_norm = float("nan")

    arm_ideal = np.asarray(ideal_arr[KP_WRIST], dtype=np.float64) - np.asarray(
        ideal_arr[KP_SHOULDER], dtype=np.float64
    )
    arm_user = user_wrist - user_shoulder
    arm_dir_delta = _angle_between(arm_ideal, arm_user)

    elbow_ideal = _angle_at_vertex(ideal_arr, KP_SHOULDER, KP_ELBOW, KP_WRIST)
    elbow_user = _angle_at_vertex(user_arr, KP_SHOULDER, KP_ELBOW, KP_WRIST)
    if _finite(elbow_ideal, elbow_user):
        elbow_delta = abs(elbow_user - elbow_ideal)
    else:
        elbow_delta = float("nan")

    return BarFarAnchorMetrics(
        anchor=anchor,
        wrist_error_px=wrist_error_px,
        torso_length=torso_length,
        wrist_error_norm=wrist_error_norm,
        wrist_error_x_px=wrist_error_x_px,
        wrist_error_x_norm=wrist_error_x_norm,
        wrist_error_y_px=wrist_error_y_px,
        wrist_error_y_norm=wrist_error_y_norm,
        delta_x_wrist=delta_x_wrist,
        arm_dir_delta=arm_dir_delta,
        elbow_angle_ideal=elbow_ideal,
        elbow_angle_user=elbow_user,
        elbow_angle_delta=elbow_delta,
    )
