
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Keypoints COCO lado derecho usados en el pipeline RDL.
KP_RIGHT_SHOULDER = 6
KP_RIGHT_HIP = 12
KP_RIGHT_KNEE = 14
KP_RIGHT_ANKLE = 16

SPINE_ANCHORS: tuple[str, ...] = ("ecc_0", "ecc_25", "ecc_50", "ecc_75", "ecc_100")

ANCHOR_TO_SEGMENT: dict[str, str] = {
    "ecc_25": "ecc_0_to_ecc_25",
    "ecc_50": "ecc_25_to_ecc_50",
    "ecc_75": "ecc_50_to_ecc_75",
    "ecc_100": "ecc_75_to_ecc_100",
}

SEGMENT_TO_ANCHOR: dict[str, str] = {v: k for k, v in ANCHOR_TO_SEGMENT.items()}

TORSO_DROP_LEVE = 0.08
TORSO_DROP_MEDIA = 0.15
TORSO_DROP_GRAVE = 0.22
TORSO_LOW_MIN_PX = 6.0


def _finite_point(kps: np.ndarray, idx: int) -> bool:
    try:
        p = np.asarray(kps[idx], dtype=np.float64)
    except Exception:
        return False
    return bool(p.shape == (2,) and np.isfinite(p).all())


def _dist(kps: np.ndarray, a: int, b: int) -> float:
    if not (_finite_point(kps, a) and _finite_point(kps, b)):
        return float("nan")
    return float(np.linalg.norm(np.asarray(kps[a], dtype=np.float64) - np.asarray(kps[b], dtype=np.float64)))


def _body_scale(user_kps: np.ndarray, ideal_aligned_kps: np.ndarray) -> float:
    candidates = [
        _dist(user_kps, KP_RIGHT_HIP, KP_RIGHT_KNEE),
        _dist(ideal_aligned_kps, KP_RIGHT_HIP, KP_RIGHT_KNEE),
        _dist(user_kps, KP_RIGHT_HIP, KP_RIGHT_SHOULDER),
        _dist(ideal_aligned_kps, KP_RIGHT_HIP, KP_RIGHT_SHOULDER),
        _dist(user_kps, KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
        _dist(ideal_aligned_kps, KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
    ]
    vals = [v for v in candidates if np.isfinite(v) and v > 1e-6]
    if not vals:
        return 1.0
    return float(np.median(vals))


def _torso_angle_deg(kps: np.ndarray) -> float:
    if not (_finite_point(kps, KP_RIGHT_HIP) and _finite_point(kps, KP_RIGHT_SHOULDER)):
        return float("nan")
    hip = np.asarray(kps[KP_RIGHT_HIP], dtype=np.float64)
    shoulder = np.asarray(kps[KP_RIGHT_SHOULDER], dtype=np.float64)
    dx = float(shoulder[0] - hip[0])
    dy_up = float(-(shoulder[1] - hip[1]))
    return float(math.degrees(math.atan2(dy_up, dx)))


def shoulder_drop_from_top_norm(
    kps: np.ndarray,
    kps_top: np.ndarray,
    *,
    scale: float | None = None,
) -> float:
    if not (
        _finite_point(kps, KP_RIGHT_SHOULDER)
        and _finite_point(kps_top, KP_RIGHT_SHOULDER)
        and _finite_point(kps, KP_RIGHT_HIP)
        and _finite_point(kps_top, KP_RIGHT_HIP)
    ):
        return float("nan")
    if scale is None or not (isinstance(scale, (int, float)) and np.isfinite(scale) and scale > 1e-6):
        scale = _body_scale(kps, kps_top)
    y_anchor = float(np.asarray(kps[KP_RIGHT_SHOULDER], dtype=np.float64)[1])
    y_top = float(np.asarray(kps_top[KP_RIGHT_SHOULDER], dtype=np.float64)[1])
    return float((y_anchor - y_top) / max(float(scale), 1e-6))


def severity_from_torso_low_norm(value: float, *, low_px: float = 0.0) -> str:
    if not np.isfinite(value) or not np.isfinite(low_px):
        return "none"
    if low_px < TORSO_LOW_MIN_PX:
        return "none"
    if value >= TORSO_DROP_GRAVE:
        return "grave"
    if value >= TORSO_DROP_MEDIA:
        return "media"
    if value >= TORSO_DROP_LEVE:
        return "leve"
    return "none"


def align_ideal_to_user_torso_for_spine_geometry(
    ideal_kps: np.ndarray,
    user_kps: np.ndarray,
) -> np.ndarray:
    ideal_arr = np.asarray(ideal_kps, dtype=np.float64)
    user_arr = np.asarray(user_kps, dtype=np.float64)
    if ideal_arr.shape != (17, 2) or user_arr.shape != (17, 2):
        raise ValueError("invalid_kps_shape_expected_17x2")
    if not (_finite_point(ideal_arr, KP_RIGHT_HIP) and _finite_point(ideal_arr, KP_RIGHT_SHOULDER)):
        raise ValueError("ideal_hip_or_shoulder_invalid")
    if not (_finite_point(user_arr, KP_RIGHT_HIP) and _finite_point(user_arr, KP_RIGHT_SHOULDER)):
        raise ValueError("user_hip_or_shoulder_invalid")

    ideal_hip = np.asarray(ideal_arr[KP_RIGHT_HIP], dtype=np.float64)
    user_hip = np.asarray(user_arr[KP_RIGHT_HIP], dtype=np.float64)
    ideal_torso = float(np.linalg.norm(ideal_arr[KP_RIGHT_SHOULDER] - ideal_hip))
    user_torso = float(np.linalg.norm(user_arr[KP_RIGHT_SHOULDER] - user_hip))
    if not (np.isfinite(ideal_torso) and np.isfinite(user_torso) and ideal_torso > 1e-9 and user_torso > 1e-9):
        raise ValueError("invalid_torso_length_for_alignment")
    scale = user_torso / ideal_torso
    return (ideal_arr - ideal_hip) * scale + user_hip


def compute_spine_anchor_geometry(
    *,
    user_kps: np.ndarray,
    ideal_aligned_kps: np.ndarray,
    anchor: str,
    user_frame: int | None = None,
    ideal_frame: int | None = None,
) -> dict[str, Any]:
    required = (KP_RIGHT_SHOULDER, KP_RIGHT_HIP, KP_RIGHT_KNEE)
    if not all(_finite_point(user_kps, i) for i in required):
        return {
            "anchor": anchor,
            "segment": ANCHOR_TO_SEGMENT.get(anchor),
            "status": "inconclusive",
            "reason": "user_keypoints_invalid",
            "user_frame": user_frame,
            "ideal_frame": ideal_frame,
        }
    if not all(_finite_point(ideal_aligned_kps, i) for i in required):
        return {
            "anchor": anchor,
            "segment": ANCHOR_TO_SEGMENT.get(anchor),
            "status": "inconclusive",
            "reason": "ideal_keypoints_invalid_after_alignment",
            "user_frame": user_frame,
            "ideal_frame": ideal_frame,
        }

    user_shoulder = np.asarray(user_kps[KP_RIGHT_SHOULDER], dtype=np.float64)
    ideal_shoulder = np.asarray(ideal_aligned_kps[KP_RIGHT_SHOULDER], dtype=np.float64)
    scale = _body_scale(user_kps, ideal_aligned_kps)
    shoulder_low_px = float(user_shoulder[1] - ideal_shoulder[1])
    shoulder_low_norm = float(shoulder_low_px / max(scale, 1e-6))
    torso_angle_user = _torso_angle_deg(user_kps)
    torso_angle_ideal = _torso_angle_deg(ideal_aligned_kps)
    torso_angle_delta = (
        float(torso_angle_user - torso_angle_ideal)
        if np.isfinite(torso_angle_user) and np.isfinite(torso_angle_ideal)
        else float("nan")
    )
    severity = severity_from_torso_low_norm(shoulder_low_norm, low_px=shoulder_low_px)

    return {
        "anchor": anchor,
        "segment": ANCHOR_TO_SEGMENT.get(anchor),
        "status": "ok",
        "user_frame": int(user_frame) if user_frame is not None else None,
        "ideal_frame": int(ideal_frame) if ideal_frame is not None else None,
        "body_scale": scale,
        "shoulder_low_px": shoulder_low_px,
        "shoulder_low_norm": shoulder_low_norm,
        "torso_low_failed": severity != "none",
        "torso_low_severity": severity,
        "torso_angle_user_deg": torso_angle_user,
        "torso_angle_ideal_deg": torso_angle_ideal,
        "torso_angle_delta_deg": torso_angle_delta,
        "thresholds": {
            "leve": TORSO_DROP_LEVE,
            "media": TORSO_DROP_MEDIA,
            "grave": TORSO_DROP_GRAVE,
            "min_px": TORSO_LOW_MIN_PX,
        },
    }


def geometry_by_segment(geometry_by_anchor: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for anchor, item in geometry_by_anchor.items():
        segment = str(item.get("segment") or ANCHOR_TO_SEGMENT.get(anchor) or "")
        if segment:
            out[segment] = dict(item)
    return out
