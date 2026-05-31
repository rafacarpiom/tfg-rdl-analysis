
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KP_SHOULDER = 6
KP_ELBOW = 8
KP_WRIST = 10

BENT_ARMS_ANCHORS: tuple[str, ...] = (
    "ecc_0", "ecc_25", "ecc_50", "ecc_75", "ecc_100",
    "bottom",
    "con_25", "con_50", "con_75", "con_100",
)


@dataclass(frozen=True)
class BentArmsAnchorMetrics:
    anchor: str
    angle_elbow: float


def _angle_deg(p0: np.ndarray, v: np.ndarray, p1: np.ndarray) -> float:
    a = np.asarray(p0, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    cos_t = float(np.dot(a, b) / (na * nb))
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


def compute_bent_arms_anchor_metrics(
    user_kps: np.ndarray,
    anchor: str,
) -> BentArmsAnchorMetrics:
    kps = np.asarray(user_kps, dtype=np.float64)
    if kps.shape != (17, 2):
        return BentArmsAnchorMetrics(anchor=anchor, angle_elbow=float("nan"))
    shoulder = kps[KP_SHOULDER]
    elbow = kps[KP_ELBOW]
    wrist = kps[KP_WRIST]
    if not (np.all(np.isfinite(shoulder)) and np.all(np.isfinite(elbow)) and np.all(np.isfinite(wrist))):
        return BentArmsAnchorMetrics(anchor=anchor, angle_elbow=float("nan"))
    angle = _angle_deg(shoulder, elbow, wrist)
    return BentArmsAnchorMetrics(anchor=anchor, angle_elbow=angle)
