
from __future__ import annotations

import numpy as np

from src.pose_cleaning.constants import KP_R_HIP, KP_R_KNEE, KP_R_SHOULDER


def compute_right_torso_femur_ratio(kps_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(kps_xy, dtype=np.float64)
    T = xy.shape[0]
    ratio = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        shoulder = xy[t, KP_R_SHOULDER]
        hip = xy[t, KP_R_HIP]
        knee = xy[t, KP_R_KNEE]
        if not (
            np.isfinite(shoulder).all()
            and np.isfinite(hip).all()
            and np.isfinite(knee).all()
        ):
            continue
        L_torso = float(np.linalg.norm(shoulder - hip))
        L_femur = float(np.linalg.norm(hip - knee))
        if L_femur > 1e-6:
            ratio[t] = L_torso / L_femur
    return ratio

