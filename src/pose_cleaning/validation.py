
from __future__ import annotations

import numpy as np

from src.pose_cleaning.constants import EXPECTED_K, RIGHT_CHAIN


def validate_pose_arrays(kps_xy: np.ndarray, kps_score: np.ndarray) -> None:
    if kps_xy.ndim != 3 or kps_xy.shape[1:] != (EXPECTED_K, 2):
        raise ValueError(f"kps_xy debe tener shape (T, 17, 2); recibido {kps_xy.shape}")
    if kps_score.ndim != 2 or kps_score.shape[1] != EXPECTED_K:
        raise ValueError(f"kps_score debe tener shape (T, 17); recibido {kps_score.shape}")
    if kps_score.shape[0] != kps_xy.shape[0]:
        raise ValueError(
            "kps_xy y kps_score tienen distinto número de frames: "
            f"{kps_xy.shape[0]} vs {kps_score.shape[0]}"
        )


def compute_mask_valid(kps_xy: np.ndarray) -> np.ndarray:
    return np.isfinite(kps_xy).all(axis=2)


def compute_mask_valid_frames(mask_valid: np.ndarray) -> np.ndarray:
    if mask_valid.ndim != 2:
        raise ValueError(f"mask_valid debe tener shape (T, K); recibido {mask_valid.shape}")
    return np.any(mask_valid, axis=1)


def compute_mask_valid_right_chain(kps_xy: np.ndarray) -> np.ndarray:
    mask = compute_mask_valid(kps_xy)
    return mask[:, RIGHT_CHAIN]


def compute_mask_valid_frames_right_chain(mask_valid_right_chain: np.ndarray) -> np.ndarray:
    if mask_valid_right_chain.ndim != 2:
        raise ValueError(
            "mask_valid_right_chain debe tener shape (T, len(RIGHT_CHAIN)); "
            f"recibido {mask_valid_right_chain.shape}"
        )
    return np.all(mask_valid_right_chain, axis=1)
