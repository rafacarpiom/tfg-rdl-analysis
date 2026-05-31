
from __future__ import annotations

from typing import Any

import numpy as np

from src.biomechanics.normalization.config import NormalizationConfig
from src.biomechanics.normalization.validation import validate_keypoints_sequence_2d


def normalize_pose_sequence(
    kps_xy: np.ndarray,
    *,
    config: NormalizationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or NormalizationConfig()
    if cfg.method != "pelvis_torso_scale":
        raise ValueError(f"Unsupported normalization method: {cfg.method!r}.")
    if cfg.apply_rotation:
        raise NotImplementedError(
            "Rotation-based normalization is not implemented because the default pipeline preserves orientation differences."
        )

    mode = str(cfg.sequence_scale_mode)
    if mode not in {"fixed_median", "framewise"}:
        raise ValueError(
            f"Unsupported sequence_scale_mode: {mode!r}. Expected 'fixed_median' or 'framewise'."
        )

    arr = validate_keypoints_sequence_2d(kps_xy, expected_keypoints=17)

    pelvis_idx = int(cfg.pelvis_idx)
    shoulder_idx = int(cfg.shoulder_idx)
    eps = float(cfg.eps)
    total_frame_count = int(arr.shape[0])

    if pelvis_idx < 0 or pelvis_idx >= arr.shape[1]:
        raise ValueError(f"pelvis_idx out of range: {pelvis_idx}")
    if shoulder_idx < 0 or shoulder_idx >= arr.shape[1]:
        raise ValueError(f"shoulder_idx out of range: {shoulder_idx}")

    origins = np.full((total_frame_count, 2), np.nan, dtype=np.float64)
    raw_scales = np.full((total_frame_count,), np.nan, dtype=np.float64)
    scales = np.full((total_frame_count,), np.nan, dtype=np.float64)
    normalized = np.full_like(arr, np.nan, dtype=np.float64)
    mask_valid_normalized = np.zeros((total_frame_count,), dtype=bool)
    warnings: list[str] = []

    pelvis_pts = arr[:, pelvis_idx, :]
    shoulder_pts = arr[:, shoulder_idx, :]
    pelvis_finite = np.isfinite(pelvis_pts).all(axis=1)
    shoulder_finite = np.isfinite(shoulder_pts).all(axis=1)
    torso_lengths = np.linalg.norm(shoulder_pts - pelvis_pts, axis=1)
    mask_valid_candidate = pelvis_finite & shoulder_finite & (torso_lengths > eps)

    origins[mask_valid_candidate] = pelvis_pts[mask_valid_candidate]
    raw_scales[mask_valid_candidate] = torso_lengths[mask_valid_candidate]

    if mode == "fixed_median":
        valid_scales = raw_scales[np.isfinite(raw_scales) & (raw_scales > eps)]
        if valid_scales.size == 0:
            warnings.append("NO_VALID_REFERENCE_SEGMENTS_FOR_SEQUENCE_NORMALIZATION")
            return {
                "kps_xy_normalized": normalized,
                "mask_valid_normalized": mask_valid_normalized,
                "origins": origins,
                "scales": scales,
                "raw_scales": raw_scales,
                "method": "pelvis_torso_scale",
                "sequence_scale_mode": mode,
                "pelvis_idx": pelvis_idx,
                "shoulder_idx": shoulder_idx,
                "rotation_applied": False,
                "valid_frame_count": 0,
                "total_frame_count": total_frame_count,
                "valid_frame_ratio": 0.0,
                "warnings": warnings,
            }
        fixed_scale = float(np.median(valid_scales))
        if not np.isfinite(fixed_scale) or fixed_scale <= eps:
            warnings.append("INVALID_FIXED_MEDIAN_SCALE_FOR_SEQUENCE_NORMALIZATION")
            return {
                "kps_xy_normalized": normalized,
                "mask_valid_normalized": mask_valid_normalized,
                "origins": origins,
                "scales": scales,
                "raw_scales": raw_scales,
                "method": "pelvis_torso_scale",
                "sequence_scale_mode": mode,
                "pelvis_idx": pelvis_idx,
                "shoulder_idx": shoulder_idx,
                "rotation_applied": False,
                "valid_frame_count": 0,
                "total_frame_count": total_frame_count,
                "valid_frame_ratio": 0.0,
                "warnings": warnings,
            }
        normalized[mask_valid_candidate] = (
            arr[mask_valid_candidate] - origins[mask_valid_candidate][:, None, :]
        ) / fixed_scale
        scales[mask_valid_candidate] = fixed_scale
        mask_valid_normalized = mask_valid_candidate.copy()
    elif mode == "framewise":
        valid_idx = np.where(mask_valid_candidate)[0]
        if valid_idx.size > 0:
            normalized[valid_idx] = (
                arr[valid_idx] - origins[valid_idx][:, None, :]
            ) / raw_scales[valid_idx][:, None, None]
            scales[valid_idx] = raw_scales[valid_idx]
            mask_valid_normalized[valid_idx] = True

    if np.any(mask_valid_normalized):
        pelvis_norm = normalized[mask_valid_normalized, pelvis_idx, :]
        pelvis_ok = np.all(np.isclose(pelvis_norm, 0.0, atol=1e-6), axis=1)
        if not np.all(pelvis_ok):
            fail_idx = np.where(mask_valid_normalized)[0][~pelvis_ok]
            mask_valid_normalized[fail_idx] = False
            normalized[fail_idx] = np.nan
            scales[fail_idx] = np.nan
            warnings.append("NORMALIZED_PELVIS_NOT_ZERO")

    if mode == "framewise" and np.any(mask_valid_normalized):
        valid_idx = np.where(mask_valid_normalized)[0]
        shoulder_norm = normalized[valid_idx, shoulder_idx, :]
        pelvis_norm = normalized[valid_idx, pelvis_idx, :]
        dists = np.linalg.norm(shoulder_norm - pelvis_norm, axis=1)
        dist_ok = np.isclose(dists, 1.0, atol=1e-6)
        if not np.all(dist_ok):
            fail_idx = valid_idx[~dist_ok]
            mask_valid_normalized[fail_idx] = False
            normalized[fail_idx] = np.nan
            scales[fail_idx] = np.nan
            warnings.append("NORMALIZED_TORSO_SCALE_NOT_ONE")

    if warnings:
        warnings = list(dict.fromkeys(warnings))

    valid_frame_count = int(np.count_nonzero(mask_valid_normalized))
    valid_frame_ratio = float(valid_frame_count / total_frame_count) if total_frame_count > 0 else 0.0

    return {
        "kps_xy_normalized": normalized,
        "mask_valid_normalized": mask_valid_normalized,
        "origins": origins,
        "scales": scales,
        "raw_scales": raw_scales,
        "method": "pelvis_torso_scale",
        "sequence_scale_mode": mode,
        "pelvis_idx": pelvis_idx,
        "shoulder_idx": shoulder_idx,
        "rotation_applied": False,
        "valid_frame_count": valid_frame_count,
        "total_frame_count": total_frame_count,
        "valid_frame_ratio": valid_frame_ratio,
        "warnings": warnings,
    }

