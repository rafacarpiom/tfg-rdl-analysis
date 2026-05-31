
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pose_cleaning.config import PoseCleaningConfig
from src.pose_cleaning.constants import RIGHT_CHAIN
from src.pose_cleaning.diagnostics import build_cleaning_diagnostics
from src.pose_cleaning.filters import (
    filter_by_confidence,
)
from src.pose_cleaning.interpolation import interpolate_short_gaps
from src.pose_cleaning.io import default_clean_output_path, load_pose_npz_raw, save_pose_npz_clean
from src.pose_cleaning.smoothing import smooth_keypoints_savgol
from src.pose_cleaning.validation import (
    compute_mask_valid,
    compute_mask_valid_frames_right_chain,
    compute_mask_valid_right_chain,
    validate_pose_arrays,
)


def _reconstruct_scores(
    kps_score: np.ndarray,
    mask_valid: np.ndarray,
    mask_valid_right_chain: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(kps_score, dtype=np.float64).copy()
    _ = mask_valid
    for pos, k in enumerate(RIGHT_CHAIN):
        scores[~mask_valid_right_chain[:, pos], k] = 0.0
    return scores


def clean_pose_data(
    pose_data: dict,
    *,
    config: PoseCleaningConfig | None = None,
) -> dict:
    cfg = config if config is not None else PoseCleaningConfig()
    if "kps_xy" not in pose_data or "kps_score" not in pose_data:
        raise KeyError("pose_data must contain 'kps_xy' and 'kps_score'.")
    kps_xy = np.asarray(pose_data["kps_xy"], dtype=np.float64)
    kps_score = np.asarray(pose_data["kps_score"], dtype=np.float64)
    validate_pose_arrays(kps_xy, kps_score)

    xy_conf, _mask_conf = filter_by_confidence(
        kps_xy,
        kps_score,
        score_thr=cfg.score_thr,
        keypoint_indices=RIGHT_CHAIN,
    )
    xy_interp_1 = interpolate_short_gaps(
        xy_conf,
        max_gap=cfg.max_gap,
    )
    xy_interp_2 = interpolate_short_gaps(
        xy_interp_1,
        max_gap=cfg.max_gap,
    )
    xy_smooth = xy_interp_2
    for _ in range(max(0, int(cfg.smoothing_passes))):
        xy_smooth = smooth_keypoints_savgol(
            xy_smooth,
            window_length=cfg.savgol_window,
            polyorder=cfg.savgol_polyorder,
            min_valid_ratio=cfg.min_valid_ratio_smoothing,
        )
    kps_xy_clean = xy_smooth
    non_chain = [i for i in range(kps_xy.shape[1]) if i not in RIGHT_CHAIN]
    kps_xy_clean[:, non_chain] = kps_xy[:, non_chain]

    mask_valid = compute_mask_valid(kps_xy_clean)
    mask_valid_right_chain = compute_mask_valid_right_chain(kps_xy_clean)
    mask_valid_frames = compute_mask_valid_frames_right_chain(mask_valid_right_chain)
    kps_score_clean = _reconstruct_scores(kps_score, mask_valid, mask_valid_right_chain)
    diagnostics = build_cleaning_diagnostics(
        kps_xy,
        kps_xy_clean,
        mask_valid_right_chain,
        mask_valid_frames,
    )
    result = dict(pose_data)
    result.update(
        {
            "kps_xy_clean": kps_xy_clean,
            "kps_score_clean": kps_score_clean,
            "mask_valid": mask_valid,
            "mask_valid_frames": mask_valid_frames,
            "mask_valid_right_chain": mask_valid_right_chain,
            "cleaning_diagnostics": diagnostics,
        }
    )
    return result


def clean_pose_npz(
    input_path: str,
    output_path: str | None = None,
    config: PoseCleaningConfig | None = None,
) -> None:
    cfg = config if config is not None else PoseCleaningConfig()
    raw = load_pose_npz_raw(input_path)
    cleaned = clean_pose_data(raw, config=cfg)

    out = Path(output_path) if output_path is not None else default_clean_output_path(input_path)
    save_pose_npz_clean(
        out,
        raw=raw,
        kps_xy_clean=np.asarray(cleaned["kps_xy_clean"], dtype=np.float64),
        kps_score_clean=np.asarray(cleaned["kps_score_clean"], dtype=np.float64),
        mask_valid=np.asarray(cleaned["mask_valid"]),
        mask_valid_frames=np.asarray(cleaned["mask_valid_frames"]),
        mask_valid_right_chain=np.asarray(cleaned["mask_valid_right_chain"]),
        cleaning_diagnostics=cleaned["cleaning_diagnostics"],
    )
