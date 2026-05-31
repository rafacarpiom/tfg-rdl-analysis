
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.biomechanics.rdl.analysis_context.constants import RDL_REFERENCE_DEFAULT_NAME


def _valid_rep_indices(segmentation_result: dict[str, Any]) -> list[int]:
    reps = segmentation_result.get("reps", [])
    if not isinstance(reps, list):
        return []
    valid: list[int] = []
    for i, rep in enumerate(reps):
        if isinstance(rep, dict) and rep.get("anchor_valid", True) is True:
            valid.append(i)
    return valid


def load_rdl_reference(
    reference_dir: str | Path,
    *,
    ideal_valid_rep_index: int = 1,
) -> dict[str, Any]:
    reference_path = Path(reference_dir).expanduser().resolve()
    seg_path = reference_path / "ideal_segmentation_result.json"
    norm_path = reference_path / "ideal_pose_sequence_normalized.npz"
    meta_path = reference_path / "ideal_pose_sequence_normalized_meta.json"

    for path in (seg_path, norm_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required reference file: {path}")

    with seg_path.open("r", encoding="utf-8") as f:
        segmentation_result = json.load(f)
    if not isinstance(segmentation_result, dict):
        raise ValueError("Reference segmentation_result must be a dict.")
    reps = segmentation_result.get("reps")
    if not isinstance(reps, list) or len(reps) == 0:
        raise ValueError("Reference segmentation_result must contain non-empty 'reps'.")

    valid_indices = _valid_rep_indices(segmentation_result)
    if len(valid_indices) <= int(ideal_valid_rep_index):
        raise ValueError(
            "Reference PM-Ideal expected at least "
            f"{int(ideal_valid_rep_index) + 1} valid repetitions, got {len(valid_indices)}."
        )
    selected_rep_raw_index = int(valid_indices[int(ideal_valid_rep_index)])

    with np.load(norm_path, allow_pickle=True) as data:
        required = {"kps_xy_normalized", "mask_valid_normalized"}
        missing = [k for k in required if k not in data.files]
        if missing:
            raise ValueError(f"Missing keys in reference normalization NPZ: {missing}")

        kps_xy_normalized = np.asarray(data["kps_xy_normalized"], dtype=np.float64)
        mask_valid_normalized = np.asarray(data["mask_valid_normalized"], dtype=bool)
        frame_idx = np.asarray(data["frame_idx"], dtype=np.int64) if "frame_idx" in data.files else None
        fps = float(np.asarray(data["fps"]).squeeze()) if "fps" in data.files else None
        kps_xy_clean = np.asarray(data["kps_xy_clean"], dtype=np.float64) if "kps_xy_clean" in data.files else None
        kps_score_clean = (
            np.asarray(data["kps_score_clean"], dtype=np.float64) if "kps_score_clean" in data.files else None
        )
        origins = (
            np.asarray(data["normalization_origins"], dtype=np.float64)
            if "normalization_origins" in data.files
            else None
        )
        scales = (
            np.asarray(data["normalization_scales"], dtype=np.float64)
            if "normalization_scales" in data.files
            else None
        )
        raw_scales = np.asarray(data["raw_scales"], dtype=np.float64) if "raw_scales" in data.files else None

    if kps_xy_normalized.ndim != 3 or kps_xy_normalized.shape[1:] != (17, 2):
        raise ValueError(f"Invalid reference normalized shape: {kps_xy_normalized.shape}")
    t = int(kps_xy_normalized.shape[0])
    if mask_valid_normalized.shape != (t,):
        raise ValueError(
            f"Invalid reference mask_valid_normalized shape: {mask_valid_normalized.shape}, expected ({t},)"
        )

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    reference_name = (
        meta.get("reference_name", reference_path.name)
        if isinstance(meta, dict)
        else reference_path.name or RDL_REFERENCE_DEFAULT_NAME
    )
    return {
        "reference_name": str(reference_name),
        "reference_dir": str(reference_path),
        "segmentation_path": str(seg_path),
        "normalization_path": str(norm_path),
        "normalization_meta_path": str(meta_path),
        "segmentation_result": segmentation_result,
        "normalization": {
            "kps_xy_normalized": kps_xy_normalized,
            "mask_valid_normalized": mask_valid_normalized,
            "frame_idx": frame_idx,
            "fps": fps,
            "kps_xy_clean": kps_xy_clean,
            "kps_score_clean": kps_score_clean,
            "origins": origins,
            "scales": scales,
            "raw_scales": raw_scales,
            "meta": meta if isinstance(meta, dict) else {},
        },
        "selected_rep_raw_index": int(selected_rep_raw_index),
        "selected_valid_rep_index": int(ideal_valid_rep_index),
        "num_valid_reps": int(len(valid_indices)),
        "warnings": [],
    }

