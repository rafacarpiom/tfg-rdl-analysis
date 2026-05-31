
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_RAW_KEYS: tuple[str, ...] = (
    "kps_xy",
    "kps_score",
    "bbox_xyxy",
    "frame_idx",
    "fps",
    "meta",
)


def load_pose_npz_raw(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ no encontrado: {path}")

    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in REQUIRED_RAW_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"NPZ incompleto. Faltan claves: {missing}")
        return {key: data[key] for key in REQUIRED_RAW_KEYS}


def default_clean_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    stem = path.stem
    if stem.endswith("_clean"):
        return path
    return path.with_name(f"{stem}_clean.npz")


def save_pose_npz_clean(
    output_path: str | Path,
    *,
    raw: dict[str, Any],
    kps_xy_clean: np.ndarray,
    kps_score_clean: np.ndarray,
    mask_valid: np.ndarray,
    mask_valid_frames: np.ndarray,
    mask_valid_right_chain: np.ndarray,
    cleaning_diagnostics: dict[str, Any] | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kps_xy": raw["kps_xy"],
        "kps_score": raw["kps_score"],
        "bbox_xyxy": raw["bbox_xyxy"],
        "frame_idx": raw["frame_idx"],
        "fps": raw["fps"],
        "meta": raw["meta"],
        "kps_xy_clean": kps_xy_clean,
        "kps_score_clean": kps_score_clean,
        "mask_valid": mask_valid.astype(bool),
        "mask_valid_frames": mask_valid_frames.astype(bool),
        "mask_valid_right_chain": mask_valid_right_chain.astype(bool),
        "cleaning_diagnostics": np.array(
            [cleaning_diagnostics if cleaning_diagnostics is not None else {}],
            dtype=object,
        ),
    }
    np.savez(str(path), **payload)
