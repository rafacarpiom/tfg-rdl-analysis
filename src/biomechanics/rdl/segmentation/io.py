
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_KEYS = {"kps_xy", "kps_score", "bbox_xyxy", "frame_idx", "fps", "meta"}


def _sanitize_video_id(stem: str) -> str:
    for suffix in ("_rtmpose_clean", "_rtmpose_probe", "_rtmpose"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _extract_meta(meta_raw: np.ndarray) -> dict[str, Any]:
    if np.asarray(meta_raw).size == 0:
        return {}
    try:
        value = meta_raw.item()
    except Exception:
        value = meta_raw
    return value if isinstance(value, dict) else {"raw_meta": str(value)}


def load_pose_npz(npz_path: str | Path) -> dict[str, Any]:
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")
    with np.load(str(path), allow_pickle=True) as data:
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            raise ValueError(f"Missing NPZ keys: {sorted(missing)}")

        has_clean_pose = "kps_xy_clean" in data.files and "kps_score_clean" in data.files
        if has_clean_pose:
            kps_xy = np.asarray(data["kps_xy_clean"])
            kps_score = np.asarray(data["kps_score_clean"])
            pose_source = "clean"
        else:
            kps_xy = np.asarray(data["kps_xy"])
            kps_score = np.asarray(data["kps_score"])
            pose_source = "raw"

        bbox_xyxy = np.asarray(data["bbox_xyxy"])
        frame_idx = np.asarray(data["frame_idx"])
        fps = float(np.asarray(data["fps"]).item())
        meta = _extract_meta(data["meta"])
        mask_valid = np.asarray(data["mask_valid"]) if "mask_valid" in data.files else None
        mask_valid_frames = np.asarray(data["mask_valid_frames"]) if "mask_valid_frames" in data.files else None
        mask_valid_right_chain = (
            np.asarray(data["mask_valid_right_chain"]) if "mask_valid_right_chain" in data.files else None
        )
        cleaning_diagnostics = None
        if "cleaning_diagnostics" in data.files:
            try:
                cleaning_diagnostics = data["cleaning_diagnostics"].item()
            except Exception:
                cleaning_diagnostics = data["cleaning_diagnostics"]

    T = kps_xy.shape[0]
    if kps_xy.ndim != 3 or kps_xy.shape[1:] != (17, 2):
        raise ValueError(f"Invalid kps_xy shape: {kps_xy.shape}")
    if kps_score.ndim != 2 or kps_score.shape != (T, 17):
        raise ValueError(f"Invalid kps_score shape: {kps_score.shape}")
    if bbox_xyxy.ndim != 2 or bbox_xyxy.shape != (T, 4):
        raise ValueError(f"Invalid bbox_xyxy shape: {bbox_xyxy.shape}")
    if frame_idx.ndim != 1 or frame_idx.shape[0] != T:
        raise ValueError(f"Invalid frame_idx shape: {frame_idx.shape}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps value: {fps}")

    return {
        "npz_path": str(path),
        "video_id": _sanitize_video_id(path.stem),
        "kps_xy": kps_xy,
        "kps_score": kps_score,
        "bbox_xyxy": bbox_xyxy,
        "frame_idx": frame_idx,
        "fps": fps,
        "meta": meta,
        "pose_source": pose_source,
        "has_clean_pose": bool(has_clean_pose),
        "mask_valid": mask_valid,
        "mask_valid_frames": mask_valid_frames,
        "mask_valid_right_chain": mask_valid_right_chain,
        "cleaning_diagnostics": cleaning_diagnostics,
    }


def validate_pose_data_dict(pose_data: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_KEYS - set(pose_data.keys())
    if missing:
        raise ValueError(f"Missing pose_data keys: {sorted(missing)}")

    has_clean_pose = "kps_xy_clean" in pose_data and "kps_score_clean" in pose_data
    if has_clean_pose:
        kps_xy = np.asarray(pose_data["kps_xy_clean"])
        kps_score = np.asarray(pose_data["kps_score_clean"])
        pose_source = "clean"
    else:
        kps_xy = np.asarray(pose_data["kps_xy"])
        kps_score = np.asarray(pose_data["kps_score"])
        pose_source = "raw"

    bbox_xyxy = np.asarray(pose_data["bbox_xyxy"])
    frame_idx = np.asarray(pose_data["frame_idx"])
    fps = float(np.asarray(pose_data["fps"]).item())
    meta_raw = pose_data.get("meta", {})
    meta = meta_raw if isinstance(meta_raw, dict) else {"raw_meta": str(meta_raw)}

    T = kps_xy.shape[0]
    if kps_xy.ndim != 3 or kps_xy.shape[1:] != (17, 2):
        raise ValueError(f"Invalid kps_xy shape: {kps_xy.shape}")
    if kps_score.ndim != 2 or kps_score.shape != (T, 17):
        raise ValueError(f"Invalid kps_score shape: {kps_score.shape}")
    if bbox_xyxy.ndim != 2 or bbox_xyxy.shape != (T, 4):
        raise ValueError(f"Invalid bbox_xyxy shape: {bbox_xyxy.shape}")
    if frame_idx.ndim != 1 or frame_idx.shape[0] != T:
        raise ValueError(f"Invalid frame_idx shape: {frame_idx.shape}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps value: {fps}")

    video_id = str(pose_data.get("video_id") or _sanitize_video_id(str(meta.get("video_id", "unknown_video"))))

    return {
        "npz_path": str(pose_data.get("npz_path", "")),
        "video_id": video_id,
        "kps_xy": kps_xy,
        "kps_score": kps_score,
        "bbox_xyxy": bbox_xyxy,
        "frame_idx": frame_idx,
        "fps": fps,
        "meta": meta,
        "pose_source": pose_source,
        "has_clean_pose": bool(has_clean_pose),
        "mask_valid": np.asarray(pose_data["mask_valid"]) if "mask_valid" in pose_data else None,
        "mask_valid_frames": np.asarray(pose_data["mask_valid_frames"]) if "mask_valid_frames" in pose_data else None,
        "mask_valid_right_chain": (
            np.asarray(pose_data["mask_valid_right_chain"]) if "mask_valid_right_chain" in pose_data else None
        ),
        "cleaning_diagnostics": pose_data.get("cleaning_diagnostics"),
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_segmentation_json(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def save_segmentation_debug_npz(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signals = result["signals"]
    events = result["event_candidates"]
    reps = result.get("reps", [])
    np.savez(
        str(path),
        signal_raw=np.asarray(signals["signal_raw"], dtype=np.float64),
        signal_smooth=np.asarray(signals["signal_smooth"], dtype=np.float64),
        valid_mask_raw=np.asarray(signals["valid_mask_raw"], dtype=np.int8),
        interp_mask=np.asarray(signals["interp_mask"], dtype=np.int8),
        valid_mask=np.asarray(signals["valid_mask"], dtype=np.int8),
        bottom_candidates=np.asarray(events["bottom_indices_candidates"], dtype=np.int64),
        top_candidates=np.asarray(events["top_indices_candidates"], dtype=np.int64),
        bottom_valid=np.asarray(events["bottom_indices_valid"], dtype=np.int64),
        top_valid=np.asarray(events["top_indices_valid"], dtype=np.int64),
        rep_top_start=np.asarray([r["top_start"] for r in reps], dtype=np.int64),
        rep_bottom=np.asarray([r["bottom"] for r in reps], dtype=np.int64),
        rep_top_end=np.asarray([r["top_end"] for r in reps], dtype=np.int64),
    )
    return path

