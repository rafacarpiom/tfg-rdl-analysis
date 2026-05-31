
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

KP_NOSE = 0
TORSO_KPS = (5, 6, 11, 12)  # left_shoulder, right_shoulder, left_hip, right_hip


def _is_finite_xy(point: np.ndarray) -> bool:
    return bool(point.shape == (2,) and np.isfinite(point).all())


def _frame_scale_from_torso(torso_points: list[np.ndarray]) -> float | None:
    if len(torso_points) < 2:
        return None
    xs = np.array([p[0] for p in torso_points], dtype=np.float64)
    span_x = float(np.nanmax(xs) - np.nanmin(xs))
    if np.isfinite(span_x) and span_x > 1e-6:
        return span_x
    return None


def estimate_subject_facing_from_pose(
    pose_data: dict[str, Any],
    *,
    score_threshold: float = 0.30,
    min_valid_frames: int = 10,
    margin: float = 0.03,
) -> dict[str, Any]:
    if "kps_xy" not in pose_data:
        raise KeyError("pose_data missing required key: kps_xy")
    if "kps_score" not in pose_data:
        raise KeyError("pose_data missing required key: kps_score")
    kps_xy = np.asarray(pose_data["kps_xy"], dtype=np.float64)
    kps_score = np.asarray(pose_data["kps_score"], dtype=np.float64)
    bbox_raw = pose_data.get("bbox_xyxy")
    bbox = np.asarray(bbox_raw, dtype=np.float64) if bbox_raw is not None else None

    scores_norm: list[float] = []
    total_frames = int(kps_xy.shape[0])

    for t in range(total_frames):
        if kps_xy.shape[1] <= max(TORSO_KPS) or kps_score.shape[1] <= max(TORSO_KPS):
            break

        nose_score = float(kps_score[t, KP_NOSE]) if np.isfinite(kps_score[t, KP_NOSE]) else -np.inf
        if nose_score < score_threshold:
            continue
        nose_xy = np.asarray(kps_xy[t, KP_NOSE], dtype=np.float64)
        if not _is_finite_xy(nose_xy):
            continue

        torso_points: list[np.ndarray] = []
        for kp in TORSO_KPS:
            score = float(kps_score[t, kp]) if np.isfinite(kps_score[t, kp]) else -np.inf
            if score < score_threshold:
                continue
            xy = np.asarray(kps_xy[t, kp], dtype=np.float64)
            if _is_finite_xy(xy):
                torso_points.append(xy)
        if len(torso_points) < 2:
            continue

        torso_center_x = float(np.mean([p[0] for p in torso_points]))
        raw_score = float(nose_xy[0] - torso_center_x)

        scale: float | None = None
        if bbox is not None and t < bbox.shape[0] and bbox.shape[1] >= 4:
            x1, _y1, x2, _y2 = bbox[t]
            width = float(x2 - x1) if np.isfinite([x1, x2]).all() else float("nan")
            if np.isfinite(width) and width > 1e-6:
                scale = width
        if scale is None:
            scale = _frame_scale_from_torso(torso_points)
        if scale is None:
            continue

        score_norm = float(raw_score / scale)
        if np.isfinite(score_norm):
            scores_norm.append(score_norm)

    valid_frames = int(len(scores_norm))
    median_score: float | None = float(np.median(scores_norm)) if scores_norm else None

    if scores_norm:
        arr = np.asarray(scores_norm, dtype=np.float64)
        positive_ratio = float(np.mean(arr > margin))
        negative_ratio = float(np.mean(arr < -margin))
        confidence = float(max(positive_ratio, negative_ratio))
    else:
        confidence = 0.0

    warning: str | None = None
    if valid_frames < min_valid_frames:
        facing = "unknown"
        warning = "No hay suficientes frames fiables para estimar la orientación."
    elif median_score is None:
        facing = "unknown"
        warning = "No hay suficientes frames fiables para estimar la orientación."
    elif median_score > margin:
        facing = "right"
    elif median_score < -margin:
        facing = "left"
    else:
        facing = "unknown"
        warning = "La orientación no supera el margen mínimo de decisión."

    return {
        "method": "nose_vs_torso_center",
        "facing": facing,
        "confidence": confidence,
        "median_score": median_score,
        "valid_frames": int(valid_frames),
        "score_threshold": float(score_threshold),
        "min_valid_frames": int(min_valid_frames),
        "margin": float(margin),
        "warning": warning,
    }


def estimate_subject_facing_from_npz(
    npz_path: Path,
    *,
    score_threshold: float = 0.30,
    min_valid_frames: int = 10,
    margin: float = 0.03,
) -> dict[str, Any]:
    with np.load(str(npz_path), allow_pickle=True) as npz:
        if "kps_xy" not in npz.files:
            raise KeyError(f"NPZ incompleto: falta clave 'kps_xy' en {npz_path}")
        if "kps_score" not in npz.files:
            raise KeyError(f"NPZ incompleto: falta clave 'kps_score' en {npz_path}")
        kps_xy = np.asarray(npz["kps_xy"], dtype=np.float64)
        kps_score = np.asarray(npz["kps_score"], dtype=np.float64)
        bbox = np.asarray(npz["bbox_xyxy"], dtype=np.float64) if "bbox_xyxy" in npz.files else None

    return estimate_subject_facing_from_pose(
        {
            "kps_xy": kps_xy,
            "kps_score": kps_score,
            "bbox_xyxy": bbox,
        },
        score_threshold=score_threshold,
        min_valid_frames=min_valid_frames,
        margin=margin,
    )
