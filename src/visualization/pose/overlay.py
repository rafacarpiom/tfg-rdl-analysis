
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

RIGHT_CHAIN: tuple[int, ...] = (6, 8, 10, 12, 14, 16)
RIGHT_CHAIN_EDGES: tuple[tuple[int, int], ...] = (
    (6, 8),
    (8, 10),
    (6, 12),
    (12, 14),
    (14, 16),
)

_COLOR_SKELETON = (180, 180, 180)
_COLOR_KEYPOINT = (255, 120, 40)
_COLOR_RIGHT_EDGE = (0, 220, 255)
_COLOR_RIGHT_KEYPOINT = (30, 60, 255)
_COLOR_TEXT = (255, 255, 255)
_COLOR_TEXT_BG = (15, 15, 15)


def load_pose_arrays_for_visualization(
    npz_path: str | Path,
    *,
    pose_source: str = "clean",
) -> dict:
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")
    source_req = str(pose_source).strip().lower()
    if source_req not in {"clean", "raw", "auto"}:
        raise ValueError(f"Invalid pose_source={pose_source!r}. Use clean/raw/auto.")

    with np.load(str(path), allow_pickle=True) as npz:
        has_clean = "kps_xy_clean" in npz.files and "kps_score_clean" in npz.files
        if source_req == "clean":
            if not has_clean:
                raise ValueError(f"Requested clean pose but missing clean arrays in {path}.")
            src = "clean"
        elif source_req == "raw":
            src = "raw"
        else:
            src = "clean" if has_clean else "raw"

        if src == "clean":
            kps_xy = np.asarray(npz["kps_xy_clean"], dtype=np.float64)
            kps_score = np.asarray(npz["kps_score_clean"], dtype=np.float64)
        else:
            if "kps_xy" not in npz.files or "kps_score" not in npz.files:
                raise ValueError(f"Missing raw pose arrays in {path}.")
            kps_xy = np.asarray(npz["kps_xy"], dtype=np.float64)
            kps_score = np.asarray(npz["kps_score"], dtype=np.float64)

        frame_idx = np.asarray(npz["frame_idx"], dtype=np.int64) if "frame_idx" in npz.files else None
        fps = float(np.asarray(npz["fps"]).item()) if "fps" in npz.files else None
        mask_valid_frames = (
            np.asarray(npz["mask_valid_frames"], dtype=bool) if "mask_valid_frames" in npz.files else None
        )
        mask_valid_right_chain = (
            np.asarray(npz["mask_valid_right_chain"], dtype=bool)
            if "mask_valid_right_chain" in npz.files
            else None
        )
        meta: dict[str, Any] = {}
        if "meta" in npz.files:
            try:
                v = npz["meta"].item()
                if isinstance(v, dict):
                    meta = v
            except Exception:
                meta = {}

    if kps_xy.ndim != 3 or kps_xy.shape[1:] != (17, 2):
        raise ValueError(f"Invalid kps_xy shape: {kps_xy.shape}")
    if kps_score.ndim != 2 or kps_score.shape != (kps_xy.shape[0], 17):
        raise ValueError(f"Invalid kps_score shape: {kps_score.shape}")

    return {
        "npz_path": str(path),
        "pose_source": src,
        "kps_xy": kps_xy,
        "kps_score": kps_score,
        "frame_idx": frame_idx,
        "fps": fps,
        "mask_valid_frames": mask_valid_frames,
        "mask_valid_right_chain": mask_valid_right_chain,
        "meta": meta,
    }


def read_video_frame(
    video_path: str | Path,
    frame_idx: int,
) -> np.ndarray:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        idx = int(frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {idx} from video: {path}")
        return frame
    finally:
        cap.release()


def _is_visible(kps: np.ndarray, scores: np.ndarray | None, idx: int, thr: float) -> bool:
    if idx < 0 or idx >= kps.shape[0]:
        return False
    if not np.isfinite(kps[idx]).all():
        return False
    if scores is not None and (idx >= scores.shape[0] or not np.isfinite(scores[idx]) or scores[idx] < thr):
        return False
    return True


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_score: np.ndarray | None = None,
    *,
    score_threshold: float = 0.30,
    draw_skeleton: bool = True,
    draw_keypoints: bool = True,
    draw_labels: bool = True,
    draw_scores: bool = False,
    highlight_right_chain: bool = True,
    title: str | None = None,
) -> np.ndarray:
    out = np.asarray(frame_bgr).copy()
    kps = np.asarray(keypoints_xy, dtype=np.float64)
    if kps.ndim != 2 or kps.shape != (17, 2):
        raise ValueError(f"Expected keypoints shape (17,2), got {kps.shape}.")
    scores = None if keypoints_score is None else np.asarray(keypoints_score, dtype=np.float64)

    if draw_skeleton:
        for i, j in COCO_SKELETON:
            if _is_visible(kps, scores, i, score_threshold) and _is_visible(kps, scores, j, score_threshold):
                p1 = tuple(np.round(kps[i]).astype(int))
                p2 = tuple(np.round(kps[j]).astype(int))
                cv2.line(out, p1, p2, _COLOR_SKELETON, 2, cv2.LINE_AA)

    if highlight_right_chain:
        for i, j in RIGHT_CHAIN_EDGES:
            if _is_visible(kps, scores, i, score_threshold) and _is_visible(kps, scores, j, score_threshold):
                p1 = tuple(np.round(kps[i]).astype(int))
                p2 = tuple(np.round(kps[j]).astype(int))
                cv2.line(out, p1, p2, _COLOR_RIGHT_EDGE, 3, cv2.LINE_AA)

    if draw_keypoints:
        for idx in range(kps.shape[0]):
            if not _is_visible(kps, scores, idx, score_threshold):
                continue
            pt = tuple(np.round(kps[idx]).astype(int))
            color = _COLOR_RIGHT_KEYPOINT if idx in RIGHT_CHAIN and highlight_right_chain else _COLOR_KEYPOINT
            cv2.circle(out, pt, 4, color, -1, cv2.LINE_AA)
            if draw_labels or draw_scores:
                chunks: list[str] = []
                if draw_labels:
                    chunks.append(str(idx))
                if draw_scores and scores is not None and idx < scores.shape[0] and np.isfinite(scores[idx]):
                    chunks.append(f"{float(scores[idx]):.2f}")
                if chunks:
                    text = ":".join(chunks)
                    tx, ty = int(pt[0] + 5), int(pt[1] - 5)
                    cv2.rectangle(out, (tx - 1, ty - 11), (tx + 7 * len(text), ty + 3), _COLOR_TEXT_BG, -1)
                    cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _COLOR_TEXT, 1, cv2.LINE_AA)

    if title:
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), _COLOR_TEXT_BG, -1)
        cv2.putText(out, str(title), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_TEXT, 1, cv2.LINE_AA)
    return out


def render_pose_frame_overlay(
    *,
    video_path: str | Path,
    npz_path: str | Path,
    frame_idx: int,
    output_path: str | Path,
    pose_source: str = "clean",
    score_threshold: float = 0.30,
    draw_labels: bool = True,
    draw_scores: bool = False,
    highlight_right_chain: bool = True,
) -> dict:
    pose = load_pose_arrays_for_visualization(npz_path, pose_source=pose_source)
    idx = int(frame_idx)
    if idx < 0 or idx >= pose["kps_xy"].shape[0]:
        raise ValueError(f"frame_idx={idx} out of range [0, {pose['kps_xy'].shape[0]-1}].")

    frame = read_video_frame(video_path, idx)
    kps = np.asarray(pose["kps_xy"][idx], dtype=np.float64)
    scores = np.asarray(pose["kps_score"][idx], dtype=np.float64)
    title = f"frame={idx} source={pose['pose_source']}"
    rendered = draw_pose_overlay(
        frame,
        kps,
        scores,
        score_threshold=float(score_threshold),
        draw_skeleton=True,
        draw_keypoints=True,
        draw_labels=bool(draw_labels),
        draw_scores=bool(draw_scores),
        highlight_right_chain=bool(highlight_right_chain),
        title=title,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), rendered)
    if not ok:
        raise ValueError(f"Failed to write overlay PNG: {out_path}")

    visible_mask = np.array(
        [_is_visible(kps, scores, i, float(score_threshold)) for i in range(kps.shape[0])],
        dtype=bool,
    )
    right_chain_finite = bool(np.isfinite(kps[list(RIGHT_CHAIN)]).all())
    mask_valid_frame = None
    if pose["mask_valid_frames"] is not None and idx < pose["mask_valid_frames"].shape[0]:
        mask_valid_frame = bool(pose["mask_valid_frames"][idx])
    mask_valid_right_chain = None
    if pose["mask_valid_right_chain"] is not None and idx < pose["mask_valid_right_chain"].shape[0]:
        mask_valid_right_chain = [bool(x) for x in pose["mask_valid_right_chain"][idx].tolist()]

    return {
        "video_path": str(video_path),
        "npz_path": str(npz_path),
        "frame_idx": idx,
        "output_path": str(out_path),
        "pose_source": str(pose["pose_source"]),
        "score_threshold": float(score_threshold),
        "num_finite_keypoints": int(np.isfinite(kps).all(axis=1).sum()),
        "num_visible_keypoints": int(visible_mask.sum()),
        "right_chain_all_finite": right_chain_finite,
        "mask_valid_frame": mask_valid_frame,
        "mask_valid_right_chain": mask_valid_right_chain,
    }


def render_pose_frame_range_overlay(
    *,
    video_path: str | Path,
    npz_path: str | Path,
    start_frame: int,
    end_frame: int,
    output_dir: str | Path,
    step: int = 1,
    pose_source: str = "clean",
    score_threshold: float = 0.30,
    draw_labels: bool = True,
    draw_scores: bool = False,
    highlight_right_chain: bool = True,
) -> list[dict]:
    start = int(start_frame)
    end = int(end_frame)
    jump = int(step)
    if start > end:
        raise ValueError(f"start_frame must be <= end_frame. Got {start} > {end}.")
    if jump < 1:
        raise ValueError(f"step must be >= 1. Got {jump}.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    source_tag = str(pose_source).lower()
    for idx in range(start, end + 1, jump):
        file_name = f"frame_{idx:06d}_{source_tag}_debug.png"
        result = render_pose_frame_overlay(
            video_path=video_path,
            npz_path=npz_path,
            frame_idx=idx,
            output_path=out_dir / file_name,
            pose_source=pose_source,
            score_threshold=score_threshold,
            draw_labels=draw_labels,
            draw_scores=draw_scores,
            highlight_right_chain=highlight_right_chain,
        )
        results.append(result)
    return results
