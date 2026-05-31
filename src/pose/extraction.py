
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from src.pose.detection import YoloPersonDetector
from src.pose.selection import select_primary_person
from src.pose.model import RTMPoseModel
from src.utils.paths import RTMPOSE_CHECKPOINT, RTMPOSE_CONFIG, YOLO_WEIGHTS

K = 17
LOGGER = logging.getLogger(__name__)


def _zero_frame_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((K, 2), dtype=np.float64),
        np.zeros(K, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
    )


def _validate_input_paths(video_path: Path, config_path: Path, checkpoint_path: Path, yolo_weights: Path) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Vídeo no encontrado: {video_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config RTMPose no encontrado: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint RTMPose no encontrado: {checkpoint_path}")
    if not yolo_weights.is_file():
        raise FileNotFoundError(f"Pesos de YOLO no encontrados: {yolo_weights}")


def extract_video_pose(
    video_path: str,
    verbose: bool = False,
    *,
    config_path: str = str(RTMPOSE_CONFIG),
    checkpoint_path: str = str(RTMPOSE_CHECKPOINT),
    yolo_weights: str = str(YOLO_WEIGHTS),
    yolo_conf_threshold: float = 0.25,
) -> dict:
    video_path_obj = Path(video_path)
    config_path_obj = Path(config_path)
    checkpoint_path_obj = Path(checkpoint_path)
    yolo_weights_obj = Path(yolo_weights)
    _validate_input_paths(video_path_obj, config_path_obj, checkpoint_path_obj, yolo_weights_obj)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YoloPersonDetector(
        weights_path=str(yolo_weights_obj),
        device=device,
        conf_threshold=yolo_conf_threshold,
    )
    pose_model = RTMPoseModel(
        config_path=str(config_path_obj),
        checkpoint_path=str(checkpoint_path_obj),
        device=device,
    )

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path_obj}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_kps: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_bbox: list[np.ndarray] = []
    frame_indices: list[int] = []

    # Métricas de calidad para trazabilidad.
    frames_read = 0
    person_detected_frames = 0
    missing_person_frames = 0
    yolo_failed_frames = 0
    pose_success_frames = 0
    pose_failed_frames = 0
    invalid_keypoint_shape_frames = 0
    invalid_score_shape_frames = 0

    previous_box: list[float] | None = None

    t = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1

            try:
                person_boxes = detector.detect(frame)
            except Exception as exc:
                LOGGER.warning("Fallo en YOLO en frame %s: %s", t, exc)
                yolo_failed_frames += 1
                person_boxes = []
            best_box = select_primary_person(
                person_boxes,
                frame.shape,
                previous_box=previous_box,
            )

            if best_box is None:
                missing_person_frames += 1
                kps_zeros, score_zeros, bbox_zeros = _zero_frame_data()
                all_kps.append(kps_zeros)
                all_scores.append(score_zeros)
                all_bbox.append(bbox_zeros)
                frame_indices.append(t)
                t += 1
                if verbose and t % 100 == 0:
                    LOGGER.info("Frame %s/%s", t, total_frames)
                continue

            person_detected_frames += 1
            previous_box = best_box
            bbox_arr = np.array(best_box, dtype=np.float64)
            all_bbox.append(bbox_arr)

            try:
                keypoints, scores = pose_model.predict(frame, best_box)
            except Exception as exc:
                LOGGER.warning("Fallo en RTMPose en frame %s: %s", t, exc)
                pose_failed_frames += 1
                keypoints, scores = None, None

            if keypoints is None or scores is None:
                kps_zeros, score_zeros, _ = _zero_frame_data()
                all_kps.append(kps_zeros)
                all_scores.append(score_zeros)
            else:
                keypoints = np.asarray(keypoints, dtype=np.float64)
                scores = np.asarray(scores, dtype=np.float64)
                kps_shape_ok = keypoints.shape == (K, 2)
                score_shape_ok = scores.shape == (K,)

                if not kps_shape_ok:
                    LOGGER.warning("Shape inválido de keypoints en frame %s: %s", t, keypoints.shape)
                    invalid_keypoint_shape_frames += 1
                if not score_shape_ok:
                    LOGGER.warning("Shape inválido de scores en frame %s: %s", t, scores.shape)
                    invalid_score_shape_frames += 1

                if kps_shape_ok and score_shape_ok:
                    pose_success_frames += 1
                    all_kps.append(keypoints)
                    all_scores.append(scores)
                else:
                    pose_failed_frames += 1
                    kps_zeros, score_zeros, _ = _zero_frame_data()
                    all_kps.append(kps_zeros)
                    all_scores.append(score_zeros)

            frame_indices.append(t)
            t += 1
            if verbose and t % 100 == 0:
                LOGGER.info("Frame %s/%s", t, total_frames)
    finally:
        cap.release()
    if not all_kps:
        raise RuntimeError(f"No se pudieron procesar frames para: {video_path_obj}")

    kps_xy = np.stack(all_kps, axis=0)
    kps_score = np.stack(all_scores, axis=0)
    bbox_xyxy = np.stack(all_bbox, axis=0)
    frame_idx = np.array(frame_indices, dtype=np.int64)

    person_detected_ratio = (person_detected_frames / frames_read) if frames_read > 0 else 0.0
    pose_success_ratio = (pose_success_frames / frames_read) if frames_read > 0 else 0.0

    meta = {
        "model_type": "rtmpose",
        "input_resolution": "256x192",
        "video_path": str(video_path_obj.resolve()),
        "device": device,
        "num_frames": len(all_kps),
        "num_keypoints": K,
        "fps": fps,
        "config_path": str(config_path_obj.resolve()),
        "checkpoint_path": str(checkpoint_path_obj.resolve()),
        "yolo_weights": str(yolo_weights_obj.resolve()),
        "quality": {
            "frames_read": int(frames_read),
            "person_detected_frames": int(person_detected_frames),
            "missing_person_frames": int(missing_person_frames),
            "yolo_failed_frames": int(yolo_failed_frames),
            "pose_success_frames": int(pose_success_frames),
            "pose_failed_frames": int(pose_failed_frames),
            "invalid_keypoint_shape_frames": int(invalid_keypoint_shape_frames),
            "invalid_score_shape_frames": int(invalid_score_shape_frames),
            "person_detected_ratio": float(person_detected_ratio),
            "pose_success_ratio": float(pose_success_ratio),
            "yolo_conf_threshold": float(yolo_conf_threshold),
        },
    }

    return {
        "kps_xy": kps_xy,
        "kps_score": kps_score,
        "bbox_xyxy": bbox_xyxy,
        "frame_idx": frame_idx,
        "fps": fps,
        "meta": meta,
    }


def extract_video_to_npz(
    video_path: str,
    output_path: str,
    verbose: bool = True,
    *,
    config_path: str = str(RTMPOSE_CONFIG),
    checkpoint_path: str = str(RTMPOSE_CHECKPOINT),
    yolo_weights: str = str(YOLO_WEIGHTS),
    yolo_conf_threshold: float = 0.25,
) -> dict:
    result = extract_video_pose(
        video_path=video_path,
        verbose=verbose,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        yolo_weights=yolo_weights,
        yolo_conf_threshold=yolo_conf_threshold,
    )
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path_obj),
        kps_xy=result["kps_xy"],
        kps_score=result["kps_score"],
        bbox_xyxy=result["bbox_xyxy"],
        frame_idx=result["frame_idx"],
        fps=np.float64(result["fps"]),
        meta=np.array([result["meta"]], dtype=object),
    )
    return result
