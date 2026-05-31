
from __future__ import annotations

import numpy as np
from mmpose.apis import inference_topdown, init_model


class RTMPoseModel:

    def __init__(self, config_path: str, checkpoint_path: str, device: str) -> None:
        self.model = init_model(config_path, checkpoint_path, device=device)

    def predict(
        self,
        frame: np.ndarray,
        bbox: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(bbox) != 4:
            raise ValueError(f"bbox inválida. Se esperaban 4 valores y llegaron {len(bbox)}.")
        bboxes = np.array([bbox], dtype=np.float32)
        pose_results = inference_topdown(self.model, frame, bboxes)
        if not pose_results:
            raise RuntimeError("RTMPose no devolvió resultados de inferencia.")
        keypoints = np.asarray(pose_results[0].pred_instances.keypoints[0], dtype=np.float64)
        scores = np.asarray(pose_results[0].pred_instances.keypoint_scores[0], dtype=np.float64)
        return keypoints[:, :2], scores
