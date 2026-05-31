
from __future__ import annotations

import numpy as np


def _is_valid_box(box: list[float]) -> bool:
    if len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    return bool(np.isfinite([x1, y1, x2, y2]).all() and (x2 - x1) > 0 and (y2 - y1) > 0)


def select_primary_person(
    person_boxes: list[list[float]],
    frame_shape,
    previous_box: list[float] | None = None,
) -> list[float] | None:
    if not person_boxes:
        return None

    h, w = frame_shape[:2]
    if h <= 0 or w <= 0:
        return None
    frame_center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    frame_diag = float(np.hypot(w, h))
    if frame_diag <= 1e-9:
        return None

    valid_boxes = [box for box in person_boxes if _is_valid_box(box)]
    if not valid_boxes:
        return None
    if len(valid_boxes) == 1:
        return [float(v) for v in valid_boxes[0]]

    prev_valid = previous_box is not None and _is_valid_box(previous_box)
    prev_center = None
    if prev_valid:
        px1, py1, px2, py2 = previous_box  # type: ignore[misc]
        prev_center = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0], dtype=np.float64)

    frame_area = float(h * w)
    best_score = -float("inf")
    best_box: list[float] | None = None

    for box in valid_boxes:
        x1, y1, x2, y2 = box
        width = max(0.0, float(x2 - x1))
        height = max(0.0, float(y2 - y1))
        box_area = width * height
        area_score = float(box_area / frame_area) if frame_area > 0 else 0.0

        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
        center_dist_norm = float(np.linalg.norm(box_center - frame_center) / frame_diag)
        center_score = 1.0 - center_dist_norm

        if prev_valid and prev_center is not None:
            prev_dist_norm = float(np.linalg.norm(box_center - prev_center) / frame_diag)
            continuity_score = 1.0 - prev_dist_norm
            score = 0.45 * area_score + 0.25 * center_score + 0.30 * continuity_score
        else:
            score = 0.65 * area_score + 0.35 * center_score

        if score > best_score:
            best_score = score
            best_box = [float(v) for v in box]
    return best_box
