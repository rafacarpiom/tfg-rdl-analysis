
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RIGHT_CHAIN = (6, 8, 10, 12, 14, 16)
LEFT_CHAIN = (5, 7, 9, 11, 13, 15)
LEFT_EAR = 3
RIGHT_EAR = 4
NOSE = 0


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ndarray_summary(arr: np.ndarray) -> dict[str, Any]:
    return {"_type": "ndarray", "shape": list(arr.shape), "dtype": str(arr.dtype)}


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _ndarray_summary(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [to_jsonable(x) for x in sorted(obj, key=lambda x: str(x))]
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    return obj


def save_json(path: str | Path, data: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def plot_skeleton(ax, keypoints, *, color: str = "C0", label: str | None = None, alpha: float = 1.0, linewidth: float = 2.0) -> None:
    pts = np.asarray(keypoints, dtype=np.float64)
    if pts.shape != (17, 2):
        return

    def _finite_point(idx: int) -> bool:
        return bool(0 <= idx < pts.shape[0] and np.isfinite(pts[idx]).all())

    def _side_visibility_score(side_chain: tuple[int, ...], ear_idx: int) -> float:
        indices = list(side_chain) + [ear_idx]
        finite_count = float(sum(1 for idx in indices if _finite_point(idx)))
        chain_length = 0.0
        for i, j in zip(side_chain[:-1], side_chain[1:]):
            if _finite_point(i) and _finite_point(j):
                chain_length += float(np.linalg.norm(pts[i] - pts[j]))
        return finite_count + 0.01 * chain_length

    left_score = _side_visibility_score(LEFT_CHAIN, LEFT_EAR)
    right_score = _side_visibility_score(RIGHT_CHAIN, RIGHT_EAR)
    side_chain = LEFT_CHAIN if left_score >= right_score else RIGHT_CHAIN
    side_ear = LEFT_EAR if side_chain is LEFT_CHAIN else RIGHT_EAR

    side_order = (side_chain[5], side_chain[4], side_chain[3], side_chain[0], side_chain[1], side_chain[2])
    draw_edges = (
        (side_order[0], side_order[1]),
        (side_order[1], side_order[2]),
        (side_order[2], side_order[3]),
        (side_order[3], side_order[4]),
        (side_order[4], side_order[5]),
        (side_order[3], side_ear),
        (side_ear, NOSE),
    )
    for i, j in draw_edges:
        if _finite_point(i) and _finite_point(j):
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=color, alpha=alpha, linewidth=linewidth)

    draw_points = list(side_order) + [side_ear, NOSE]
    finite_points = [idx for idx in draw_points if _finite_point(idx)]
    if finite_points:
        ax.scatter(pts[finite_points, 0], pts[finite_points, 1], s=24, color=color, alpha=alpha, label=label, zorder=3)


def set_equal_xy(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
