
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseCleaningConfig:
    score_thr: float = 0.4
    max_gap: int = 5
    savgol_window: int = 21
    savgol_polyorder: int = 2
    smoothing_passes: int = 1
    velocity_factor: float = 2.0
    min_valid_ratio_smoothing: float = 0.75

