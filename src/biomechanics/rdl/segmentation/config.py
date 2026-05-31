
from __future__ import annotations

from dataclasses import dataclass, field

from .validation import RDLValidationConfig


@dataclass(frozen=True)
class RDLSegmentationConfig:
    thr_conf: float = 0.30
    max_gap_interp: int = 5
    savgol_window_length: int = 11
    savgol_polyorder: int = 2
    min_peak_distance: int = 15
    min_prominence_ratio: float = 0.05
    double_bottom_window_seconds: float = 0.15
    double_bottom_bridge_factor: float = 2.0
    local_prominence_ratio: float = 0.15
    phase_top_margin_ratio: float = 0.10
    phase_bottom_margin_ratio: float = 0.015
    allow_boundary_events: bool = True
    boundary_hold_seconds: float = 0.5
    boundary_requires_stable_top: bool = True
    anchor_percentages: tuple[int, ...] = (0, 25, 50, 75, 100)
    anchor_method: str = "signal_progress"
    anchor_min_valid_frames: int = 3
    anchor_min_phase_rom_deg: float = 5.0
    anchor_allow_nearest_valid_fallback: bool = True
    validation: RDLValidationConfig = field(default_factory=RDLValidationConfig)

