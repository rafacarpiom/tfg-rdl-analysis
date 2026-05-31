
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationConfig:
    method: str = "pelvis_torso_scale"
    pelvis_idx: int = 12
    shoulder_idx: int = 6
    eps: float = 1e-6
    apply_rotation: bool = False
    sequence_scale_mode: str = "fixed_median"

