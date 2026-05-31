
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.biomechanics.rdl import RDLSegmentationConfig
from src.utils.paths import (
    OUTPUTS,
    rdl_reference_dir,
    RTMPOSE_CHECKPOINT,
    RTMPOSE_CONFIG,
    YOLO_WEIGHTS,
)


@dataclass(frozen=True)
class PoseExtractionConfig:
    # Configuración de modelos/pesos para extracción de pose.
    config_path: Path = RTMPOSE_CONFIG
    checkpoint_path: Path = RTMPOSE_CHECKPOINT
    yolo_weights: Path = YOLO_WEIGHTS
    verbose: bool = False


@dataclass(frozen=True)
class OrientationConfig:
    target_facing: str = "right"
    score_threshold: float = 0.30
    min_valid_frames: int = 10
    margin: float = 0.03


@dataclass(frozen=True)
class FullAnalysisConfig:
    # Entrada principal del flujo inicial.
    video_path: Path
    # Subconfiguración de pose.
    pose: PoseExtractionConfig = field(default_factory=PoseExtractionConfig)
    # Configuración de estimación de orientación lateral.
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    # Configuración de segmentación RDL.
    segmentation: RDLSegmentationConfig = field(default_factory=RDLSegmentationConfig)
    # Referencia ideal fija para construcción del contexto RDL.
    reference_dir: Path = rdl_reference_dir("PM-Ideal")
    ideal_valid_rep_index: int = 1

