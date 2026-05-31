
from .detector import detect_neck_movement_error
from .metrics import (
    NECK_DIRECTION_SIGN,
    NECK_SEGMENTS,
    ORDERED_ANCHORS,
    NeckMovementMetrics,
    NeckPoseState,
    classify_neck_direction,
    compute_neck_movement_segments,
    compute_neck_pose_state,
    compute_neck_segment_metrics,
    wrap_to_180,
)
from .rules import (
    AnchorRuling,
    DETECTION_ANCHORS,
    NECK_B_LEVE_MAX_DEG,
    NECK_B_MEDIA_MAX_DEG,
    NECK_B_NONE_MAX_DEG,
    RepNeckMovementVerdict,
    detect_neck_movement,
    rule_segment,
)

# Alias temporales de compatibilidad; importados desde capa de compatibilidad de spine_flexion.
from .rules import RepSpineFlexionVerdict, detect_spine_flexion

__all__ = [
    "ORDERED_ANCHORS",
    "NECK_SEGMENTS",
    "NECK_DIRECTION_SIGN",
    "NeckPoseState",
    "NeckMovementMetrics",
    "compute_neck_pose_state",
    "compute_neck_segment_metrics",
    "compute_neck_movement_segments",
    "classify_neck_direction",
    "wrap_to_180",
    "DETECTION_ANCHORS",
    "NECK_B_NONE_MAX_DEG",
    "NECK_B_LEVE_MAX_DEG",
    "NECK_B_MEDIA_MAX_DEG",
    "AnchorRuling",
    "RepNeckMovementVerdict",
    "detect_neck_movement",
    "rule_segment",
    "detect_neck_movement_error",
    "RepSpineFlexionVerdict",
    "detect_spine_flexion",
]
