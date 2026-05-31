
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.biomechanics.rdl.analysis_context.anchor_pairing import build_rdl_anchor_pairs
from src.biomechanics.rdl.analysis_context.reference_loader import load_rdl_reference


def build_rdl_analysis_context(
    *,
    user_pose_clean: dict,
    user_segmentation_result: dict,
    user_pose_sequence_normalization: dict,
    reference_dir: str | Path,
    ideal_valid_rep_index: int = 1,
) -> dict[str, Any]:
    reference = load_rdl_reference(
        reference_dir,
        ideal_valid_rep_index=ideal_valid_rep_index,
    )
    anchor_pairs = build_rdl_anchor_pairs(
        user_pose_clean=user_pose_clean,
        user_segmentation_result=user_segmentation_result,
        user_pose_sequence_normalization=user_pose_sequence_normalization,
        reference=reference,
    )

    user_video_id = (
        user_segmentation_result.get("video_id")
        if isinstance(user_segmentation_result, dict) and user_segmentation_result.get("video_id")
        else (user_pose_sequence_normalization.get("meta") or {}).get("video_id")
        if isinstance(user_pose_sequence_normalization, dict)
        else None
    )
    if not user_video_id:
        user_video_id = "unknown"

    combined_warnings = sorted(
        set(
            list(reference.get("warnings", []))
            + list(anchor_pairs.get("warnings", []))
        )
    )

    return {
        "exercise": "RDL",
        "user": {
            "video_id": str(user_video_id),
            "pose_clean": user_pose_clean,
            "segmentation_result": user_segmentation_result,
            "normalization": user_pose_sequence_normalization,
        },
        "reference": reference,
        "anchor_pairs": anchor_pairs,
        "context_meta": {
            "reference_name": reference["reference_name"],
            "reference_dir": reference["reference_dir"],
            "ideal_rep_raw_index": reference["selected_rep_raw_index"],
            "ideal_valid_rep_index": reference["selected_valid_rep_index"],
            "num_user_reps": anchor_pairs["num_user_reps"],
            "num_paired_repetitions": anchor_pairs["num_paired_repetitions"],
            "anchor_names": anchor_pairs["anchor_names"],
            "warnings": combined_warnings,
        },
        "warnings": combined_warnings,
    }

