
from __future__ import annotations

from typing import Any

import numpy as np

from src.biomechanics.rdl.analysis_context.anchor_resolver import resolve_rdl_anchor_frames
from src.biomechanics.rdl.analysis_context.constants import RDL_ANCHOR_NAMES, RIGHT_CHAIN_KEYPOINTS


def _frame_has_finite_chain(kps: np.ndarray) -> bool:
    return bool(np.isfinite(kps[list(RIGHT_CHAIN_KEYPOINTS), :]).all())


def _validate_sequence_shapes(kps_clean: np.ndarray, kps_norm: np.ndarray, mask_norm: np.ndarray, label: str) -> None:
    if kps_clean.ndim != 3 or kps_clean.shape[1:] != (17, 2):
        raise ValueError(f"{label} clean keypoints must have shape (T, 17, 2), got {kps_clean.shape}.")
    if kps_norm.ndim != 3 or kps_norm.shape[1:] != (17, 2):
        raise ValueError(f"{label} normalized keypoints must have shape (T, 17, 2), got {kps_norm.shape}.")
    if mask_norm.ndim != 1:
        raise ValueError(f"{label} mask_valid_normalized must have shape (T,), got {mask_norm.shape}.")
    if kps_clean.shape[0] != kps_norm.shape[0] or kps_norm.shape[0] != mask_norm.shape[0]:
        raise ValueError(
            f"{label} sequence lengths mismatch: clean={kps_clean.shape[0]}, norm={kps_norm.shape[0]}, mask={mask_norm.shape[0]}"
        )


def build_rdl_anchor_pairs(
    *,
    user_pose_clean: dict,
    user_segmentation_result: dict,
    user_pose_sequence_normalization: dict,
    reference: dict,
) -> dict:
    user_kps_clean = np.asarray(
        user_pose_clean.get("kps_xy_clean", user_pose_clean.get("kps_xy")),
        dtype=np.float64,
    )
    user_kps_norm = np.asarray(user_pose_sequence_normalization["kps_xy_normalized"], dtype=np.float64)
    user_mask_norm = np.asarray(user_pose_sequence_normalization["mask_valid_normalized"], dtype=bool)
    _validate_sequence_shapes(user_kps_clean, user_kps_norm, user_mask_norm, "User")
    t_user = int(user_kps_norm.shape[0])

    ideal_segmentation = reference["segmentation_result"]
    ideal_rep_index = int(reference["selected_rep_raw_index"])
    ideal_reps = ideal_segmentation.get("reps", [])
    if not isinstance(ideal_reps, list) or ideal_rep_index >= len(ideal_reps):
        raise ValueError("Invalid selected_rep_raw_index for reference reps.")
    ideal_rep = ideal_reps[ideal_rep_index]

    ideal_kps_norm = np.asarray(reference["normalization"]["kps_xy_normalized"], dtype=np.float64)
    ideal_mask_norm = np.asarray(reference["normalization"]["mask_valid_normalized"], dtype=bool)
    ideal_kps_clean_raw = reference["normalization"].get("kps_xy_clean")
    ideal_kps_clean = np.asarray(ideal_kps_clean_raw, dtype=np.float64) if ideal_kps_clean_raw is not None else None
    if ideal_kps_norm.ndim != 3 or ideal_kps_norm.shape[1:] != (17, 2):
        raise ValueError("Reference normalized keypoints must have shape (T, 17, 2).")
    if ideal_mask_norm.shape != (ideal_kps_norm.shape[0],):
        raise ValueError("Reference mask_valid_normalized shape mismatch.")
    if ideal_kps_clean is not None and ideal_kps_clean.shape != ideal_kps_norm.shape:
        raise ValueError("Reference clean keypoints shape mismatch with normalized sequence.")
    t_ideal = int(ideal_kps_norm.shape[0])

    ideal_resolved = resolve_rdl_anchor_frames(ideal_rep)

    user_reps = user_segmentation_result.get("reps", [])
    if not isinstance(user_reps, list):
        user_reps = []
    valid_user_rep_items = [
        (i, rep) for i, rep in enumerate(user_reps) if isinstance(rep, dict) and rep.get("anchor_valid", True) is True
    ]

    warnings: list[str] = []
    paired_repetitions: list[dict[str, Any]] = []
    if not valid_user_rep_items:
        warnings.append("NO_VALID_USER_REPS_FOR_CONTEXT")
        return {
            "anchor_names": list(RDL_ANCHOR_NAMES),
            "paired_repetitions": [],
            "num_user_reps": 0,
            "num_paired_repetitions": 0,
            "ideal_rep_raw_index": int(ideal_rep_index),
            "ideal_valid_rep_index": int(reference["selected_valid_rep_index"]),
            "warnings": warnings,
        }

    for order, (user_rep_raw_index, user_rep) in enumerate(valid_user_rep_items):
        user_resolved = resolve_rdl_anchor_frames(user_rep)
        rep_warnings: list[str] = []
        anchors_payload: dict[str, Any] = {}
        valid_count = 0
        invalid_count = 0

        for anchor_name in RDL_ANCHOR_NAMES:
            user_frame = user_resolved["frames"][anchor_name]
            ideal_frame = ideal_resolved["frames"][anchor_name]
            pair_warnings: list[str] = []

            user_anchor_valid = bool(user_resolved["valid"][anchor_name] and user_frame is not None)
            ideal_anchor_valid = bool(ideal_resolved["valid"][anchor_name] and ideal_frame is not None)

            if not user_anchor_valid:
                pair_warnings.append(f"USER_ANCHOR_INVALID:{anchor_name}")
            if not ideal_anchor_valid:
                pair_warnings.append(f"IDEAL_ANCHOR_INVALID:{anchor_name}")

            if user_anchor_valid and not (0 <= int(user_frame) < t_user):
                user_anchor_valid = False
                pair_warnings.append(f"USER_FRAME_OUT_OF_RANGE:{anchor_name}")
            if ideal_anchor_valid and not (0 <= int(ideal_frame) < t_ideal):
                ideal_anchor_valid = False
                pair_warnings.append(f"IDEAL_FRAME_OUT_OF_RANGE:{anchor_name}")

            if user_anchor_valid:
                uf = int(user_frame)
                if not bool(user_mask_norm[uf]) or not _frame_has_finite_chain(user_kps_norm[uf]):
                    user_anchor_valid = False
                    pair_warnings.append(f"USER_NORMALIZED_FRAME_INVALID:{anchor_name}")
            if ideal_anchor_valid:
                inf = int(ideal_frame)
                if not bool(ideal_mask_norm[inf]) or not _frame_has_finite_chain(ideal_kps_norm[inf]):
                    ideal_anchor_valid = False
                    pair_warnings.append(f"IDEAL_NORMALIZED_FRAME_INVALID:{anchor_name}")

            pair_valid = bool(user_anchor_valid and ideal_anchor_valid)
            if pair_valid:
                valid_count += 1
            else:
                invalid_count += 1
                rep_warnings.extend(pair_warnings)

            uf = int(user_frame) if user_anchor_valid else None
            inf = int(ideal_frame) if ideal_anchor_valid else None
            anchors_payload[anchor_name] = {
                "anchor": anchor_name,
                "valid": pair_valid,
                "user_frame": int(user_frame) if user_frame is not None else None,
                "ideal_frame": int(ideal_frame) if ideal_frame is not None else None,
                "user_source": user_resolved["source"][anchor_name],
                "ideal_source": ideal_resolved["source"][anchor_name],
                "user_kps_clean": user_kps_clean[uf] if uf is not None else None,
                "user_kps_normalized": user_kps_norm[uf] if uf is not None else None,
                "ideal_kps_normalized": ideal_kps_norm[inf] if inf is not None else None,
                "ideal_kps_clean": ideal_kps_clean[inf] if (ideal_kps_clean is not None and inf is not None) else None,
                "warnings": pair_warnings,
            }

        paired_repetitions.append(
            {
                "rep_index": int(order),
                "user_rep_raw_index": int(user_rep_raw_index),
                "user_rep_order": int(order + 1),
                "ideal_rep_raw_index": int(ideal_rep_index),
                "ideal_valid_rep_index": int(reference["selected_valid_rep_index"]),
                "anchors": anchors_payload,
                "num_valid_anchor_pairs": int(valid_count),
                "num_invalid_anchor_pairs": int(invalid_count),
                "warnings": sorted(set(rep_warnings)),
            }
        )

    return {
        "anchor_names": list(RDL_ANCHOR_NAMES),
        "paired_repetitions": paired_repetitions,
        "num_user_reps": int(len(valid_user_rep_items)),
        "num_paired_repetitions": int(len(paired_repetitions)),
        "ideal_rep_raw_index": int(ideal_rep_index),
        "ideal_valid_rep_index": int(reference["selected_valid_rep_index"]),
        "warnings": sorted(set(warnings)),
    }

