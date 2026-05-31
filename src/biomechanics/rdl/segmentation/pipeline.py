
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.biomechanics.rdl.segmentation.boundary_refinement import refine_rdl_candidate_boundaries
from src.biomechanics.rdl.segmentation.config import RDLSegmentationConfig
from src.biomechanics.rdl.segmentation.events import (
    add_boundary_top_bottom_events,
    center_top_bottom_plateaus,
    classify_top_bottom_events,
    collapse_double_bottoms,
    detect_top_bottom_candidates,
    enforce_top_bottom_alternation,
    merge_micro_oscillations,
)
from src.biomechanics.rdl.segmentation.io import (
    load_pose_npz,
    save_segmentation_debug_npz,
    save_segmentation_json,
    validate_pose_data_dict,
)
from src.biomechanics.rdl.segmentation.repetitions import build_rdl_repetitions
from src.biomechanics.rdl.segmentation.signal import build_rdl_signal
from src.biomechanics.rdl.segmentation.validation import (
    NO_VALID_RDL_SIGNAL,
    NO_VALID_REPS,
    PARTIAL_VIDEO,
    SEGMENTATION_STATUS_OK,
    TOO_NOISY,
    WRONG_MOVEMENT_OR_NOT_RDL,
    append_edge_artifact_candidates,
    compute_global_signal_quality,
    select_consistent_rep_block,
    validate_rdl_rep_candidates,
)


def _as_float_list(arr: np.ndarray) -> list[float]:
    return [float(x) if np.isfinite(x) else float("nan") for x in arr.tolist()]


def _as_bool_list(arr: np.ndarray) -> list[bool]:
    return [bool(x) for x in arr.tolist()]


def _recover_entry_top_event(
    signal: np.ndarray,
    bottom_valid: list[int],
    top_valid: list[int],
    top_candidates: np.ndarray,
    valid_mask: np.ndarray,
    *,
    global_rom: float,
    top_band: float,
) -> tuple[list[int], dict[str, Any]]:
    debug: dict[str, Any] = {
        "applied": False,
        "recovered_index": None,
        "reason": "not_needed",
        "evaluated_candidates": [],
    }
    if not bottom_valid or not top_valid or int(bottom_valid[0]) >= int(top_valid[0]):
        return list(top_valid), debug

    y = np.asarray(signal, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(y)
    first_bottom = int(bottom_valid[0])
    candidates = [
        int(m)
        for m in np.asarray(top_candidates, dtype=np.int64).tolist()
        if 0 < int(m) < first_bottom and np.isfinite(y[int(m)])
    ]
    if not candidates:
        debug["reason"] = "no_top_candidate_before_first_bottom"
        return list(top_valid), debug

    min_drop = max(35.0, 0.5 * float(global_rom))
    accepted: list[int] = []
    for m in candidates:
        segment = y[m : first_bottom + 1]
        segment_valid = valid[m : first_bottom + 1]
        dy = np.diff(segment)
        dy = dy[np.isfinite(dy)]
        denom = float(np.sum(np.abs(dy)))
        trend_score = float(np.sum(np.maximum(-dy, 0.0)) / (denom + 1e-8)) if dy.size else 0.0
        valid_ratio = float(np.mean(segment_valid)) if segment_valid.size else 0.0
        right_drop = float(y[m] - y[first_bottom]) if np.isfinite(y[first_bottom]) else float("nan")
        top_ok = bool(np.isfinite(y[m]) and y[m] >= float(top_band) - 8.0)
        accepted_candidate = bool(
            np.isfinite(right_drop)
            and right_drop >= min_drop
            and trend_score >= 0.70
            and valid_ratio >= 0.60
            and top_ok
        )
        debug["evaluated_candidates"].append(
            {
                "index": int(m),
                "right_drop": right_drop,
                "trend_score": trend_score,
                "valid_ratio": valid_ratio,
                "signal_value": float(y[m]),
                "top_threshold": float(top_band) - 8.0,
                "accepted": accepted_candidate,
            }
        )
        if accepted_candidate:
            accepted.append(m)

    if not accepted:
        debug["reason"] = "no_candidate_passed_entry_top_checks"
        return list(top_valid), debug

    recovered = int(accepted[-1])
    recovered_valid = sorted(set(int(x) for x in top_valid) | {recovered})
    debug["applied"] = True
    debug["recovered_index"] = recovered
    debug["reason"] = "entry_top_recovered"
    return recovered_valid, debug


def run_rdl_segmentation(
    npz_path: str | Path,
    *,
    config: RDLSegmentationConfig | None = None,
) -> dict[str, Any]:
    cfg = config if config is not None else RDLSegmentationConfig()
    pose = load_pose_npz(npz_path)
    return _run_rdl_segmentation_core(pose, cfg)


def run_rdl_segmentation_from_pose(
    pose_data: dict[str, Any],
    *,
    config: RDLSegmentationConfig | None = None,
) -> dict[str, Any]:
    cfg = config if config is not None else RDLSegmentationConfig()
    pose = validate_pose_data_dict(pose_data)
    return _run_rdl_segmentation_core(pose, cfg)


def _run_rdl_segmentation_core(
    pose: dict[str, Any],
    cfg: RDLSegmentationConfig,
) -> dict[str, Any]:
    signal = build_rdl_signal(
        pose["kps_xy"],
        pose["kps_score"],
        thr_conf=cfg.thr_conf,
        max_gap_interp=cfg.max_gap_interp,
        savgol_window_length=cfg.savgol_window_length,
        savgol_polyorder=cfg.savgol_polyorder,
    )
    global_quality = compute_global_signal_quality(
        signal.signal_smooth,
        signal.valid_mask,
        config=cfg.validation,
    )
    local_thr = float(cfg.local_prominence_ratio) * float(signal.rom)

    base_result: dict[str, Any] = {
        "video_id": pose["video_id"],
        "exercise": "RDL",
        "fps": float(pose["fps"]),
        "pose_source": pose["pose_source"],
        "has_clean_pose": bool(pose["has_clean_pose"]),
        "signal_name": "rdl_hip_angle_k6_k12_k14_deg",
        "config": asdict(cfg),
        "signals": {
            "signal_raw": _as_float_list(signal.signal_raw),
            "signal_smooth": _as_float_list(signal.signal_smooth),
            "valid_mask_raw": _as_bool_list(signal.valid_mask_raw),
            "interp_mask": _as_bool_list(signal.interp_mask),
            "valid_mask": _as_bool_list(signal.valid_mask),
        },
        "validation_debug": {"global_signal_quality": global_quality},
        "event_candidates": {},
        "rep_candidates": [],
    }

    if global_quality["status"] != SEGMENTATION_STATUS_OK:
        return {
            **base_result,
            "segmentation_status": NO_VALID_RDL_SIGNAL,
            "summary": {
                "rom": float(signal.rom),
                "p5": float(signal.p5) if np.isfinite(signal.p5) else float("nan"),
                "p95": float(signal.p95) if np.isfinite(signal.p95) else float("nan"),
                "num_reps": 0,
                "num_rep_candidates": 0,
                "anchor_method": str(cfg.anchor_method),
                "num_reps_with_valid_anchors": 0,
                "num_reps_with_invalid_anchors": 0,
                "savgol_fallback_used": bool(signal.savgol_fallback_used),
            },
            "event_candidates": {
                "bottom_indices_candidates": [],
                "top_indices_candidates": [],
                "bottom_indices_valid": [],
                "top_indices_valid": [],
                "bottom_details": [],
                "top_details": [],
            },
            "reps": [],
        }

    bottom_raw, top_raw = detect_top_bottom_candidates(
        signal.signal_smooth,
        min_distance=cfg.min_peak_distance,
        min_prominence_ratio=cfg.min_prominence_ratio,
    )
    bottom_alt, top_alt = enforce_top_bottom_alternation(signal.signal_smooth, bottom_raw, top_raw)
    bottom_merged, top_merged = merge_micro_oscillations(signal.signal_smooth, bottom_alt, top_alt, thr=local_thr)
    bottom_boundary, top_boundary = add_boundary_top_bottom_events(
        signal.signal_smooth,
        bottom_merged,
        top_merged,
        thr=local_thr,
        allow_boundary_events=cfg.allow_boundary_events,
        boundary_hold_frames=int(round(float(cfg.boundary_hold_seconds) * float(pose["fps"]))),
        boundary_requires_stable_top=cfg.boundary_requires_stable_top,
        top_band=float(global_quality["top_band"]),
        valid_mask=signal.valid_mask,
    )
    bottom_alt2, top_alt2 = enforce_top_bottom_alternation(signal.signal_smooth, bottom_boundary, top_boundary)
    double_bottom_window_frames = max(1, int(round(float(cfg.double_bottom_window_seconds) * float(pose["fps"]))))
    bottom_collapsed, top_collapsed = collapse_double_bottoms(
        signal.signal_smooth,
        bottom_alt2,
        top_alt2,
        min_separation_frames=double_bottom_window_frames,
        bridge_factor=cfg.double_bottom_bridge_factor,
    )
    bottom_final, top_final = center_top_bottom_plateaus(
        signal.signal_smooth,
        bottom_collapsed,
        top_collapsed,
        rom=signal.rom,
    )
    bottom_final, top_final = collapse_double_bottoms(
        signal.signal_smooth,
        bottom_final,
        top_final,
        min_separation_frames=double_bottom_window_frames,
        bridge_factor=cfg.double_bottom_bridge_factor,
    )
    bottom_valid, bottom_details, top_valid, top_details = classify_top_bottom_events(
        signal.signal_smooth,
        bottom_final,
        top_final,
        rom=signal.rom,
        thr=local_thr,
    )
    top_valid, entry_top_debug = _recover_entry_top_event(
        signal.signal_smooth,
        bottom_valid,
        top_valid,
        top_final,
        signal.valid_mask,
        global_rom=float(global_quality["global_rom"]),
        top_band=float(global_quality["top_band"]),
    )
    if entry_top_debug.get("applied"):
        recovered_idx = int(entry_top_debug["recovered_index"])
        if not any(int(d.get("index", -1)) == recovered_idx for d in top_details):
            top_details.append(
                {
                    "kind": "top",
                    "index": recovered_idx,
                    "value": float(signal.signal_smooth[recovered_idx]),
                    "left_neighbor_index": None,
                    "right_neighbor_index": int(bottom_valid[0]) if bottom_valid else None,
                    "left_amplitude": float("nan"),
                    "right_amplitude": float(signal.signal_smooth[recovered_idx] - signal.signal_smooth[int(bottom_valid[0])]) if bottom_valid else float("nan"),
                    "local_amplitude": float(signal.signal_smooth[recovered_idx] - signal.signal_smooth[int(bottom_valid[0])]) if bottom_valid else float("nan"),
                    "is_valid": True,
                    "reason": "entry_top_recovered",
                    "label": "ENTRY_TOP",
                }
            )

    rep_candidates_raw = build_rdl_repetitions(
        signal.signal_smooth,
        bottom_valid,
        top_valid,
        fps=float(pose["fps"]),
        config=cfg,
    )
    rep_candidates_refined, boundary_refinement_debug = refine_rdl_candidate_boundaries(
        rep_candidates_raw,
        signal.signal_smooth,
        signal.valid_mask,
        fps=float(pose["fps"]),
        global_quality=global_quality,
        config=cfg,
    )
    rep_candidates, candidate_validation_debug = validate_rdl_rep_candidates(
        rep_candidates_refined,
        signal.signal_smooth,
        fps=float(pose["fps"]),
        valid_mask=signal.valid_mask,
        valid_mask_raw=signal.valid_mask_raw,
        interp_mask=signal.interp_mask,
        global_quality=global_quality,
        config=cfg.validation,
    )
    final_reps, block_debug = select_consistent_rep_block(rep_candidates)
    rep_candidates = append_edge_artifact_candidates(
        rep_candidates,
        final_reps,
        signal.signal_smooth,
        fps=float(pose["fps"]),
    )
    num_candidates_with_valid_anchors = int(sum(1 for r in rep_candidates if bool(r.get("anchor_valid", False))))
    num_candidates_with_invalid_anchors = int(len(rep_candidates) - num_candidates_with_valid_anchors)
    num_reps_with_valid_anchors = int(sum(1 for r in final_reps if bool(r.get("anchor_valid", False))))
    num_reps_with_invalid_anchors = int(len(final_reps) - num_reps_with_valid_anchors)

    if final_reps:
        segmentation_status = SEGMENTATION_STATUS_OK
    else:
        labels = {str(c.get("label", "")) for c in rep_candidates}
        reasons = {str(c.get("reason", "")) for c in rep_candidates}
        if labels & {"PARTIAL_START", "PARTIAL_END", "POST_EXERCISE_NOISE"}:
            segmentation_status = PARTIAL_VIDEO
        elif labels & {"UNSTABLE_SIGNAL"} or reasons & {"unstable_signal", "abrupt_signal_jump"}:
            segmentation_status = TOO_NOISY
        elif labels & {"WRONG_MOVEMENT_OR_NOT_RDL"}:
            segmentation_status = WRONG_MOVEMENT_OR_NOT_RDL
        else:
            segmentation_status = NO_VALID_REPS

    return {
        **base_result,
        "segmentation_status": segmentation_status,
        "summary": {
            "rom": float(signal.rom),
            "p5": float(signal.p5) if np.isfinite(signal.p5) else float("nan"),
            "p95": float(signal.p95) if np.isfinite(signal.p95) else float("nan"),
            "num_bottom_candidates_raw": int(bottom_raw.size),
            "num_top_candidates_raw": int(top_raw.size),
            "num_bottom_after_alternation": int(bottom_alt.size),
            "num_top_after_alternation": int(top_alt.size),
            "num_bottom_after_merge": int(bottom_merged.size),
            "num_top_after_merge": int(top_merged.size),
            "num_bottom_after_double_bottom_collapse": int(bottom_collapsed.size),
            "num_top_after_double_bottom_collapse": int(top_collapsed.size),
            "num_bottom_final": int(bottom_final.size),
            "num_top_final": int(top_final.size),
            "num_bottom_valid": int(len(bottom_valid)),
            "num_top_valid": int(len(top_valid)),
            "num_rep_candidates": len(rep_candidates),
            "num_reps": len(final_reps),
            "anchor_method": str(cfg.anchor_method),
            "num_rep_candidates_with_valid_anchors": num_candidates_with_valid_anchors,
            "num_rep_candidates_with_invalid_anchors": num_candidates_with_invalid_anchors,
            "num_reps_with_valid_anchors": num_reps_with_valid_anchors,
            "num_reps_with_invalid_anchors": num_reps_with_invalid_anchors,
            "savgol_fallback_used": bool(signal.savgol_fallback_used),
        },
        "event_candidates": {
            "bottom_indices_candidates": [int(x) for x in bottom_final],
            "top_indices_candidates": [int(x) for x in top_final],
            "bottom_indices_valid": [int(x) for x in bottom_valid],
            "top_indices_valid": [int(x) for x in top_valid],
            "bottom_details": bottom_details,
            "top_details": top_details,
        },
        "rep_candidates": rep_candidates,
        "validation_debug": {
            "global_signal_quality": global_quality,
            "entry_top_recovery": entry_top_debug,
            "boundary_refinement": boundary_refinement_debug,
            "candidate_validation": candidate_validation_debug,
            "consistent_block": block_debug,
        },
        "reps": final_reps,
    }


def segment_rdl_reps(
    npz_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    config: RDLSegmentationConfig | None = None,
    save_json_flag: bool = True,
    save_debug_npz: bool = True,
) -> dict[str, Any]:
    result = run_rdl_segmentation(npz_path, config=config)
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = Path("outputs/segmentation") / str(result["video_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(result["video_id"]) + "_rdl_segmentation"
    if save_json_flag:
        save_segmentation_json(result, out_dir / f"{stem}.json")
    if save_debug_npz:
        save_segmentation_debug_npz(result, out_dir / f"{stem}_debug_arrays.npz")
    return result

