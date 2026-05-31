
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.biomechanics.normalization import NormalizationConfig, normalize_pose_sequence
from src.utils.paths import rdl_reference_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize full ideal RDL sequence keypoints.")
    parser.add_argument(
        "--clean-npz",
        type=Path,
        default=rdl_reference_dir("PM-Ideal") / "ideal_rtmpose_clean.npz",
        help="Path to ideal_rtmpose_clean.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=rdl_reference_dir("PM-Ideal"),
        help="Directory to write normalized ideal artifacts.",
    )
    parser.add_argument("--reference-name", type=str, default="PM-Ideal")
    parser.add_argument("--sequence-scale-mode", type=str, default="fixed_median", choices=["fixed_median", "framewise"])
    return parser.parse_args()


def _load_npz(clean_npz_path: Path) -> dict[str, Any]:
    with np.load(clean_npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _to_float_maybe(x: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(x).squeeze())
    except Exception:
        return default


def main() -> None:
    args = _parse_args()
    clean_npz_path = args.clean_npz.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clean_npz_path.is_file():
        raise FileNotFoundError(f"clean_npz_path not found: {clean_npz_path}")

    npz = _load_npz(clean_npz_path)

    kps_xy_key = "kps_xy_clean" if "kps_xy_clean" in npz else "kps_xy"
    kps_score_key = "kps_score_clean" if "kps_score_clean" in npz else "kps_score"

    kps_xy = np.asarray(npz[kps_xy_key], dtype=float)
    kps_score = np.asarray(npz[kps_score_key], dtype=float)

    pose_source = "clean" if kps_xy_key == "kps_xy_clean" else "raw_fallback"

    fps = _to_float_maybe(npz.get("fps"), default=0.0)
    warnings: list[str] = []

    T = int(kps_xy.shape[0]) if kps_xy.ndim >= 1 else 0
    frame_idx = None
    if "frame_idx" in npz:
        frame_idx = np.asarray(npz["frame_idx"], dtype=int)
    else:
        frame_idx = np.arange(T, dtype=int)
        warnings.append("FRAME_IDX_MISSING_USING_ARANGE")

    config = NormalizationConfig(sequence_scale_mode=args.sequence_scale_mode)
    norm_result = normalize_pose_sequence(kps_xy, config=config)

    kps_xy_normalized = np.asarray(norm_result["kps_xy_normalized"], dtype=float)
    mask_valid_normalized = np.asarray(norm_result["mask_valid_normalized"], dtype=bool)
    origins = np.asarray(norm_result["origins"], dtype=float)
    scales = np.asarray(norm_result["scales"], dtype=float)
    raw_scales = np.asarray(norm_result["raw_scales"], dtype=float)

    warnings.extend([str(w) for w in norm_result.get("warnings", [])])

    valid_frame_count = int(mask_valid_normalized.sum())
    total_frame_count = int(mask_valid_normalized.shape[0])
    valid_frame_ratio = float(mask_valid_normalized.mean()) if total_frame_count > 0 else 0.0

    normalization_method = str(norm_result.get("method", "pelvis_torso_scale"))
    sequence_scale_mode = str(norm_result.get("sequence_scale_mode", args.sequence_scale_mode))

    out_npz = out_dir / "ideal_pose_sequence_normalized.npz"
    out_meta_json = out_dir / "ideal_pose_sequence_normalized_meta.json"

    npz_payload: dict[str, Any] = {
        "kps_xy_clean": kps_xy,
        "kps_score_clean": kps_score,
        "kps_xy_normalized": kps_xy_normalized,
        "mask_valid_normalized": mask_valid_normalized,
        "normalization_origins": origins,
        "normalization_scales": scales,
        "raw_scales": raw_scales,
        "frame_idx": frame_idx,
        "fps": fps,
        "pose_source": np.asarray(pose_source, dtype=object),
        "normalization_method": np.asarray(normalization_method, dtype=object),
        "sequence_scale_mode": np.asarray(sequence_scale_mode, dtype=object),
    }

    # Opcionales del NPZ original (si existen)
    if "mask_valid_frames" in npz:
        npz_payload["mask_valid_frames"] = np.asarray(npz["mask_valid_frames"])
    if "mask_valid_right_chain" in npz:
        npz_payload["mask_valid_right_chain"] = np.asarray(npz["mask_valid_right_chain"])

    np.savez_compressed(out_npz, **npz_payload)

    meta: dict[str, Any] = {
        "reference_name": args.reference_name,
        "exercise": "RDL",
        "source_clean_npz_path": str(clean_npz_path),
        "output_npz": out_npz.name,
        "pose_source": pose_source,
        "num_frames": int(kps_xy.shape[0]),
        "fps": fps,
        "normalization_method": normalization_method,
        "sequence_scale_mode": sequence_scale_mode,
        "normalization_rotation_applied": False,
        "valid_frame_count": valid_frame_count,
        "total_frame_count": total_frame_count,
        "valid_frame_ratio": valid_frame_ratio,
        "warnings": warnings,
    }

    with out_meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"clean_npz_used: {clean_npz_path.as_posix()}")
    print(f"output_npz_generated: {out_npz.as_posix()}")
    print(f"meta_json_generated: {out_meta_json.as_posix()}")
    print(f"num_frames: {int(kps_xy.shape[0])}")
    print(f"valid_frame_count: {valid_frame_count}")
    print(f"valid_frame_ratio: {valid_frame_ratio}")
    print(f"sequence_scale_mode: {sequence_scale_mode}")
    print(f"warnings: {warnings}")


if __name__ == "__main__":
    main()

