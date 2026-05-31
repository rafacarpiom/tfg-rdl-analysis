
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.biomechanics.normalization import NormalizationConfig, normalize_pose_sequence
from src.biomechanics.rdl import build_rdl_analysis_context, run_rdl_segmentation_from_pose
from src.biomechanics.rdl.feedback.aggregation import aggregate_rdl_feedback_evidence
from src.biomechanics.rdl.feedback.builder import build_rdl_feedback_report
from src.biomechanics.rdl.feedback.evidence_normalizer import normalize_rdl_detector_evidence
from src.biomechanics.rdl.detectors import run_rdl_detectors
from src.biomechanics.rdl.segmentation.io import save_segmentation_debug_npz, save_segmentation_json
from src.pipeline.config import FullAnalysisConfig
from src.pipeline import run_full_analysis
from src.pipeline.results import FullAnalysisArtifacts, PipelineResult
from src.pipeline import status as pipeline_status
from src.visualization.rdl import export_rdl_debug_bundle
from src.visualization.rdl.detectors import (
    export_asymmetry_debug,
    export_bar_far_debug,
    export_bent_arms_debug,
    export_hip_hinge_debug,
    export_knee_dominant_debug,
    export_lockout_debug,
    export_neck_movement_debug,
    export_rom_debug,
    export_spine_flexion_debug,
)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
OUTPUTS = Path("outputs")


def _list_available_videos() -> list[Path]:
    data_root = Path("data")
    if not data_root.is_dir():
        return []
    videos = [p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def _ask_video_paths() -> list[Path]:
    videos = _list_available_videos()
    if not videos:
        raw = input("\nNo se encontraron videos en data/. Introduce ruta manual: ").strip()
        video_path = Path(raw).expanduser()
        if not video_path.is_file():
            raise FileNotFoundError(f"Ruta de video inexistente: {video_path}")
        return [video_path]

    print("\nVideos disponibles (data/):")
    print("0. TODOS")
    for idx, path in enumerate(videos, start=1):
        print(f"{idx}. {path.as_posix()}")

    while True:
        raw = input("\nSelecciona numero de video: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Entrada invalida. Introduce un numero.")
            continue
        if choice == 0:
            return videos
        if 1 <= choice <= len(videos):
            return [videos[choice - 1]]
        print(f"Numero fuera de rango (0..{len(videos)}).")


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _opt_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "si", "s"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano invalido: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline RDL")
    parser.add_argument("--all", action="store_true", help="Procesar todos los videos en data/")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Ruta a un video concreto (no combinar con --all). Si se omite y no hay --all, menu interactivo.",
    )
    parser.add_argument("--export-detectors", type=_opt_bool, default=None, help="Exportar debug_detectors (true/false)")
    parser.add_argument("--save-raw", type=_opt_bool, default=None, help="Guardar RAW npz (true/false)")
    parser.add_argument("--save-clean", type=_opt_bool, default=None, help="Guardar CLEAN npz (true/false)")
    parser.add_argument("--save-seg", type=_opt_bool, default=None, help="Guardar segmentacion adicional (true/false)")
    parser.add_argument("--save-norm", type=_opt_bool, default=None, help="Guardar normalizacion adicional (true/false)")
    parser.add_argument(
        "--reuse-clean-npz",
        type=_opt_bool,
        default=True,
        help="Reutilizar outputs/npz/*_rtmpose_clean.npz si existe (true/false)",
    )
    return parser.parse_args()


def _load_pose_clean_npz(npz_path: Path, video_id: str) -> dict[str, Any]:
    with np.load(str(npz_path), allow_pickle=True) as npz:
        if "kps_xy_clean" in npz.files:
            kps_xy_clean = np.asarray(npz["kps_xy_clean"], dtype=np.float64)
        elif "kps_xy" in npz.files:
            kps_xy_clean = np.asarray(npz["kps_xy"], dtype=np.float64)
        else:
            raise ValueError(f"{npz_path} sin kps_xy_clean ni kps_xy")

        if "kps_score_clean" in npz.files:
            kps_score_clean = np.asarray(npz["kps_score_clean"], dtype=np.float64)
        elif "kps_score" in npz.files:
            kps_score_clean = np.asarray(npz["kps_score"], dtype=np.float64)
        else:
            kps_score_clean = np.full((kps_xy_clean.shape[0], 17), np.nan, dtype=np.float64)

        bbox = np.asarray(npz["bbox_xyxy"], dtype=np.float64) if "bbox_xyxy" in npz.files else np.zeros((kps_xy_clean.shape[0], 4), dtype=np.float64)
        frame_idx = (
            np.asarray(npz["frame_idx"], dtype=np.int64)
            if "frame_idx" in npz.files
            else np.arange(kps_xy_clean.shape[0], dtype=np.int64)
        )
        fps = float(np.asarray(npz["fps"]).reshape(-1)[0]) if "fps" in npz.files else 0.0
        meta = {}
        if "meta" in npz.files:
            raw_meta = npz["meta"]
            try:
                if isinstance(raw_meta, np.ndarray) and raw_meta.size > 0:
                    candidate = raw_meta.reshape(-1)[0]
                    if isinstance(candidate, dict):
                        meta = candidate
            except Exception:
                meta = {}
        cleaning_diagnostics = {}
        if "cleaning_diagnostics" in npz.files:
            raw_cd = npz["cleaning_diagnostics"]
            try:
                if isinstance(raw_cd, np.ndarray) and raw_cd.size > 0:
                    candidate = raw_cd.reshape(-1)[0]
                    if isinstance(candidate, dict):
                        cleaning_diagnostics = candidate
            except Exception:
                cleaning_diagnostics = {}

    pose_clean = {
        "kps_xy": kps_xy_clean,
        "kps_score": kps_score_clean,
        "kps_xy_clean": kps_xy_clean,
        "kps_score_clean": kps_score_clean,
        "bbox_xyxy": bbox,
        "frame_idx": frame_idx,
        "fps": fps,
        "meta": {"video_id": video_id, **meta},
        "cleaning_diagnostics": cleaning_diagnostics,
    }
    return pose_clean


def _run_from_pose_clean(video_path: Path, pose_clean: dict[str, Any], *, source_note: str) -> dict[str, Any]:
    config = FullAnalysisConfig(video_path=video_path)
    kps_for_norm = np.asarray(
        pose_clean["kps_xy_clean"] if "kps_xy_clean" in pose_clean else pose_clean["kps_xy"],
        dtype=np.float64,
    )

    segmentation_result = run_rdl_segmentation_from_pose(
        pose_clean,
        config=config.segmentation,
    )
    seg_summary = segmentation_result.get("summary", {}) if isinstance(segmentation_result.get("summary"), dict) else {}
    num_valid_reps = int(seg_summary.get("num_reps_with_valid_anchors", seg_summary.get("num_reps", 0)) or 0)
    if num_valid_reps <= 0:
        pipeline_result = PipelineResult(
            status=pipeline_status.NO_REPS_DETECTED,
            ok=False,
            user_message=(
                "No se han detectado repeticiones completas. Graba una serie de 5 a 6 repeticiones "
                "completas, sin pausas largas y desde una vista lateral."
            ),
            artifacts={"video_path": str(video_path), "processed_video_path": str(video_path)},
            quality={},
            repetitions=[],
            feedback={},
            technical_warnings=["segmentation_no_valid_repetitions"],
            errors=[],
        ).to_dict()
        return FullAnalysisArtifacts(
            video_path=video_path,
            processed_video_path=video_path,
            pose_raw={},
            pose_clean=pose_clean,
            segmentation_result=segmentation_result,
            user_pose_sequence_normalization={},
            orientation={},
            pose_meta=pose_clean.get("meta", {}) if isinstance(pose_clean.get("meta"), dict) else {},
            cleaning_diagnostics=pose_clean.get("cleaning_diagnostics", {}) if isinstance(pose_clean.get("cleaning_diagnostics"), dict) else {},
            analysis_context={},
            detector_results={},
            feedback_evidence={},
            feedback_aggregation={},
            feedback_report={},
            pipeline_result=pipeline_result,
        ).to_runtime_dict()
    norm_result = normalize_pose_sequence(
        kps_for_norm,
        config=NormalizationConfig(sequence_scale_mode="fixed_median"),
    )
    user_pose_sequence_normalization = {
        "kps_xy_normalized": norm_result["kps_xy_normalized"],
        "mask_valid_normalized": norm_result["mask_valid_normalized"],
        "origins": norm_result["origins"],
        "scales": norm_result["scales"],
        "raw_scales": norm_result["raw_scales"],
        "frame_idx": np.asarray(pose_clean.get("frame_idx", np.arange(kps_for_norm.shape[0])), dtype=np.int64),
        "fps": float(pose_clean.get("fps", 0.0)),
        "pose_source": "clean_cached_npz",
        "meta": {
            "video_id": str(segmentation_result.get("video_id", video_path.stem)),
            "num_frames": int(kps_for_norm.shape[0]),
            "fps": float(pose_clean.get("fps", 0.0)),
            "normalization_method": str(norm_result.get("method", "pelvis_torso_scale")),
            "sequence_scale_mode": str(norm_result.get("sequence_scale_mode", "fixed_median")),
            "normalization_rotation_applied": bool(norm_result.get("rotation_applied", False)),
            "valid_frame_count": int(norm_result.get("valid_frame_count", 0)),
            "total_frame_count": int(norm_result.get("total_frame_count", kps_for_norm.shape[0])),
            "valid_frame_ratio": float(norm_result.get("valid_frame_ratio", 0.0)),
            "warnings": list(norm_result.get("warnings", [])),
            "source_note": source_note,
        },
    }

    analysis_context = build_rdl_analysis_context(
        user_pose_clean=pose_clean,
        user_segmentation_result=segmentation_result,
        user_pose_sequence_normalization=user_pose_sequence_normalization,
        reference_dir=config.reference_dir,
        ideal_valid_rep_index=config.ideal_valid_rep_index,
    )
    detector_results = run_rdl_detectors(analysis_context)
    detector_map = detector_results.get("detectors") if isinstance(detector_results, dict) and isinstance(detector_results.get("detectors"), dict) else detector_results
    feedback_evidence = normalize_rdl_detector_evidence(
        detector_results=detector_map if isinstance(detector_map, dict) else {},
        analysis_context=analysis_context,
    )
    feedback_aggregation = aggregate_rdl_feedback_evidence(
        feedback_evidence=feedback_evidence,
        analysis_context=analysis_context,
    )
    feedback_report = build_rdl_feedback_report(
        feedback_aggregation=feedback_aggregation,
        feedback_evidence=feedback_evidence,
        analysis_context=analysis_context,
    )

    orientation = {
        "facing": "cached_clean_npz",
        "confidence": None,
        "target_facing": config.orientation.target_facing,
        "flip_applied": False,
        "processed_video_path": str(video_path),
        "source_note": source_note,
    }
    return FullAnalysisArtifacts(
        video_path=video_path,
        processed_video_path=video_path,
        pose_raw={},
        pose_clean=pose_clean,
        segmentation_result=segmentation_result,
        user_pose_sequence_normalization=user_pose_sequence_normalization,
        orientation=orientation,
        pose_meta=pose_clean.get("meta", {}) if isinstance(pose_clean.get("meta"), dict) else {},
        cleaning_diagnostics=pose_clean.get("cleaning_diagnostics", {}) if isinstance(pose_clean.get("cleaning_diagnostics"), dict) else {},
        analysis_context=analysis_context,
        detector_results=detector_results,
        feedback_evidence=feedback_evidence,
        feedback_aggregation=feedback_aggregation,
        feedback_report=feedback_report,
    ).to_runtime_dict()


def _analyze_video(video_path: Path, *, reuse_clean_npz: bool) -> dict[str, Any]:
    clean_npz = OUTPUTS / "npz" / f"{video_path.stem}_rtmpose_clean.npz"
    if reuse_clean_npz and clean_npz.is_file():
        print(f"Reutilizando clean npz existente: {clean_npz.as_posix()}")
        pose_clean = _load_pose_clean_npz(clean_npz, video_path.stem)
        return _run_from_pose_clean(video_path, pose_clean, source_note="reused_outputs_npz_clean")
    return run_full_analysis(video_path)


def _save_segmentation_outputs(segmentation_result: dict[str, Any]) -> dict[str, Path]:
    video_id = str(segmentation_result.get("video_id", "unknown_video"))
    out_dir = OUTPUTS / "segmentation" / "rdl" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "segmentation_result.json"
    debug_npz_path = out_dir / "segmentation_debug.npz"
    save_segmentation_json(segmentation_result, json_path)
    save_segmentation_debug_npz(segmentation_result, debug_npz_path)

    paths: dict[str, Path] = {
        "out_dir": out_dir,
        "json_path": json_path,
        "debug_npz_path": debug_npz_path,
    }
    return paths


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "s", "si"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Entrada invalida. Responde y/n.")


def _save_pose_raw_outputs(artifacts: dict[str, Any]) -> Path | None:
    pose_raw = artifacts.get("pose_raw")
    if not isinstance(pose_raw, dict):
        return None
    video_id = str((artifacts.get("segmentation_result") or {}).get("video_id", "unknown_video"))
    out = OUTPUTS / "npz" / f"{video_id}_rtmpose.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if "kps_xy" not in pose_raw or "kps_score" not in pose_raw or "bbox_xyxy" not in pose_raw:
        return None
    frame_idx = np.asarray(pose_raw.get("frame_idx", np.arange(np.asarray(pose_raw["kps_xy"]).shape[0])), dtype=np.int64)
    np.savez(
        str(out),
        kps_xy=np.asarray(pose_raw["kps_xy"]),
        kps_score=np.asarray(pose_raw["kps_score"]),
        bbox_xyxy=np.asarray(pose_raw["bbox_xyxy"]),
        frame_idx=frame_idx,
        fps=np.float64(float(pose_raw.get("fps", 0.0))),
        meta=np.array([pose_raw.get("meta", {})], dtype=object),
    )
    return out


def _save_pose_clean_outputs(artifacts: dict[str, Any]) -> Path | None:
    pose_clean = artifacts.get("pose_clean")
    if not isinstance(pose_clean, dict):
        return None
    video_id = str((artifacts.get("segmentation_result") or {}).get("video_id", "unknown_video"))
    out = OUTPUTS / "npz" / f"{video_id}_rtmpose_clean.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if "kps_xy" not in pose_clean or "kps_score" not in pose_clean or "bbox_xyxy" not in pose_clean:
        return None
    frame_idx = np.asarray(
        pose_clean.get("frame_idx", np.arange(np.asarray(pose_clean["kps_xy"]).shape[0])),
        dtype=np.int64,
    )
    payload: dict[str, Any] = {
        "kps_xy": np.asarray(pose_clean["kps_xy"]),
        "kps_score": np.asarray(pose_clean["kps_score"]),
        "bbox_xyxy": np.asarray(pose_clean["bbox_xyxy"]),
        "frame_idx": frame_idx,
        "fps": np.float64(float(pose_clean.get("fps", 0.0))),
        "meta": np.array([pose_clean.get("meta", {})], dtype=object),
    }
    for key in ("kps_xy_clean", "kps_score_clean", "mask_valid", "mask_valid_frames", "mask_valid_right_chain"):
        if key in pose_clean:
            payload[key] = np.asarray(pose_clean[key])
    if "cleaning_diagnostics" in pose_clean:
        payload["cleaning_diagnostics"] = np.array([pose_clean["cleaning_diagnostics"]], dtype=object)
    np.savez(str(out), **payload)
    return out


def _save_normalization_outputs(artifacts: dict[str, Any]) -> tuple[Path | None, Path | None]:
    norm = artifacts.get("user_pose_sequence_normalization")
    if not isinstance(norm, dict):
        return None, None
    seg = artifacts.get("segmentation_result") if isinstance(artifacts.get("segmentation_result"), dict) else {}
    video_id = str(seg.get("video_id", "unknown_video"))
    out_dir = OUTPUTS / "normalization" / "rdl" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "user_pose_sequence_normalized.npz"
    meta_path = out_dir / "user_pose_sequence_normalized_meta.json"

    np.savez_compressed(
        npz_path,
        kps_xy_normalized=np.asarray(norm.get("kps_xy_normalized")),
        mask_valid_normalized=np.asarray(norm.get("mask_valid_normalized")),
        normalization_origins=np.asarray(norm.get("origins")),
        normalization_scales=np.asarray(norm.get("scales")),
        raw_scales=np.asarray(norm.get("raw_scales")),
        frame_idx=np.asarray(norm.get("frame_idx")),
        fps=np.float64(float(norm.get("fps", 0.0))),
        pose_source=np.asarray(norm.get("pose_source", "unknown"), dtype=object),
        normalization_method=np.asarray((norm.get("meta") or {}).get("normalization_method", "pelvis_torso_scale"), dtype=object),
        sequence_scale_mode=np.asarray((norm.get("meta") or {}).get("sequence_scale_mode", "fixed_median"), dtype=object),
    )
    import json

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(norm.get("meta", {}), f, ensure_ascii=False, indent=2)
    return npz_path, meta_path


def _print_pipeline_summary(artifacts: dict[str, Any]) -> None:
    orientation = artifacts.get("orientation", {}) if isinstance(artifacts.get("orientation"), dict) else {}
    pose_raw = artifacts.get("pose_raw", {}) if isinstance(artifacts.get("pose_raw"), dict) else {}
    pose_clean = artifacts.get("pose_clean", {}) if isinstance(artifacts.get("pose_clean"), dict) else {}
    seg = artifacts.get("segmentation_result", {}) if isinstance(artifacts.get("segmentation_result"), dict) else {}
    summary = seg.get("summary", {}) if isinstance(seg.get("summary"), dict) else {}
    norm = (
        artifacts.get("user_pose_sequence_normalization", {})
        if isinstance(artifacts.get("user_pose_sequence_normalization"), dict)
        else {}
    )
    norm_meta = norm.get("meta", {}) if isinstance(norm.get("meta"), dict) else {}
    analysis_context = (
        artifacts.get("analysis_context", {})
        if isinstance(artifacts.get("analysis_context"), dict)
        else {}
    )
    context_meta = analysis_context.get("context_meta", {}) if isinstance(analysis_context.get("context_meta"), dict) else {}
    detector_results = artifacts.get("detector_results", {}) if isinstance(artifacts.get("detector_results"), dict) else {}
    detector_summary = detector_results.get("summary", {}) if isinstance(detector_results.get("summary"), dict) else {}
    detectors = detector_results.get("detectors", {}) if isinstance(detector_results.get("detectors"), dict) else {}
    feedback_evidence = artifacts.get("feedback_evidence", {}) if isinstance(artifacts.get("feedback_evidence"), dict) else {}
    feedback_summary = feedback_evidence.get("summary", {}) if isinstance(feedback_evidence.get("summary"), dict) else {}
    feedback_aggregation = artifacts.get("feedback_aggregation", {}) if isinstance(artifacts.get("feedback_aggregation"), dict) else {}
    aggregation_summary = feedback_aggregation.get("summary", {}) if isinstance(feedback_aggregation.get("summary"), dict) else {}
    feedback_report = artifacts.get("feedback_report", {}) if isinstance(artifacts.get("feedback_report"), dict) else {}

    pipeline_result = artifacts.get("pipeline_result", {}) if isinstance(artifacts.get("pipeline_result"), dict) else {}
    pipeline_status_value = str(pipeline_result.get("status", pipeline_status.OK))
    pipeline_message = str(pipeline_result.get("user_message", ""))
    if pipeline_status_value == pipeline_status.OK:
        print("\nPipeline completado correctamente.")
    elif pipeline_status_value == pipeline_status.PARTIAL_ANALYSIS:
        print("\nPipeline completado parcialmente.")
    else:
        print("\nPipeline finalizado sin análisis completo.")
    if pipeline_message:
        print(f"pipeline.user_message={pipeline_message}")
    print(f"pipeline.status={pipeline_status_value}")
    print(f"\nvideo_path={artifacts.get('video_path', 'n/d')}")
    print(f"processed_video_path={artifacts.get('processed_video_path', 'n/d')}")
    pose_raw_kps = pose_raw.get("kps_xy")
    pose_clean_kps = pose_clean.get("kps_xy_clean")
    pose_raw_shape = list(pose_raw_kps.shape) if hasattr(pose_raw_kps, "shape") else pose_raw.get("kps_xy_shape", "n/d")
    pose_clean_shape = (
        list(pose_clean_kps.shape)
        if hasattr(pose_clean_kps, "shape")
        else pose_clean.get("kps_xy_clean_shape", "n/d")
    )
    print(f"pose_raw.kps_xy_shape={pose_raw_shape}")
    print(f"pose_clean.kps_xy_clean_shape={pose_clean_shape}")
    print(f"orientation.facing={orientation.get('facing', 'n/d')}")
    print(f"orientation.confidence={orientation.get('confidence', 'n/d')}")
    print(f"orientation.flip_applied={orientation.get('flip_applied', 'n/d')}")
    print(f"segmentation.exercise={seg.get('exercise', 'n/d')}")
    print(f"segmentation.pose_source={seg.get('pose_source', 'n/d')}")
    print(f"segmentation.has_clean_pose={seg.get('has_clean_pose', 'n/d')}")
    print(f"segmentation.segmentation_status={seg.get('segmentation_status', 'n/d')}")
    print(f"segmentation.summary.num_reps={summary.get('num_reps', 'n/d')}")
    print(
        "segmentation.summary.num_reps_with_valid_anchors="
        f"{summary.get('num_reps_with_valid_anchors', 'n/d')}"
    )
    print(
        "segmentation.summary.num_reps_with_invalid_anchors="
        f"{summary.get('num_reps_with_invalid_anchors', 'n/d')}"
    )
    print(f"segmentation.summary.anchor_method={summary.get('anchor_method', 'n/d')}")
    print(f"normalization.valid_frame_count={norm_meta.get('valid_frame_count', 'n/d')}")
    print(f"normalization.valid_frame_ratio={norm_meta.get('valid_frame_ratio', 'n/d')}")
    print(f"normalization.sequence_scale_mode={norm_meta.get('sequence_scale_mode', 'n/d')}")
    print("\nContexto RDL:")
    print(f"context.reference_name={context_meta.get('reference_name', 'n/d')}")
    print(f"context.reference_dir={context_meta.get('reference_dir', 'n/d')}")
    print(f"context.ideal_valid_rep_index={context_meta.get('ideal_valid_rep_index', 'n/d')}")
    print(f"context.ideal_rep_raw_index={context_meta.get('ideal_rep_raw_index', 'n/d')}")
    print(f"context.num_user_reps={context_meta.get('num_user_reps', 'n/d')}")
    print(f"context.num_paired_repetitions={context_meta.get('num_paired_repetitions', 'n/d')}")
    print(f"context.anchor_names={context_meta.get('anchor_names', [])}")
    print(f"context.warnings={context_meta.get('warnings', [])}")
    print("\nDetectores RDL:")
    print(f"detectors.num_detectors={detector_summary.get('num_detectors', 0)}")
    print(f"detectors.num_detected={detector_summary.get('num_detected', 0)}")
    print(f"detectors.detected_errors={detector_summary.get('detected_errors', [])}")
    print(f"detectors.max_severity={detector_summary.get('max_severity', 'none')}")
    for detector_name in ("bent_arms", "asymmetry", "bar_far", "hip_hinge", "knee_dominant", "lockout", "neck_movement", "rom", "spine_flexion"):
        det = detectors.get(detector_name, {}) if isinstance(detectors.get(detector_name), dict) else {}
        print(
            f"detector.{detector_name}="
            f"detected:{det.get('detected', False)} "
            f"severity:{det.get('severity', 'none')} "
            f"score:{det.get('score', 0.0)} "
            f"reps:{det.get('num_reps_detected', 0)}/{det.get('num_reps_analyzed', 0)}"
        )
    print("\nFeedback evidence:")
    print(f"feedback.num_items={feedback_summary.get('num_items', 0)}")
    print(f"feedback.detectors_with_evidence={feedback_summary.get('detectors_with_evidence', [])}")
    print(f"feedback.reps_with_evidence={feedback_summary.get('reps_with_evidence', [])}")
    print(f"feedback.phases_with_evidence={feedback_summary.get('phases_with_evidence', [])}")
    print(f"feedback.max_severity={feedback_summary.get('max_severity', 'none')}")
    print("\nFeedback aggregation:")
    print(f"aggregation.status={feedback_aggregation.get('status', 'n/d')}")
    print(f"aggregation.num_issues={aggregation_summary.get('num_issues', 0)}")
    print(f"aggregation.max_severity={aggregation_summary.get('max_severity', 'none')}")
    print(f"aggregation.primary_error_codes={aggregation_summary.get('primary_error_codes', [])}")
    print(f"aggregation.secondary_error_codes={aggregation_summary.get('secondary_error_codes', [])}")
    print(f"aggregation.affected_reps={aggregation_summary.get('affected_reps', [])}")
    print(f"aggregation.affected_phases={aggregation_summary.get('affected_phases', [])}")
    print("\nFeedback report:")
    print(f"report.headline={feedback_report.get('headline', 'n/d')}")
    print(f"report.num_main_feedback={len(feedback_report.get('main_feedback', [])) if isinstance(feedback_report.get('main_feedback'), list) else 0}")
    print(f"report.num_secondary_feedback={len(feedback_report.get('secondary_feedback', [])) if isinstance(feedback_report.get('secondary_feedback'), list) else 0}")
    print(f"report.num_observations={len(feedback_report.get('observations', [])) if isinstance(feedback_report.get('observations'), list) else 0}")


def _export_detector_debug_outputs(
    *,
    analysis_context: dict[str, Any],
    detector_results: dict[str, Any],
    video_id: str,
) -> None:
    out_root = OUTPUTS / "debug_detectors" / "rdl" / "repeticiones"
    exporters: tuple[tuple[str, Any], ...] = (
        ("bent_arms", export_bent_arms_debug),
        ("asymmetry", export_asymmetry_debug),
        ("bar_far", export_bar_far_debug),
        ("hip_hinge", export_hip_hinge_debug),
        ("knee_dominant", export_knee_dominant_debug),
        ("lockout", export_lockout_debug),
        ("neck_movement", export_neck_movement_debug),
        ("rom", export_rom_debug),
        ("spine_flexion", export_spine_flexion_debug),
    )
    print("\nExportando debug_detectors...")
    for detector_name, exporter in exporters:
        result = detector_results.get(detector_name)
        if not isinstance(result, dict):
            print(f"- {detector_name}: sin resultado, omitido")
            continue
        detector_out_dir = out_root / video_id / detector_name
        try:
            if detector_name == "bent_arms":
                payload = exporter(
                    analysis_context=analysis_context,
                    bent_arms_result=result,
                    output_dir=detector_out_dir,
                )
            elif detector_name == "neck_movement":
                payload = exporter(
                    analysis_context=analysis_context,
                    neck_movement_result=result,
                    output_dir=detector_out_dir,
                )
            elif detector_name == "rom":
                payload = exporter(
                    analysis_context=analysis_context,
                    rom_result=result,
                    output_dir=detector_out_dir,
                )
            elif detector_name == "spine_flexion":
                payload = exporter(
                    analysis_context=analysis_context,
                    spine_flexion_result=result,
                    output_dir=detector_out_dir,
                )
            elif detector_name == "asymmetry":
                payload = exporter(
                    analysis_context=analysis_context,
                    asymmetry_result=result,
                    output_dir=detector_out_dir,
                )
            elif detector_name == "bar_far":
                payload = exporter(
                    analysis_context=analysis_context,
                    bar_far_result=result,
                    output_dir=detector_out_dir,
                )
            elif detector_name == "hip_hinge":
                payload = exporter(
                    analysis_context=analysis_context,
                    hip_hinge_result=result,
                    output_dir=detector_out_dir,
                )
            elif detector_name == "knee_dominant":
                payload = exporter(
                    analysis_context=analysis_context,
                    knee_dominant_result=result,
                    output_dir=detector_out_dir,
                )
            elif detector_name == "lockout":
                payload = exporter(
                    analysis_context=analysis_context,
                    lockout_result=result,
                    output_dir=detector_out_dir,
                )
            else:
                payload = exporter(
                    analysis_context=analysis_context,
                    detector_result=result,  # reserva (sin uso actual)
                    output_dir=detector_out_dir,
                )
            print(f"- {detector_name}: ok -> {payload.get('output_dir', detector_out_dir.as_posix())}")
        except Exception as exc:
            print(f"- {detector_name}: ERROR {exc}")


def _run_single_video_pipeline(
    video_path: Path,
    *,
    index: int = 1,
    total: int = 1,
    frames_remaining_before: int = 0,
    export_detectors_opt: bool | None = None,
    save_raw_opt: bool | None = None,
    save_clean_opt: bool | None = None,
    save_seg_opt: bool | None = None,
    save_norm_opt: bool | None = None,
    reuse_clean_npz: bool = True,
) -> None:
    print(
        f"\n=== Video {index}/{total} ===\n"
        f"video={video_path.as_posix()}\n"
        f"frames_pendientes_antes={frames_remaining_before}"
    )
    try:
        artifacts = _analyze_video(video_path, reuse_clean_npz=reuse_clean_npz)
    except Exception as exc:
        print(f"\n[ERROR] Fallo durante la ejecucion del pipeline: {exc}")
        return

    _print_pipeline_summary(artifacts)
    pipeline_result = artifacts.get("pipeline_result", {}) if isinstance(artifacts.get("pipeline_result"), dict) else {}
    pipeline_status_value = str(pipeline_result.get("status", pipeline_status.OK))
    if pipeline_status_value not in {pipeline_status.OK, pipeline_status.PARTIAL_ANALYSIS}:
        print(
            f"\n[INFO] Se omiten exportaciones/análisis adicional porque pipeline.status={pipeline_status_value}."
        )
        return

    segmentation_result = artifacts.get("segmentation_result")
    if not isinstance(segmentation_result, dict):
        print("\n[ERROR] segmentation_result ausente o invalido en artifacts.")
        return

    analysis_context = artifacts.get("analysis_context") if isinstance(artifacts.get("analysis_context"), dict) else {}
    ctx_user = analysis_context.get("user") if isinstance(analysis_context.get("user"), dict) else {}
    seg = artifacts.get("segmentation_result") if isinstance(artifacts.get("segmentation_result"), dict) else {}
    video_id = (
        str(ctx_user.get("video_id"))
        if ctx_user.get("video_id")
        else str(seg.get("video_id"))
        if seg.get("video_id")
        else Path(video_path).stem
    )
    debug_output_dir = OUTPUTS / "debug_runs" / "rdl" / video_id
    debug_bundle = export_rdl_debug_bundle(
        artifacts=artifacts,
        output_dir=debug_output_dir,
    )
    print("\nBundle debug generado:")
    print(f"- carpeta: {debug_bundle['output_dir']}")
    print("- archivos:")
    for filename in debug_bundle.get("files", []):
        print(f"  - {filename}")
    if "13_feedback_report.txt" in debug_bundle.get("files", []):
        print(f"- feedback_report_txt: {Path(debug_bundle['output_dir']) / '13_feedback_report.txt'}")
    if debug_bundle.get("warnings"):
        print(f"- warnings: {debug_bundle['warnings']}")

    detector_results = artifacts.get("detector_results", {}) if isinstance(artifacts.get("detector_results"), dict) else {}
    detectors = detector_results.get("detectors", {}) if isinstance(detector_results.get("detectors"), dict) else {}
    export_detectors = export_detectors_opt if export_detectors_opt is not None else _ask_yes_no(
        "Exportar imagenes/resumen de TODOS los detectores en outputs/debug_detectors?",
        default=True,
    )
    if export_detectors:
        _export_detector_debug_outputs(
            analysis_context=analysis_context,
            detector_results=detectors,
            video_id=video_id,
        )

    print("\nExportaciones adicionales opcionales:")
    print("- raw NPZ")
    print("- clean NPZ")
    print("- segmentación JSON/debug NPZ")
    print("- normalización NPZ/meta JSON")

    save_raw = save_raw_opt if save_raw_opt is not None else _ask_yes_no("Guardar pose RAW NPZ?", default=False)
    save_clean = save_clean_opt if save_clean_opt is not None else _ask_yes_no("Guardar pose CLEAN NPZ?", default=False)
    save_seg = save_seg_opt if save_seg_opt is not None else _ask_yes_no(
        "Guardar segmentación adicional (JSON + debug NPZ)?",
        default=False,
    )
    save_norm = save_norm_opt if save_norm_opt is not None else _ask_yes_no(
        "Guardar normalización adicional (NPZ + meta JSON)?",
        default=False,
    )

    if not any([save_raw, save_clean, save_seg, save_norm]):
        print("\nSin exportaciones adicionales: no se guardaron archivos pesados extra.")
        return

    if save_raw:
        raw_path = _save_pose_raw_outputs(artifacts)
        print(f"- raw_npz: {raw_path.as_posix() if raw_path else 'n/d'}")
    if save_clean:
        clean_path = _save_pose_clean_outputs(artifacts)
        print(f"- clean_npz: {clean_path.as_posix() if clean_path else 'n/d'}")
    if save_seg:
        try:
            output_paths = _save_segmentation_outputs(segmentation_result)
            print(f"- segmentation_json: {output_paths['json_path'].as_posix()}")
            print(f"- segmentation_debug_npz: {output_paths['debug_npz_path'].as_posix()}")
        except Exception as exc:
            print(f"\n[ERROR] Fallo al guardar artefactos de segmentacion: {exc}")
            return
    if save_norm:
        norm_npz, norm_meta = _save_normalization_outputs(artifacts)
        print(f"- normalization_npz: {norm_npz.as_posix() if norm_npz else 'n/d'}")
        print(f"- normalization_meta: {norm_meta.as_posix() if norm_meta else 'n/d'}")


def _run_full_pipeline(args: argparse.Namespace) -> None:
    try:
        if bool(args.all) and args.video is not None:
            raise ValueError("No uses --all y --video a la vez.")
        if args.all:
            video_paths = _list_available_videos()
            if not video_paths:
                raise FileNotFoundError("No se encontraron videos en data/")
        elif args.video is not None:
            video_path = Path(args.video).expanduser()
            if not video_path.is_file():
                raise FileNotFoundError(f"Video no encontrado: {video_path}")
            video_paths = [video_path.resolve()]
        else:
            video_paths = _ask_video_paths()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return

    total_videos = len(video_paths)
    frame_counts = [_video_frame_count(v) for v in video_paths]
    total_frames = int(sum(frame_counts))
    print(
        f"\nBatch seleccionado: {total_videos} video(s), "
        f"frames_totales_estimados={total_frames}"
    )

    frames_done = 0
    for idx, (video_path, n_frames) in enumerate(zip(video_paths, frame_counts), start=1):
        frames_remaining_before = max(total_frames - frames_done, 0)
        _run_single_video_pipeline(
            video_path,
            index=idx,
            total=total_videos,
            frames_remaining_before=frames_remaining_before,
            export_detectors_opt=args.export_detectors,
            save_raw_opt=args.save_raw,
            save_clean_opt=args.save_clean,
            save_seg_opt=args.save_seg,
            save_norm_opt=args.save_norm,
            reuse_clean_npz=bool(args.reuse_clean_npz),
        )
        frames_done += max(int(n_frames), 0)
        frames_remaining_after = max(total_frames - frames_done, 0)
        print(
            f"frames_video={n_frames} "
            f"frames_procesados_aprox={frames_done}/{total_frames} "
            f"frames_pendientes_aprox={frames_remaining_after}"
        )


def main() -> None:
    print("\n=== Pipeline TFG ===")
    args = _parse_args()
    _run_full_pipeline(args)


if __name__ == "__main__":
    main()
