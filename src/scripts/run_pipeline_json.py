
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.biomechanics.normalization import NormalizationConfig, normalize_pose_sequence
from src.biomechanics.rdl import build_rdl_analysis_context, run_rdl_segmentation_from_pose
from src.biomechanics.rdl.detectors.pipeline import run_rdl_detectors
from src.pipeline.full_analysis import FullAnalysisConfig, run_full_analysis
from src.pipeline.results import PipelineResult
from src.pipeline import status as pipeline_status
from src.biomechanics.rdl.feedback.evidence_normalizer.normalizer import normalize_rdl_detector_evidence
from src.biomechanics.rdl.feedback.aggregation.aggregator import aggregate_rdl_feedback_evidence
from src.biomechanics.rdl.feedback.builder.builder import build_rdl_feedback_report

KNOWN_STATUSES = {
    "ok",
    "partial_analysis",
    "no_person_detected",
    "insufficient_pose_quality",
    "unknown_orientation",
    "wrong_exercise",
    "no_reps_detected",
    "invalid_segmentation",
    "invalid_anchors",
    "failed",
}


def _empty_result(status: str, ok: bool, user_message: str, video_path: str) -> dict[str, Any]:
    return {
        "status": status,
        "ok": ok,
        "user_message": user_message,
        "artifacts": {"video_path": video_path},
        "quality": {},
        "repetitions": [],
        "feedback": {},
        "technical_warnings": [],
        "errors": [],
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_pose_clean_npz(npz_path: Path) -> dict[str, Any]:
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

    video_id = npz_path.stem.replace("_rtmpose_clean", "").replace("_rtmpose", "")
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


def _run_from_npz(npz_path: Path, config: FullAnalysisConfig) -> dict[str, Any]:
    print("[pipeline] Cargando NPZ clean...", flush=True)
    pose_clean = _load_pose_clean_npz(npz_path)
    kps_for_norm = np.asarray(pose_clean["kps_xy_clean"], dtype=np.float64)

    print("[pipeline] Segmentando repeticiones RDL...", flush=True)
    segmentation_result = run_rdl_segmentation_from_pose(pose_clean, config=config.segmentation)
    seg_summary = segmentation_result.get("summary", {}) if isinstance(segmentation_result.get("summary"), dict) else {}
    num_valid_reps = int(seg_summary.get("num_reps_with_valid_anchors", seg_summary.get("num_reps", 0)) or 0)
    print(f"[pipeline] Segmentacion lista: reps_validas={num_valid_reps}", flush=True)

    if num_valid_reps <= 0:
        pipeline_result = PipelineResult(
            status=pipeline_status.NO_REPS_DETECTED,
            ok=False,
            user_message=(
                "No se han detectado repeticiones completas. Graba una serie de 5 a 6 repeticiones "
                "completas, sin pausas largas y desde una vista lateral."
            ),
            artifacts={"npz_path": str(npz_path)},
            quality={},
            repetitions=[],
            feedback={},
            technical_warnings=[],
            errors=[],
        ).to_dict()
        return {
            "npz_path": npz_path,
            "pose_clean": pose_clean,
            "segmentation_result": segmentation_result,
            "pipeline_result": pipeline_result,
        }

    print("[pipeline] Normalizando secuencia de pose...", flush=True)
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
        "pose_source": "npz_clean",
        "meta": {
            "video_id": str(segmentation_result.get("video_id", npz_path.stem)),
            "num_frames": int(kps_for_norm.shape[0]),
            "fps": float(pose_clean.get("fps", 0.0)),
            "normalization_method": str(norm_result.get("method", "pelvis_torso_scale")),
            "sequence_scale_mode": str(norm_result.get("sequence_scale_mode", "fixed_median")),
            "normalization_rotation_applied": bool(norm_result.get("rotation_applied", False)),
            "valid_frame_count": int(norm_result.get("valid_frame_count", 0)),
            "total_frame_count": int(norm_result.get("total_frame_count", kps_for_norm.shape[0])),
            "valid_frame_ratio": float(norm_result.get("valid_frame_ratio", 0.0)),
            "warnings": list(norm_result.get("warnings", [])),
            "source_note": "ui_npz",
        },
    }

    print("[pipeline] Construyendo contexto de analisis...", flush=True)
    analysis_context = build_rdl_analysis_context(
        user_pose_clean=pose_clean,
        user_segmentation_result=segmentation_result,
        user_pose_sequence_normalization=user_pose_sequence_normalization,
        reference_dir=config.reference_dir,
        ideal_valid_rep_index=config.ideal_valid_rep_index,
    )

    print("[pipeline] Ejecutando detectores...", flush=True)
    detector_results = run_rdl_detectors(analysis_context)
    detector_map = (
        detector_results.get("detectors")
        if isinstance(detector_results, dict) and isinstance(detector_results.get("detectors"), dict)
        else detector_results
    )
    print("[pipeline] Generando feedback...", flush=True)
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

    pipeline_result = PipelineResult(
        status=pipeline_status.OK,
        ok=True,
        user_message="Análisis completado correctamente.",
        artifacts={"npz_path": str(npz_path)},
        quality={},
        repetitions=list(segmentation_result.get("reps", [])),
        feedback=feedback_report,
        technical_warnings=[],
        errors=[],
    ).to_dict()

    return {
        "npz_path": npz_path,
        "pose_clean": pose_clean,
        "segmentation_result": segmentation_result,
        "analysis_context": analysis_context,
        "detector_results": detector_results,
        "feedback_evidence": feedback_evidence,
        "feedback_aggregation": feedback_aggregation,
        "feedback_report": feedback_report,
        "pipeline_result": pipeline_result,
    }


def _extract_pipeline_payload(runtime: dict[str, Any], source_path: str) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        return _empty_result(status="failed", ok=False, user_message="Salida invalida del pipeline.", video_path=source_path)
    pipeline_result = runtime.get("pipeline_result")
    if not isinstance(pipeline_result, dict):
        return _empty_result(status="failed", ok=False, user_message="Resultado de pipeline ausente.", video_path=source_path)
    return pipeline_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RDL pipeline for UI")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--npz", type=str, default=None, help="Path to clean NPZ file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--exercise", type=str, default="RDL", help="Exercise type")
    args = parser.parse_args()

    output_path = Path(args.output)

    try:
        print("[pipeline] Proceso Python del pipeline en marcha.", flush=True)
        if args.npz:
            print(f"[pipeline] Ejecutando desde NPZ: {args.npz}", flush=True)
            npz_path = Path(args.npz)
            if not npz_path.is_file():
                payload = _empty_result(
                    status="failed",
                    ok=False,
                    user_message=f"Fichero NPZ no encontrado: {args.npz}",
                    video_path=str(args.npz),
                )
                _write_json(output_path, payload)
                return 1

            config = FullAnalysisConfig(video_path=npz_path)
            runtime = _run_from_npz(npz_path, config)
            payload = _extract_pipeline_payload(runtime, str(args.npz))
            _write_json(output_path, payload)
            print(f"[pipeline] Estado final: {payload.get('status', 'unknown')}", flush=True)
            return 0 if payload.get("status") in KNOWN_STATUSES else 1

        elif args.video:
            print(f"[pipeline] Ejecutando desde vídeo: {args.video}", flush=True)
            print("[pipeline] Fase pose: YOLO + RTMPose (puede tardar varios minutos)...", flush=True)
            config = FullAnalysisConfig(video_path=Path(args.video))
            runtime = run_full_analysis(config)
            print("[pipeline] Pipeline de vídeo finalizado.", flush=True)
            payload = _extract_pipeline_payload(runtime, str(args.video))
            _write_json(output_path, payload)
            print(f"[pipeline] Estado final: {payload.get('status', 'unknown')}", flush=True)
            return 0 if payload.get("status") in KNOWN_STATUSES else 1

        else:
            payload = _empty_result(
                status="failed",
                ok=False,
                user_message="Debe proporcionar --video o --npz",
                video_path="",
            )
            _write_json(output_path, payload)
            return 1

    except Exception as exc:
        print(f"[pipeline] Error inesperado: {type(exc).__name__}: {exc}", flush=True)
        source = args.npz if args.npz else args.video if args.video else ""
        payload = _empty_result(
            status="failed",
            ok=False,
            user_message="Ha ocurrido un error interno durante el analisis.",
            video_path=source,
        )
        payload["errors"] = [
            {
                "stage": "run_pipeline_json",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        ]
        try:
            _write_json(output_path, payload)
        finally:
            return 1


if __name__ == "__main__":
    sys.exit(main())