
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.biomechanics.normalization import NormalizationConfig, normalize_pose_sequence
from src.biomechanics.rdl.feedback.aggregation import aggregate_rdl_feedback_evidence
from src.biomechanics.rdl.feedback.builder import build_rdl_feedback_report
from src.biomechanics.rdl.feedback.evidence_normalizer import normalize_rdl_detector_evidence
from src.biomechanics.rdl.detectors import run_rdl_detectors
from . import status
from .contracts import (
    validate_anchors_contract,
    validate_pose_clean_contract_data,
    validate_pose_raw_contract_data,
    validate_segmentation_contract,
)
from .config import FullAnalysisConfig
from .results import FullAnalysisArtifacts, PipelineResult
from .validation import validate_video_input
from src.biomechanics.rdl import build_rdl_analysis_context, run_rdl_segmentation_from_pose
from src.pose.extraction import extract_video_pose
from src.pose.orientation import estimate_subject_facing_from_pose
from src.pose_cleaning import clean_pose_data
from src.preprocessing.video_orientation import ensure_video_facing


def run_full_analysis(config_or_video_path: FullAnalysisConfig | str | Path) -> dict[str, Any]:
    # 1) Normalizar entrada a config tipada.
    config = _coerce_config(config_or_video_path)
    video_path_obj = config.video_path
    technical_warnings: list[str] = []
    stage_errors: list[dict[str, Any]] = []

    # 2) Validaciones básicas de entrada.
    in_val = validate_video_input(video_path_obj)
    technical_warnings.extend(in_val.technical_warnings)
    if not in_val.ok:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=in_val.status,
            user_message=in_val.user_message,
            technical_warnings=technical_warnings,
            errors=stage_errors,
        )

    # 3) Pose preliminar sobre vídeo original (en memoria).
    try:
        pose_probe = extract_video_pose(
            video_path=str(video_path_obj),
            verbose=config.pose.verbose,
            config_path=str(config.pose.config_path),
            checkpoint_path=str(config.pose.checkpoint_path),
            yolo_weights=str(config.pose.yolo_weights),
        )
    except Exception as exc:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=status.FAILED,
            user_message="Ha ocurrido un error interno durante el análisis.",
            technical_warnings=technical_warnings,
            errors=[{"stage": "pose_estimation", "type": type(exc).__name__, "message": str(exc)}],
        )
    pose_probe["video_id"] = video_path_obj.stem
    raw_val = validate_pose_raw_contract_data(pose_probe)
    technical_warnings.extend(raw_val.technical_warnings)
    if not raw_val.ok:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=raw_val.status,
            user_message=raw_val.user_message
            if raw_val.status != status.NO_PERSON_DETECTED
            else "No se ha detectado una persona de forma fiable en el vídeo. Graba el ejercicio con el cuerpo completo visible.",
            technical_warnings=technical_warnings,
            errors=stage_errors,
        )

    # 4) Estimar orientación lateral desde keypoints del probe (en memoria).
    try:
        orientation = estimate_subject_facing_from_pose(
            pose_probe,
            score_threshold=config.orientation.score_threshold,
            min_valid_frames=config.orientation.min_valid_frames,
            margin=config.orientation.margin,
        )
    except Exception as exc:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=status.UNKNOWN_ORIENTATION,
            user_message="No se ha podido determinar de forma fiable la orientación del sujeto en el vídeo.",
            technical_warnings=technical_warnings + ["orientation_estimation_failed"],
            errors=[{"stage": "orientation", "type": type(exc).__name__, "message": str(exc)}],
        )
    facing = str(orientation.get("facing", "")).lower()
    if facing in {"", "unknown", "none"}:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=status.UNKNOWN_ORIENTATION,
            user_message="No se ha podido determinar de forma fiable la orientación del sujeto en el vídeo.",
            technical_warnings=technical_warnings + ["unknown_orientation"],
            errors=stage_errors,
        )

    # 5) Aplicar flip si la orientación detectada lo requiere.
    try:
        processed_video, flip_applied = ensure_video_facing(
            video_path_obj,
            detected_facing=orientation["facing"],
            target_facing=config.orientation.target_facing,
        )
    except Exception as exc:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=status.UNKNOWN_ORIENTATION,
            user_message="No se ha podido determinar de forma fiable la orientación del sujeto en el vídeo.",
            technical_warnings=technical_warnings + ["video_orientation_preprocess_failed"],
            errors=[{"stage": "orientation_preprocess", "type": type(exc).__name__, "message": str(exc)}],
        )

    # 6) Pose raw final (en memoria).
    if not flip_applied:
        pose_raw = dict(pose_probe)
        pose_meta = pose_probe.get("meta", {})
    else:
        try:
            pose_raw = extract_video_pose(
                video_path=str(processed_video),
                verbose=config.pose.verbose,
                config_path=str(config.pose.config_path),
                checkpoint_path=str(config.pose.checkpoint_path),
                yolo_weights=str(config.pose.yolo_weights),
            )
        except Exception as exc:
            return _terminal_runtime(
                video_path=video_path_obj,
                st=status.FAILED,
                user_message="Ha ocurrido un error interno durante el análisis.",
                technical_warnings=technical_warnings,
                errors=[{"stage": "pose_estimation_final", "type": type(exc).__name__, "message": str(exc)}],
            )
        pose_raw["video_id"] = video_path_obj.stem
        pose_meta = pose_raw.get("meta", {})
        raw_val2 = validate_pose_raw_contract_data(pose_raw)
        technical_warnings.extend(raw_val2.technical_warnings)
        if not raw_val2.ok:
            return _terminal_runtime(
                video_path=video_path_obj,
                st=raw_val2.status,
                user_message="No se ha detectado una persona de forma fiable en el vídeo. Graba el ejercicio con el cuerpo completo visible.",
                technical_warnings=technical_warnings,
                errors=stage_errors,
            )

    # 7) Pose cleaning: transformar raw -> clean (en memoria).
    try:
        pose_clean = clean_pose_data(pose_raw)
    except Exception as exc:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=status.FAILED,
            user_message="Ha ocurrido un error interno durante el análisis.",
            technical_warnings=technical_warnings,
            errors=[{"stage": "pose_cleaning", "type": type(exc).__name__, "message": str(exc)}],
        )
    cleaning_diagnostics = pose_clean.get("cleaning_diagnostics", {})
    clean_val = validate_pose_clean_contract_data(pose_clean)
    technical_warnings.extend(clean_val.technical_warnings)
    if not clean_val.ok:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=clean_val.status,
            user_message=clean_val.user_message,
            technical_warnings=technical_warnings,
            errors=stage_errors,
        )

    orientation = {
        **orientation,
        "target_facing": config.orientation.target_facing,
        "flip_applied": bool(flip_applied),
        "processed_video_path": str(processed_video),
    }

    # 8) Segmentación RDL sobre pose clean en memoria.
    try:
        segmentation_result = run_rdl_segmentation_from_pose(
            pose_clean,
            config=config.segmentation,
        )
    except Exception as exc:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=status.INVALID_SEGMENTATION,
            user_message="El movimiento no se ha podido segmentar correctamente en repeticiones válidas.",
            technical_warnings=technical_warnings + ["segmentation_runtime_failed"],
            errors=[{"stage": "segmentation", "type": type(exc).__name__, "message": str(exc)}],
        )
    seg_val = validate_segmentation_contract(segmentation_result)
    technical_warnings.extend(seg_val.technical_warnings)
    if not seg_val.ok:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=seg_val.status,
            user_message=seg_val.user_message,
            technical_warnings=technical_warnings,
            errors=stage_errors,
            segmentation_result=segmentation_result,
            pose_raw=pose_raw,
            pose_clean=pose_clean,
            orientation=orientation,
            pose_meta=pose_meta,
            cleaning_diagnostics=cleaning_diagnostics,
        )

    # 9) Normalización completa de secuencia en memoria.
    pose_source = "clean" if "kps_xy_clean" in pose_clean else "raw_fallback"
    kps_for_norm = np.asarray(
        pose_clean["kps_xy_clean"] if "kps_xy_clean" in pose_clean else pose_clean["kps_xy"],
        dtype=np.float64,
    )
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
        "pose_source": pose_source,
        "meta": {
            "video_id": str(segmentation_result.get("video_id", video_path_obj.stem)),
            "num_frames": int(kps_for_norm.shape[0]),
            "fps": float(pose_clean.get("fps", 0.0)),
            "normalization_method": str(norm_result.get("method", "pelvis_torso_scale")),
            "sequence_scale_mode": str(norm_result.get("sequence_scale_mode", "fixed_median")),
            "normalization_rotation_applied": bool(norm_result.get("rotation_applied", False)),
            "valid_frame_count": int(norm_result.get("valid_frame_count", 0)),
            "total_frame_count": int(norm_result.get("total_frame_count", kps_for_norm.shape[0])),
            "valid_frame_ratio": float(norm_result.get("valid_frame_ratio", 0.0)),
            "warnings": list(norm_result.get("warnings", [])),
        },
    }

    # 10) Construir contexto RDL usuario + referencia ideal en memoria.
    try:
        analysis_context = build_rdl_analysis_context(
            user_pose_clean=pose_clean,
            user_segmentation_result=segmentation_result,
            user_pose_sequence_normalization=user_pose_sequence_normalization,
            reference_dir=config.reference_dir,
            ideal_valid_rep_index=config.ideal_valid_rep_index,
        )
    except Exception as exc:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=status.FAILED,
            user_message="Ha ocurrido un error interno durante el análisis.",
            technical_warnings=technical_warnings + ["analysis_context_build_failed"],
            errors=[{"stage": "analysis_context", "type": type(exc).__name__, "message": str(exc)}],
            segmentation_result=segmentation_result,
            pose_raw=pose_raw,
            pose_clean=pose_clean,
            user_pose_sequence_normalization=user_pose_sequence_normalization,
            orientation=orientation,
            pose_meta=pose_meta,
            cleaning_diagnostics=cleaning_diagnostics,
        )
    anchors_val = validate_anchors_contract(analysis_context)
    technical_warnings.extend(anchors_val.technical_warnings)
    if not anchors_val.ok:
        return _terminal_runtime(
            video_path=video_path_obj,
            st=anchors_val.status,
            user_message=anchors_val.user_message,
            technical_warnings=technical_warnings,
            errors=stage_errors,
            segmentation_result=segmentation_result,
            pose_raw=pose_raw,
            pose_clean=pose_clean,
            user_pose_sequence_normalization=user_pose_sequence_normalization,
            orientation=orientation,
            pose_meta=pose_meta,
            cleaning_diagnostics=cleaning_diagnostics,
            analysis_context=analysis_context,
        )

    # 11) Ejecutar detectores RDL sobre analysis_context en memoria.
    detector_results = run_rdl_detectors(analysis_context)
    detector_map = detector_results.get("detectors") if isinstance(detector_results, dict) and isinstance(detector_results.get("detectors"), dict) else detector_results
    partial_warnings = _collect_detector_partial_warnings(detector_map if isinstance(detector_map, dict) else {})
    technical_warnings.extend(partial_warnings)
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

    final_status = status.PARTIAL_ANALYSIS if partial_warnings else status.OK
    user_message = (
        "El análisis se ha completado parcialmente. Algunos aspectos no se han evaluado por falta de datos fiables."
        if final_status == status.PARTIAL_ANALYSIS
        else "Análisis completado correctamente."
    )
    pipeline_result = PipelineResult(
        status=final_status,
        ok=final_status == status.OK,
        user_message=user_message,
        artifacts={
            "video_path": str(video_path_obj),
            "processed_video_path": str(processed_video),
            "video_id": str(segmentation_result.get("video_id", video_path_obj.stem)),
        },
        quality={
            "pose_source": pose_source,
            "normalization_valid_frame_ratio": float(
                (user_pose_sequence_normalization.get("meta") or {}).get("valid_frame_ratio", 0.0)
            ),
        },
        repetitions=list(segmentation_result.get("reps", [])) if isinstance(segmentation_result.get("reps"), list) else [],
        feedback=feedback_report if isinstance(feedback_report, dict) else {},
        technical_warnings=sorted(set(technical_warnings)),
        errors=stage_errors,
    ).to_dict()

    # 12) Devolver artefactos estructurados para CLI/tests/UI.
    return FullAnalysisArtifacts(
        video_path=video_path_obj,
        processed_video_path=Path(processed_video),
        pose_raw=pose_raw,
        pose_clean=pose_clean,
        segmentation_result=segmentation_result,
        user_pose_sequence_normalization=user_pose_sequence_normalization,
        orientation=orientation,
        pose_meta=pose_meta,
        cleaning_diagnostics=cleaning_diagnostics,
        analysis_context=analysis_context,
        detector_results=detector_results,
        feedback_evidence=feedback_evidence,
        feedback_aggregation=feedback_aggregation,
        feedback_report=feedback_report,
        pipeline_result=pipeline_result,
    ).to_runtime_dict()


def _coerce_config(config_or_video_path: FullAnalysisConfig | str | Path) -> FullAnalysisConfig:
    # Mantiene compatibilidad: permite llamar con str/Path.
    if isinstance(config_or_video_path, FullAnalysisConfig):
        return config_or_video_path
    return FullAnalysisConfig(video_path=Path(config_or_video_path))


def _load_cleaning_diagnostics(clean_npz_path: Path) -> dict[str, Any]:
    # Stub de compatibilidad; el pipeline en memoria obtiene diagnósticos desde clean_pose_data.
    _ = clean_npz_path
    return {}


def _collect_detector_partial_warnings(detector_map: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for name, result in detector_map.items():
        if not isinstance(result, dict):
            out.append(f"detector_{name}_invalid_result")
            continue
        if result.get("evaluable") is False:
            out.append(f"detector_{name}_not_evaluable")
        for w in result.get("warnings", []):
            sw = str(w)
            if sw.startswith("DETECTOR_EXCEPTION:"):
                out.append(f"detector_{name}_exception")
            elif "skipped" in sw.lower():
                out.append(f"detector_{name}_skipped")
    return sorted(set(out))


def _terminal_runtime(
    *,
    video_path: Path,
    st: str,
    user_message: str,
    technical_warnings: list[str],
    errors: list[dict[str, Any]],
    pose_raw: dict[str, Any] | None = None,
    pose_clean: dict[str, Any] | None = None,
    segmentation_result: dict[str, Any] | None = None,
    user_pose_sequence_normalization: dict[str, Any] | None = None,
    orientation: dict[str, Any] | None = None,
    pose_meta: dict[str, Any] | None = None,
    cleaning_diagnostics: dict[str, Any] | None = None,
    analysis_context: dict[str, Any] | None = None,
    detector_results: dict[str, Any] | None = None,
    feedback_evidence: dict[str, Any] | None = None,
    feedback_aggregation: dict[str, Any] | None = None,
    feedback_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline_result = PipelineResult(
        status=st,
        ok=st == status.OK,
        user_message=user_message,
        artifacts={"video_path": str(video_path)},
        quality={},
        repetitions=list((segmentation_result or {}).get("reps", [])) if isinstance(segmentation_result, dict) else [],
        feedback=feedback_report if isinstance(feedback_report, dict) else {},
        technical_warnings=sorted(set(technical_warnings)),
        errors=errors,
    ).to_dict()
    return FullAnalysisArtifacts(
        video_path=video_path,
        processed_video_path=video_path,
        pose_raw=pose_raw or {},
        pose_clean=pose_clean or {},
        segmentation_result=segmentation_result or {},
        user_pose_sequence_normalization=user_pose_sequence_normalization or {},
        orientation=orientation or {},
        pose_meta=pose_meta or {},
        cleaning_diagnostics=cleaning_diagnostics or {},
        analysis_context=analysis_context or {},
        detector_results=detector_results or {},
        feedback_evidence=feedback_evidence or {},
        feedback_aggregation=feedback_aggregation or {},
        feedback_report=feedback_report or {},
        pipeline_result=pipeline_result,
    ).to_runtime_dict()
