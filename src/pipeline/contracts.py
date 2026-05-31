
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import status
from .results import ValidationResult


def _ok(details: dict[str, Any] | None = None, warnings: list[str] | None = None) -> ValidationResult:
    return ValidationResult(
        ok=True,
        status=status.OK,
        user_message="Contrato válido.",
        technical_warnings=warnings or [],
        details=details or {},
    )


def _fail(st: str, user_message: str, warning: str, details: dict[str, Any] | None = None) -> ValidationResult:
    return ValidationResult(
        ok=False,
        status=st,
        user_message=user_message,
        technical_warnings=[warning],
        details=details or {},
    )


def validate_raw_npz_contract(npz_path: str | Path) -> ValidationResult:
    p = Path(npz_path)
    if not p.is_file():
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "No se ha generado un archivo de pose raw válido.", "raw_npz_missing", {"path": str(p)})
    try:
        with np.load(str(p), allow_pickle=True) as npz:
            required = ("kps_xy", "kps_score", "frame_idx")
            for key in required:
                if key not in npz.files:
                    return _fail(status.INSUFFICIENT_POSE_QUALITY, "La estimación de pose raw no contiene todos los datos necesarios.", f"raw_npz_missing_{key}", {"path": str(p)})
            kps = np.asarray(npz["kps_xy"])
            score = np.asarray(npz["kps_score"])
            frame_idx = np.asarray(npz["frame_idx"])
    except Exception as exc:
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "No se pudo leer la pose raw generada.", f"raw_npz_load_failed:{exc}", {"path": str(p)})
    if kps.ndim != 3 or kps.shape[-2:] != (17, 2):
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "La pose raw tiene un formato inesperado.", "raw_npz_invalid_kps_shape", {"shape": list(kps.shape)})
    if score.ndim != 2 or score.shape[0] != kps.shape[0]:
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "La pose raw no tiene puntuaciones coherentes por frame.", "raw_npz_invalid_score_shape", {"score_shape": list(score.shape), "kps_shape": list(kps.shape)})
    if frame_idx.shape[0] != kps.shape[0]:
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "La pose raw no tiene índices de frame coherentes.", "raw_npz_invalid_frame_idx", {"frame_idx_shape": list(frame_idx.shape), "kps_shape": list(kps.shape)})
    if kps.shape[0] < 3:
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "No hay suficientes frames de pose raw para continuar.", "raw_npz_too_few_frames", {"num_frames": int(kps.shape[0])})
    return _ok({"num_frames": int(kps.shape[0])})


def validate_clean_npz_contract(npz_path: str | Path) -> ValidationResult:
    p = Path(npz_path)
    if not p.is_file():
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "No se ha generado un archivo de pose clean válido.", "clean_npz_missing", {"path": str(p)})
    try:
        with np.load(str(p), allow_pickle=True) as npz:
            key = "kps_xy_clean" if "kps_xy_clean" in npz.files else ("kps_xy" if "kps_xy" in npz.files else "")
            if not key:
                return _fail(status.INSUFFICIENT_POSE_QUALITY, "La pose clean no contiene keypoints limpios.", "clean_npz_missing_kps", {"path": str(p)})
            kps = np.asarray(npz[key])
    except Exception as exc:
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "No se pudo leer la pose clean generada.", f"clean_npz_load_failed:{exc}", {"path": str(p)})
    if kps.ndim != 3 or kps.shape[-2:] != (17, 2):
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "La pose clean tiene un formato inesperado.", "clean_npz_invalid_kps_shape", {"shape": list(kps.shape)})
    if kps.shape[0] < 3:
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "No hay suficientes frames limpios para segmentar repeticiones.", "clean_npz_too_few_frames", {"num_frames": int(kps.shape[0])})
    return _ok({"num_frames": int(kps.shape[0])})


def validate_pose_raw_contract_data(pose_raw: dict[str, Any]) -> ValidationResult:
    if not isinstance(pose_raw, dict):
        return _fail(status.NO_PERSON_DETECTED, "No se han obtenido datos de pose válidos.", "pose_raw_not_dict")
    if "kps_xy" not in pose_raw or "kps_score" not in pose_raw:
        return _fail(status.NO_PERSON_DETECTED, "No se ha detectado una persona de forma fiable en el vídeo.", "pose_raw_missing_kps_or_score")
    try:
        kps = np.asarray(pose_raw["kps_xy"], dtype=np.float64)
        score = np.asarray(pose_raw["kps_score"], dtype=np.float64)
    except Exception:
        return _fail(status.NO_PERSON_DETECTED, "No se han obtenido datos de pose válidos.", "pose_raw_array_cast_failed")
    if kps.ndim != 3 or kps.shape[-2:] != (17, 2) or score.ndim != 2 or score.shape[0] != kps.shape[0]:
        return _fail(status.NO_PERSON_DETECTED, "No se ha detectado una persona de forma fiable en el vídeo.", "pose_raw_invalid_shape")
    finite_ratio = float(np.isfinite(kps).all(axis=(1, 2)).mean()) if kps.shape[0] else 0.0
    if kps.shape[0] < 3 or finite_ratio < 0.05:
        return _fail(status.NO_PERSON_DETECTED, "No se ha detectado una persona de forma fiable en el vídeo.", "pose_raw_low_valid_frames", {"finite_ratio": finite_ratio})
    return _ok({"num_frames": int(kps.shape[0]), "finite_ratio": finite_ratio})


def validate_pose_clean_contract_data(pose_clean: dict[str, Any], *, min_valid_pose_ratio: float = 0.10) -> ValidationResult:
    if not isinstance(pose_clean, dict):
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "La calidad de la estimación de pose no es suficiente para realizar un análisis fiable.", "pose_clean_not_dict")
    key = "kps_xy_clean" if "kps_xy_clean" in pose_clean else ("kps_xy" if "kps_xy" in pose_clean else "")
    if not key:
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "La calidad de la estimación de pose no es suficiente para realizar un análisis fiable.", "pose_clean_missing_kps")
    try:
        kps = np.asarray(pose_clean[key], dtype=np.float64)
    except Exception:
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "La calidad de la estimación de pose no es suficiente para realizar un análisis fiable.", "pose_clean_array_cast_failed")
    if kps.ndim != 3 or kps.shape[-2:] != (17, 2):
        return _fail(status.INSUFFICIENT_POSE_QUALITY, "La calidad de la estimación de pose no es suficiente para realizar un análisis fiable.", "pose_clean_invalid_shape", {"shape": list(kps.shape)})
    valid_ratio = float(np.isfinite(kps).all(axis=(1, 2)).mean()) if kps.shape[0] else 0.0
    warnings: list[str] = []
    if valid_ratio < min_valid_pose_ratio:
        return _fail(
            status.INSUFFICIENT_POSE_QUALITY,
            "La calidad de la estimación de pose no es suficiente para realizar un análisis fiable. Repite la grabación con mejor iluminación, cuerpo completo visible y menos oclusiones.",
            "low_valid_pose_ratio",
            {"valid_ratio": valid_ratio, "min_valid_pose_ratio": min_valid_pose_ratio},
        )
    diagnostics = pose_clean.get("cleaning_diagnostics") if isinstance(pose_clean.get("cleaning_diagnostics"), dict) else {}
    right_chain_pct = diagnostics.get("right_chain_valid_frame_pct")
    if right_chain_pct is not None:
        try:
            rc = float(right_chain_pct)
            if rc < 0.20:
                warnings.append("right_chain_quality_below_threshold")
        except Exception:
            pass
    return _ok({"num_frames": int(kps.shape[0]), "valid_ratio": valid_ratio}, warnings=warnings)


def validate_segmentation_contract(segmentation_result: dict[str, Any]) -> ValidationResult:
    if not isinstance(segmentation_result, dict):
        return _fail(status.INVALID_SEGMENTATION, "El movimiento no se ha podido segmentar correctamente en repeticiones válidas.", "segmentation_not_dict")
    reps = segmentation_result.get("reps")
    if not isinstance(reps, list):
        return _fail(status.INVALID_SEGMENTATION, "El movimiento no se ha podido segmentar correctamente en repeticiones válidas.", "segmentation_missing_reps")
    valid_reps = [r for r in reps if isinstance(r, dict) and bool(r.get("anchor_valid", True))]
    if not valid_reps:
        return _fail(
            status.NO_REPS_DETECTED,
            "No se han detectado repeticiones completas. Graba una serie de 5 a 6 repeticiones completas, sin pausas largas y desde una vista lateral.",
            "segmentation_no_valid_repetitions",
        )
    warnings: list[str] = []
    for idx, rep in enumerate(valid_reps, start=1):
        start = rep.get("start")
        end = rep.get("end")
        if isinstance(start, int) and isinstance(end, int) and end <= start:
            warnings.append(f"segmentation_invalid_bounds_rep_{idx}")
    return _ok({"num_valid_reps": len(valid_reps)}, warnings=warnings)


def validate_anchors_contract(analysis_context: dict[str, Any]) -> ValidationResult:
    anchor_pairs = analysis_context.get("anchor_pairs") if isinstance(analysis_context, dict) else None
    paired = anchor_pairs.get("paired_repetitions") if isinstance(anchor_pairs, dict) else None
    if not isinstance(paired, list) or not paired:
        return _fail(
            status.INVALID_ANCHORS,
            "No se han podido obtener puntos de análisis fiables dentro de las repeticiones detectadas.",
            "anchors_missing_paired_repetitions",
        )
    warnings: list[str] = []
    valid_rep_count = 0
    required = {"ecc_0", "ecc_50", "ecc_100", "bottom"}
    for rep in paired:
        if not isinstance(rep, dict):
            continue
        rep_order = rep.get("user_rep_order")
        anchors = rep.get("anchors") if isinstance(rep.get("anchors"), dict) else {}
        valid_names = {name for name, info in anchors.items() if isinstance(info, dict) and bool(info.get("valid", False))}
        if valid_names:
            valid_rep_count += 1
        missing = sorted(required - valid_names)
        for m in missing:
            warnings.append(f"missing_anchor_{m}_rep_{rep_order}")
    if valid_rep_count == 0:
        return _fail(
            status.INVALID_ANCHORS,
            "No se han podido obtener puntos de análisis fiables dentro de las repeticiones detectadas.",
            "anchors_no_valid_repetition",
        )
    return _ok({"num_reps_with_valid_anchors": valid_rep_count}, warnings=warnings)

