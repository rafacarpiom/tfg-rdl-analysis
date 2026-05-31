
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.visualization.rdl.debug.analysis_context import export_analysis_context_debug
from src.visualization.rdl.debug.common import ensure_dir, save_json, to_jsonable
from src.visualization.rdl.debug.normalization import export_normalization_debug
from src.visualization.rdl.debug.segmentation import export_segmentation_debug


def _as_runtime(artifacts: Any) -> dict[str, Any]:
    if hasattr(artifacts, "to_runtime_dict"):
        return artifacts.to_runtime_dict()
    if isinstance(artifacts, dict):
        return artifacts
    raise TypeError("artifacts must be runtime dict or object with to_runtime_dict()")


def _as_lightweight(artifacts: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    if hasattr(artifacts, "to_dict"):
        return artifacts.to_dict()
    if isinstance(artifacts, dict):
        base = {
            "video_path": runtime.get("video_path"),
            "processed_video_path": runtime.get("processed_video_path"),
            "orientation": runtime.get("orientation", {}),
            "segmentation_result": {
                "video_id": (runtime.get("segmentation_result") or {}).get("video_id"),
                "segmentation_status": (runtime.get("segmentation_result") or {}).get("segmentation_status"),
                "summary": (runtime.get("segmentation_result") or {}).get("summary", {}),
            },
            "user_pose_sequence_normalization": {"meta": (runtime.get("user_pose_sequence_normalization") or {}).get("meta", {})},
            "analysis_context": {
                "context_meta": (runtime.get("analysis_context") or {}).get("context_meta", {}),
                "warnings": (runtime.get("analysis_context") or {}).get("warnings", []),
            },
            "feedback_evidence": {
                "summary": (runtime.get("feedback_evidence") or {}).get("summary", {}),
                "warnings": (runtime.get("feedback_evidence") or {}).get("warnings", []),
            },
            "feedback_aggregation": {
                "status": (runtime.get("feedback_aggregation") or {}).get("status"),
                "summary": (runtime.get("feedback_aggregation") or {}).get("summary", {}),
                "warnings": (runtime.get("feedback_aggregation") or {}).get("warnings", []),
            },
            "feedback_report": {
                "status": (runtime.get("feedback_report") or {}).get("status"),
                "summary": (runtime.get("feedback_report") or {}).get("summary", {}),
                "headline": (runtime.get("feedback_report") or {}).get("headline"),
                "warnings": (runtime.get("feedback_report") or {}).get("warnings", []),
            },
        }
        return to_jsonable(base)
    return {}


def _save_pose_clean_npz_for_debug(
    *,
    pose_clean: dict[str, Any],
    output_path: Path,
) -> bool:
    if not isinstance(pose_clean, dict) or not pose_clean:
        return False
    payload: dict[str, Any] = {}
    for key in (
        "kps_xy_clean",
        "kps_score_clean",
        "kps_xy",
        "kps_score",
        "mask_valid",
        "mask_valid_frames",
        "mask_valid_right_chain",
        "bbox_xyxy",
        "frame_idx",
    ):
        if key in pose_clean and pose_clean.get(key) is not None:
            payload[key] = np.asarray(pose_clean[key])
    if "fps" in pose_clean and pose_clean.get("fps") is not None:
        payload["fps"] = np.float64(float(pose_clean["fps"]))
    meta = pose_clean.get("meta")
    if isinstance(meta, dict):
        payload["meta"] = np.array([meta], dtype=object)
        payload["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object)
    diagnostics = pose_clean.get("cleaning_diagnostics")
    if isinstance(diagnostics, dict):
        payload["cleaning_diagnostics"] = np.array([diagnostics], dtype=object)
        payload["cleaning_diagnostics_json"] = np.asarray(json.dumps(diagnostics, ensure_ascii=False), dtype=object)
    has_primary = ("kps_xy_clean" in payload and "kps_score_clean" in payload) or ("kps_xy" in payload and "kps_score" in payload)
    if not has_primary:
        return False
    np.savez_compressed(str(output_path), **payload)
    return True


def _save_analysis_context_anchor_cache_for_debug(
    *,
    analysis_context: dict[str, Any],
    output_path: Path,
) -> bool:
    if not isinstance(analysis_context, dict):
        return False
    anchor_pairs = analysis_context.get("anchor_pairs")
    if not isinstance(anchor_pairs, dict):
        return False
    paired = anchor_pairs.get("paired_repetitions")
    if not isinstance(paired, list):
        return False
    payload = {
        "analysis_context": np.array([analysis_context], dtype=object),
        "context_meta_json": np.asarray(json.dumps(analysis_context.get("context_meta", {}), ensure_ascii=False), dtype=object),
    }
    np.savez_compressed(str(output_path), **payload)
    return True


def export_rdl_debug_bundle(*, artifacts, output_dir: str | Path) -> dict:
    out_dir = ensure_dir(output_dir)
    runtime = _as_runtime(artifacts)
    light = _as_lightweight(artifacts, runtime)
    warnings: list[str] = []
    files: list[str] = []
    p00 = out_dir / "00_pipeline_summary.json"
    save_json(p00, light if isinstance(light, dict) else {"summary": light})
    files.append(p00.name)
    seg = runtime.get("segmentation_result", {})
    p01 = out_dir / "01_segmentation_result.json"
    save_json(p01, seg if isinstance(seg, dict) else {"segmentation_result": seg})
    files.append(p01.name)
    p02 = out_dir / "02_segmentation_debug.png"
    export_segmentation_debug(segmentation_result=seg, output_path=p02)
    files.append(p02.name)
    p04 = out_dir / "04_normalization_debug.png"
    norm_summary = export_normalization_debug(
        pose_clean=runtime.get("pose_clean", {}),
        user_pose_sequence_normalization=runtime.get("user_pose_sequence_normalization", {}),
        segmentation_result=seg,
        output_path=p04,
    )
    files.append(p04.name)
    p03 = out_dir / "03_normalization_summary.json"
    save_json(p03, norm_summary)
    files.append(p03.name)
    p06 = out_dir / "06_analysis_context_debug.png"
    ac_summary = export_analysis_context_debug(analysis_context=runtime.get("analysis_context", {}), output_path=p06)
    files.append(p06.name)
    p05 = out_dir / "05_analysis_context_summary.json"
    save_json(p05, ac_summary)
    files.append(p05.name)
    p07 = out_dir / "07_user_pose_clean.npz"
    pose_clean_saved = _save_pose_clean_npz_for_debug(
        pose_clean=runtime.get("pose_clean", {}) if isinstance(runtime.get("pose_clean"), dict) else {},
        output_path=p07,
    )
    if pose_clean_saved:
        files.append(p07.name)
    else:
        warnings.append("POSE_CLEAN_MISSING_FOR_DEBUG_BUNDLE")
    p07b = out_dir / "07_analysis_context_anchor_cache.npz"
    anchor_cache_saved = _save_analysis_context_anchor_cache_for_debug(
        analysis_context=runtime.get("analysis_context", {}) if isinstance(runtime.get("analysis_context"), dict) else {},
        output_path=p07b,
    )
    if anchor_cache_saved:
        files.append(p07b.name)
    else:
        warnings.append("ANALYSIS_CONTEXT_ANCHOR_CACHE_MISSING")
    detector_results = runtime.get("detector_results", {}) if isinstance(runtime.get("detector_results"), dict) else {}
    detector_summary = detector_results.get("summary", {}) if isinstance(detector_results.get("summary"), dict) else {}
    detector_map_in = detector_results.get("detectors", {}) if isinstance(detector_results.get("detectors"), dict) else {}
    detector_map_out: dict[str, dict[str, Any]] = {}
    for name, result in detector_map_in.items():
        if not isinstance(result, dict):
            continue
        detector_map_out[str(name)] = {
            "detected": result.get("detected"),
            "severity": result.get("severity"),
            "score": result.get("score"),
            "num_reps_analyzed": result.get("num_reps_analyzed"),
            "num_reps_detected": result.get("num_reps_detected"),
            "warnings": result.get("warnings", []),
        }
    p08 = out_dir / "08_detector_results_summary.json"
    save_json(
        p08,
        {
            "summary": detector_summary,
            "detectors": detector_map_out,
            "warnings": detector_results.get("warnings", []),
        },
    )
    files.append(p08.name)
    feedback_evidence = runtime.get("feedback_evidence")
    if isinstance(feedback_evidence, dict):
        p10 = out_dir / "10_feedback_evidence.json"
        save_json(p10, to_jsonable(feedback_evidence))
        files.append(p10.name)
    else:
        warnings.append("FEEDBACK_EVIDENCE_MISSING")
    feedback_aggregation = runtime.get("feedback_aggregation")
    if isinstance(feedback_aggregation, dict):
        p11 = out_dir / "11_feedback_aggregation.json"
        save_json(p11, to_jsonable(feedback_aggregation))
        files.append(p11.name)
    else:
        warnings.append("FEEDBACK_AGGREGATION_MISSING")
    feedback_report = runtime.get("feedback_report")
    if isinstance(feedback_report, dict):
        p12 = out_dir / "12_feedback_report.json"
        save_json(p12, to_jsonable(feedback_report))
        files.append(p12.name)
        p13 = out_dir / "13_feedback_report.txt"
        p13.write_text(str(feedback_report.get("plain_text", "")), encoding="utf-8")
        files.append(p13.name)
    else:
        warnings.append("FEEDBACK_REPORT_MISSING")
    warnings.extend(list(norm_summary.get("warnings", [])))
    warnings.extend(list(ac_summary.get("warnings", [])))
    return {"output_dir": str(out_dir), "files": files, "warnings": sorted(set(warnings))}
