
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FullAnalysisArtifacts:
    # Input original y versión usada tras preprocessing.
    video_path: Path
    processed_video_path: Path

    # Artefactos en memoria.
    pose_raw: dict[str, Any]
    pose_clean: dict[str, Any]
    segmentation_result: dict[str, Any]
    user_pose_sequence_normalization: dict[str, Any]
    orientation: dict[str, Any]
    pose_meta: dict[str, Any]
    cleaning_diagnostics: dict[str, Any]
    analysis_context: dict[str, Any] | None = None
    detector_results: dict[str, Any] | None = None
    feedback_evidence: dict[str, Any] | None = None
    feedback_aggregation: dict[str, Any] | None = None
    feedback_report: dict[str, Any] | None = None
    pipeline_result: dict[str, Any] | None = None

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "processed_video_path": str(self.processed_video_path),
            "pose_raw": self.pose_raw,
            "pose_clean": self.pose_clean,
            "segmentation_result": self.segmentation_result,
            "user_pose_sequence_normalization": self.user_pose_sequence_normalization,
            "analysis_context": self.analysis_context,
            "detector_results": self.detector_results,
            "feedback_evidence": self.feedback_evidence,
            "feedback_aggregation": self.feedback_aggregation,
            "feedback_report": self.feedback_report,
            "pipeline_result": self.pipeline_result,
            "orientation": self.orientation,
            "pose_meta": self.pose_meta,
            "cleaning_diagnostics": self.cleaning_diagnostics,
        }

    def to_dict(self) -> dict[str, Any]:
        # Serialización estable para CLI/tests/UI.
        pose_raw_kps = self.pose_raw.get("kps_xy")
        pose_raw_scores = self.pose_raw.get("kps_score")
        pose_clean_kps = self.pose_clean.get("kps_xy_clean")
        norm_kps = self.user_pose_sequence_normalization.get("kps_xy_normalized")
        seg_summary = self.segmentation_result.get("summary", {})
        ctx_meta = {}
        ctx_warnings: list[Any] = []
        num_paired = 0
        if isinstance(self.analysis_context, dict):
            raw_meta = self.analysis_context.get("context_meta")
            if isinstance(raw_meta, dict):
                ctx_meta = raw_meta
            raw_warnings = self.analysis_context.get("warnings")
            if isinstance(raw_warnings, list):
                ctx_warnings = raw_warnings
            anchor_pairs = self.analysis_context.get("anchor_pairs")
            if isinstance(anchor_pairs, dict):
                num_paired = int(anchor_pairs.get("num_paired_repetitions", 0))
        detector_results_light = None
        if isinstance(self.detector_results, dict):
            detectors_in = self.detector_results.get("detectors", {})
            detector_map: dict[str, Any] = {}
            if isinstance(detectors_in, dict):
                for name, result in detectors_in.items():
                    if not isinstance(result, dict):
                        continue
                    detector_map[str(name)] = {
                        "detected": result.get("detected"),
                        "severity": result.get("severity"),
                        "score": result.get("score"),
                        "num_reps_analyzed": result.get("num_reps_analyzed"),
                        "num_reps_detected": result.get("num_reps_detected"),
                        "warnings": result.get("warnings", []),
                    }
            detector_results_light = {
                "summary": self.detector_results.get("summary", {}),
                "detectors": detector_map,
                "warnings": self.detector_results.get("warnings", []),
            }
        feedback_evidence_light = None
        if isinstance(self.feedback_evidence, dict):
            feedback_evidence_light = {
                "summary": self.feedback_evidence.get("summary", {}),
                "warnings": self.feedback_evidence.get("warnings", []),
                "evidence_items_preview": list(self.feedback_evidence.get("evidence_items", []))[:5]
                if isinstance(self.feedback_evidence.get("evidence_items"), list)
                else [],
            }
        feedback_aggregation_light = None
        if isinstance(self.feedback_aggregation, dict):
            feedback_aggregation_light = {
                "status": self.feedback_aggregation.get("status"),
                "summary": self.feedback_aggregation.get("summary", {}),
                "primary_focus": list(self.feedback_aggregation.get("primary_focus", []))[:5]
                if isinstance(self.feedback_aggregation.get("primary_focus"), list)
                else [],
                "secondary_focus": list(self.feedback_aggregation.get("secondary_focus", []))[:5]
                if isinstance(self.feedback_aggregation.get("secondary_focus"), list)
                else [],
                "warnings": self.feedback_aggregation.get("warnings", []),
            }
        feedback_report_light = None
        if isinstance(self.feedback_report, dict):
            feedback_report_light = {
                "summary": self.feedback_report.get("summary", {}),
                "headline": self.feedback_report.get("headline"),
                "main_feedback": [
                    {
                        "priority": i.get("priority"),
                        "error_code": i.get("error_code"),
                        "title": i.get("title"),
                        "severity": i.get("severity"),
                        "where": i.get("where"),
                    }
                    for i in list(self.feedback_report.get("main_feedback", []))[:5]
                    if isinstance(i, dict)
                ],
                "warnings": self.feedback_report.get("warnings", []),
            }
        return {
            "video_path": str(self.video_path),
            "processed_video_path": str(self.processed_video_path),
            "pose_raw": {
                "kps_xy_shape": list(pose_raw_kps.shape) if hasattr(pose_raw_kps, "shape") else None,
                "kps_score_shape": list(pose_raw_scores.shape) if hasattr(pose_raw_scores, "shape") else None,
                "fps": self.pose_raw.get("fps"),
                "meta": self.pose_raw.get("meta", {}),
            },
            "pose_clean": {
                "kps_xy_clean_shape": list(pose_clean_kps.shape) if hasattr(pose_clean_kps, "shape") else None,
                "cleaning_diagnostics": self.cleaning_diagnostics,
            },
            "segmentation_result": {
                "video_id": self.segmentation_result.get("video_id"),
                "exercise": self.segmentation_result.get("exercise"),
                "segmentation_status": self.segmentation_result.get("segmentation_status"),
                "summary": seg_summary if isinstance(seg_summary, dict) else {},
            },
            "user_pose_sequence_normalization": {
                "kps_xy_normalized_shape": list(norm_kps.shape) if hasattr(norm_kps, "shape") else None,
                "meta": self.user_pose_sequence_normalization.get("meta", {}),
            },
            "analysis_context": {
                "context_meta": ctx_meta,
                "num_paired_repetitions": num_paired,
                "warnings": ctx_warnings,
            },
            "detector_results": detector_results_light,
            "feedback_evidence": feedback_evidence_light,
            "feedback_aggregation": feedback_aggregation_light,
            "feedback_report": feedback_report_light,
            "pipeline_result": self.pipeline_result or {},
            "pose_meta": self.pose_meta,
            "cleaning_diagnostics": self.cleaning_diagnostics,
            "orientation": self.orientation,
        }


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    status: str
    user_message: str
    technical_warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "status": str(self.status),
            "user_message": str(self.user_message),
            "technical_warnings": [str(w) for w in self.technical_warnings],
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PipelineResult:
    status: str
    ok: bool
    user_message: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    repetitions: list[Any] = field(default_factory=list)
    feedback: dict[str, Any] = field(default_factory=dict)
    technical_warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "ok": bool(self.ok),
            "user_message": str(self.user_message),
            "artifacts": dict(self.artifacts),
            "quality": dict(self.quality),
            "repetitions": list(self.repetitions),
            "feedback": dict(self.feedback),
            "technical_warnings": [str(w) for w in self.technical_warnings],
            "errors": [dict(e) for e in self.errors if isinstance(e, dict)],
        }

