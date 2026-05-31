
from __future__ import annotations

from typing import Any, Callable

from .constants import PHASE_UNKNOWN, SEVERITY_RANK
from .extractors import (
    extract_asymmetry_evidence,
    extract_bar_far_evidence,
    extract_bent_arms_evidence,
    extract_hip_hinge_evidence,
    extract_knee_dominant_evidence,
    extract_lockout_evidence,
    extract_neck_movement_evidence,
    extract_non_detected_debug_evidence,
    extract_rom_evidence,
    extract_spine_flexion_evidence,
)
from .schema import EvidenceItem, EvidenceResult, evidence_result_to_dict


def _max_severity(severities: list[str]) -> str:
    if not severities:
        return "none"
    best = "none"
    for sev in severities:
        s = str(sev or "none").lower()
        if SEVERITY_RANK.get(s, 0) > SEVERITY_RANK.get(best, 0):
            best = s
    return best


def normalize_rdl_detector_evidence(
    *,
    detector_results: dict[str, Any],
    analysis_context: dict[str, Any] | None = None,
    include_non_detected: bool = False,
) -> dict[str, Any]:
    detectors = detector_results if isinstance(detector_results, dict) else {}
    warnings: list[str] = []
    evidence_items: list[EvidenceItem] = []

    extractors: dict[str, Callable[[dict[str, Any], dict[str, Any] | None], list[EvidenceItem]]] = {
        "bent_arms": extract_bent_arms_evidence,
        "asymmetry": extract_asymmetry_evidence,
        "bar_far": extract_bar_far_evidence,
        "hip_hinge": extract_hip_hinge_evidence,
        "knee_dominant": extract_knee_dominant_evidence,
        "lockout": extract_lockout_evidence,
        "neck_movement": extract_neck_movement_evidence,
        "rom": extract_rom_evidence,
        "spine_flexion": extract_spine_flexion_evidence,
    }

    for detector_name, extractor in extractors.items():
        result = detectors.get(detector_name)
        if not isinstance(result, dict):
            warnings.append(f"EVIDENCE_DETECTOR_RESULT_MISSING:{detector_name}")
            continue
        try:
            items = extractor(result, analysis_context)
            evidence_items.extend(items)
            if include_non_detected:
                evidence_items.extend(extract_non_detected_debug_evidence(detector_name, result))
        except Exception as exc:
            warnings.append(f"EVIDENCE_EXTRACTOR_EXCEPTION:{detector_name}:{exc}")

    severities = [item.severity for item in evidence_items]
    detectors_with_evidence = sorted(set(item.detector for item in evidence_items))
    reps_with_evidence = sorted(
        set(item.user_rep_order for item in evidence_items if item.user_rep_order is not None)
    )
    phases_with_evidence = sorted(set(item.phase or PHASE_UNKNOWN for item in evidence_items))
    summary = {
        "num_items": len(evidence_items),
        "num_detected_items": sum(1 for item in evidence_items if item.detected),
        "detectors_present": sorted(detectors.keys()),
        "detectors_with_evidence": detectors_with_evidence,
        "reps_with_evidence": reps_with_evidence,
        "phases_with_evidence": phases_with_evidence,
        "max_severity": _max_severity(severities),
        "warnings_count": len(warnings),
    }

    result = EvidenceResult(
        exercise="RDL",
        evidence_items=evidence_items,
        summary=summary,
        warnings=warnings,
    )
    return evidence_result_to_dict(result)
