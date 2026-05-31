
from __future__ import annotations

from typing import Any


def _progress(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)

from src.biomechanics.rdl.detectors.arms import detect_bent_arms
from src.biomechanics.rdl.detectors.asymmetry import detect_asymmetry
from src.biomechanics.rdl.detectors.bar_far import detect_bar_far
from src.biomechanics.rdl.detectors.hip_hinge import detect_hip_hinge
from src.biomechanics.rdl.detectors.knee_dominant import detect_knee_dominant_error
from src.biomechanics.rdl.detectors.lockout import detect_no_lockout_error
from src.biomechanics.rdl.detectors.neck_movement import detect_neck_movement_error
from src.biomechanics.rdl.detectors.rom import detect_short_rom_error
from src.biomechanics.rdl.detectors.spine_flexion import detect_spine_flexion_error

_SEVERITY_RANK = {
    "none": 0,
    "posible": 1,
    "leve": 2,
    "media": 3,
    "grave": 4,
}


def _max_severity(results: dict[str, dict]) -> str:
    best = "none"
    for result in results.values():
        sev = str(result.get("severity", "none"))
        if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(best, 0):
            best = sev
    return best


def _detector_exception_result(name: str, exc: Exception) -> dict[str, Any]:
    return {
        "detector": name,
        "detected": False,
        "evaluable": False,
        "message": None,
        "severity": "none",
        "score": 0.0,
        "num_reps_analyzed": 0,
        "num_reps_detected": 0,
        "rep_results": [],
        "warnings": [f"DETECTOR_EXCEPTION:{exc}"],
    }


def run_rdl_detectors(
    analysis_context: dict,
) -> dict:
    warnings: list[str] = []
    detectors: dict[str, dict] = {}

    _progress("Detector: bent_arms...")
    try:
        bent_arms_result = detect_bent_arms(analysis_context)
    except Exception as exc:
        bent_arms_result = _detector_exception_result("bent_arms", exc)
    detectors["bent_arms"] = bent_arms_result

    _progress("Detector: asymmetry...")
    try:
        asymmetry_result = detect_asymmetry(analysis_context)
    except Exception as exc:
        asymmetry_result = _detector_exception_result("asymmetry", exc)
    detectors["asymmetry"] = asymmetry_result

    _progress("Detector: bar_far...")
    try:
        bar_far_result = detect_bar_far(analysis_context)
    except Exception as exc:
        bar_far_result = _detector_exception_result("bar_far", exc)
    detectors["bar_far"] = bar_far_result

    _progress("Detector: hip_hinge...")
    try:
        hip_hinge_result = detect_hip_hinge(analysis_context)
    except Exception as exc:
        hip_hinge_result = _detector_exception_result("hip_hinge", exc)
    detectors["hip_hinge"] = hip_hinge_result

    _progress("Detector: knee_dominant...")
    try:
        knee_dominant_result = detect_knee_dominant_error(analysis_context)
    except Exception as exc:
        knee_dominant_result = _detector_exception_result("knee_dominant", exc)
    detectors["knee_dominant"] = knee_dominant_result

    _progress("Detector: lockout...")
    try:
        lockout_result = detect_no_lockout_error(analysis_context)
    except Exception as exc:
        lockout_result = _detector_exception_result("lockout", exc)
    detectors["lockout"] = lockout_result

    _progress("Detector: neck_movement...")
    try:
        neck_movement_result = detect_neck_movement_error(analysis_context)
    except Exception as exc:
        neck_movement_result = _detector_exception_result("neck_movement", exc)
    detectors["neck_movement"] = neck_movement_result

    _progress("Detector: rom...")
    try:
        rom_result = detect_short_rom_error(analysis_context)
    except Exception as exc:
        rom_result = _detector_exception_result("rom", exc)
    detectors["rom"] = rom_result

    _progress("Detector: spine_flexion...")
    try:
        spine_flexion_result = detect_spine_flexion_error(
            analysis_context=analysis_context,
            detector_results=detectors,
        )
    except Exception as exc:
        spine_flexion_result = _detector_exception_result("spine_flexion", exc)
    detectors["spine_flexion"] = spine_flexion_result
    _progress("Detectores finalizados.")

    for result in detectors.values():
        detector_warnings = result.get("warnings")
        if isinstance(detector_warnings, list):
            warnings.extend([str(x) for x in detector_warnings])

    detected_errors = [name for name, result in detectors.items() if bool(result.get("detected", False))]

    return {
        "exercise": "RDL",
        "detectors": detectors,
        "summary": {
            "num_detectors": len(detectors),
            "num_detected": len(detected_errors),
            "detected_errors": detected_errors,
            "max_severity": _max_severity(detectors),
        },
        "warnings": sorted(set(warnings)),
    }
