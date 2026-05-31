
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

CoherenceVerdict = Literal["coherent", "incoherent", "inconclusive"]

Severity = Literal["none", "leve", "media", "grave"]

# --- Ranking de severidad ------------------------------------------------------
_SEVERITY_RANK: dict[str, int] = {
    "none": 0, "leve": 1, "media": 2, "grave": 3,
}
_RANK_TO_SEVERITY: tuple[Severity, ...] = ("none", "leve", "media", "grave")


def _coerce_severity(value: Any) -> Severity:
    v = str(value or "none").lower()
    if v not in _SEVERITY_RANK:
        return "none"
    return v  # type: ignore[return-value]


def _max(a: Severity, b: Severity) -> Severity:
    return _RANK_TO_SEVERITY[max(_SEVERITY_RANK[a], _SEVERITY_RANK[b])]


def _min(a: Severity, b: Severity) -> Severity:
    return _RANK_TO_SEVERITY[min(_SEVERITY_RANK[a], _SEVERITY_RANK[b])]


# --- Parámetros de confirmación -------------------------------------------
SEGMENT_ORDER: tuple[str, ...] = (
    "ecc_0_to_ecc_25",
    "ecc_25_to_ecc_50",
    "ecc_50_to_ecc_75",
    "ecc_75_to_ecc_100",
)

SEGMENT_TO_END_ANCHOR: dict[str, str] = {
    "ecc_0_to_ecc_25": "ecc_25",
    "ecc_25_to_ecc_50": "ecc_50",
    "ecc_50_to_ecc_75": "ecc_75",
    "ecc_75_to_ecc_100": "ecc_100",
}

MIN_SEGMENTS_FOR_REP: int = 2
INFERENCE_CAP_SEVERITY: Severity = "grave"

HIP_JUSTIFICATION_FACTOR: float = 0.6
# 3A: mínimo hip-back ideal (norm) para estimar pendiente hombro–cadera vs PM-Ideal.
MIN_IDEAL_HIP_BACK_FOR_COHERENCE: float = 0.04
# Caída excesiva de hombro (norm) sobre lo esperado = s_u - s_i * (h_u / h_i).
# Calibrado en corpus: Rafa-Baseline ~0.18-0.25; flexión funcional suele >0.30.
SHOULDER_HIP_EXCESS_LEVE: float = 0.26
SHOULDER_HIP_EXCESS_MEDIA: float = 0.34
SHOULDER_HIP_EXCESS_GRAVE: float = 0.42


# --- Reglas por segmento ---------------------------------------------------

@dataclass
class SegmentRuling:

    segment: str
    severity: Severity = "none"
    triggered: bool = False
    possible: bool = False
    reason: str = ""

    # Detectores hermanos.
    hip_hinge_severity: Severity = "none"
    hip_hinge_failed: bool = False
    knee_dominant_failed: bool = False
    neck_movement_failed: bool = False
    neck_direction: str = "neutral"

    # Evidencia geométrica directa de columna.
    anchor: str = ""
    torso_low_failed: bool = False
    torso_low_severity: Severity = "none"
    torso_low_norm: float | None = None
    torso_low_px: float | None = None
    torso_angle_delta_deg: float | None = None
    geometry_status: str = "missing"

    trace: list[str] = field(default_factory=list)


@dataclass
class RepSpineFlexionVerdict:

    detected: bool
    severity: Severity
    method: str = "inferred_by_geometry_and_exclusion"
    n_segments_triggered: int = 0
    triggered_segments: list[str] = field(default_factory=list)
    possible_segments: list[str] = field(default_factory=list)
    per_segment: dict[str, SegmentRuling] = field(default_factory=dict)
    per_anchor: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


# --- Auxiliares --------------------------------------------------------------

def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_rep_anchor_result(result: dict | None, anchor: str) -> dict | None:
    if not isinstance(result, dict):
        return None
    for key in ("anchor_rulings", "anchor_results", "per_anchor", "anchor_metrics"):
        block = result.get(key)
        if isinstance(block, dict) and isinstance(block.get(anchor), dict):
            return block.get(anchor)
    return None


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _hip_delta_back(hip_rep_result: dict | None, segment: str) -> float:
    if not isinstance(hip_rep_result, dict):
        return 0.0
    end_anchor = SEGMENT_TO_END_ANCHOR.get(segment, segment.split("_to_", 1)[-1])
    ruling = _get_rep_anchor_result(hip_rep_result, end_anchor)
    d = None
    if ruling is not None:
        d = _read_field(ruling, "delta_hip_back", None)
    if d is None:
        metrics = hip_rep_result.get("anchor_metrics")
        if isinstance(metrics, dict):
            m = metrics.get(end_anchor)
            if isinstance(m, dict):
                d = m.get("delta_hip_back")
    if d is None or not _finite(float(d)):
        return 0.0
    return float(d)


def _hip_segment_severity(
    hip_rep_result: dict | None,
    segment: str,
) -> tuple[Severity, bool, str]:
    if not isinstance(hip_rep_result, dict):
        return "none", False, "hip_hinge_rep_result_unavailable"
    end_anchor = SEGMENT_TO_END_ANCHOR.get(segment, segment.split("_to_", 1)[-1])
    ruling = _get_rep_anchor_result(hip_rep_result, end_anchor)
    if ruling is None:
        return "none", False, f"hip_hinge_anchor_no_data_{end_anchor}"
    failed = bool(_read_field(ruling, "failed", False))
    if not failed:
        return "none", False, f"hip_hinge_{end_anchor}_failed=False"
    anchor_sev = _coerce_severity(_read_field(ruling, "severity", None))
    rep_sev = _coerce_severity(_read_field(hip_rep_result, "severity", "none"))
    sev = anchor_sev if anchor_sev != "none" else rep_sev
    return sev, True, f"hip_hinge_{end_anchor}_failed=True_sev={sev}"


def _knee_segment_failed(
    knee_rep_result: dict | None,
    segment: str,
) -> tuple[bool, str]:
    if not isinstance(knee_rep_result, dict):
        return False, "knee_dominant_rep_result_unavailable"
    end_anchor = SEGMENT_TO_END_ANCHOR.get(segment, segment.split("_to_", 1)[-1])
    ruling = _get_rep_anchor_result(knee_rep_result, end_anchor)
    if ruling is None:
        return False, f"knee_dominant_anchor_no_data_{end_anchor}"
    failed = bool(_read_field(ruling, "failed", False))
    return failed, f"knee_dominant_{end_anchor}_failed={failed}"


def _neck_segment_ruling(
    neck_rep_result: dict | None,
    segment: str,
) -> tuple[bool, str, str]:
    if not isinstance(neck_rep_result, dict):
        return False, "neutral", "neck_movement_rep_result_unavailable"

    end_anchor = SEGMENT_TO_END_ANCHOR.get(segment, segment.split("_to_", 1)[-1])
    ruling = None
    segment_block = neck_rep_result.get("segment_results")
    if isinstance(segment_block, dict):
        ruling = segment_block.get(segment)
    if ruling is None:
        segment_block = neck_rep_result.get("per_segment")
        if isinstance(segment_block, dict):
            ruling = segment_block.get(segment)
    source = f"segment_{segment}"
    if ruling is None:
        ruling = _get_rep_anchor_result(neck_rep_result, end_anchor)
        source = f"anchor_{end_anchor}"
    if ruling is None:
        return False, "neutral", f"neck_movement_no_granular_data_{segment}"

    failed = bool(_read_field(ruling, "failed", False)) or bool(_read_field(ruling, "confirmed", False))
    sev = _coerce_severity(_read_field(ruling, "severity", "none"))
    if sev != "none":
        failed = True
    direction = str(
        _read_field(ruling, "neck_direction", None)
        or _read_field(ruling, "direction", None)
        or _read_field(neck_rep_result, "neck_direction", "neutral")
    )
    return failed, direction, f"neck_movement_{source}_failed={failed}_sev={sev}_dir={direction}"


def _geometry_segment_evidence(
    spine_geometry: dict[str, Any] | None,
    segment: str,
) -> tuple[dict[str, Any] | None, str]:
    if not spine_geometry:
        return None, "spine_geometry_unavailable"
    item = spine_geometry.get(segment)
    if item is None:
        anchor = SEGMENT_TO_END_ANCHOR.get(segment)
        item = spine_geometry.get(anchor) if anchor else None
    if not isinstance(item, dict):
        return None, f"spine_geometry_no_data_{segment}"
    return item, "spine_geometry_ok"


def _severity_from_geometry(item: dict[str, Any] | None) -> tuple[Severity, bool, str, float | None, float | None, float | None, str, str]:
    if not item:
        return "none", False, "", None, None, None, "missing", "missing_geometry"
    status = str(item.get("status", "ok"))
    anchor = str(item.get("anchor", ""))
    if status != "ok":
        return "none", False, anchor, None, None, None, status, str(item.get("reason", status))
    sev = _coerce_severity(item.get("torso_low_severity", "none"))
    failed = bool(item.get("torso_low_failed", False)) or sev != "none"
    norm = item.get("shoulder_low_norm")
    px = item.get("shoulder_low_px")
    angle_delta = item.get("torso_angle_delta_deg")
    return (
        sev,
        failed,
        anchor,
        float(norm) if isinstance(norm, (int, float)) else None,
        float(px) if isinstance(px, (int, float)) else None,
        float(angle_delta) if isinstance(angle_delta, (int, float)) else None,
        status,
        f"torso_low_failed={failed}_sev={sev}_norm={norm}_px={px}",
    )


def _capped_severity(torso_sev: Severity, hip_sev: Severity) -> Severity:
    return _min(_max(torso_sev, hip_sev), INFERENCE_CAP_SEVERITY)


def _read_geometry_float(geom: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(geom, dict):
        return None
    v = geom.get(key)
    if isinstance(v, (int, float)) and _finite(float(v)):
        return float(v)
    return None


def _shoulder_hip_coherence(
    geom: dict[str, Any] | None,
) -> tuple[CoherenceVerdict, float | None, list[str]]:
    traces: list[str] = []
    s_u = _read_geometry_float(geom, "user_shoulder_drop_from_top_norm")
    s_i = _read_geometry_float(geom, "ideal_shoulder_drop_from_top_norm")
    h_u_raw = _read_geometry_float(geom, "user_hip_back_norm")
    h_i_raw = _read_geometry_float(geom, "ideal_hip_back_norm")
    traces.append(
        f"coherence s_u={s_u} s_i={s_i} h_u_raw={h_u_raw} h_i_raw={h_i_raw}"
    )
    if s_u is None or s_i is None or h_u_raw is None or h_i_raw is None:
        return "inconclusive", None, traces + ["coherence_missing_fields"]
    h_u = abs(h_u_raw)
    h_i = abs(h_i_raw)
    traces.append(f"coherence h_u_mag={h_u:.4f} h_i_mag={h_i:.4f}")
    if h_i < MIN_IDEAL_HIP_BACK_FOR_COHERENCE:
        return "inconclusive", None, traces + [f"coherence_hip_ideal_mag_below_{MIN_IDEAL_HIP_BACK_FOR_COHERENCE}"]
    if h_u < 1e-6:
        return "inconclusive", None, traces + ["coherence_user_hip_back_near_zero"]
    expected_s = s_i * (h_u / h_i)
    excess = s_u - expected_s
    traces.append(f"coherence_expected_shoulder={expected_s:.4f} excess={excess:.4f}")
    if excess < SHOULDER_HIP_EXCESS_LEVE:
        return "coherent", excess, traces
    return "incoherent", excess, traces


def _severity_from_shoulder_hip_excess(excess: float, torso_sev: Severity) -> Severity:
    if excess >= SHOULDER_HIP_EXCESS_GRAVE:
        return _max(torso_sev, "grave")
    if excess >= SHOULDER_HIP_EXCESS_MEDIA:
        return _max(torso_sev, "media")
    if excess >= SHOULDER_HIP_EXCESS_LEVE:
        return _max(torso_sev, "leve")
    return torso_sev


# --- Regla de inferencia por segmento --------------------------------------

def rule_segment(
    segment: str,
    *,
    hip_result: dict | None,
    knee_result: dict | None,
    neck_result: dict | None,
    spine_geometry: dict[str, Any] | None = None,
) -> SegmentRuling:
    hip_sev, hip_failed, hip_trace = _hip_segment_severity(hip_result, segment)
    knee_failed, knee_trace = _knee_segment_failed(knee_result, segment)
    neck_failed, neck_dir, neck_trace = _neck_segment_ruling(neck_result, segment)
    geom, geom_lookup_trace = _geometry_segment_evidence(spine_geometry, segment)
    (
        torso_sev,
        torso_failed,
        anchor,
        torso_norm,
        torso_px,
        angle_delta,
        geometry_status,
        geom_trace,
    ) = _severity_from_geometry(geom)

    trace = [geom_lookup_trace, geom_trace, hip_trace, knee_trace, neck_trace]

    base = SegmentRuling(
        segment=segment,
        severity="none",
        triggered=False,
        possible=False,
        reason="",
        hip_hinge_severity=hip_sev,
        hip_hinge_failed=hip_failed,
        knee_dominant_failed=knee_failed,
        neck_movement_failed=neck_failed,
        neck_direction=neck_dir,
        anchor=anchor or SEGMENT_TO_END_ANCHOR.get(segment, ""),
        torso_low_failed=torso_failed,
        torso_low_severity=torso_sev,
        torso_low_norm=torso_norm,
        torso_low_px=torso_px,
        torso_angle_delta_deg=angle_delta,
        geometry_status=geometry_status,
        trace=trace,
    )

    # PASO 1 — Geometría
    if geometry_status != "ok":
        base.reason = "spine_geometry_inconclusive"
        return base

    if not torso_failed:
        base.reason = "torso_not_lower_than_ideal"
        return base

    # PASO 2 — Exclusión knee / neck (posible, no confirmado)
    if knee_failed or neck_failed:
        base.possible = True
        base.triggered = False
        if torso_sev != "none":
            base.severity = torso_sev
        base.reason = "possible_spine_flexion_explained_by_knee_or_neck"
        return base

    # PASO 3A — Hip hinge NO falla: coherencia hombro–cadera vs patrón PM-Ideal
    if not hip_failed:
        coherence, excess, coh_trace = _shoulder_hip_coherence(geom)
        trace.extend(coh_trace)
        if coherence == "coherent":
            base.triggered = False
            base.possible = False
            base.severity = "none"
            base.reason = "shoulder_drop_coherent_with_hip_pattern_vs_ideal"
            trace.append("step=3A hip_ok coherent → no spine_flexion")
            base.trace = trace
            return base
        if coherence == "incoherent" and excess is not None:
            sev = _min(
                _severity_from_shoulder_hip_excess(excess, torso_sev),
                INFERENCE_CAP_SEVERITY,
            )
            if sev == "none":
                sev = "leve"
            base.severity = sev
            base.triggered = True
            base.reason = "shoulder_drop_exceeds_hip_coherence_vs_ideal"
            trace.append(f"step=3A hip_ok incoherent excess={excess:.4f} severity={sev}")
            base.trace = trace
            return base

        base.triggered = False
        base.possible = False
        base.severity = "none"
        base.reason = "shoulder_hip_coherence_not_computed_no_spine_confirmation"
        trace.append("step=3A hip_ok coherence_inconclusive → no spine_flexion (sin fallback)")
        base.trace = trace
        return base

    # PASO 3B — Hip hinge SÍ falla: bisagra insuficiente vs ideal → no justifica el hombro bajo
    delta_hip_back = _hip_delta_back(hip_result, segment)
    trace.append(f"delta_hip_back={delta_hip_back:.4f}")
    sev = _capped_severity(torso_sev, hip_sev)
    if sev == "none":
        sev = "leve"
    base.severity = sev
    base.triggered = True
    base.reason = "shoulder_low_with_insufficient_hip_hinge"
    trace.append(f"step=3B hip_failed → spine_flexion severity={sev}")
    base.trace = trace
    return base


# --- Inferencia por repetición --------------------------------------------

def detect_spine_flexion(
    *,
    hip_result: dict | None,
    knee_result: dict | None,
    neck_result: dict | None,
    spine_geometry: dict[str, Any] | None = None,
    segment_order: tuple[str, ...] = SEGMENT_ORDER,
) -> RepSpineFlexionVerdict:
    per_segment: dict[str, SegmentRuling] = {}
    triggered_segments: list[str] = []
    rep_severity: Severity = "none"

    for segment in segment_order:
        r = rule_segment(
            segment,
            hip_result=hip_result,
            knee_result=knee_result,
            neck_result=neck_result,
            spine_geometry=spine_geometry,
        )
        per_segment[segment] = r
        if r.triggered:
            triggered_segments.append(segment)
            rep_severity = _max(rep_severity, r.severity)

    possible_segments = [s for s, r in per_segment.items() if r.possible]

    if len(triggered_segments) >= MIN_SEGMENTS_FOR_REP:
        rep_detected = True
        rep_severity_final = _persistent_severity(
            [per_segment[s].severity for s in triggered_segments]
        )
    else:
        rep_detected = False
        rep_severity_final = "none"

    geometry_summary = _summarise_geometry(spine_geometry)
    evidence = {
        "spine_geometry": geometry_summary,
        "hip_hinge": _summarise_verdict(hip_result, "hip_hinge"),
        "knee_dominant": _summarise_verdict(knee_result, "knee_dominant"),
        "neck_movement": _summarise_verdict(neck_result, "neck_movement"),
    }

    trace = [
        f"triggered_segments = {triggered_segments}",
        f"possible_segments = {possible_segments}",
        f"min_segments_for_rep = {MIN_SEGMENTS_FOR_REP}",
        f"raw_max_severity = {rep_severity}",
        f"inference_cap = {INFERENCE_CAP_SEVERITY}",
        f"hip_justification_factor = {HIP_JUSTIFICATION_FACTOR}",
        f"final_severity = {rep_severity_final}",
    ]

    return RepSpineFlexionVerdict(
        detected=rep_detected,
        severity=rep_severity_final,
        method="inferred_by_geometry_and_exclusion",
        n_segments_triggered=len(triggered_segments),
        triggered_segments=triggered_segments,
        possible_segments=possible_segments,
        per_segment=per_segment,
        per_anchor=_build_per_anchor(per_segment, spine_geometry),
        evidence=evidence,
        trace=trace,
    )


def _persistent_severity(severities: list[Severity]) -> Severity:
    if not severities:
        return "none"
    n_grave = sum(1 for s in severities if s == "grave")
    n_media = sum(1 for s in severities if s == "media")
    n_leve = sum(1 for s in severities if s == "leve")

    if n_grave >= 2 or (n_grave >= 1 and n_media >= 1):
        return "grave"
    if (n_grave >= 1 and n_leve >= 1) or n_media >= 1 or n_leve >= 2:
        return "media"
    if n_leve >= 1:
        return "leve"
    return "none"


def _suppression_reason(r: SegmentRuling) -> str:
    if r.triggered:
        return "none"
    if r.possible:
        return "possible_spine_flexion_explained_by_knee_or_neck"
    if r.knee_dominant_failed:
        return "explained_by_knee_dominant"
    if r.neck_movement_failed:
        return "explained_by_neck_movement"
    if not r.hip_hinge_failed and r.torso_low_failed:
        if r.reason == "shoulder_drop_coherent_with_hip_pattern_vs_ideal":
            return "hip_shoulder_coherent"
        if r.reason == "shoulder_hip_coherence_not_computed_no_spine_confirmation":
            return "coherence_not_computed"
        return "hip_hinge_ok"
    return "none"


def _anchor_dict_from_ruling(r: SegmentRuling) -> dict[str, Any]:
    candidate = r.triggered or r.possible
    return {
        "torso_drop_vs_ideal": r.torso_low_norm,
        "torso_drop_px": r.torso_low_px,
        "torso_low": r.torso_low_failed,
        "hip_hinge_failed": r.hip_hinge_failed,
        "knee_dominant_failed": r.knee_dominant_failed,
        "neck_movement_failed": r.neck_movement_failed,
        "suppression_reason": _suppression_reason(r),
        "severity": (
            r.severity
            if r.triggered
            else (r.torso_low_severity if r.possible and r.torso_low_severity != "none" else "none")
        ),
        "torso_low_severity": r.torso_low_severity,
        "triggered": r.triggered,
        "possible": r.possible,
        "spine_candidate": candidate,
        "result": (
            f"spine_flexion_{r.severity}" if r.triggered else ("possible" if r.possible else "none")
        ),
        "segment": r.segment,
        "geometry_status": r.geometry_status,
    }


def _build_per_anchor(
    per_segment: dict[str, SegmentRuling],
    spine_geometry: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_anchor = {}
    if isinstance(spine_geometry, dict):
        raw = spine_geometry.get("_by_anchor", {})
        if isinstance(raw, dict):
            by_anchor = raw
    for anchor in ("ecc_0", "ecc_25", "ecc_50", "ecc_75", "ecc_100"):
        segment = None
        for s, end_anchor in SEGMENT_TO_END_ANCHOR.items():
            if end_anchor == anchor:
                segment = s
                break
        if segment and segment in per_segment:
            out[anchor] = _anchor_dict_from_ruling(per_segment[segment])
            continue
        geom = by_anchor.get(anchor, {}) if isinstance(by_anchor, dict) else {}
        torso_sev = _coerce_severity(geom.get("torso_low_severity", "none")) if isinstance(geom, dict) else "none"
        out[anchor] = {
            "torso_drop_vs_ideal": geom.get("shoulder_low_norm") if isinstance(geom, dict) else None,
            "torso_drop_px": geom.get("shoulder_low_px") if isinstance(geom, dict) else None,
            "torso_low": bool(geom.get("torso_low_failed", False)) if isinstance(geom, dict) else False,
            "hip_hinge_failed": False,
            "knee_dominant_failed": False,
            "neck_movement_failed": False,
            "suppression_reason": "none",
            "severity": "none",
            "torso_low_severity": torso_sev,
            "triggered": False,
            "possible": False,
            "spine_candidate": False,
            "result": "none",
            "segment": segment,
            "geometry_status": geom.get("status", "missing") if isinstance(geom, dict) else "missing",
        }
    return out


def _summarise_geometry(spine_geometry: dict[str, Any] | None) -> dict[str, Any]:
    if not spine_geometry:
        return {"available": False}
    out: dict[str, Any] = {"available": True, "segments": {}}
    for segment in SEGMENT_ORDER:
        item = spine_geometry.get(segment)
        if not isinstance(item, dict):
            continue
        out["segments"][segment] = {
            "anchor": item.get("anchor"),
            "status": item.get("status"),
            "torso_low_failed": item.get("torso_low_failed"),
            "torso_low_severity": item.get("torso_low_severity"),
            "shoulder_low_norm": item.get("shoulder_low_norm"),
            "shoulder_low_px": item.get("shoulder_low_px"),
            "user_frame": item.get("user_frame"),
            "ideal_frame": item.get("ideal_frame"),
        }
    return out


def _summarise_verdict(verdict: Any, name: str) -> dict[str, Any]:
    if verdict is None:
        return {"available": False, "name": name}
    summary: dict[str, Any] = {
        "available": True,
        "name": name,
        "detected": bool(_read_field(verdict, "detected", False)),
        "severity": _read_field(verdict, "severity", "none"),
    }
    if name == "neck_movement":
        summary["subtype"] = _read_field(verdict, "subtype", "none")
        summary["neck_direction"] = _read_field(verdict, "neck_direction", "neutral")
    return summary
