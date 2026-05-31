
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from src.biomechanics.rdl.detectors.hip_hinge.metrics import HIP_HINGE_ANCHORS, HipBackMetrics

# ── Tipos ────────────────────────────────────────────────────────────────────

Severity = Literal["none", "leve", "media", "grave"]
Confidence = Literal["baja", "media", "alta"]
DominantPhase = Literal["inicio bajada", "final bajada", "ninguna"]

# Anclas evaluadas (imports retrocompatibles).
ORDERED_ANCHORS: tuple[str, ...] = HIP_HINGE_ANCHORS

# Umbrales (delta_hip_back normalizado; usuario − ideal).
DELTA_FAIL_THR: float = 0.08
SEV_THR: dict[str, float] = {
    "leve":  0.08,
    "media": 0.15,
    "grave": 0.25,
}


# ── Clases de datos ──────────────────────────────────────────────────────────────

@dataclass
class AnchorRuling:

    anchor: str
    failed: bool
    delta_hip_back: float
    trace: list[str] = field(default_factory=list)


@dataclass
class RepHipHingeVerdict:

    detected: bool
    severity: Severity
    confidence: Confidence
    dominant_phase: DominantPhase
    magnitude: float
    mean_deficit: float
    max_deficit: float
    n_failed: int
    per_anchor: dict[str, AnchorRuling] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


# ── Auxiliares ──────────────────────────────────────────────────────────────────

def _finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _severity_from_magnitude(mag: float) -> Severity:
    if not _finite(mag):
        return "none"
    if mag > SEV_THR["grave"]:
        return "grave"
    if mag > SEV_THR["media"]:
        return "media"
    if mag > SEV_THR["leve"]:
        return "leve"
    return "none"


def _confidence(n_failed: int) -> Confidence:
    ratio = n_failed / float(len(ORDERED_ANCHORS))
    if ratio > 0.7:
        return "alta"
    if ratio >= 0.4:
        return "media"
    return "baja"


def _dominant_phase(failed_anchors: list[str]) -> DominantPhase:
    if not failed_anchors:
        return "ninguna"
    inicio = 0
    final = 0
    for a in failed_anchors:
        if a == "ecc_0":
            inicio += 1
        elif a == "ecc_25":
            inicio += 1
        elif a == "ecc_50":
            inicio += 1
            final += 1
        elif a == "ecc_75":
            final += 1
        elif a == "ecc_100":
            final += 1
        elif a == "bottom":
            final += 1
    if inicio > final:
        return "inicio bajada"
    return "final bajada"


def _has_consecutive_failures(
    ordered_rulings: list[AnchorRuling], min_run: int = 2
) -> bool:
    run = 0
    for r in ordered_rulings:
        if r.failed:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


# ── Regla por ancla ────────────────────────────────────────────────────────

def rule_anchor(m: HipBackMetrics) -> AnchorRuling:
    trace: list[str] = []
    d = m.delta_hip_back
    d_txt = f"{d:+.3f}" if _finite(d) else "nan"
    trace.append(
        f"Δhip_back = {d_txt}   (fail si > {DELTA_FAIL_THR:+.2f})"
    )

    failed = _finite(d) and d > DELTA_FAIL_THR
    if failed:
        trace.append("  →  FALLO (user no se desplaza hacia atrás lo suficiente)")
    else:
        trace.append("  →  no-fail")
    return AnchorRuling(
        anchor=m.anchor,
        failed=failed,
        delta_hip_back=float(d) if _finite(d) else float("nan"),
        trace=trace,
    )


# ── Clasificación por repetición ─────────────────────────────────────────────────

def detect_hip_hinge_from_trajectory(
    anchor_metrics: dict[str, HipBackMetrics],
) -> RepHipHingeVerdict:
    trace: list[str] = []

    per_anchor: dict[str, AnchorRuling] = {}
    ordered: list[AnchorRuling] = []
    for a in ORDERED_ANCHORS:
        m = anchor_metrics.get(a)
        if m is None:
            ruling = AnchorRuling(
                anchor=a, failed=False,
                delta_hip_back=float("nan"),
                trace=[f"{a}: anchor no disponible"],
            )
        else:
            ruling = rule_anchor(m)
        per_anchor[a] = ruling
        ordered.append(ruling)

    failed_rulings = [r for r in ordered if r.failed]
    n_failed = len(failed_rulings)

    persistence_ok = _has_consecutive_failures(ordered, min_run=2)
    trace.append(
        "Persistencia: "
        + ("OK (≥2 anchors consecutivos)" if persistence_ok
           else "no hay 2 consecutivos → ruido")
    )

    if not persistence_ok or n_failed == 0:
        return RepHipHingeVerdict(
            detected=False,
            severity="none",
            confidence=_confidence(n_failed),
            dominant_phase=_dominant_phase([r.anchor for r in failed_rulings]),
            magnitude=0.0,
            mean_deficit=0.0,
            max_deficit=0.0,
            n_failed=n_failed,
            per_anchor=per_anchor,
            trace=trace,
        )

    deltas = [abs(r.delta_hip_back) for r in failed_rulings if _finite(r.delta_hip_back)]
    # delta_hip_back de anclas fallidas es positivo (> umbral);
    # abs() es red de seguridad — en la práctica no cambia valores.
    if not deltas:
        return RepHipHingeVerdict(
            detected=False,
            severity="none",
            confidence=_confidence(n_failed),
            dominant_phase=_dominant_phase([r.anchor for r in failed_rulings]),
            magnitude=0.0,
            mean_deficit=0.0,
            max_deficit=0.0,
            n_failed=n_failed,
            per_anchor=per_anchor,
            trace=trace,
        )

    mean_deficit = float(sum(deltas) / len(deltas))
    max_deficit = float(max(deltas))
    magnitude = 0.7 * mean_deficit + 0.3 * max_deficit

    trace.append(
        f"mean Δhip_back = {mean_deficit:.3f}   max Δhip_back = {max_deficit:.3f}"
    )
    trace.append(
        f"magnitud = 0.7·mean + 0.3·max = {magnitude:.3f}"
    )

    severity = _severity_from_magnitude(magnitude)
    trace.append(f"severidad (por magnitud agregada) = {severity}")

    return RepHipHingeVerdict(
        detected=True,
        severity=severity,
        confidence=_confidence(n_failed),
        dominant_phase=_dominant_phase([r.anchor for r in failed_rulings]),
        magnitude=magnitude,
        mean_deficit=mean_deficit,
        max_deficit=max_deficit,
        n_failed=n_failed,
        per_anchor=per_anchor,
        trace=trace,
    )


# Alias retrocompatible (nombre antiguo en __init__.py).
classify_rep = detect_hip_hinge_from_trajectory
