
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from src.biomechanics.rdl.detectors.bar_far.metrics import BarFarAnchorMetrics

Severity = Literal["none", "leve", "media", "grave"]
Confidence = Literal["normal", "alta"]

# ── Umbrales ────────────────────────────────────────────────────────────────

NORM_THR: dict[str, float] = {
    "leve":  0.20,
    "media": 0.40,
    "grave": 0.60,
}

ELBOW_COMPENSATION_THR: float = 25.0  # grados
ARM_DIR_REINFORCE_THR:  float = 10.0  # grados (solo evidencia secundaria)
NOISE_NORM_MAX:         float = 0.10
NOISE_ELBOW_MAX:        float = 15.0  # grados
NOISE_ARM_DIR_MAX:      float = 8.0   # grados

_PROMOTE: dict[Severity, Severity] = {
    "none":  "leve",
    "leve":  "media",
    "media": "grave",
    "grave": "grave",
}


@dataclass
class BarFarAnchorVerdict:

    base_severity: Severity
    severity: Severity
    confidence: Confidence
    trace: list[str] = field(default_factory=list)
    applied_rules: list[str] = field(default_factory=list)


# ── Helpers internos ──────────────────────────────────────────────────────────

def _finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _base_severity(norm: float) -> Severity:
    if not _finite(norm):
        return "none"
    if norm > NORM_THR["grave"]:
        return "grave"
    if norm > NORM_THR["media"]:
        return "media"
    if norm > NORM_THR["leve"]:
        return "leve"
    return "none"


# ── API pública ────────────────────────────────────────────────────────────────

def classify_bar_far_anchor(metrics: BarFarAnchorMetrics) -> BarFarAnchorVerdict:
    x_norm = metrics.wrist_error_x_norm
    dx = metrics.delta_x_wrist
    elbow_d = metrics.elbow_angle_delta
    arm_d = metrics.arm_dir_delta

    trace: list[str] = []
    applied: list[str] = []

    # ── 1. Clasificación base (solo horizontal) ────────────────────────────
    base = _base_severity(x_norm)
    n_txt = f"{x_norm:.3f}" if _finite(x_norm) else "nan"
    trace.append("CLASIFICACIÓN BASE (componente horizontal)")
    if base == "grave":
        trace.append(f"  n_x = {n_txt} > {NORM_THR['grave']}  →  grave")
    elif base == "media":
        trace.append(f"  n_x = {n_txt} > {NORM_THR['media']}  →  media")
    elif base == "leve":
        trace.append(f"  n_x = {n_txt} > {NORM_THR['leve']}  →  leve")
    else:
        trace.append(f"  n_x = {n_txt} ≤ {NORM_THR['leve']}  →  none")

    sev: Severity = base
    confidence: Confidence = "normal"

    # ── 2. Filtro de ruido ─────────────────────────────────────────────────────
    if (
        _finite(x_norm) and _finite(elbow_d) and _finite(arm_d)
        and x_norm < NOISE_NORM_MAX
        and elbow_d < NOISE_ELBOW_MAX
        and arm_d < NOISE_ARM_DIR_MAX
    ):
        old = sev
        sev = "none"
        trace.append("AJUSTE · ruido")
        trace.append(
            f"  n_x<{NOISE_NORM_MAX} & Δcodo<{NOISE_ELBOW_MAX}° "
            f"& Δdir<{NOISE_ARM_DIR_MAX}°  →  none"
        )
        if old != "none":
            trace.append(f"  {old} → none")
        applied.append("noise_filter")

    # ── 3. Compensación ─────────────────────────────────────────────────────
    if (
        _finite(x_norm) and _finite(elbow_d)
        and x_norm < NORM_THR["media"]
        and elbow_d > ELBOW_COMPENSATION_THR
        and sev in ("none", "leve")
    ):
        old = sev
        sev = _PROMOTE[sev]
        trace.append("AJUSTE · compensación")
        trace.append(
            f"  Δcodo={elbow_d:.1f}° > {ELBOW_COMPENSATION_THR}° "
            f"& n_x<{NORM_THR['media']}  →  compensación detectada"
        )
        trace.append(f"  {old} → {sev}")
        applied.append("compensation")

    # ── 4. Ajuste de dirección firmada ──────────────────────────────────────
    if _finite(x_norm) and _finite(dx):
        if x_norm > NORM_THR["media"] and dx > 0.0:
            confidence = "alta"
            trace.append("AJUSTE · dirección signada (empeora)")
            trace.append(
                f"  n_x>{NORM_THR['media']} & Δx_muñeca>0"
                "  →  confirma barra más alejada"
            )
            applied.append("signed_direction_worse")
        elif NORM_THR["leve"] < x_norm <= NORM_THR["media"] and dx > 0.0:
            trace.append("AJUSTE · zona gris (dirección)")
            trace.append(
                f"  {NORM_THR['leve']}<n_x<={NORM_THR['media']} & Δx_muñeca>0"
                "  →  leve real (se mantiene)"
            )
            applied.append("grey_zone_forward")
        elif x_norm > NORM_THR["leve"] and dx < 0.0:
            old = sev
            if sev == "grave":
                sev = "media"
            elif sev == "media":
                sev = "leve"
            elif sev == "leve":
                sev = "none"
            confidence = "normal"
            trace.append("AJUSTE · dirección signada (mejora)")
            trace.append(
                f"  n_x>{NORM_THR['leve']} & Δx_muñeca<0"
                "  →  muñeca más cerca del cuerpo"
            )
            if old != sev:
                trace.append(f"  {old} → {sev}")
            applied.append("signed_direction_better")

    # ── 5. Evidencia secundaria de dirección (solo magnitud) ────────────────────
    if (
        _finite(x_norm) and _finite(arm_d)
        and x_norm > NORM_THR["media"]
        and arm_d > ARM_DIR_REINFORCE_THR
    ):
        confidence = "alta"
        trace.append("AJUSTE · refuerzo secundario (|Δdir|)")
        trace.append(
            f"  |Δdir|={arm_d:.1f}° > {ARM_DIR_REINFORCE_THR}°"
            "  → evidencia adicional"
        )
        applied.append("direction_magnitude_support")

    return BarFarAnchorVerdict(
        base_severity=base,
        severity=sev,
        confidence=confidence,
        trace=trace,
        applied_rules=applied,
    )
