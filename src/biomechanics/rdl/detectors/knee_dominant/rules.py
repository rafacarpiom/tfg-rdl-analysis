
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from src.biomechanics.rdl.detectors.knee_dominant.metrics import KNEE_DOMINANT_ANCHORS, KneeDominantMetrics

# ── Tipos ────────────────────────────────────────────────────────────────────

Severity = Literal["none", "leve", "media", "grave"]

ORDERED_ANCHORS: tuple[str, ...] = (
    *KNEE_DOMINANT_ANCHORS,
)

# Umbral único por ancla (grados).
KNEE_EXCESS_THR: float = 10.0

# Cortes de severidad por magnitud agregada (grados).
SEV_THR: dict[str, float] = {
    "leve":  10.0,
    "media": 14.0,
    "grave": 18.0,
}


# ── Clases de datos ──────────────────────────────────────────────────────────────

@dataclass
class AnchorRuling:

    anchor: str
    failed: bool
    delta_knee: float
    delta_hip: float             # solo informativo, no decide
    reject_reason: str = ""      # "" si failed=True o no evaluado
    trace: list[str] = field(default_factory=list)


@dataclass
class RepKneeDominantVerdict:

    detected: bool
    severity: Severity
    confidence: float              # n_failed / len(ORDERED_ANCHORS)  ∈ [0, 1]
    phase: str                     # always "ecc"
    magnitude: float               # magnitud agregada (grados)
    mean_knee: float               # mean delta_knee over failed anchors
    max_knee: float                # max  delta_knee over failed anchors
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


def classify_knee_anchor(delta_knee: float) -> Severity:
    if not _finite(delta_knee):
        return "none"
    if delta_knee > 18.0:
        return "grave"
    if delta_knee > 14.0:
        return "media"
    if delta_knee > 10.0:
        return "leve"
    return "none"


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

def rule_anchor(m: KneeDominantMetrics) -> AnchorRuling:
    trace: list[str] = []
    dk = m.delta_knee
    dh = m.delta_hip

    dk_txt = f"{dk:+.2f}°" if _finite(dk) else "nan"
    dh_txt = f"{dh:+.2f}°" if _finite(dh) else "nan"
    trace.append(f"Δknee = {dk_txt}   Δhip = {dh_txt}  (Δhip informativo)")

    if not _finite(dk):
        trace.append("  →  Δknee no finito  →  no-fail")
        return AnchorRuling(
            anchor=m.anchor, failed=False,
            delta_knee=dk, delta_hip=dh,
            reject_reason="datos_insuficientes", trace=trace,
        )

    if dk <= KNEE_EXCESS_THR:
        trace.append(
            f"  Δknee ≤ {KNEE_EXCESS_THR:+.0f}°  →  no supera umbral  →  no-fail"
        )
        return AnchorRuling(
            anchor=m.anchor, failed=False,
            delta_knee=dk, delta_hip=dh,
            reject_reason="knee_bajo_umbral", trace=trace,
        )

    trace.append(
        f"  Δknee > {KNEE_EXCESS_THR:+.0f}°  →  FALLO (exceso de flexión de rodilla)"
    )
    return AnchorRuling(
        anchor=m.anchor, failed=True,
        delta_knee=dk, delta_hip=dh,
        reject_reason="",
        trace=trace,
    )


# ── Clasificación por repetición ─────────────────────────────────────────────────

def detect_knee_dominant(
    anchor_metrics: dict[str, KneeDominantMetrics],
) -> RepKneeDominantVerdict:
    trace: list[str] = []

    per_anchor: dict[str, AnchorRuling] = {}
    ordered: list[AnchorRuling] = []
    for a in ORDERED_ANCHORS:
        m = anchor_metrics.get(a)
        if m is None:
            ruling = AnchorRuling(
                anchor=a, failed=False,
                delta_knee=float("nan"), delta_hip=float("nan"),
                reject_reason="anchor_no_disponible",
                trace=[f"{a}: anchor no disponible"],
            )
        else:
            ruling = rule_anchor(m)
        per_anchor[a] = ruling
        ordered.append(ruling)

    failed_rulings = [r for r in ordered if r.failed]
    n_failed = len(failed_rulings)
    confidence = round(n_failed / float(len(ORDERED_ANCHORS)), 3)

    persistence_ok = _has_consecutive_failures(ordered, min_run=2)
    trace.append(
        "Persistencia: "
        + ("OK (≥2 anchors consecutivos)" if persistence_ok
           else "no hay 2 consecutivos → ruido")
    )

    if not persistence_ok or n_failed == 0:
        return RepKneeDominantVerdict(
            detected=False,
            severity="none",
            confidence=confidence,
            phase="ecc",
            magnitude=0.0,
            mean_knee=0.0,
            max_knee=0.0,
            n_failed=n_failed,
            per_anchor=per_anchor,
            trace=trace,
        )

    knees = [r.delta_knee for r in failed_rulings if _finite(r.delta_knee)]
    if not knees:
        return RepKneeDominantVerdict(
            detected=False,
            severity="none",
            confidence=confidence,
            phase="ecc",
            magnitude=0.0,
            mean_knee=0.0,
            max_knee=0.0,
            n_failed=n_failed,
            per_anchor=per_anchor,
            trace=trace,
        )

    mean_knee = float(sum(knees) / len(knees))
    max_knee  = float(max(knees))
    magnitude = 0.7 * mean_knee + 0.3 * max_knee

    trace.append(
        f"mean Δknee = {mean_knee:.2f}°   max Δknee = {max_knee:.2f}°"
    )
    trace.append(
        f"magnitud = 0.7·mean + 0.3·max = {magnitude:.2f}°"
    )

    severity = _severity_from_magnitude(magnitude)
    trace.append(f"severidad (por magnitud agregada) = {severity}")

    return RepKneeDominantVerdict(
        detected=True,
        severity=severity,
        confidence=confidence,
        phase="ecc",
        magnitude=magnitude,
        mean_knee=mean_knee,
        max_knee=max_knee,
        n_failed=n_failed,
        per_anchor=per_anchor,
        trace=trace,
    )


# Alias retrocompatible.
classify_rep = detect_knee_dominant
