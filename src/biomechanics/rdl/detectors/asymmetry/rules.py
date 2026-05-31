
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["none", "posible", "leve", "media", "grave"]

_ORDER: tuple[Severity, ...] = ("none", "posible", "leve", "media", "grave")

# ── Umbrales ────────────────────────────────────────────────────────────────

# forward_diff_norm por frame (umbrales elevados: menos falsos positivos)
NORM_THR: dict[str, float] = {
    "leve":  0.11,   # antes media
    "media": 0.18,   # antes grave
    "grave": 0.26,   # grave solo con asimetría muy marcada
}

# Umbrales de ratio por fase
RATIO_THR: dict[str, float] = {
    "leve":  0.45,
    "media": 0.72,
    "grave": 0.85,
}

# Consistencia temporal del lado dominante entre frames asimétricos.
CONSISTENCY_THR: float = 0.70

# Desbalance lateral en TODOS los frames válidos — filtra consistencia alta
# solo por pocos frames asimétricos.
SIDE_IMBALANCE_THR_MEDIA: float = 0.55
SIDE_IMBALANCE_THR_GRAVE: float = 0.68

# Estabilidad: desv. típica de forward_diff_norm en porción asimétrica.
# Por encima → señal ruidosa, bajar un nivel.
STABILITY_STD_THR: float = 0.05

# Degradación por estabilidad solo si ratio no trivial y dirección pasa gates;
# si no, la varianza ya quedó filtrada y no contar dos veces.
STABILITY_MIN_RATIO: float = 0.40


# ── Clasificación por frame ──────────────────────────────────────────────────

def frame_severity(fwd_norm: float, confident: bool) -> Severity:
    if fwd_norm != fwd_norm:  # protección NaN
        return "none"
    if not confident:
        return "posible" if fwd_norm > NORM_THR["grave"] else "none"
    if fwd_norm > NORM_THR["grave"]:
        return "grave"
    if fwd_norm > NORM_THR["media"]:
        return "media"
    if fwd_norm > NORM_THR["leve"]:
        return "leve"
    return "none"


# ── Solo ratio de fase (agregado legacy) ───────────────────────────────────────

def phase_severity(ratios: dict[str, float]) -> tuple[Severity, str]:
    if not ratios:
        return "none", ""
    worst_phase = max(ratios, key=lambda p: ratios[p])
    worst_ratio = ratios[worst_phase]
    above_media = [p for p, r in ratios.items() if r > RATIO_THR["media"]]
    if worst_ratio > RATIO_THR["grave"] or len(above_media) >= 2:
        return "grave", worst_phase
    if worst_ratio > RATIO_THR["media"]:
        return "media", worst_phase
    if worst_ratio > RATIO_THR["leve"]:
        return "leve", worst_phase
    return "none", worst_phase


# ── Clasificador de grupo multi-señal ────────────────────────────────────

@dataclass(frozen=True)
class GroupClassification:

    severity: Severity
    magnitude: float            # 0.7·media + 0.3·máx de fwd_norm primario
    ratio: float                # frames asimétricos / válidos
    consistency: float          # ratio lado dominante entre frames asimétricos
    side_imbalance: float       # abs(L-R) / frames válidos
    stability_std: float        # desv. típica solo en frames asimétricos
    dominant_side: str          # "L" | "R" | "none"
    dominant_phase: str
    confidence_label: str       # "high" | "mixed" | "low"


def classify_group(
    *,
    magnitude: float,
    ratio: float,
    consistency: float,
    side_imbalance: float,
    stability_std: float,
    dominant_side: str,
    dominant_phase: str,
    confidence_label: str,
    multi_phase_above_media: bool = False,
) -> GroupClassification:
    direction_ok_media = (
        consistency >= CONSISTENCY_THR
        and side_imbalance >= SIDE_IMBALANCE_THR_MEDIA
    )
    direction_ok_grave = (
        consistency >= CONSISTENCY_THR
        and side_imbalance >= SIDE_IMBALANCE_THR_GRAVE
    )

    severity: Severity = "none"

    grave_ok = (
        magnitude > NORM_THR["grave"]
        and ratio > RATIO_THR["grave"]
        and direction_ok_grave
    )
    media_ok = (
        magnitude > NORM_THR["media"]
        and ratio > RATIO_THR["media"]
        and direction_ok_media
    )
    leve_ok = magnitude > NORM_THR["leve"] and ratio > RATIO_THR["leve"]

    if grave_ok:
        severity = "grave"
    elif media_ok:
        severity = "media"
    elif leve_ok:
        severity = "leve"
    else:
        severity = "none"

    # Magnitud/ratio en rango MEDIA pero falla gate MEDIA → LEVE (variación normal), no escalar.
    media_like = magnitude > NORM_THR["media"] and ratio > RATIO_THR["media"]
    if media_like and not media_ok and severity in ("none", "posible"):
        severity = "leve"

    # ── Penalización por inestabilidad ──────────────────────────────────────────────────
    # Degradar solo si señal sustancial Y pasa gate de dirección;
    # si no, no hay señal real y se contaría doble frente al filtro previo.
    if (
        severity in ("grave", "media")
        and stability_std > STABILITY_STD_THR
        and ratio > STABILITY_MIN_RATIO
        and direction_ok_media
    ):
        severity = _demote(severity)

    # Baja confianza limita severidad a "posible".
    if confidence_label == "low" and severity not in ("none", "posible"):
        severity = "posible"

    return GroupClassification(
        severity=severity,
        magnitude=magnitude,
        ratio=ratio,
        consistency=consistency,
        side_imbalance=side_imbalance,
        stability_std=stability_std,
        dominant_side=dominant_side,
        dominant_phase=dominant_phase,
        confidence_label=confidence_label,
    )


def _demote(sev: Severity) -> Severity:
    demote_map: dict[Severity, Severity] = {
        "grave":   "media",
        "media":   "leve",
        "leve":    "posible",
        "posible": "none",
        "none":    "none",
    }
    return demote_map[sev]


# ── Utilidades ─────────────────────────────────────────────────────────────────

def max_severity(a: Severity, b: Severity) -> Severity:
    return _ORDER[max(_ORDER.index(a), _ORDER.index(b))]


def stability_label(std: float) -> str:
    if std != std:  # NaN
        return "low"
    return "high" if std <= STABILITY_STD_THR else "low"


def severity_color(sev: Severity) -> str:
    return {
        "none":    "#2ecc71",
        "posible": "#f39c12",
        "leve":    "#f1c40f",
        "media":   "#e67e22",
        "grave":   "#e74c3c",
    }.get(sev, "gray")
