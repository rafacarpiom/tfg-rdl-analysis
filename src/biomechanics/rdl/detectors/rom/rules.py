
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from src.biomechanics.rdl.detectors.rom.metrics import RomMetrics

Severity = Literal["none", "leve", "media", "grave"]

# Umbrales relativos (ideal conocido).
SEV_THR_NORM: dict[str, float] = {
    "leve":  0.85,
    "media": 0.75,
    "grave": 0.65,
}

# Reserva absoluta (sin ideal). Valores en radianes.
SEV_THR_ABS_RAD: dict[str, float] = {
    "leve":  math.radians(60.0),
    "media": math.radians(50.0),
    "grave": math.radians(40.0),
}


@dataclass
class RepShortRomVerdict:

    detected: bool
    severity: Severity
    magnitude: float        # ROM_norm si hay ideal, si no |ROM_user| en radianes
    confidence: float
    rom_user_abs: float
    rom_ideal_abs: float
    rom_norm: float
    used_ideal: bool
    trace: list[str]


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def detect_short_rom(m: RomMetrics) -> RepShortRomVerdict:
    trace: list[str] = []

    ru = m.rom_user_abs
    ri = m.rom_ideal_abs
    rn = m.rom_norm
    trace.append(
        f"|ROM_user| = {math.degrees(ru):.2f}°"
        if _finite(ru) else "|ROM_user| = nan"
    )
    if _finite(ri):
        trace.append(
            f"|ROM_ideal| = {math.degrees(ri):.2f}°   "
            f"ROM_norm = {rn:.3f}"
            if _finite(rn) else
            f"|ROM_ideal| = {math.degrees(ri):.2f}°   ROM_norm = nan"
        )

    if _finite(rn):
        # Clasificación relativa.
        severity: Severity
        if rn < SEV_THR_NORM["grave"]:
            severity = "grave"
        elif rn < SEV_THR_NORM["media"]:
            severity = "media"
        elif rn < SEV_THR_NORM["leve"]:
            severity = "leve"
        else:
            severity = "none"
        magnitude = float(rn)
        used_ideal = True
        trace.append(
            "clasificación relativa (umbrales "
            f"leve<{SEV_THR_NORM['leve']:.2f}  "
            f"media<{SEV_THR_NORM['media']:.2f}  "
            f"grave<{SEV_THR_NORM['grave']:.2f})"
        )
    elif _finite(ru):
        # Reserva absoluta.
        severity = "none"
        if ru < SEV_THR_ABS_RAD["grave"]:
            severity = "grave"
        elif ru < SEV_THR_ABS_RAD["media"]:
            severity = "media"
        elif ru < SEV_THR_ABS_RAD["leve"]:
            severity = "leve"
        magnitude = float(ru)
        used_ideal = False
        trace.append("clasificación absoluta (sin ideal disponible)")
    else:
        return RepShortRomVerdict(
            detected=False, severity="none",
            magnitude=float("nan"), confidence=0.0,
            rom_user_abs=float("nan"), rom_ideal_abs=float("nan"),
            rom_norm=float("nan"), used_ideal=False,
            trace=trace + ["datos insuficientes para ROM"],
        )

    detected = severity != "none"
    trace.append(f"severidad = {severity}   detected = {detected}")
    return RepShortRomVerdict(
        detected=detected,
        severity=severity,
        magnitude=magnitude,
        confidence=1.0,
        rom_user_abs=float(ru) if _finite(ru) else float("nan"),
        rom_ideal_abs=float(ri) if _finite(ri) else float("nan"),
        rom_norm=float(rn) if _finite(rn) else float("nan"),
        used_ideal=used_ideal,
        trace=trace,
    )
