
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from src.biomechanics.rdl.detectors.lockout.metrics import LockoutMetrics

Severity = Literal["none", "leve", "media", "grave"]

# Umbrales en grados (error_lockout = max(0, θ_user − θ_ideal)).
LOCKOUT_LEVE_DEG:  float = 5.0
LOCKOUT_MEDIA_DEG: float = 7.5
LOCKOUT_GRAVE_DEG: float = 10.0


@dataclass
class RepNoLockoutVerdict:
    detected: bool
    severity: Severity
    magnitude: float       # error_lockout_pos in degrees
    confidence: float
    error_lockout_deg: float
    theta_end_user_deg: float
    theta_end_ideal_deg: float
    trace: list[str]


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def detect_no_lockout(m: LockoutMetrics) -> RepNoLockoutVerdict:
    trace: list[str] = []
    if not _finite(m.error_lockout):
        return RepNoLockoutVerdict(
            detected=False, severity="none",
            magnitude=float("nan"), confidence=0.0,
            error_lockout_deg=float("nan"),
            theta_end_user_deg=float("nan"),
            theta_end_ideal_deg=float("nan"),
            trace=["error_lockout = nan (keypoints insuficientes)"],
        )

    err_deg_raw = math.degrees(m.error_lockout)
    err_deg = max(0.0, err_deg_raw)
    teu_deg = math.degrees(m.theta_end_user) if _finite(m.theta_end_user) else float("nan")
    tei_deg = math.degrees(m.theta_end_ideal) if _finite(m.theta_end_ideal) else float("nan")
    trace.append(
        f"θ_end_user = {teu_deg:+.2f}°   θ_end_ideal = {tei_deg:+.2f}°   "
        f"error_lockout_raw = {err_deg_raw:+.2f}°   error_lockout = {err_deg:.2f}°"
    )

    if err_deg > LOCKOUT_GRAVE_DEG:
        severity: Severity = "grave"
    elif err_deg >= LOCKOUT_MEDIA_DEG:
        severity = "media"
    elif err_deg >= LOCKOUT_LEVE_DEG:
        severity = "leve"
    else:
        severity = "none"

    detected = severity != "none"
    trace.append(
        f"umbrales: leve≥{LOCKOUT_LEVE_DEG:.1f}°  media≥{LOCKOUT_MEDIA_DEG:.1f}°  "
        f"grave>{LOCKOUT_GRAVE_DEG:.1f}°  →  severidad={severity}"
    )
    return RepNoLockoutVerdict(
        detected=detected,
        severity=severity,
        magnitude=float(err_deg),
        confidence=1.0,
        error_lockout_deg=float(err_deg),
        theta_end_user_deg=float(teu_deg) if _finite(teu_deg) else float("nan"),
        theta_end_ideal_deg=float(tei_deg) if _finite(tei_deg) else float("nan"),
        trace=trace,
    )
