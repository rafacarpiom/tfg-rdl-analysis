from __future__ import annotations

import math

from src.biomechanics.rdl.detectors.lockout.metrics import LockoutMetrics
from src.biomechanics.rdl.detectors.lockout.rules import detect_no_lockout


def _metrics_error_deg(err_deg: float) -> LockoutMetrics:
    err_rad = math.radians(err_deg)
    return LockoutMetrics(
        theta_end_user=0.0,
        theta_end_ideal=0.0,
        error_lockout=err_rad,
    )


def test_lockout_severity_none_below_leve():
    v = detect_no_lockout(_metrics_error_deg(4.9))
    assert v.severity == "none"
    assert not v.detected


def test_lockout_severity_leve():
    v = detect_no_lockout(_metrics_error_deg(5.0))
    assert v.severity == "leve"
    assert v.detected
    v2 = detect_no_lockout(_metrics_error_deg(7.4))
    assert v2.severity == "leve"


def test_lockout_severity_media():
    # 7.6 evita borde float (degrees(radians(7.5)) puede quedar < 7.5).
    v = detect_no_lockout(_metrics_error_deg(7.6))
    assert v.severity == "media"
    v2 = detect_no_lockout(_metrics_error_deg(10.0))
    assert v2.severity == "media"


def test_lockout_severity_grave():
    v = detect_no_lockout(_metrics_error_deg(10.01))
    assert v.severity == "grave"
