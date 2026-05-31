
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from src.biomechanics.rdl.detectors.arms.metrics import (
    BENT_ARMS_ANCHORS,
    BentArmsAnchorMetrics,
)

Severity = Literal["none", "leve", "media", "grave"]

BENT_ARMS_FAIL_THR = 160.0
BENT_ARMS_MEDIA_THR = 150.0
BENT_ARMS_GRAVE_THR = 140.0
BENT_ARMS_MIN_FAILED_ANCHORS = 2


@dataclass
class BentArmsAnchorRuling:
    anchor: str
    angle_elbow: float
    severity: Severity
    failed: bool
    grave: bool
    trace: list[str] = field(default_factory=list)


@dataclass
class BentArmsRepVerdict:
    detected: bool
    severity: Severity
    magnitude: float
    confidence: float
    n_failed: int
    failed_anchors: list[str]
    per_anchor: dict[str, BentArmsAnchorRuling] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def rule_bent_arms_anchor(
    metrics: BentArmsAnchorMetrics,
) -> BentArmsAnchorRuling:
    trace: list[str] = []
    a = metrics.angle_elbow
    severity: Severity = "none"
    if _finite(a):
        trace.append(f"angle_elbow={a:.1f} fail<{BENT_ARMS_FAIL_THR:.1f}")
        failed = a < BENT_ARMS_FAIL_THR
        grave = a < BENT_ARMS_GRAVE_THR
        if grave:
            severity = "grave"
        elif a < BENT_ARMS_MEDIA_THR:
            severity = "media"
        elif failed:
            severity = "leve"
    else:
        trace.append("anchor no disponible: angle_elbow=nan")
        failed = False
        grave = False
        severity = "none"
    return BentArmsAnchorRuling(
        anchor=metrics.anchor,
        angle_elbow=float(a) if _finite(a) else float("nan"),
        severity=severity,
        failed=failed,
        grave=grave,
        trace=trace,
    )


def detect_bent_arms_from_metrics(
    anchor_metrics: dict[str, BentArmsAnchorMetrics],
) -> BentArmsRepVerdict:
    trace: list[str] = []
    per_anchor: dict[str, BentArmsAnchorRuling] = {}
    evaluated_anchors = 0
    for a in BENT_ARMS_ANCHORS:
        m = anchor_metrics.get(a)
        if m is None:
            per_anchor[a] = BentArmsAnchorRuling(
                anchor=a, angle_elbow=float("nan"),
                severity="none",
                failed=False, grave=False,
                trace=[f"{a}: anchor no disponible"],
            )
        else:
            per_anchor[a] = rule_bent_arms_anchor(m)
            if _finite(m.angle_elbow):
                evaluated_anchors += 1

    failed = [r for r in per_anchor.values() if r.failed]
    n_failed = len(failed)
    failed_anchors = [r.anchor for r in failed]
    trace.append(f"failed={n_failed} evaluated={evaluated_anchors}")

    confidence = (n_failed / float(evaluated_anchors)) if evaluated_anchors > 0 else 0.0

    if n_failed < BENT_ARMS_MIN_FAILED_ANCHORS:
        return BentArmsRepVerdict(
            detected=False, severity="none",
            magnitude=0.0, confidence=confidence,
            n_failed=n_failed, failed_anchors=failed_anchors,
            per_anchor=per_anchor,
            trace=trace + [f"n_failed<{BENT_ARMS_MIN_FAILED_ANCHORS}"],
        )

    deficits = [BENT_ARMS_FAIL_THR - r.angle_elbow for r in failed if _finite(r.angle_elbow)]
    magnitude = float(sum(deficits) / len(deficits)) if deficits else 0.0
    rank = {"none": 0, "leve": 1, "media": 2, "grave": 3}
    severity = max((r.severity for r in failed), key=lambda s: rank.get(s, 0), default="none")
    trace.append(f"magnitude={magnitude:.3f}")
    trace.append(f"severity={severity}")

    return BentArmsRepVerdict(
        detected=True,
        severity=severity,
        magnitude=magnitude,
        confidence=confidence,
        n_failed=n_failed,
        failed_anchors=failed_anchors,
        per_anchor=per_anchor,
        trace=trace,
    )
