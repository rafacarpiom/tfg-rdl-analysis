
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from src.biomechanics.rdl.detectors.neck_movement.metrics import (
    NECK_DIRECTION_SIGN,
    NECK_SEGMENTS,
    ORDERED_ANCHORS,
    NeckMovementMetrics,
    NeckPoseState,
    classify_neck_direction,
    wrap_to_180,
)

Severity = Literal["none", "leve", "media", "grave"]

DETECTION_ANCHORS: tuple[str, ...] = (
    "ecc_25",
    "ecc_50",
    "ecc_75",
    "ecc_100",
)
SEGMENT_ORDER: tuple[str, ...] = tuple(
    f"{a}_to_{b}" for a, b in NECK_SEGMENTS
)

# Umbrales de contexto por compatibilidad/debug.
THRESHOLD_BETA_MIN: float = math.radians(3.0)
THRESHOLD_GAMMA_SAFE: float = math.radians(3.0)
HINGE_MIN_RATIO: float = 0.35
HINGE_MIN_PX: float = 5.0
FEMUR_MOVEMENT_MIN: float = math.radians(3.0)

# Umbrales para deriva/A (solo debug).
# Más estricto que el baseline (15/25/40): el detector era demasiado permisivo.
NECK_A_NONE_MAX_DEG: float = 6.0
NECK_A_LEVE_MAX_DEG: float = 12.0
NECK_A_MEDIA_MAX_DEG: float = 22.0

# Umbrales del clasificador B (comparación absoluta usuario vs ideal).
NECK_B_NONE_MAX_DEG: float = 12.0
NECK_B_LEVE_MAX_DEG: float = 22.0
NECK_B_MEDIA_MAX_DEG: float = 35.0

# Umbral de dirección (ambos clasificadores).
NECK_NONE_MAX_DEG: float = NECK_B_NONE_MAX_DEG   # alias legacy más abajo
NECK_LEVE_MAX_DEG: float = NECK_B_LEVE_MAX_DEG
NECK_MEDIA_MAX_DEG: float = NECK_B_MEDIA_MAX_DEG
NECK_DIRECTION_THRESHOLD_DEG: float = NECK_B_NONE_MAX_DEG
NECK_DIRECTION_THRESHOLD: float = math.radians(NECK_DIRECTION_THRESHOLD_DEG)

# Alias usados por pipeline/gráficos; métrica relativa eje cara/torso, no flexión de espalda.
BACK_CURL_LEVE_DEG: float = NECK_NONE_MAX_DEG
BACK_CURL_MEDIA_DEG: float = NECK_LEVE_MAX_DEG
BACK_CURL_GRAVE_DEG: float = NECK_MEDIA_MAX_DEG
BACK_CURL_LEVE: float = math.radians(BACK_CURL_LEVE_DEG)
BACK_CURL_MEDIA: float = math.radians(BACK_CURL_MEDIA_DEG)
BACK_CURL_GRAVE: float = math.radians(BACK_CURL_GRAVE_DEG)
BACK_CURL_USER_MIN_FOR_FLAG_DEG: float = NECK_NONE_MAX_DEG
BACK_CURL_USER_MIN_FOR_FLAG: float = math.radians(BACK_CURL_USER_MIN_FOR_FLAG_DEG)
EXCESS_BACK_CURL_LEVE_DEG: float = NECK_NONE_MAX_DEG
EXCESS_BACK_CURL_MEDIA_DEG: float = NECK_LEVE_MAX_DEG
EXCESS_BACK_CURL_GRAVE_DEG: float = NECK_MEDIA_MAX_DEG
EXCESS_BACK_CURL_LEVE: float = math.radians(EXCESS_BACK_CURL_LEVE_DEG)
EXCESS_BACK_CURL_MEDIA: float = math.radians(EXCESS_BACK_CURL_MEDIA_DEG)
EXCESS_BACK_CURL_GRAVE: float = math.radians(EXCESS_BACK_CURL_GRAVE_DEG)

# Persistencia del clasificador absoluto; constantes para configurar el resumen debug desde un solo sitio.
ABS_EXCESS_LEVE_DEG: float = NECK_NONE_MAX_DEG
ABS_EXCESS_MEDIA_DEG: float = NECK_LEVE_MAX_DEG
ABS_EXCESS_GRAVE_DEG: float = NECK_MEDIA_MAX_DEG
ABS_EXCESS_LEVE: float = math.radians(ABS_EXCESS_LEVE_DEG)
ABS_EXCESS_MEDIA: float = math.radians(ABS_EXCESS_MEDIA_DEG)
ABS_EXCESS_GRAVE: float = math.radians(ABS_EXCESS_GRAVE_DEG)
ABS_MIN_ANCHORS_LEVE: int = 2
ABS_MIN_ANCHORS_MEDIA: int = 2
ABS_MIN_ANCHORS_GRAVE: int = 2
MIN_CONSECUTIVE_FAILED_SEGMENTS: int = 2
LAST_SEGMENT_NAME: str = "ecc_75_to_ecc_100"

NeckSubtype = Literal[
    "neck_flexion_down",
    "neck_extension_up",
    "mixed_or_unclear",
    "none",
    "inconclusive",
]
NeckDirection = Literal["down", "up", "neutral", "mixed", "unclear"]

# Constantes legacy para lectores JSON antiguos; no deciden.
D_LEVE: float = 1.15
D_MEDIA: float = 1.35
D_GRAVE: float = 1.60
EXCESS_BETA_LEVE: float = math.radians(4.0)
EXCESS_BETA_MEDIA: float = math.radians(6.0)
EXCESS_BETA_GRAVE: float = math.radians(9.0)
EXCESS_BETA_GAMMA_ZERO_GRAVE: float = math.radians(8.0)
EXCESS_BETA_SMALL_GAMMA_EXTREME: float = math.radians(10.0)
GAMMA_NEAR_ZERO: float = math.radians(0.5)
SMALL_GAMMA_USER: float = math.radians(3.0)
NO_HINGE_TORSO_FEMUR_GRAVE_RATIO: float = 2.0
BETA_DIFF_LEVE: float = math.radians(4.0)
BETA_DIFF_MEDIA: float = math.radians(8.0)
BETA_DIFF_GRAVE: float = math.radians(12.0)


@dataclass
class AnchorRuling:

    anchor: str
    segment: str
    failed: bool
    severity: Severity
    delta_beta_diff: float
    delta_beta_user: float
    delta_beta_ideal: float
    delta_px_diff: float
    delta_px_user: float
    delta_px_ideal: float
    ratio_delta: float | None
    resultado_local: Severity = "none"
    d_ratio: float | None = None
    exceso_beta: float = float("nan")
    beta_esperado: float = float("nan")
    fase_evaluada: str = ""
    decision_phase: str = ""
    spine_flag: bool = False
    confirmed: bool = False
    reject_reason: str = ""
    no_hinge_flag: bool = False
    knee_dominant_flag: bool = False
    torso_global_diff: float = float("nan")
    delta_back_curl_user: float = float("nan")
    delta_back_curl_ideal: float = float("nan")
    excess_back_curl: float = float("nan")
    excess_back_curl_anchor: float = float("nan")
    sustained_chepa_flag: bool = False
    last_segment_nodding_flag: bool = False
    signed_excess_anchor: float = float("nan")
    signed_excess_delta: float = float("nan")
    classifier_A_value: float = float("nan")
    classifier_A_severity: Severity = "none"
    classifier_B_value: float = float("nan")
    classifier_B_severity: Severity = "none"
    theta_head_start: float = float("nan")
    theta_head_end: float = float("nan")
    theta_torso_start: float = float("nan")
    theta_torso_end: float = float("nan")
    neck_relative_start: float = float("nan")
    neck_relative_end: float = float("nan")
    delta_neck_relative: float = float("nan")
    selected_face_axis_start: str = "none"
    selected_face_axis_end: str = "none"
    selected_face_keypoints_start: tuple[int, int] = (-1, -1)
    selected_face_keypoints_end: tuple[int, int] = (-1, -1)
    nose_confidence_start: float = float("nan")
    nose_confidence_end: float = float("nan")
    face_ref_confidence_start: float = float("nan")
    face_ref_confidence_end: float = float("nan")
    neck_direction: str = "neutral"
    subtype: str = "none"
    user_angle: float = float("nan")
    ideal_angle: float = float("nan")
    B: float = float("nan")
    drift_from_start: float = float("nan")
    direction: str = "neutral"
    status: str = "ok"
    confidence: str = "unknown"
    trace: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.resultado_local == "none" and self.severity != "none":
            self.resultado_local = self.severity
        if self.d_ratio is None and self.ratio_delta is not None:
            self.d_ratio = self.ratio_delta
        if not self.fase_evaluada and self.decision_phase:
            self.fase_evaluada = self.decision_phase


@dataclass
class RepNeckMovementVerdict:

    detected: bool
    severity: Severity
    confidence: float
    phase: str
    magnitude: float
    n_failed: int
    failed_anchors: list[str] = field(default_factory=list)
    failed_segments: list[str] = field(default_factory=list)
    per_anchor: dict[str, AnchorRuling] = field(default_factory=dict)
    per_segment: dict[str, AnchorRuling] = field(default_factory=dict)
    subtype: str = "none"
    neck_direction: str = "neutral"
    mean_signed_excess: float = float("nan")
    method: str = "face_axis_relative_to_torso"
    trace: list[str] = field(default_factory=list)


RepSpineFlexionVerdict = RepNeckMovementVerdict


def _finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _rad_to_deg(v: float) -> float:
    return float(math.degrees(v)) if _finite(v) else float("nan")


def _severity_rank(s: Severity) -> int:
    return {"none": 0, "leve": 1, "media": 2, "grave": 3}[s]


def _rank_to_severity(rank: int) -> Severity:
    return ("none", "leve", "media", "grave")[max(0, min(3, rank))]


def _max_severity(a: Severity, b: Severity) -> Severity:
    return _rank_to_severity(max(_severity_rank(a), _severity_rank(b)))


def classify_neck_magnitude(value_rad: float) -> Severity:
    if not _finite(value_rad):
        return "none"
    deg = abs(_rad_to_deg(value_rad))
    if deg < NECK_B_NONE_MAX_DEG:
        return "none"
    if deg < NECK_B_LEVE_MAX_DEG:
        return "leve"
    if deg < NECK_B_MEDIA_MAX_DEG:
        return "media"
    return "grave"


def classify_neck_a_severity(magnitude_increase_rad: float) -> Severity:
    if not _finite(magnitude_increase_rad):
        return "none"
    deg = _rad_to_deg(magnitude_increase_rad)
    if deg < NECK_A_NONE_MAX_DEG:
        return "none"
    if deg < NECK_A_LEVE_MAX_DEG:
        return "leve"
    if deg < NECK_A_MEDIA_MAX_DEG:
        return "media"
    return "grave"


def classify_back_curl_severity(delta_back_curl_user_rad: float) -> Severity:
    return classify_neck_a_severity(delta_back_curl_user_rad)


def classify_anchor_severity(d_value: float) -> Severity:
    if not _finite(d_value):
        return "none"
    d = abs(float(d_value))
    if d < D_LEVE:
        return "none"
    if d < D_MEDIA:
        return "leve"
    if d < D_GRAVE:
        return "media"
    return "grave"


def classify_phase4_severity(
    d_value: float,
    excess_beta: float,
    gamma_user: float,
) -> Severity:
    _ = d_value, gamma_user
    return classify_neck_magnitude(excess_beta)


def _direction_label(value: float) -> str:
    return classify_neck_direction(
        value,
        threshold_deg=NECK_DIRECTION_THRESHOLD_DEG,
    )


def _subtype_from_direction(direction: str, severity: Severity) -> str:
    if severity == "none":
        return "none"
    if direction == "down":
        return "neck_flexion_down"
    if direction == "up":
        return "neck_extension_up"
    return "mixed_or_unclear"


def _confidence_label(*values: float) -> str:
    finite = [float(v) for v in values if _finite(v)]
    if not finite:
        return "unknown"
    m = min(finite)
    if m >= 0.75:
        return "alta"
    if m >= 0.50:
        return "media"
    return "baja"


def _consecutive_error_pair(rulings: list[AnchorRuling]) -> bool:
    previous_failed = False
    for r in rulings:
        failed = bool(r.failed and r.status == "ok")
        if failed and previous_failed:
            return True
        previous_failed = failed
    return False


def _empty_ruling(
    *,
    anchor: str,
    segment: str,
    decision_phase: str,
    reject_reason: str,
    extra_trace: list[str],
    m: NeckMovementMetrics | None = None,
) -> AnchorRuling:
    return AnchorRuling(
        anchor=anchor,
        segment=segment,
        failed=False,
        severity="none",
        delta_beta_diff=m.delta_beta_diff if m else float("nan"),
        delta_beta_user=m.delta_beta_user if m else float("nan"),
        delta_beta_ideal=m.delta_beta_ideal if m else float("nan"),
        delta_px_diff=m.delta_px_diff if m else float("nan"),
        delta_px_user=m.delta_px_user if m else float("nan"),
        delta_px_ideal=m.delta_px_ideal if m else float("nan"),
        ratio_delta=m.ratio_delta if m else None,
        decision_phase=decision_phase,
        reject_reason=reject_reason,
        trace=extra_trace,
        theta_head_start=m.theta_head_start if m else float("nan"),
        theta_head_end=m.theta_head_end if m else float("nan"),
        theta_torso_start=m.theta_torso_start if m else float("nan"),
        theta_torso_end=m.theta_torso_end if m else float("nan"),
        neck_relative_start=m.neck_relative_start if m else float("nan"),
        neck_relative_end=m.neck_relative_end if m else float("nan"),
        delta_neck_relative=m.delta_neck_relative_user if m else float("nan"),
        selected_face_axis_start=m.selected_face_axis_start if m else "none",
        selected_face_axis_end=m.selected_face_axis_end if m else "none",
        selected_face_keypoints_start=m.selected_face_keypoints_start if m else (-1, -1),
        selected_face_keypoints_end=m.selected_face_keypoints_end if m else (-1, -1),
        nose_confidence_start=m.nose_confidence_start if m else float("nan"),
        nose_confidence_end=m.nose_confidence_end if m else float("nan"),
        face_ref_confidence_start=m.face_ref_confidence_start if m else float("nan"),
        face_ref_confidence_end=m.face_ref_confidence_end if m else float("nan"),
        user_angle=m.neck_relative_end if m else float("nan"),
        ideal_angle=m.ideal_neck_relative_end if m else float("nan"),
        B=m.classifier_B_value if m else float("nan"),
        drift_from_start=m.classifier_A_value if m else float("nan"),
        direction="neutral",
        status="inconclusive",
        confidence="baja",
    )


def rule_segment(m: NeckMovementMetrics) -> AnchorRuling:
    base_trace = [
        f"segment = {m.segment}",
        "method = face_axis_relative_to_torso",
        f"selected_face_axis_start = {m.selected_face_axis_start}",
        f"selected_face_axis_end = {m.selected_face_axis_end}",
        f"theta_head_start = {_rad_to_deg(m.theta_head_start):+.2f}°",
        f"theta_head_end = {_rad_to_deg(m.theta_head_end):+.2f}°",
        f"theta_torso_start = {_rad_to_deg(m.theta_torso_start):+.2f}°",
        f"theta_torso_end = {_rad_to_deg(m.theta_torso_end):+.2f}°",
        f"neck_relative_start = {_rad_to_deg(m.neck_relative_start):+.2f}°",
        f"neck_relative_end = {_rad_to_deg(m.neck_relative_end):+.2f}°",
        f"drift_from_start/A = {_rad_to_deg(m.classifier_A_value):+.2f}°",
        f"B = user(anchor) - ideal(anchor) = {_rad_to_deg(m.classifier_B_value):+.2f}°",
    ]

    if not m.is_conclusive:
        return _empty_ruling(
            anchor=m.end_anchor,
            segment=m.segment,
            decision_phase="inconclusive_face_keypoints",
            reject_reason=m.inconclusive_reason or "face_axis_not_reliable",
            extra_trace=base_trace + [
                "No hay nariz + oreja/ojo fiables para este tramo."
            ],
            m=m,
        )

    severity_a = classify_neck_a_severity(m.classifier_A_value)
    severity_b = classify_neck_magnitude(m.classifier_B_value)
    severity = severity_b
    decision_phase = "classifier_B_anchor_vs_ideal"
    direction = _direction_label(m.classifier_B_value)
    subtype = _subtype_from_direction(direction, severity)
    failed = severity != "none"
    reject_reason = "" if failed else "neck_relative_below_threshold"
    confidence = _confidence_label(
        m.user_end.nose_confidence,
        m.user_end.face_ref_confidence,
        m.user_end.shoulder_confidence,
        m.user_end.hip_confidence,
        m.ideal_end.nose_confidence,
        m.ideal_end.face_ref_confidence,
        m.ideal_end.shoulder_confidence,
        m.ideal_end.hip_confidence,
    )

    trace = base_trace + [
        f"A drift severity (debug only) = {severity_a}",
        f"B severity (decides) = {severity_b}",
        f"final anchor severity = {severity}",
        f"direction from B = {direction}",
        f"direction_sign = {NECK_DIRECTION_SIGN:+.0f}",
    ]

    return AnchorRuling(
        anchor=m.end_anchor,
        segment=m.segment,
        failed=failed,
        severity=severity,
        delta_beta_diff=m.delta_beta_diff,
        delta_beta_user=m.delta_beta_user,
        delta_beta_ideal=m.delta_beta_ideal,
        delta_px_diff=m.delta_px_diff,
        delta_px_user=m.delta_px_user,
        delta_px_ideal=m.delta_px_ideal,
        ratio_delta=m.ratio_delta,
        d_ratio=m.ratio_delta,
        torso_global_diff=(
            m.user_end.beta - m.ideal_end.beta
            if _finite(m.user_end.beta) and _finite(m.ideal_end.beta)
            else float("nan")
        ),
        delta_back_curl_user=m.classifier_A_value,
        delta_back_curl_ideal=m.delta_neck_relative_ideal,
        excess_back_curl=m.excess_back_curl,
        excess_back_curl_anchor=m.classifier_B_value,
        signed_excess_anchor=m.classifier_B_value,
        signed_excess_delta=m.classifier_A_value,
        classifier_A_value=m.classifier_A_value,
        classifier_A_severity=severity_a,
        classifier_B_value=m.classifier_B_value,
        classifier_B_severity=severity_b,
        theta_head_start=m.theta_head_start,
        theta_head_end=m.theta_head_end,
        theta_torso_start=m.theta_torso_start,
        theta_torso_end=m.theta_torso_end,
        neck_relative_start=m.neck_relative_start,
        neck_relative_end=m.neck_relative_end,
        delta_neck_relative=m.delta_neck_relative_user,
        selected_face_axis_start=m.selected_face_axis_start,
        selected_face_axis_end=m.selected_face_axis_end,
        selected_face_keypoints_start=m.selected_face_keypoints_start,
        selected_face_keypoints_end=m.selected_face_keypoints_end,
        nose_confidence_start=m.nose_confidence_start,
        nose_confidence_end=m.nose_confidence_end,
        face_ref_confidence_start=m.face_ref_confidence_start,
        face_ref_confidence_end=m.face_ref_confidence_end,
        neck_direction=direction,
        subtype=subtype,
        user_angle=m.neck_relative_end,
        ideal_angle=m.ideal_neck_relative_end,
        B=m.classifier_B_value,
        drift_from_start=m.classifier_A_value,
        direction=direction,
        status="ok",
        confidence=confidence,
        resultado_local=severity,
        fase_evaluada=decision_phase,
        decision_phase=decision_phase,
        spine_flag=failed,
        confirmed=failed,
        reject_reason=reject_reason,
        trace=trace,
    )


def classify_sustained_absolute_chepa(
    segment_rulings: list[AnchorRuling],
) -> tuple[Severity, dict[str, int], list[str]]:
    finite = [
        abs(float(r.classifier_B_value))
        for r in segment_rulings
        if _finite(r.classifier_B_value)
    ]
    n_grave = sum(1 for v in finite if v >= ABS_EXCESS_GRAVE)
    n_media = sum(1 for v in finite if v >= ABS_EXCESS_MEDIA)
    n_leve = sum(1 for v in finite if v >= ABS_EXCESS_LEVE)
    severity: Severity = "none"
    if n_grave >= ABS_MIN_ANCHORS_GRAVE:
        severity = "grave"
    elif n_media >= ABS_MIN_ANCHORS_MEDIA:
        severity = "media"
    elif n_leve >= ABS_MIN_ANCHORS_LEVE:
        severity = "leve"
    counts = {
        "n_grave_25deg": n_grave,
        "n_media_15deg": n_media,
        "n_leve_8deg": n_leve,
        "n_anchors_with_data": len(finite),
    }
    trace = [
        "classifier_B_abs: abs(user_neck_relative - ideal_neck_relative)",
        f"n>=25°={n_grave}, n>=15°={n_media}, n>=8°={n_leve}",
        f"classifier_B_abs_severity={severity}",
    ]
    return severity, counts, trace


def _anchor0_ruling(first_metric: NeckMovementMetrics | None) -> AnchorRuling:
    if first_metric is None:
        return AnchorRuling(
            anchor="ecc_0",
            segment="start",
            failed=False,
            severity="none",
            delta_beta_diff=float("nan"),
            delta_beta_user=float("nan"),
            delta_beta_ideal=float("nan"),
            delta_px_diff=float("nan"),
            delta_px_user=float("nan"),
            delta_px_ideal=float("nan"),
            ratio_delta=None,
            decision_phase="missing_anchor",
            reject_reason="anchor_ecc_0_no_disponible",
            status="inconclusive",
            confidence="baja",
        )

    u: NeckPoseState = first_metric.user_start
    i: NeckPoseState = first_metric.ideal_start
    b_value = (
        math.radians(
            wrap_to_180(math.degrees(u.neck_relative - i.neck_relative))
        ) * first_metric.forward_sign
        if _finite(u.neck_relative) and _finite(i.neck_relative)
        else float("nan")
    )
    status = "ok" if u.status == "ok" and i.status == "ok" and _finite(b_value) else "inconclusive"
    severity = classify_neck_magnitude(b_value) if status == "ok" else "none"
    direction = _direction_label(b_value) if status == "ok" else "neutral"
    confidence = _confidence_label(
        u.nose_confidence,
        u.face_ref_confidence,
        u.shoulder_confidence,
        u.hip_confidence,
        i.nose_confidence,
        i.face_ref_confidence,
        i.shoulder_confidence,
        i.hip_confidence,
    )
    return AnchorRuling(
        anchor="ecc_0",
        segment="start",
        failed=severity != "none",
        severity=severity,
        delta_beta_diff=float("nan"),
        delta_beta_user=float("nan"),
        delta_beta_ideal=float("nan"),
        delta_px_diff=float("nan"),
        delta_px_user=float("nan"),
        delta_px_ideal=float("nan"),
        ratio_delta=None,
        decision_phase="classifier_B_anchor_vs_ideal" if status == "ok" else "inconclusive_keypoints",
        reject_reason="" if severity != "none" else ("neck_relative_below_threshold" if status == "ok" else "keypoints_not_reliable"),
        classifier_A_value=0.0 if status == "ok" else float("nan"),
        classifier_A_severity="none",
        classifier_B_value=b_value,
        classifier_B_severity=severity,
        neck_relative_start=u.neck_relative,
        neck_relative_end=u.neck_relative,
        selected_face_axis_start=u.selected_face_axis,
        selected_face_axis_end=u.selected_face_axis,
        selected_face_keypoints_start=u.selected_face_keypoints,
        selected_face_keypoints_end=u.selected_face_keypoints,
        nose_confidence_start=u.nose_confidence,
        nose_confidence_end=u.nose_confidence,
        face_ref_confidence_start=u.face_ref_confidence,
        face_ref_confidence_end=u.face_ref_confidence,
        neck_direction=direction,
        subtype=_subtype_from_direction(direction, severity),
        user_angle=u.neck_relative,
        ideal_angle=i.neck_relative,
        B=b_value,
        drift_from_start=0.0 if status == "ok" else float("nan"),
        direction=direction,
        status=status,
        confidence=confidence,
        confirmed=severity != "none",
        trace=[
            "anchor = ecc_0",
            "B = neck_relative_user(ecc_0) - neck_relative_ideal(ecc_0)",
            f"B = {_rad_to_deg(b_value):+.2f}°",
            f"severity = {severity}",
            f"direction = {direction}",
        ],
    )


def detect_neck_movement(
    segment_metrics: dict[str, NeckMovementMetrics],
) -> RepNeckMovementVerdict:
    per_segment: dict[str, AnchorRuling] = {}
    ordered: list[AnchorRuling] = []

    for segment in SEGMENT_ORDER:
        m = segment_metrics.get(segment)
        if m is None:
            start, _, end = segment.partition("_to_")
            ruling = AnchorRuling(
                anchor=end or segment,
                segment=segment,
                failed=False,
                severity="none",
                delta_beta_diff=float("nan"),
                delta_beta_user=float("nan"),
                delta_beta_ideal=float("nan"),
                delta_px_diff=float("nan"),
                delta_px_user=float("nan"),
                delta_px_ideal=float("nan"),
                ratio_delta=None,
                decision_phase="missing_segment",
                reject_reason=f"segmento_no_disponible_desde_{start}",
                trace=[f"{segment}: segmento no disponible"],
            )
        else:
            ruling = rule_segment(m)
        per_segment[segment] = ruling
        ordered.append(ruling)

    first_metric = next(iter(segment_metrics.values()), None)
    anchor0 = _anchor0_ruling(first_metric)
    anchors_ordered = [anchor0, *ordered]

    # La referencia ideal suele empezar con cabeza ligeramente baja en ecc_0.
    # Excluimos ecc_0 de la agregación para evitar falsos positivos en el ancla inicial
    # (severidad / anclas fallidas / confianza).
    ignored_for_decision = {"ecc_0"}
    anchors_for_decision = [r for r in anchors_ordered if r.anchor not in ignored_for_decision]

    conclusive = [
        r for r in anchors_for_decision
        if r.decision_phase == "classifier_B_anchor_vs_ideal" and r.status == "ok"
    ]
    confirmed = [r for r in conclusive if r.failed]
    if not conclusive:
        return RepNeckMovementVerdict(
            detected=False,
            severity="none",
            confidence=0.0,
            phase="ecc",
            magnitude=0.0,
            n_failed=0,
            failed_anchors=[],
            failed_segments=[],
            per_anchor={r.anchor: r for r in anchors_ordered},
            per_segment=per_segment,
            subtype="inconclusive",
            neck_direction="unclear",
            mean_signed_excess=float("nan"),
            trace=["No hay anchors concluyentes por keypoints de torso/cabeza."],
        )

    severity: Severity = "none"
    for r in confirmed:
        severity = _max_severity(severity, r.severity)

    signed_pool: list[float] = [
        float(r.classifier_B_value)
        for r in (confirmed or conclusive)
        if _finite(r.classifier_B_value)
    ]
    mean_signed = (
        float(sum(signed_pool) / len(signed_pool)) if signed_pool else float("nan")
    )

    if severity == "none":
        subtype = "none"
        neck_direction = "neutral"
    else:
        down_errors = [
            r for r in confirmed
            if _finite(r.classifier_B_value)
            and float(r.classifier_B_value) > NECK_DIRECTION_THRESHOLD
        ]
        up_errors = [
            r for r in confirmed
            if _finite(r.classifier_B_value)
            and float(r.classifier_B_value) < -NECK_DIRECTION_THRESHOLD
        ]
        if down_errors and up_errors and len(down_errors) == len(up_errors):
            neck_direction = "mixed"
            subtype = "mixed_or_unclear"
        elif len(down_errors) > len(up_errors):
            neck_direction = "down"
            subtype = "neck_flexion_down"
        elif len(up_errors) > len(down_errors):
            neck_direction = "up"
            subtype = "neck_extension_up"
        else:
            neck_direction = "mixed"
            subtype = "mixed_or_unclear"

        dominant = down_errors if neck_direction == "down" else up_errors if neck_direction == "up" else confirmed
        if len(dominant) < 2 and severity != "none":
            degraded = _rank_to_severity(_severity_rank(severity) - 1)
            severity = degraded
            if severity == "none":
                neck_direction = "neutral"
                subtype = "none"

    magnitude = 0.0
    for r in conclusive:
        if _finite(r.classifier_B_value):
            magnitude = max(magnitude, abs(float(r.classifier_B_value)))

    failed_segments = [] if severity == "none" else [r.segment for r in confirmed]
    failed_anchors = [] if severity == "none" else [r.anchor for r in confirmed]
    confidence = round(len(confirmed) / float(len(conclusive)), 3) if conclusive else 0.0
    if _consecutive_error_pair(anchors_for_decision):
        confidence = min(1.0, round(confidence + 0.15, 3))
    elif len(confirmed) == 1:
        confidence = min(confidence, 0.45)

    trace = [
        "method = face_axis_relative_to_torso",
        "decision = classifier_B_anchor_vs_ideal",
        f"ignored_anchors_for_decision = {sorted(ignored_for_decision)}",
        "drift_from_start/A = debug_only",
        f"failed_anchors = {failed_anchors}",
        f"final_severity = {severity}",
        f"mean_signed = {_rad_to_deg(mean_signed):+.2f}° -> {subtype}",
        f"max_abs_B = {_rad_to_deg(magnitude):+.2f}°",
    ]

    per_anchor: dict[str, AnchorRuling] = {
        r.anchor: r for r in sorted(
            anchors_ordered,
            key=lambda r: ORDERED_ANCHORS.index(r.anchor)
            if r.anchor in ORDERED_ANCHORS else 999,
        )
    }

    return RepNeckMovementVerdict(
        detected=severity != "none",
        severity=severity,
        confidence=confidence,
        phase="ecc",
        magnitude=float(magnitude),
        n_failed=0 if severity == "none" else len(confirmed),
        failed_anchors=failed_anchors,
        failed_segments=failed_segments,
        per_anchor=per_anchor,
        per_segment=per_segment,
        subtype=subtype,
        neck_direction=neck_direction,
        mean_signed_excess=mean_signed,
        trace=trace,
    )


classify_rep = detect_neck_movement
detect_spine_flexion = detect_neck_movement
rule_anchor = rule_segment
