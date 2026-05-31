
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Keypoints faciales COCO-17.
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4

# Cadena derecha usada en el pipeline RDL.
KP_SHOULDER = 6
KP_HIP = 12
KP_KNEE = 14

ORDERED_ANCHORS: tuple[str, ...] = (
    "ecc_0",
    "ecc_25",
    "ecc_50",
    "ecc_75",
    "ecc_100",
)

NECK_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("ecc_0", "ecc_25"),
    ("ecc_25", "ecc_50"),
    ("ecc_50", "ecc_75"),
    ("ecc_75", "ecc_100"),
)

DEFAULT_FACE_CONFIDENCE: float = 0.30
DEFAULT_BODY_CONFIDENCE: float = 0.30

# Convención de signo en vista lateral.
# Con y creciendo hacia abajo, desviación positiva = flexión cervical abajo.
# Si el dataset se grabó desde el lado opuesto e invierte etiquetas, cambiar esta constante a -1.
NECK_DIRECTION_SIGN: float = 1.0


@dataclass(frozen=True)
class FaceAxisSelection:

    valid: bool
    axis: str
    nose_idx: int = KP_NOSE
    ref_idx: int = -1
    ref_name: str = ""
    nose_confidence: float = float("nan")
    ref_confidence: float = float("nan")
    reason: str = ""


@dataclass(frozen=True)
class NeckPoseState:

    anchor: str
    beta: float
    gamma: float
    px: float
    theta_head: float = float("nan")
    theta_torso: float = float("nan")
    neck_relative: float = float("nan")
    face_axis_valid: bool = False
    torso_axis_valid: bool = False
    status: str = "inconclusive"
    inconclusive_reason: str = ""
    selected_face_axis: str = "none"
    selected_face_keypoints: tuple[int, int] = (-1, -1)
    nose_confidence: float = float("nan")
    face_ref_confidence: float = float("nan")
    shoulder_confidence: float = float("nan")
    hip_confidence: float = float("nan")
    face_axis_reason: str = ""
    # Alias legacy/debug; reflejan la métrica relativa firmada cara/torso
    # para mantener legibles JSON y gráficos aguas abajo.
    theta_back: float = float("nan")
    neck_ref_idx: int = -1
    signed_neck_offset: float = float("nan")
    neck_lateral_offset: float = float("nan")


@dataclass(frozen=True)
class NeckMovementMetrics:

    segment: str
    start_anchor: str
    end_anchor: str
    anchor: str
    user_start: NeckPoseState
    user_end: NeckPoseState
    ideal_start: NeckPoseState
    ideal_end: NeckPoseState
    delta_beta_user: float
    delta_beta_ideal: float
    delta_beta_diff: float
    delta_gamma_user: float
    delta_gamma_ideal: float
    delta_px_user: float
    delta_px_ideal: float
    delta_px_user_aligned: float
    delta_px_ideal_aligned: float
    delta_px_diff: float
    ratio_user: float | None
    ratio_ideal: float | None
    ratio_delta: float | None
    has_descent: bool
    has_hinge: bool
    theta_head_start: float = float("nan")
    theta_head_end: float = float("nan")
    theta_torso_start: float = float("nan")
    theta_torso_end: float = float("nan")
    neck_relative_start: float = float("nan")
    neck_relative_end: float = float("nan")
    ideal_neck_relative_start: float = float("nan")
    ideal_neck_relative_end: float = float("nan")
    delta_neck_relative_user: float = float("nan")
    delta_neck_relative_ideal: float = float("nan")
    classifier_A_value: float = float("nan")
    classifier_B_value: float = float("nan")
    selected_face_axis_start: str = "none"
    selected_face_axis_end: str = "none"
    selected_face_keypoints_start: tuple[int, int] = (-1, -1)
    selected_face_keypoints_end: tuple[int, int] = (-1, -1)
    nose_confidence_start: float = float("nan")
    nose_confidence_end: float = float("nan")
    face_ref_confidence_start: float = float("nan")
    face_ref_confidence_end: float = float("nan")
    is_conclusive: bool = False
    inconclusive_reason: str = ""
    # Campos de compatibilidad para renderizado/resumen existente.
    theta_back_user_baseline: float = float("nan")
    theta_back_ideal_baseline: float = float("nan")
    delta_back_curl_user_from_baseline: float = float("nan")
    delta_back_curl_ideal_from_baseline: float = float("nan")
    excess_back_curl: float = float("nan")
    excess_back_curl_anchor: float = float("nan")
    signed_neck_user_start: float = float("nan")
    signed_neck_user_end: float = float("nan")
    signed_neck_ideal_start: float = float("nan")
    signed_neck_ideal_end: float = float("nan")
    forward_sign: float = NECK_DIRECTION_SIGN
    signed_excess_anchor: float = float("nan")
    signed_excess_delta: float = float("nan")


def _finite_point(p: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(p, dtype=np.float64)).all())


def _score(scores: np.ndarray | None, idx: int) -> float:
    if scores is None:
        return 1.0
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape[0] <= idx or not math.isfinite(float(arr[idx])):
        return float("nan")
    return float(arr[idx])


def _keypoint_ok(
    kps: np.ndarray,
    scores: np.ndarray | None,
    idx: int,
    min_confidence: float,
) -> bool:
    return _finite_point(kps[idx]) and _score(scores, idx) >= min_confidence


def _angle_from_vector(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    if not _finite_point(v) or float(np.linalg.norm(v)) <= 1e-9:
        return float("nan")
    return float(math.atan2(float(v[1]), float(v[0])))


def _angle_from_pelvis(distal: np.ndarray, pelvis: np.ndarray) -> float:
    if not (_finite_point(distal) and _finite_point(pelvis)):
        return float("nan")
    return _angle_from_vector(np.asarray(distal, dtype=np.float64) - np.asarray(pelvis, dtype=np.float64))


def wrap_to_180(angle_deg: float) -> float:
    if not math.isfinite(float(angle_deg)):
        return float("nan")
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _wrap_delta_rad(angle_delta: float) -> float:
    if not math.isfinite(float(angle_delta)):
        return float("nan")
    return math.radians(wrap_to_180(math.degrees(float(angle_delta))))


def select_face_axis(
    kps: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    min_confidence: float = DEFAULT_FACE_CONFIDENCE,
) -> FaceAxisSelection:
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape != (17, 2):
        raise ValueError(f"Expected kps shape (17, 2), got {kps.shape}.")

    nose_conf = _score(scores, KP_NOSE)
    if not _keypoint_ok(kps, scores, KP_NOSE, min_confidence):
        return FaceAxisSelection(
            valid=False,
            axis="none",
            nose_confidence=nose_conf,
            reason="nose_not_reliable",
        )

    for idx, name in ((KP_R_EAR, "right_ear"), (KP_L_EAR, "left_ear")):
        conf = _score(scores, idx)
        if _keypoint_ok(kps, scores, idx, min_confidence):
            return FaceAxisSelection(
                valid=True,
                axis=f"{name}_to_nose",
                ref_idx=idx,
                ref_name=name,
                nose_confidence=nose_conf,
                ref_confidence=conf,
                reason="ok",
            )

    for idx, name in ((KP_R_EYE, "right_eye"), (KP_L_EYE, "left_eye")):
        conf = _score(scores, idx)
        if _keypoint_ok(kps, scores, idx, min_confidence):
            return FaceAxisSelection(
                valid=True,
                axis=f"{name}_to_nose",
                ref_idx=idx,
                ref_name=name,
                nose_confidence=nose_conf,
                ref_confidence=conf,
                reason="ok_fallback_eye",
            )

    return FaceAxisSelection(
        valid=False,
        axis="none",
        nose_confidence=nose_conf,
        reason="no_reliable_ear_or_eye",
    )


def compute_head_orientation(
    kps: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    min_confidence: float = DEFAULT_FACE_CONFIDENCE,
) -> tuple[float, FaceAxisSelection]:
    kps = np.asarray(kps, dtype=np.float64)
    selection = select_face_axis(kps, scores, min_confidence=min_confidence)
    if not selection.valid:
        return float("nan"), selection
    head_vec = kps[KP_NOSE] - kps[selection.ref_idx]
    return _angle_from_vector(head_vec), selection


def compute_torso_orientation(
    kps: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    min_confidence: float = DEFAULT_BODY_CONFIDENCE,
) -> float:
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape != (17, 2):
        raise ValueError(f"Expected kps shape (17, 2), got {kps.shape}.")
    if not (
        _keypoint_ok(kps, scores, KP_HIP, min_confidence)
        and _keypoint_ok(kps, scores, KP_SHOULDER, min_confidence)
    ):
        return float("nan")
    return _angle_from_vector(kps[KP_SHOULDER] - kps[KP_HIP])


def compute_neck_relative(theta_head: float, theta_torso: float) -> float:
    if not (math.isfinite(float(theta_head)) and math.isfinite(float(theta_torso))):
        return float("nan")
    return math.radians(wrap_to_180(math.degrees(theta_head - theta_torso)))


def classify_neck_direction(value_rad: float, *, threshold_deg: float = 8.0) -> str:
    if not math.isfinite(float(value_rad)):
        return "unclear"
    calibrated = float(value_rad) * NECK_DIRECTION_SIGN
    threshold = math.radians(float(threshold_deg))
    if calibrated > threshold:
        return "down"
    if calibrated < -threshold:
        return "up"
    return "neutral"


def compute_neck_pose_state(
    kps: np.ndarray,
    anchor: str,
    scores: np.ndarray | None = None,
    *,
    min_face_confidence: float = DEFAULT_FACE_CONFIDENCE,
    min_body_confidence: float = DEFAULT_BODY_CONFIDENCE,
) -> NeckPoseState:
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape != (17, 2):
        raise ValueError(f"Expected kps shape (17, 2), got {kps.shape}.")

    shoulder = kps[KP_SHOULDER]
    pelvis = kps[KP_HIP]
    knee = kps[KP_KNEE]
    beta = _angle_from_pelvis(shoulder, pelvis)
    gamma = _angle_from_pelvis(knee, pelvis)
    px = float(pelvis[0]) if _finite_point(pelvis) else float("nan")
    shoulder_confidence = _score(scores, KP_SHOULDER)
    hip_confidence = _score(scores, KP_HIP)
    torso_axis_valid = bool(
        _keypoint_ok(kps, scores, KP_HIP, min_body_confidence)
        and _keypoint_ok(kps, scores, KP_SHOULDER, min_body_confidence)
    )
    theta_torso = compute_torso_orientation(
        kps,
        scores,
        min_confidence=min_body_confidence,
    )
    theta_head, face_axis = compute_head_orientation(
        kps,
        scores,
        min_confidence=min_face_confidence,
    )
    neck_relative = compute_neck_relative(theta_head, theta_torso)
    pose_ok = bool(torso_axis_valid and face_axis.valid and math.isfinite(neck_relative))
    if pose_ok:
        status = "ok"
        reason = ""
    elif not torso_axis_valid:
        status = "inconclusive"
        reason = "torso_axis_not_reliable"
    else:
        status = "inconclusive"
        reason = face_axis.reason or "face_axis_not_reliable"

    return NeckPoseState(
        anchor=anchor,
        beta=beta,
        gamma=gamma,
        px=px,
        theta_head=theta_head,
        theta_torso=theta_torso,
        neck_relative=neck_relative,
        face_axis_valid=face_axis.valid,
        torso_axis_valid=torso_axis_valid,
        status=status,
        inconclusive_reason=reason,
        selected_face_axis=face_axis.axis,
        selected_face_keypoints=(face_axis.ref_idx, KP_NOSE)
        if face_axis.valid else (-1, -1),
        nose_confidence=face_axis.nose_confidence,
        face_ref_confidence=face_axis.ref_confidence,
        shoulder_confidence=shoulder_confidence,
        hip_confidence=hip_confidence,
        face_axis_reason=face_axis.reason,
        theta_back=neck_relative,
        neck_ref_idx=face_axis.ref_idx,
        signed_neck_offset=neck_relative,
        neck_lateral_offset=neck_relative,
    )


def _sign_or_one(value: float) -> float:
    if math.isfinite(float(value)) and abs(float(value)) > 1e-9:
        return 1.0 if float(value) > 0.0 else -1.0
    return 1.0


def compute_neck_segment_metrics(
    *,
    user_start_kps: np.ndarray,
    user_end_kps: np.ndarray,
    ideal_start_kps: np.ndarray,
    ideal_end_kps: np.ndarray,
    start_anchor: str,
    end_anchor: str,
    user_baseline_kps: np.ndarray | None = None,
    ideal_baseline_kps: np.ndarray | None = None,
    user_start_scores: np.ndarray | None = None,
    user_end_scores: np.ndarray | None = None,
    ideal_start_scores: np.ndarray | None = None,
    ideal_end_scores: np.ndarray | None = None,
    user_baseline_scores: np.ndarray | None = None,
    ideal_baseline_scores: np.ndarray | None = None,
    hinge_min_ratio: float = 0.35,
    hinge_min_px: float = 5.0,
    threshold_beta_min: float = math.radians(3.0),
    threshold_gamma_safe: float = math.radians(3.0),
    forward_sign: float = NECK_DIRECTION_SIGN,
    min_face_confidence: float = DEFAULT_FACE_CONFIDENCE,
    min_body_confidence: float = DEFAULT_BODY_CONFIDENCE,
) -> NeckMovementMetrics:
    user_start = compute_neck_pose_state(
        user_start_kps, start_anchor, user_start_scores,
        min_face_confidence=min_face_confidence,
        min_body_confidence=min_body_confidence,
    )
    user_end = compute_neck_pose_state(
        user_end_kps, end_anchor, user_end_scores,
        min_face_confidence=min_face_confidence,
        min_body_confidence=min_body_confidence,
    )
    ideal_start = compute_neck_pose_state(
        ideal_start_kps, start_anchor, ideal_start_scores,
        min_face_confidence=min_face_confidence,
        min_body_confidence=min_body_confidence,
    )
    ideal_end = compute_neck_pose_state(
        ideal_end_kps, end_anchor, ideal_end_scores,
        min_face_confidence=min_face_confidence,
        min_body_confidence=min_body_confidence,
    )

    delta_beta_user_raw = _wrap_delta_rad(user_end.beta - user_start.beta)
    delta_beta_ideal_raw = _wrap_delta_rad(ideal_end.beta - ideal_start.beta)
    beta_dir = _sign_or_one(delta_beta_ideal_raw)
    delta_beta_user = float(delta_beta_user_raw * beta_dir)
    delta_beta_ideal = float(delta_beta_ideal_raw * beta_dir)
    delta_beta_diff = float(delta_beta_user - delta_beta_ideal)

    delta_gamma_user_raw = _wrap_delta_rad(user_end.gamma - user_start.gamma)
    delta_gamma_ideal_raw = _wrap_delta_rad(ideal_end.gamma - ideal_start.gamma)
    gamma_dir = _sign_or_one(delta_gamma_ideal_raw)
    delta_gamma_user = float(delta_gamma_user_raw * gamma_dir)
    delta_gamma_ideal = float(delta_gamma_ideal_raw * gamma_dir)

    delta_px_user = float(user_end.px - user_start.px)
    delta_px_ideal = float(ideal_end.px - ideal_start.px)
    px_dir = _sign_or_one(delta_px_ideal)
    delta_px_user_aligned = float(delta_px_user * px_dir)
    delta_px_ideal_aligned = float(delta_px_ideal * px_dir)
    delta_px_diff = float(delta_px_user_aligned - delta_px_ideal_aligned)

    ratio_user: float | None = None
    ratio_ideal: float | None = None
    ratio_delta: float | None = None
    if (
        math.isfinite(delta_gamma_ideal)
        and abs(delta_gamma_ideal) > threshold_gamma_safe
        and math.isfinite(delta_gamma_user)
        and abs(delta_gamma_user) > 1e-9
    ):
        ratio_user = float(delta_beta_user / delta_gamma_user)
        ratio_ideal = float(delta_beta_ideal / delta_gamma_ideal)
        if abs(ratio_ideal) > 1e-9:
            ratio_delta = float(ratio_user / ratio_ideal)

    has_descent = bool(
        math.isfinite(delta_beta_user)
        and delta_beta_user >= threshold_beta_min
    )
    _ = hinge_min_ratio
    has_hinge = bool(
        math.isfinite(delta_px_user_aligned)
        and delta_px_user_aligned >= hinge_min_px
    )

    user_baseline_state = (
        compute_neck_pose_state(
            user_baseline_kps,
            "ecc_0",
            user_baseline_scores,
            min_face_confidence=min_face_confidence,
            min_body_confidence=min_body_confidence,
        )
        if user_baseline_kps is not None else user_start
    )
    ideal_baseline_state = (
        compute_neck_pose_state(
            ideal_baseline_kps,
            "ecc_0",
            ideal_baseline_scores,
            min_face_confidence=min_face_confidence,
            min_body_confidence=min_body_confidence,
        )
        if ideal_baseline_kps is not None else ideal_start
    )

    delta_user_neck = (
        _wrap_delta_rad(user_end.neck_relative - user_baseline_state.neck_relative)
        if math.isfinite(user_end.neck_relative)
        and math.isfinite(user_baseline_state.neck_relative)
        else float("nan")
    )
    delta_ideal_neck = (
        _wrap_delta_rad(ideal_end.neck_relative - ideal_baseline_state.neck_relative)
        if math.isfinite(ideal_end.neck_relative)
        and math.isfinite(ideal_baseline_state.neck_relative)
        else float("nan")
    )

    # Clasificador A: aumento de magnitud respecto al baseline ecc_0 del usuario.
    # Usa |end_nr| - |baseline_nr|:
    #   - Positivo → magnitud creció (cabeza más lejos de neutro) → error
    #   - Negativo → magnitud disminuyó (recuperación) → sin error
    # Robusto al salto ±180° y a cabeza ya desviada al inicio (p. ej. CuelloAbajo -179°).
    classifier_a = (
        float(abs(user_end.neck_relative) - abs(user_baseline_state.neck_relative))
        if math.isfinite(user_end.neck_relative)
        and math.isfinite(user_baseline_state.neck_relative)
        else float("nan")
    )

    # Clasificador B: desviación firmada usuario vs ideal en este ancla.
    # Positivo = más desviado en dirección "primaria" (depende de forward_sign).
    # Negativo = más desviado en dirección opuesta.
    classifier_b_raw = (
        _wrap_delta_rad(user_end.neck_relative - ideal_end.neck_relative)
        if math.isfinite(user_end.neck_relative)
        and math.isfinite(ideal_end.neck_relative)
        else float("nan")
    )
    classifier_b = float(classifier_b_raw * forward_sign) if math.isfinite(classifier_b_raw) else float("nan")

    is_conclusive = bool(
        user_end.status == "ok"
        and ideal_end.status == "ok"
        and math.isfinite(classifier_b)
    )
    if user_end.status != "ok":
        reason = f"user_end_invalid:{user_end.inconclusive_reason}"
    elif ideal_end.status != "ok":
        reason = f"ideal_end_invalid:{ideal_end.inconclusive_reason}"
    else:
        reason = ""

    segment = f"{start_anchor}_to_{end_anchor}"
    return NeckMovementMetrics(
        segment=segment,
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        anchor=end_anchor,
        user_start=user_start,
        user_end=user_end,
        ideal_start=ideal_start,
        ideal_end=ideal_end,
        delta_beta_user=delta_beta_user,
        delta_beta_ideal=delta_beta_ideal,
        delta_beta_diff=delta_beta_diff,
        delta_gamma_user=delta_gamma_user,
        delta_gamma_ideal=delta_gamma_ideal,
        delta_px_user=delta_px_user,
        delta_px_ideal=delta_px_ideal,
        delta_px_user_aligned=delta_px_user_aligned,
        delta_px_ideal_aligned=delta_px_ideal_aligned,
        delta_px_diff=delta_px_diff,
        ratio_user=ratio_user,
        ratio_ideal=ratio_ideal,
        ratio_delta=ratio_delta,
        has_descent=has_descent,
        has_hinge=has_hinge,
        theta_head_start=user_start.theta_head,
        theta_head_end=user_end.theta_head,
        theta_torso_start=user_start.theta_torso,
        theta_torso_end=user_end.theta_torso,
        neck_relative_start=user_start.neck_relative,
        neck_relative_end=user_end.neck_relative,
        ideal_neck_relative_start=ideal_start.neck_relative,
        ideal_neck_relative_end=ideal_end.neck_relative,
        delta_neck_relative_user=delta_user_neck,
        delta_neck_relative_ideal=delta_ideal_neck,
        classifier_A_value=classifier_a,
        classifier_B_value=classifier_b,
        selected_face_axis_start=user_start.selected_face_axis,
        selected_face_axis_end=user_end.selected_face_axis,
        selected_face_keypoints_start=user_start.selected_face_keypoints,
        selected_face_keypoints_end=user_end.selected_face_keypoints,
        nose_confidence_start=user_start.nose_confidence,
        nose_confidence_end=user_end.nose_confidence,
        face_ref_confidence_start=user_start.face_ref_confidence,
        face_ref_confidence_end=user_end.face_ref_confidence,
        is_conclusive=is_conclusive,
        inconclusive_reason=reason,
        theta_back_user_baseline=user_baseline_state.neck_relative,
        theta_back_ideal_baseline=ideal_baseline_state.neck_relative,
        delta_back_curl_user_from_baseline=classifier_a,
        delta_back_curl_ideal_from_baseline=(
            float(delta_ideal_neck * forward_sign)
            if math.isfinite(delta_ideal_neck) else float("nan")
        ),
        excess_back_curl=(
            float((delta_user_neck - delta_ideal_neck) * forward_sign)
            if math.isfinite(delta_user_neck) and math.isfinite(delta_ideal_neck)
            else float("nan")
        ),
        excess_back_curl_anchor=classifier_b,
        signed_neck_user_start=user_start.neck_relative,
        signed_neck_user_end=user_end.neck_relative,
        signed_neck_ideal_start=ideal_start.neck_relative,
        signed_neck_ideal_end=ideal_end.neck_relative,
        forward_sign=forward_sign,
        signed_excess_anchor=classifier_b,
        signed_excess_delta=classifier_a,
    )


def compute_neck_movement_segments(
    user_kps_by_anchor: dict[str, np.ndarray],
    ideal_kps_by_anchor: dict[str, np.ndarray],
    *,
    user_scores_by_anchor: dict[str, np.ndarray] | None = None,
    ideal_scores_by_anchor: dict[str, np.ndarray] | None = None,
    hinge_min_ratio: float = 0.35,
    hinge_min_px: float = 5.0,
    threshold_beta_min: float = math.radians(3.0),
    threshold_gamma_safe: float = math.radians(3.0),
    forward_sign: float | None = None,
    min_face_confidence: float = DEFAULT_FACE_CONFIDENCE,
    min_body_confidence: float = DEFAULT_BODY_CONFIDENCE,
) -> dict[str, NeckMovementMetrics]:
    metrics: dict[str, NeckMovementMetrics] = {}
    user_scores_by_anchor = user_scores_by_anchor or {}
    ideal_scores_by_anchor = ideal_scores_by_anchor or {}
    user_baseline = user_kps_by_anchor.get("ecc_0")
    ideal_baseline = ideal_kps_by_anchor.get("ecc_0")
    user_baseline_scores = user_scores_by_anchor.get("ecc_0")
    ideal_baseline_scores = ideal_scores_by_anchor.get("ecc_0")
    fwd = NECK_DIRECTION_SIGN if forward_sign is None else float(forward_sign)

    for start_anchor, end_anchor in NECK_SEGMENTS:
        if (
            start_anchor not in user_kps_by_anchor
            or end_anchor not in user_kps_by_anchor
            or start_anchor not in ideal_kps_by_anchor
            or end_anchor not in ideal_kps_by_anchor
        ):
            continue
        m = compute_neck_segment_metrics(
            user_start_kps=user_kps_by_anchor[start_anchor],
            user_end_kps=user_kps_by_anchor[end_anchor],
            ideal_start_kps=ideal_kps_by_anchor[start_anchor],
            ideal_end_kps=ideal_kps_by_anchor[end_anchor],
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            user_baseline_kps=user_baseline,
            ideal_baseline_kps=ideal_baseline,
            user_start_scores=user_scores_by_anchor.get(start_anchor),
            user_end_scores=user_scores_by_anchor.get(end_anchor),
            ideal_start_scores=ideal_scores_by_anchor.get(start_anchor),
            ideal_end_scores=ideal_scores_by_anchor.get(end_anchor),
            user_baseline_scores=user_baseline_scores,
            ideal_baseline_scores=ideal_baseline_scores,
            hinge_min_ratio=hinge_min_ratio,
            hinge_min_px=hinge_min_px,
            threshold_beta_min=threshold_beta_min,
            threshold_gamma_safe=threshold_gamma_safe,
            forward_sign=fwd,
            min_face_confidence=min_face_confidence,
            min_body_confidence=min_body_confidence,
        )
        metrics[m.segment] = m
    return metrics
