
from __future__ import annotations

from typing import Any

import numpy as np

from .constants import PHASE_ECCENTRIC, PHASE_FULL_REP, PHASE_LOCKOUT, PHASE_UNKNOWN
from .phase_mapping import location_label_from_anchors, phase_from_anchors
from .schema import EvidenceItem


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _is_detected(rep_result: dict[str, Any]) -> bool:
    return bool(rep_result.get("detected", False))


def _get_rep_ids(rep_result: dict[str, Any], rep_index: int) -> tuple[int | None, int | None, int | None]:
    rep_raw = _safe_int(rep_result.get("user_rep_raw_index"))
    rep_order = _safe_int(rep_result.get("user_rep_order"))
    if rep_order is None:
        rep_order = rep_index + 1
    return rep_index, rep_raw, rep_order


def _rep_index_order_mismatch_warnings(
    detector: str,
    *,
    rep_index: int | None,
    user_rep_raw_index: int | None,
    user_rep_order: int | None,
) -> list[str]:
    if rep_index is None or user_rep_order is None:
        return []
    mismatch = False
    if user_rep_order == user_rep_raw_index and rep_index != (user_rep_order - 1):
        mismatch = True
    if user_rep_order != (rep_index + 1):
        mismatch = True
    if not mismatch:
        return []
    return [
        f"EVIDENCE_REP_INDEX_ORDER_MISMATCH:{detector}:rep_index={rep_index}:user_rep_order={user_rep_order}"
    ]


def _collect_anchor_frames(anchor_results: dict[str, Any], anchors: list[str]) -> list[int]:
    frames: list[int] = []
    for anchor in anchors:
        item = anchor_results.get(anchor) if isinstance(anchor_results, dict) else None
        if not isinstance(item, dict):
            continue
        f = _safe_int(item.get("user_frame"))
        if f is None:
            frames_block = item.get("frames") if isinstance(item.get("frames"), dict) else {}
            if isinstance(frames_block, dict):
                f = _safe_int(frames_block.get("user"))
                if f is None:
                    f = _safe_int(frames_block.get("user_frame"))
        if f is None:
            f = _safe_int(item.get("frame"))
        if f is None:
            f = _safe_int(item.get("user_frame_idx"))
        if f is not None:
            frames.append(f)
    return sorted(set(frames))


def _anchors_from_failed_or_non_none(rep_result: dict[str, Any], anchor_block_name: str) -> list[str]:
    failed = rep_result.get("failed_anchors")
    if isinstance(failed, list) and failed:
        return [str(a) for a in failed]
    block = rep_result.get(anchor_block_name)
    if not isinstance(block, dict):
        return []
    out: list[str] = []
    for anchor, info in block.items():
        if not isinstance(info, dict):
            continue
        sev = str(info.get("severity", "none"))
        if bool(info.get("failed", False)) or sev != "none":
            out.append(str(anchor))
    return out


def _map_anchor_to_segment(anchor: str) -> str | None:
    from src.biomechanics.rdl.feedback.segment_keys import to_canonical_segment

    raw = str(anchor).strip().replace(" ", "_")
    if raw.lower() == "bottom":
        return None
    canonical = to_canonical_segment(anchor)
    if canonical is not None:
        return canonical
    return raw


def _get_anchor_severity(anchor_rulings: dict[str, Any], anchor: str, default_severity: str = "leve") -> str:
    ruling = anchor_rulings.get(anchor)
    if not isinstance(ruling, dict):
        return default_severity

    if "severity" in ruling:
        return str(ruling.get("severity", default_severity))

    if ruling.get("failed", False):
        return default_severity

    return "none"


def _get_anchor_frame(anchor_rulings: dict[str, Any], anchor: str) -> int | None:
    ruling = anchor_rulings.get(anchor)
    if not isinstance(ruling, dict):
        return None
    return _safe_int(ruling.get("user_frame"))


def extract_bent_arms_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []

    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict) or not _is_detected(rep):
            continue

        failed_anchors = [str(a) for a in rep.get("failed_anchors", [])] if isinstance(rep.get("failed_anchors"), list) else []
        if not failed_anchors:
            continue

        anchor_rulings = rep.get("anchor_rulings") if isinstance(rep.get("anchor_rulings"), dict) else {}
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        mismatch_warnings = _rep_index_order_mismatch_warnings(
            "bent_arms",
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )

        for anchor in failed_anchors:
            segment = _map_anchor_to_segment(anchor)
            if segment is None:
                continue
            anchor_severity = _get_anchor_severity(anchor_rulings, anchor, default_severity=str(rep.get("severity", "leve")))
            anchor_frame = _get_anchor_frame(anchor_rulings, anchor)
            frames = [anchor_frame] if anchor_frame is not None else []

            items.append(
                EvidenceItem(
                    detector="bent_arms",
                    error_code="bent_arms",
                    rep_index=rep_index,
                    user_rep_raw_index=rep_raw,
                    user_rep_order=rep_order,
                    segment=segment,
                    phase=PHASE_ECCENTRIC,
                    anchors=[anchor],
                    frames=frames,
                    severity=anchor_severity,
                    score=_safe_float(rep.get("score", 0.0)),
                    detected=True,
                    confidence=_safe_float(rep.get("confidence"), default=0.0) if rep.get("confidence") is not None else None,
                    location_label="brazos",
                    summary_metrics={
                        "anchor": anchor,
                        "magnitude": rep.get("magnitude"),
                        "n_failed": rep.get("n_failed"),
                    },
                    source={"detector": "bent_arms", "rep_result_index": i, "anchor": anchor},
                    warnings=mismatch_warnings,
                )
            )

    return items


def extract_asymmetry_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []
    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict):
            continue
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        for block_name, error_code in (("arms", "asymmetry_arms"), ("legs", "asymmetry_legs")):
            block = rep.get(block_name)
            if not isinstance(block, dict):
                continue
            sev = str(block.get("severity", "none"))
            if sev == "none":
                continue
            dominant_phase = str(block.get("dominant_phase", "")).lower()
            if dominant_phase in {"ecc", "eccentric"}:
                phase = "eccentric"
            elif dominant_phase in {"con", "concentric"}:
                phase = "concentric"
            else:
                phase = PHASE_FULL_REP
            worst_frames = rep.get("worst_frames") if isinstance(rep.get("worst_frames"), dict) else {}
            this_worst = worst_frames.get(block_name) if isinstance(worst_frames.get(block_name), dict) else {}
            frames = []
            for key in ("ecc", "con"):
                f = _safe_int(this_worst.get(key))
                if f is not None and f >= 0:
                    frames.append(f)
            mismatch_warnings = _rep_index_order_mismatch_warnings(
                "asymmetry",
                rep_index=rep_index,
                user_rep_raw_index=rep_raw,
                user_rep_order=rep_order,
            )
            items.append(
                EvidenceItem(
                    detector="asymmetry",
                    error_code=error_code,
                    rep_index=rep_index,
                    user_rep_raw_index=rep_raw,
                    user_rep_order=rep_order,
                    segment=None,
                    phase=phase,
                    anchors=[],
                    frames=sorted(set(frames)),
                    severity=sev,
                    score=_safe_float(block.get("magnitude", rep.get("score", 0.0))),
                    detected=True,
                    confidence=None,
                    location_label=phase if phase != PHASE_FULL_REP else "full_rep",
                    summary_metrics={
                        "dominant_side": block.get("dominant_side"),
                        "dominant_phase": block.get("dominant_phase"),
                        "consistency": block.get("consistency"),
                        "side_imbalance": block.get("side_imbalance"),
                        "stability": block.get("stability"),
                        "magnitude": block.get("magnitude"),
                        "confidence": block.get("confidence"),
                        "severity_before_group_adjustment": block.get("severity_before_group_adjustment"),
                        "secondary_reinforcement": block.get("secondary_reinforcement"),
                        "arm_conservatism_applied": block.get("arm_conservatism_applied"),
                        "arm_conservatism_reason": block.get("arm_conservatism_reason"),
                        "arm_perspective_filter_applied": block.get("arm_perspective_filter_applied"),
                        "arm_perspective_filter_reason": block.get("arm_perspective_filter_reason"),
                        "wrist_mean_fwd_norm": block.get("wrist_mean_fwd_norm"),
                        "wrist_max_fwd_norm": block.get("wrist_max_fwd_norm"),
                        "elbow_mean_fwd_norm": block.get("elbow_mean_fwd_norm"),
                        "elbow_max_fwd_norm": block.get("elbow_max_fwd_norm"),
                        "leg_grave_rescue_applied": block.get("leg_grave_rescue_applied"),
                        "leg_grave_rescue_reason": block.get("leg_grave_rescue_reason"),
                        "severity_before_stability_or_rescue": block.get("severity_before_stability_or_rescue"),
                        "grave_blocked_by": block.get("grave_blocked_by", []),
                    },
                    source={"detector": "asymmetry", "rep_result_index": i, "block": block_name},
                    warnings=(
                        ([str(w) for w in rep.get("warnings", [])] if isinstance(rep.get("warnings"), list) else [])
                        + mismatch_warnings
                    ),
                )
            )
    return items


def extract_bar_far_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []

    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict) or not _is_detected(rep):
            continue

        failed_anchors = [str(a) for a in rep.get("failed_anchors", [])] if isinstance(rep.get("failed_anchors"), list) else []
        if not failed_anchors:
            continue

        anchor_rulings = rep.get("anchor_rulings") if isinstance(rep.get("anchor_rulings"), dict) else {}
        anchor_results = rep.get("anchor_results") if isinstance(rep.get("anchor_results"), dict) else {}
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        mismatch_warnings = _rep_index_order_mismatch_warnings(
            "bar_far",
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )

        for anchor in failed_anchors:
            segment = _map_anchor_to_segment(anchor)
            if segment is None:
                continue
            anchor_severity = _get_anchor_severity(anchor_rulings, anchor, default_severity=str(rep.get("severity", "leve")))
            anchor_frame = _get_anchor_frame(anchor_rulings, anchor)
            if anchor_frame is None and isinstance(anchor_results.get(anchor), dict):
                anchor_frame = _safe_int(anchor_results[anchor].get("user_frame"))
            frames = [anchor_frame] if anchor_frame is not None else []

            items.append(
                EvidenceItem(
                    detector="bar_far",
                    error_code="bar_far",
                    rep_index=rep_index,
                    user_rep_raw_index=rep_raw,
                    user_rep_order=rep_order,
                    segment=segment,
                    phase=PHASE_ECCENTRIC,
                    anchors=[anchor],
                    frames=frames,
                    severity=anchor_severity,
                    score=_safe_float(rep.get("score", 0.0)),
                    detected=True,
                    confidence=None,
                    location_label="barra",
                    summary_metrics={
                        "anchor": anchor,
                        "n_failed": rep.get("n_failed"),
                        "confidence": rep.get("confidence"),
                    },
                    source={"detector": "bar_far", "rep_result_index": i, "anchor": anchor},
                    warnings=mismatch_warnings,
                )
            )

    return items


def extract_hip_hinge_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []

    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict) or not _is_detected(rep):
            continue

        failed_anchors = [str(a) for a in rep.get("failed_anchors", [])] if isinstance(rep.get("failed_anchors"), list) else []
        if not failed_anchors:
            continue

        anchor_rulings = rep.get("anchor_rulings") if isinstance(rep.get("anchor_rulings"), dict) else {}
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        mismatch_warnings = _rep_index_order_mismatch_warnings(
            "hip_hinge",
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )

        for anchor in failed_anchors:
            segment = _map_anchor_to_segment(anchor)
            if segment is None:
                continue
            anchor_severity = _get_anchor_severity(anchor_rulings, anchor, default_severity=str(rep.get("severity", "leve")))
            anchor_frame = _get_anchor_frame(anchor_rulings, anchor)
            frames = [anchor_frame] if anchor_frame is not None else []

            items.append(
                EvidenceItem(
                    detector="hip_hinge",
                    error_code="hip_hinge",
                    rep_index=rep_index,
                    user_rep_raw_index=rep_raw,
                    user_rep_order=rep_order,
                    segment=segment,
                    phase=PHASE_ECCENTRIC,
                    anchors=[anchor],
                    frames=frames,
                    severity=anchor_severity,
                    score=_safe_float(rep.get("score", 0.0)),
                    detected=True,
                    confidence=_safe_float(rep.get("confidence"), default=0.0) if rep.get("confidence") is not None else None,
                    location_label=location_label_from_anchors([anchor]),
                    summary_metrics={
                        "anchor": anchor,
                        "mean_deficit": rep.get("mean_deficit"),
                        "max_deficit": rep.get("max_deficit"),
                        "magnitude": rep.get("magnitude"),
                        "dominant_phase": rep.get("dominant_phase"),
                    },
                    source={"detector": "hip_hinge", "rep_result_index": i, "anchor": anchor},
                    warnings=mismatch_warnings,
                )
            )

    return items


def extract_knee_dominant_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []

    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict) or not _is_detected(rep):
            continue

        failed_anchors = [str(a) for a in rep.get("failed_anchors", [])] if isinstance(rep.get("failed_anchors"), list) else []
        if not failed_anchors:
            continue

        anchor_rulings = rep.get("anchor_rulings") if isinstance(rep.get("anchor_rulings"), dict) else {}
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        mismatch_warnings = _rep_index_order_mismatch_warnings(
            "knee_dominant",
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )

        for anchor in failed_anchors:
            segment = _map_anchor_to_segment(anchor)
            if segment is None:
                continue
            anchor_severity = _get_anchor_severity(anchor_rulings, anchor, default_severity=str(rep.get("severity", "leve")))
            anchor_frame = _get_anchor_frame(anchor_rulings, anchor)
            frames = [anchor_frame] if anchor_frame is not None else []
            phase = phase_from_anchors([anchor])
            if phase == "concentric":
                phase = PHASE_ECCENTRIC

            items.append(
                EvidenceItem(
                    detector="knee_dominant",
                    error_code="knee_dominant",
                    rep_index=rep_index,
                    user_rep_raw_index=rep_raw,
                    user_rep_order=rep_order,
                    segment=segment,
                    phase=phase,
                    anchors=[anchor],
                    frames=frames,
                    severity=anchor_severity,
                    score=_safe_float(rep.get("score", 0.0)),
                    detected=True,
                    confidence=_safe_float(rep.get("confidence"), default=0.0) if rep.get("confidence") is not None else None,
                    location_label=location_label_from_anchors([anchor]),
                    summary_metrics={
                        "anchor": anchor,
                        "magnitude": rep.get("magnitude"),
                        "mean_deficit": rep.get("mean_deficit"),
                        "max_deficit": rep.get("max_deficit"),
                        "dominant_phase": rep.get("dominant_phase"),
                    },
                    source={"detector": "knee_dominant", "rep_result_index": i, "anchor": anchor},
                    warnings=mismatch_warnings,
                )
            )

    return items


def extract_lockout_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []
    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict) or not _is_detected(rep):
            continue
        anchor = rep.get("anchor")
        anchors = [str(anchor)] if anchor else ["con_100"]
        frame = _safe_int(rep.get("user_frame"))
        frames = [frame] if frame is not None else []
        metrics = rep.get("metrics") if isinstance(rep.get("metrics"), dict) else {}
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        mismatch_warnings = _rep_index_order_mismatch_warnings(
            "lockout",
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )
        items.append(
            EvidenceItem(
                detector="lockout",
                error_code="lockout",
                rep_index=rep_index,
                user_rep_raw_index=rep_raw,
                user_rep_order=rep_order,
                segment=None,
                phase=PHASE_LOCKOUT,
                anchors=anchors,
                frames=frames,
                severity=str(rep.get("severity", "none")),
                score=_safe_float(rep.get("score", rep.get("magnitude", 0.0))),
                detected=True,
                confidence=_safe_float(rep.get("confidence"), default=0.0) if rep.get("confidence") is not None else None,
                location_label="lockout",
                summary_metrics={
                    "theta_end_user_deg": metrics.get("theta_end_user_deg"),
                    "theta_end_ideal_deg": metrics.get("theta_end_ideal_deg"),
                    "error_lockout_deg": metrics.get("error_lockout_deg"),
                    "magnitude": rep.get("magnitude"),
                },
                source={"detector": "lockout", "rep_result_index": i},
                warnings=(
                    ([str(w) for w in rep.get("warnings", [])] if isinstance(rep.get("warnings"), list) else [])
                    + mismatch_warnings
                ),
            )
        )
    return items


def extract_neck_movement_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []

    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict) or not _is_detected(rep):
            continue

        failed_segments = rep.get("failed_segments", []) if isinstance(rep.get("failed_segments"), list) else []
        if not failed_segments:
            continue

        segment_results = rep.get("segment_results", {}) if isinstance(rep.get("segment_results"), dict) else {}
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        mismatch_warnings = _rep_index_order_mismatch_warnings(
            "neck_movement",
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )

        for segment in failed_segments:
            segment_data = segment_results.get(segment, {})
            if not isinstance(segment_data, dict):
                segment_data = {}
            segment_severity = str(segment_data.get("severity", rep.get("severity", "leve")))
            anchor = str(segment_data.get("anchor", segment))
            from src.biomechanics.rdl.feedback.segment_keys import to_canonical_segment

            segment_id = to_canonical_segment(str(segment)) or _map_anchor_to_segment(segment)
            if segment_id is None:
                continue

            items.append(
                EvidenceItem(
                    detector="neck_movement",
                    error_code="neck_movement",
                    rep_index=rep_index,
                    user_rep_raw_index=rep_raw,
                    user_rep_order=rep_order,
                    segment=segment_id,
                    phase=PHASE_ECCENTRIC,
                    anchors=[anchor],
                    frames=[],
                    severity=segment_severity,
                    score=_safe_float(rep.get("score", rep.get("magnitude_deg", 0.0))),
                    detected=True,
                    confidence=_safe_float(rep.get("confidence"), default=0.0) if rep.get("confidence") is not None else None,
                    location_label="cuello",
                    summary_metrics={
                        "segment": segment,
                        "subtype": segment_data.get("subtype", rep.get("subtype")),
                        "neck_direction": segment_data.get("neck_direction", rep.get("neck_direction")),
                        "classifier_B_deg": segment_data.get("classifier_B_deg"),
                    },
                    source={"detector": "neck_movement", "rep_result_index": i, "segment": segment},
                    warnings=mismatch_warnings,
                )
            )

    return items


def extract_rom_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []
    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict) or not _is_detected(rep):
            continue
        frames_info = rep.get("frames") if isinstance(rep.get("frames"), dict) else {}
        frames = []
        for key in ("user_start", "user_bottom"):
            f = _safe_int(frames_info.get(key))
            if f is not None:
                frames.append(f)
        metrics = rep.get("metrics") if isinstance(rep.get("metrics"), dict) else {}
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        mismatch_warnings = _rep_index_order_mismatch_warnings(
            "rom",
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )
        items.append(
            EvidenceItem(
                detector="rom",
                error_code="short_rom",
                rep_index=rep_index,
                user_rep_raw_index=rep_raw,
                user_rep_order=rep_order,
                segment=None,
                phase=PHASE_ECCENTRIC,
                anchors=["ecc_0", "bottom"],
                frames=sorted(set(frames)),
                severity=str(rep.get("severity", "none")),
                score=_safe_float(rep.get("score", 0.0)),
                detected=True,
                confidence=_safe_float(rep.get("confidence"), default=0.0) if rep.get("confidence") is not None else None,
                location_label="eccentric_range",
                summary_metrics={
                    "rom_user_deg": metrics.get("rom_user_abs_deg"),
                    "rom_ideal_deg": metrics.get("rom_ideal_abs_deg"),
                    "rom_norm": metrics.get("rom_norm"),
                    "used_ideal": rep.get("used_ideal"),
                    "score": rep.get("score"),
                },
                source={"detector": "rom", "rep_result_index": i},
                warnings=(
                    ([str(w) for w in rep.get("warnings", [])] if isinstance(rep.get("warnings"), list) else [])
                    + mismatch_warnings
                ),
            )
        )
    return items


def _spine_seg_to_anchor() -> dict[str, str]:
    return {
        "ecc_0_to_ecc_25": "ecc_25",
        "ecc_25_to_ecc_50": "ecc_50",
        "ecc_50_to_ecc_75": "ecc_75",
        "ecc_75_to_ecc_100": "ecc_100",
    }


def _spine_per_segment_map(rep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_segment = rep.get("per_segment")
    if not isinstance(per_segment, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for seg, data in per_segment.items():
        if isinstance(data, dict):
            out[str(seg)] = data
    return out


def _spine_triggered_anchors(rep: dict[str, Any], seg_to_anchor: dict[str, str]) -> list[str]:
    anchors: list[str] = []
    trig = rep.get("triggered_segments")
    if isinstance(trig, list):
        for seg in trig:
            a = seg_to_anchor.get(str(seg))
            if a:
                anchors.append(a)
    if not anchors:
        per_anchor = rep.get("per_anchor") if isinstance(rep.get("per_anchor"), dict) else {}
        for a, info in per_anchor.items():
            if isinstance(info, dict) and bool(info.get("triggered", False)):
                anchors.append(str(a))
    return sorted(set(anchors))


def _spine_frames_and_max_drop(rep: dict[str, Any], anchors: list[str]) -> tuple[list[int], float]:
    by_anchor = rep.get("spine_geometry_by_anchor") if isinstance(rep.get("spine_geometry_by_anchor"), dict) else {}
    frames: list[int] = []
    max_drop = 0.0
    for a in anchors:
        item = by_anchor.get(a) if isinstance(by_anchor.get(a), dict) else {}
        f = _safe_int(item.get("user_frame"))
        if f is not None:
            frames.append(f)
        max_drop = max(max_drop, _safe_float(item.get("shoulder_low_norm"), 0.0))
    return sorted(set(frames)), max_drop


def _spine_grave_possible_segments(rep: dict[str, Any]) -> list[str]:
    per_segment = _spine_per_segment_map(rep)
    possible_segments = rep.get("possible_segments")
    if not isinstance(possible_segments, list):
        return []
    grave_possible: list[str] = []
    for seg in possible_segments:
        seg_key = str(seg)
        data = per_segment.get(seg_key, {})
        if str(data.get("torso_low_severity", "")).lower() == "grave":
            grave_possible.append(seg_key)
    return grave_possible


def _spine_possible_segments_for_evidence(rep: dict[str, Any]) -> list[str]:
    per_segment = _spine_per_segment_map(rep)
    possible_segments = rep.get("possible_segments")
    if not isinstance(possible_segments, list):
        return []
    out: list[str] = []
    for seg in possible_segments:
        seg_key = str(seg)
        data = per_segment.get(seg_key, {})
        torso = str(data.get("torso_low_severity", data.get("severity", "none"))).lower()
        if torso in ("grave", "media", "leve"):
            out.append(seg_key)
    return out


def _spine_blocking_errors_for_segments(rep: dict[str, Any], segments: list[str]) -> list[str]:
    per_segment = _spine_per_segment_map(rep)
    blocking: list[str] = []
    for seg in segments:
        data = per_segment.get(seg, {})
        if not isinstance(data, dict):
            continue
        if bool(data.get("neck_movement_failed", False)) and "neck_movement" not in blocking:
            blocking.append("neck_movement")
        if bool(data.get("knee_dominant_failed", False)) and "knee_dominant" not in blocking:
            blocking.append("knee_dominant")
    return blocking


def extract_spine_flexion_evidence(detector_result: dict[str, Any], analysis_context: dict[str, Any] | None = None) -> list[EvidenceItem]:
    _ = analysis_context
    if not isinstance(detector_result, dict):
        return []
    rep_results = detector_result.get("rep_results")
    if not isinstance(rep_results, list):
        return []

    items: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict):
            continue

        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        per_segment = _spine_per_segment_map(rep)
        triggered_segments = rep.get("triggered_segments", []) if isinstance(rep.get("triggered_segments"), list) else []

        if _is_detected(rep):
            if not triggered_segments:
                continue

            mismatch_warnings = _rep_index_order_mismatch_warnings(
                "spine_flexion",
                rep_index=rep_index,
                user_rep_raw_index=rep_raw,
                user_rep_order=rep_order,
            )
            rep_warnings = [str(w) for w in rep.get("warnings", [])] if isinstance(rep.get("warnings"), list) else []

            for segment in triggered_segments:
                segment_data = per_segment.get(segment, {})
                if not isinstance(segment_data, dict):
                    segment_data = {}
                segment_severity = str(
                    segment_data.get("torso_low_severity", segment_data.get("severity", rep.get("severity", "leve")))
                )
                torso_low_norm = _safe_float(segment_data.get("torso_low_norm"), 0.0)

                items.append(
                    EvidenceItem(
                        detector="spine_flexion",
                        error_code="spine_flexion",
                        rep_index=rep_index,
                        user_rep_raw_index=rep_raw,
                        user_rep_order=rep_order,
                        segment=segment,
                        phase=PHASE_ECCENTRIC,
                        anchors=[segment],
                        frames=[],
                        severity=segment_severity,
                        score=torso_low_norm,
                        detected=True,
                        confidence=None,
                        location_label="tronco",
                        summary_metrics={
                            "segment": segment,
                            "torso_low_norm": torso_low_norm,
                            "torso_low_severity": segment_severity,
                            "triggered": segment_data.get("triggered", False),
                        },
                        source={"detector": "spine_flexion", "rep_result_index": i, "segment": segment},
                        warnings=rep_warnings + mismatch_warnings,
                    )
                )

        possible_segments = _spine_possible_segments_for_evidence(rep)
        if not possible_segments:
            continue

        mismatch_warnings = _rep_index_order_mismatch_warnings(
            "spine_flexion",
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )
        rep_warnings = [str(w) for w in rep.get("warnings", [])] if isinstance(rep.get("warnings"), list) else []

        for segment in possible_segments:
            segment_data = per_segment.get(segment, {})
            if not isinstance(segment_data, dict):
                segment_data = {}
            blocking_errors = _spine_blocking_errors_for_segments(rep, [segment])
            if not blocking_errors:
                if bool(segment_data.get("neck_movement_failed")):
                    blocking_errors.append("neck_movement")
                if bool(segment_data.get("knee_dominant_failed")):
                    blocking_errors.append("knee_dominant")
            torso_sev = str(segment_data.get("torso_low_severity", segment_data.get("severity", "grave")))
            torso_low_norm = _safe_float(segment_data.get("torso_low_norm"), 0.0)

            items.append(
                EvidenceItem(
                    detector="spine_flexion",
                    error_code="spine_flexion_possible",
                    rep_index=rep_index,
                    user_rep_raw_index=rep_raw,
                    user_rep_order=rep_order,
                    segment=segment,
                    phase=PHASE_ECCENTRIC,
                    anchors=[segment],
                    frames=[],
                    severity="posible",
                    score=torso_low_norm,
                    detected=True,
                    confidence=None,
                    location_label="tronco",
                    summary_metrics={
                        "segment": segment,
                        "torso_low_severity": torso_sev,
                        "torso_low_norm": torso_low_norm,
                        "blocking_errors": blocking_errors,
                        "neck_movement_failed": segment_data.get("neck_movement_failed"),
                        "knee_dominant_failed": segment_data.get("knee_dominant_failed"),
                    },
                    source={
                        "detector": "spine_flexion",
                        "rep_result_index": i,
                        "variant": "possible_grave",
                        "segment": segment,
                    },
                    warnings=rep_warnings + mismatch_warnings,
                )
            )

    return items


def extract_non_detected_debug_evidence(detector_name: str, detector_result: dict[str, Any]) -> list[EvidenceItem]:
    rep_results = detector_result.get("rep_results") if isinstance(detector_result, dict) else None
    if not isinstance(rep_results, list):
        return []
    out: list[EvidenceItem] = []
    for i, rep in enumerate(rep_results):
        if not isinstance(rep, dict) or _is_detected(rep):
            continue
        rep_index, rep_raw, rep_order = _get_rep_ids(rep, i)
        mismatch_warnings = _rep_index_order_mismatch_warnings(
            detector_name,
            rep_index=rep_index,
            user_rep_raw_index=rep_raw,
            user_rep_order=rep_order,
        )
        out.append(
            EvidenceItem(
                detector=detector_name,
                error_code=detector_name,
                rep_index=rep_index,
                user_rep_raw_index=rep_raw,
                user_rep_order=rep_order,
                phase=PHASE_UNKNOWN,
                anchors=[],
                frames=[],
                severity=str(rep.get("severity", "none")),
                score=_safe_float(rep.get("score", 0.0)),
                detected=False,
                confidence=None,
                location_label="unknown",
                summary_metrics={},
                source={"detector": detector_name, "rep_result_index": i},
                warnings=(
                    ([str(w) for w in rep.get("warnings", [])] if isinstance(rep.get("warnings"), list) else [])
                    + mismatch_warnings
                ),
            )
        )
    return out
