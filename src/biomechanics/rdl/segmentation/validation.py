
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SEGMENTATION_STATUS_OK = "OK"
NO_VALID_RDL_SIGNAL = "NO_VALID_RDL_SIGNAL"
NO_VALID_REPS = "NO_VALID_REPS"
TOO_NOISY = "TOO_NOISY"
PARTIAL_VIDEO = "PARTIAL_VIDEO"
WRONG_MOVEMENT_OR_NOT_RDL = "WRONG_MOVEMENT_OR_NOT_RDL"


@dataclass(frozen=True, slots=True)
class RDLValidationConfig:
    min_global_top_angle: float = 150.0
    min_global_rom: float = 45.0
    min_global_valid_ratio: float = 0.75
    top_band_absolute: float = 150.0
    top_band_margin_ratio: float = 0.15
    top_band_tolerance_deg: float = 2.0
    bottom_band_margin_ratio: float = 0.20
    min_excursion_deg: float = 35.0
    min_total_seconds: float = 1.0
    min_phase_seconds: float = 0.35
    min_duration_rel: float = 0.35
    max_duration_rel: float = 2.5
    min_excursion_rel: float = 0.50
    min_monotonic_ratio: float = 0.55
    min_trend_score: float = 0.70
    min_net_motion_abs: float = 20.0
    min_net_motion_excursion_ratio: float = 0.30
    monotonic_eps_abs: float = 0.75
    monotonic_eps_rom_ratio: float = 0.01
    max_delta_abs: float = 18.0
    max_delta_rom_ratio: float = 0.25
    max_p95_velocity: float = 300.0
    min_rep_valid_ratio: float = 0.85
    max_rep_interp_ratio: float = 0.20
    max_invalid_gap_seconds: float = 0.25


def _finite_signal(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=np.float64)


def _safe_float(v: float) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def _max_false_gap(mask: np.ndarray) -> int:
    max_gap = 0
    cur = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            cur = 0
        else:
            cur += 1
            max_gap = max(max_gap, cur)
    return int(max_gap)


def _ratio(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    return float(np.mean(arr)) if arr.size else 0.0


def compute_global_signal_quality(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    *,
    config: RDLValidationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RDLValidationConfig()
    y = _finite_signal(signal)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(y)
    vals = y[valid]
    if vals.size == 0:
        return {
            "status": NO_VALID_RDL_SIGNAL,
            "max_angle": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
            "global_rom": 0.0,
            "valid_ratio_global": 0.0,
            "top_band": float("nan"),
            "bottom_band": float("nan"),
            "failed_checks": ["no_valid_signal_values"],
        }
    p5 = float(np.percentile(vals, 5.0))
    p95 = float(np.percentile(vals, 95.0))
    rom = float(max(0.0, p95 - p5))
    top_band_dynamic = p95 - cfg.top_band_margin_ratio * rom
    top_band = float(max(top_band_dynamic, cfg.top_band_absolute))
    bottom_band = float(p5 + cfg.bottom_band_margin_ratio * rom)
    max_angle = float(np.nanmax(vals))
    valid_ratio_global = _ratio(valid)

    failed: list[str] = []
    if max_angle < cfg.min_global_top_angle:
        failed.append("max_angle_below_150")
    if rom < cfg.min_global_rom:
        failed.append("global_rom_below_45")
    if valid_ratio_global < cfg.min_global_valid_ratio:
        failed.append("valid_ratio_global_below_0_75")
    return {
        "status": NO_VALID_RDL_SIGNAL if failed else SEGMENTATION_STATUS_OK,
        "max_angle": max_angle,
        "p5": p5,
        "p95": p95,
        "global_rom": rom,
        "valid_ratio_global": valid_ratio_global,
        "top_band_dynamic": float(top_band_dynamic),
        "top_band": top_band,
        "bottom_band": bottom_band,
        "failed_checks": failed,
    }


def compute_rep_validation_metrics(
    rep: dict[str, Any],
    signal: np.ndarray,
    *,
    fps: float,
    valid_mask: np.ndarray,
    valid_mask_raw: np.ndarray,
    interp_mask: np.ndarray,
    global_quality: dict[str, Any],
    config: RDLValidationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RDLValidationConfig()
    y = _finite_signal(signal)
    n = y.size
    top_start = int(rep["top_start"])
    bottom = int(rep["bottom"])
    top_end = int(rep["top_end"])
    ecc_start = int(rep["ecc_start"])
    ecc_end = int(rep["ecc_end"])
    con_start = int(rep["con_start"])
    con_end = int(rep["con_end"])

    def _value(idx: int) -> float:
        if 0 <= idx < n:
            return _safe_float(y[idx])
        return float("nan")

    start_val = _value(ecc_start)
    bottom_val = _value(bottom)
    end_val = _value(con_end)
    candidate_start_value = _value(top_start)
    candidate_end_value = _value(top_end)
    top_band = float(global_quality.get("top_band", float("nan")))
    bottom_band = float(global_quality.get("bottom_band", float("nan")))
    rom = float(global_quality.get("global_rom", 0.0) or 0.0)
    excursion = (
        float(min(start_val, end_val) - bottom_val)
        if np.isfinite([start_val, bottom_val, end_val]).all()
        else float("nan")
    )

    full_total_frames = int(top_end - top_start)
    active_total_frames = int(con_end - ecc_start)
    active_ecc_frames = int(ecc_end - ecc_start)
    active_con_frames = int(con_end - con_start)
    eps = max(cfg.monotonic_eps_abs, cfg.monotonic_eps_rom_ratio * rom)
    ecc_slice = y[max(0, ecc_start) : min(n, ecc_end + 1)]
    con_slice = y[max(0, con_start) : min(n, con_end + 1)]
    ecc_dy = np.diff(ecc_slice)
    con_dy = np.diff(con_slice)
    ecc_dy = ecc_dy[np.isfinite(ecc_dy)]
    con_dy = con_dy[np.isfinite(con_dy)]
    ecc_monotonic_ratio = float(np.mean(ecc_dy < -eps)) if ecc_dy.size else 0.0
    con_monotonic_ratio = float(np.mean(con_dy > eps)) if con_dy.size else 0.0
    ecc_direction_ratio = float(np.mean(ecc_dy < 0.0)) if ecc_dy.size else 0.0
    con_direction_ratio = float(np.mean(con_dy > 0.0)) if con_dy.size else 0.0
    ecc_abs_sum = float(np.sum(np.abs(ecc_dy))) if ecc_dy.size else 0.0
    con_abs_sum = float(np.sum(np.abs(con_dy))) if con_dy.size else 0.0
    ecc_trend_score = float(np.sum(np.maximum(-ecc_dy, 0.0)) / (ecc_abs_sum + 1e-8)) if ecc_dy.size else 0.0
    con_trend_score = float(np.sum(np.maximum(con_dy, 0.0)) / (con_abs_sum + 1e-8)) if con_dy.size else 0.0
    net_drop = float(ecc_slice[0] - ecc_slice[-1]) if ecc_slice.size >= 2 and np.isfinite(ecc_slice[[0, -1]]).all() else float("nan")
    net_rise = float(con_slice[-1] - con_slice[0]) if con_slice.size >= 2 and np.isfinite(con_slice[[0, -1]]).all() else float("nan")
    segment = y[max(0, top_start) : min(n, top_end + 1)]
    diffs = np.diff(segment)
    diffs = diffs[np.isfinite(diffs)]
    max_delta_per_frame = float(np.max(np.abs(diffs))) if diffs.size else 0.0
    velocity = np.abs(diffs) * float(fps)
    p95_velocity = float(np.percentile(velocity, 95.0)) if velocity.size else 0.0
    valid_seg = np.asarray(valid_mask, dtype=bool)[max(0, top_start) : min(n, top_end + 1)]
    raw_seg = np.asarray(valid_mask_raw, dtype=bool)[max(0, top_start) : min(n, top_end + 1)]
    interp_seg = np.asarray(interp_mask, dtype=bool)[max(0, top_start) : min(n, top_end + 1)]
    valid_ratio = _ratio(valid_seg)
    raw_valid_ratio = _ratio(raw_seg)
    interp_ratio = _ratio(interp_seg)
    max_invalid_gap = _max_false_gap(valid_seg)
    return {
        "start_value": start_val,
        "bottom_value": bottom_val,
        "end_value": end_val,
        "candidate_start_value": candidate_start_value,
        "candidate_end_value": candidate_end_value,
        "top_band": top_band,
        "bottom_band": bottom_band,
        "excursion": _safe_float(excursion),
        "full_total_frames": full_total_frames,
        "active_total_frames": active_total_frames,
        "active_ecc_frames": active_ecc_frames,
        "active_con_frames": active_con_frames,
        "total_frames": active_total_frames,
        "ecc_frames": active_ecc_frames,
        "con_frames": active_con_frames,
        "ecc_monotonic_ratio": ecc_monotonic_ratio,
        "con_monotonic_ratio": con_monotonic_ratio,
        "ecc_direction_ratio": ecc_direction_ratio,
        "con_direction_ratio": con_direction_ratio,
        "ecc_trend_score": ecc_trend_score,
        "con_trend_score": con_trend_score,
        "net_drop": _safe_float(net_drop),
        "net_rise": _safe_float(net_rise),
        "eps_monotonic": float(eps),
        "max_delta_per_frame": max_delta_per_frame,
        "p95_velocity": p95_velocity,
        "valid_ratio": valid_ratio,
        "raw_valid_ratio": raw_valid_ratio,
        "interp_ratio": interp_ratio,
        "max_invalid_gap": int(max_invalid_gap),
    }


def _initial_reasonable_candidates(
    candidates: list[dict[str, Any]],
    *,
    fps: float,
    config: RDLValidationConfig,
) -> list[dict[str, Any]]:
    out = []
    for cand in candidates:
        m = cand["validation_metrics"]
        if (
            m["total_frames"] >= fps * config.min_total_seconds
            and m["ecc_frames"] >= fps * config.min_phase_seconds
            and m["con_frames"] >= fps * config.min_phase_seconds
            and np.isfinite(m["excursion"])
            and m["excursion"] >= config.min_excursion_deg
        ):
            out.append(cand)
    return out


def _apply_candidate_rules(
    cand: dict[str, Any],
    *,
    fps: float,
    median_duration: float | None,
    median_excursion: float | None,
    config: RDLValidationConfig,
) -> None:
    m = cand["validation_metrics"]
    reasons: list[str] = []
    labels: list[str] = []
    warnings: list[str] = []
    if not bool(cand.get("anchor_valid", True)):
        reasons.append("INVALID_ANCHORS")
    top_tol = config.top_band_tolerance_deg
    top_threshold = m["top_band"] - top_tol
    start_low = (not np.isfinite(m["start_value"])) or m["start_value"] < top_threshold
    end_low = (not np.isfinite(m["end_value"])) or m["end_value"] < top_threshold
    enough_excursion = np.isfinite(m["excursion"]) and m["excursion"] >= config.min_excursion_deg
    clear_drop = np.isfinite(m["net_drop"]) and m["net_drop"] >= max(
        config.min_net_motion_abs,
        config.min_net_motion_excursion_ratio * max(float(m["excursion"]), 0.0),
    )
    clear_rise = np.isfinite(m["net_rise"]) and m["net_rise"] >= max(
        config.min_net_motion_abs,
        config.min_net_motion_excursion_ratio * max(float(m["excursion"]), 0.0),
    )
    if start_low and enough_excursion and clear_drop:
        warnings.append("low_start_top")
    elif start_low:
        labels.append("PARTIAL_START")
        reasons.append("start_below_top_band")
    if end_low and enough_excursion and clear_rise:
        warnings.append("low_end_top")
    elif end_low:
        labels.append("PARTIAL_END")
        reasons.append("end_below_top_band")
    if not np.isfinite(m["bottom_value"]) or m["bottom_value"] > m["bottom_band"]:
        if np.isfinite(m["excursion"]) and m["excursion"] >= config.min_excursion_deg:
            warnings.append("bottom_not_deep_enough")
        else:
            labels.append("WRONG_MOVEMENT_OR_NOT_RDL")
            reasons.append("bottom_not_deep_enough")
    if not np.isfinite(m["excursion"]) or m["excursion"] < config.min_excursion_deg:
        reasons.append("excursion_below_35")
    if m["total_frames"] < fps * config.min_total_seconds:
        reasons.append("duration_too_short")
    if m["ecc_frames"] < fps * config.min_phase_seconds:
        reasons.append("eccentric_too_short")
    if m["con_frames"] < fps * config.min_phase_seconds:
        reasons.append("concentric_too_short")
    if median_duration and np.isfinite(median_duration):
        if m["total_frames"] < config.min_duration_rel * median_duration:
            reasons.append("duration_too_short_vs_series")
        if m["total_frames"] > config.max_duration_rel * median_duration:
            reasons.append("duration_too_long_vs_series")
    if median_excursion and np.isfinite(median_excursion):
        if m["excursion"] < config.min_excursion_rel * median_excursion:
            reasons.append("excursion_too_small_vs_series")
    net_motion_threshold = max(
        config.min_net_motion_abs,
        config.min_net_motion_excursion_ratio * max(float(m["excursion"]), 0.0),
    )
    if m["ecc_trend_score"] < config.min_trend_score and (not np.isfinite(m["net_drop"]) or m["net_drop"] < net_motion_threshold):
        reasons.append("eccentric_not_monotonic")
    elif m["ecc_trend_score"] < config.min_trend_score:
        warnings.append("slow_eccentric")
    if m["con_trend_score"] < config.min_trend_score and (not np.isfinite(m["net_rise"]) or m["net_rise"] < net_motion_threshold):
        reasons.append("concentric_not_monotonic")
    elif m["con_trend_score"] < config.min_trend_score:
        warnings.append("slow_concentric")
    max_allowed_delta = max(config.max_delta_abs, config.max_delta_rom_ratio * float(m.get("rom", 0.0) or 0.0))
    if m["max_delta_per_frame"] > max_allowed_delta:
        labels.append("UNSTABLE_SIGNAL")
        reasons.append("abrupt_signal_jump")
    if m["p95_velocity"] > config.max_p95_velocity:
        labels.append("UNSTABLE_SIGNAL")
        reasons.append("unstable_signal")
    if m["valid_ratio"] < config.min_rep_valid_ratio:
        reasons.append("low_valid_ratio")
    if m["interp_ratio"] > config.max_rep_interp_ratio:
        reasons.append("too_much_interpolation")
    if m["max_invalid_gap"] > fps * config.max_invalid_gap_seconds:
        reasons.append("long_invalid_gap")

    if not reasons:
        cand["is_valid_individual"] = True
        cand["label"] = "VALID_WITH_WARNINGS" if warnings else "VALID_CANDIDATE"
        cand["reason"] = "ok"
    else:
        cand["is_valid_individual"] = False
        cand["label"] = labels[0] if labels else "INVALID_CANDIDATE"
        cand["reason"] = reasons[0]
    cand["validation_warnings"] = warnings
    cand["discard_reasons"] = reasons


def validate_rdl_rep_candidates(
    rep_candidates: list[dict[str, Any]],
    signal: np.ndarray,
    *,
    fps: float,
    valid_mask: np.ndarray,
    valid_mask_raw: np.ndarray,
    interp_mask: np.ndarray,
    global_quality: dict[str, Any],
    config: RDLValidationConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config or RDLValidationConfig()
    candidates: list[dict[str, Any]] = []
    for idx, rep in enumerate(rep_candidates):
        cand = dict(rep)
        cand["candidate_id"] = int(idx)
        metrics = compute_rep_validation_metrics(
            cand, signal, fps=fps, valid_mask=valid_mask, valid_mask_raw=valid_mask_raw,
            interp_mask=interp_mask, global_quality=global_quality, config=cfg,
        )
        metrics["rom"] = float(global_quality.get("global_rom", 0.0) or 0.0)
        cand["validation_metrics"] = metrics
        candidates.append(cand)

    reasonable = _initial_reasonable_candidates(candidates, fps=fps, config=cfg)
    durations = [c["validation_metrics"]["total_frames"] for c in reasonable]
    excursions = [c["validation_metrics"]["excursion"] for c in reasonable]
    median_duration = float(np.median(durations)) if durations else None
    median_excursion = float(np.median(excursions)) if excursions else None
    for cand in candidates:
        _apply_candidate_rules(
            cand, fps=fps, median_duration=median_duration, median_excursion=median_excursion, config=cfg
        )
    debug = {
        "num_candidates": len(candidates),
        "num_initial_reasonable": len(reasonable),
        "valid_anchor_reps": int(sum(1 for c in candidates if bool(c.get("anchor_valid", False)))),
        "invalid_anchor_reps": int(sum(1 for c in candidates if not bool(c.get("anchor_valid", False)))),
        "median_duration_frames": median_duration,
        "median_excursion": median_excursion,
        "validation_config": asdict(cfg),
    }
    return candidates, debug


def _block_score(block: list[dict[str, Any]]) -> tuple[int, float, float]:
    durations = np.asarray([c["validation_metrics"]["total_frames"] for c in block], dtype=np.float64)
    excursions = np.asarray([c["validation_metrics"]["excursion"] for c in block], dtype=np.float64)
    warning_count = sum(1 for c in block if c.get("validation_warnings"))

    def rel_std(values: np.ndarray) -> float:
        med = float(np.median(values)) if values.size else 0.0
        if med <= 1e-9:
            return float("inf")
        return float(np.std(values) / med)

    return (len(block), -0.05 * warning_count, -rel_std(durations), -rel_std(excursions))


def select_consistent_rep_block(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [c for c in candidates if c.get("is_valid_individual")]
    if not valid:
        return [], {"selected_candidate_ids": [], "reason": "no_individually_valid_candidates"}
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_id: int | None = None
    for cand in valid:
        cid = int(cand["candidate_id"])
        if previous_id is None or cid == previous_id + 1:
            current.append(cand)
        else:
            if current:
                blocks.append(current)
            current = [cand]
        previous_id = cid
    if current:
        blocks.append(current)

    best = max(blocks, key=_block_score)
    selected_ids = {int(c["candidate_id"]) for c in best}
    for cand in candidates:
        if cand.get("is_valid_individual") and int(cand["candidate_id"]) not in selected_ids:
            cand["is_valid_individual"] = False
            cand["label"] = "OUTSIDE_CONSISTENT_BLOCK"
            cand["reason"] = "outside_consistent_block"
            cand.setdefault("discard_reasons", []).append("outside_consistent_block")

    final = [c for c in candidates if int(c["candidate_id"]) in selected_ids]
    for new_id, cand in enumerate(final):
        cand["rep_id"] = int(new_id)
        cand["validation_label"] = "VALID_WITH_WARNINGS" if cand.get("validation_warnings") else "VALID"
        cand["label"] = cand["validation_label"]
        cand["reason"] = "ok"
    return final, {
        "selected_candidate_ids": sorted(selected_ids),
        "num_blocks": len(blocks),
        "block_lengths": [len(b) for b in blocks],
        "reason": "selected_largest_most_consistent_block",
    }


def append_edge_artifact_candidates(
    candidates: list[dict[str, Any]],
    final_reps: list[dict[str, Any]],
    signal: np.ndarray,
    *,
    fps: float,
    min_edge_seconds: float = 0.25,
    min_edge_rom: float = 5.0,
) -> list[dict[str, Any]]:
    if not final_reps:
        return candidates
    y = _finite_signal(signal)
    n = y.size
    min_len = max(1, int(round(float(fps) * min_edge_seconds)))
    out = list(candidates)

    def _edge_rom(start: int, end: int) -> float:
        seg = y[max(0, start) : min(n, end + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            return 0.0
        return float(np.nanmax(seg) - np.nanmin(seg))

    def _append(start: int, end: int, label: str, reason: str) -> None:
        if end - start + 1 < min_len:
            return
        if _edge_rom(start, end) < min_edge_rom:
            return
        out.append(
            {
                "candidate_id": len(out),
                "rep_id": -1,
                "top_start": int(start),
                "bottom": int(round((start + end) / 2.0)),
                "top_end": int(end),
                "ecc_start": int(start),
                "ecc_end": int(round((start + end) / 2.0)),
                "con_start": int(round((start + end) / 2.0)),
                "con_end": int(end),
                "anchors": {},
                "anchor_details": {},
                "anchor_valid": False,
                "anchor_warnings": ["EDGE_ARTIFACT_CANDIDATE"],
                "is_valid_individual": False,
                "label": label,
                "reason": reason,
                "discard_reasons": [reason],
                "validation_metrics": {"edge_rom": _edge_rom(start, end), "total_frames": int(end - start)},
            }
        )

    first_start = int(min(r["top_start"] for r in final_reps))
    last_end = int(max(r["top_end"] for r in final_reps))
    if first_start > 0:
        earlier_candidates = [c for c in candidates if int(c.get("top_start", n + 1)) < first_start]
        if earlier_candidates:
            edge_end = min(int(c.get("top_start", first_start)) for c in earlier_candidates) - 1
        else:
            edge_end = first_start - 1
        _append(0, edge_end, "PARTIAL_START", "missing_stable_top_before_descent")
    if last_end < n - 1:
        _append(last_end + 1, n - 1, "POST_EXERCISE_NOISE", "post_exercise_noise")
    return out

