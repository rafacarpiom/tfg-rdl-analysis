
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.biomechanics.rdl.segmentation.config import RDLSegmentationConfig

_NORM_POINTS = 101

VELOCITY_SMOOTH_SECONDS = 0.10
MAIN_LOBE_MAX_GAP_SECONDS = 0.20
MAIN_LOBE_MIN_BLOCK_SECONDS = 0.08
DESCENT_VEL_ABS_MIN = 0.20
DESCENT_VEL_ROM_RATIO = 0.0015
DESCENT_VEL_P90_RATIO = 0.25
MAIN_LOBE_MIN_BLOCK_DROP_RATIO = 0.08
MAIN_LOBE_MIN_BLOCK_DROP_ABS = 8.0
MAIN_LOBE_MIN_FUTURE_DROP_RATIO = 0.45
MAIN_LOBE_MIN_FUTURE_DROP_ABS = 20.0
ECC_START_BACKTRACK_SECONDS = 0.18
ECC_START_LOW_VEL_RATIO = 0.18
ECC_FIRST_QUARTER_DROP_RATIO = 0.08
ECC_FIRST_QUARTER_DROP_ABS = 8.0
CON_START_WINDOW_SECONDS = 0.12
CON_START_MIN_RATIO = 0.55
CON_START_MIN_TREND = 0.65
CON_END_HOLD_SECONDS = 0.20
CON_END_TOP_TOLERANCE_RATIO = 0.03
CON_END_TOP_TOLERANCE_ABS = 2.0
INFERRED_START_MIN_EXCURSION_ABS = 20.0


def _finite_valid_signal(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=np.float64)


def _moving_average(values: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    win = max(1, int(win))
    if win <= 1:
        return arr.copy()
    kernel = np.ones(win, dtype=np.float64) / float(win)
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _velocity_threshold(down_smooth: np.ndarray, rom: float) -> float:
    vals = np.asarray(down_smooth, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 1e-9)]
    p90 = float(np.percentile(vals, 90.0)) if vals.size else 0.0
    return max(
        DESCENT_VEL_ABS_MIN,
        DESCENT_VEL_ROM_RATIO * float(max(rom, 0.0)),
        DESCENT_VEL_P90_RATIO * p90,
    )


def _trend_score(dy: np.ndarray, direction: str) -> float:
    dy = np.asarray(dy, dtype=np.float64)
    dy = dy[np.isfinite(dy)]
    if dy.size == 0:
        return 0.0
    denom = float(np.sum(np.abs(dy))) + 1e-8
    if direction == "down":
        return float(np.sum(np.maximum(-dy, 0.0)) / denom)
    return float(np.sum(np.maximum(dy, 0.0)) / denom)


def _segments_from_mask(mask: np.ndarray, offset: int = 0) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    segments: list[tuple[int, int]] = []
    i = 0
    while i < arr.size:
        if not arr[i]:
            i += 1
            continue
        start = i
        while i < arr.size and arr[i]:
            i += 1
        end = i - 1
        segments.append((int(start + offset), int(end + offset)))
    return segments


def _merge_close_segments(segments: list[tuple[int, int]], *, max_gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    ordered = sorted((int(a), int(b)) for a, b in segments)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if gap <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _safe_value(y: np.ndarray, idx: int) -> float:
    if 0 <= int(idx) < y.size and np.isfinite(y[int(idx)]):
        return float(y[int(idx)])
    return float("nan")


def _drop_between(y: np.ndarray, start: int, end: int) -> float:
    a = _safe_value(y, start)
    b = _safe_value(y, end)
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def _refine_ecc_start_first_quarter(y: np.ndarray, start: int, bottom: int, *, excursion: float) -> int:
    start = int(start)
    bottom = int(bottom)
    if bottom <= start + 4:
        return start
    min_drop = max(ECC_FIRST_QUARTER_DROP_ABS, ECC_FIRST_QUARTER_DROP_RATIO * float(max(excursion, 0.0)))
    last = max(start, bottom - 4)
    for t in range(start, last + 1):
        q25 = int(round(t + 0.25 * (bottom - t)))
        if q25 <= t or q25 >= y.size:
            continue
        drop = _drop_between(y, t, q25)
        if np.isfinite(drop) and drop >= min_drop:
            return int(t)
    return start


def find_active_eccentric_start(
    signal_smooth: np.ndarray,
    *,
    top_start: int,
    bottom: int,
    fps: float,
    rom: float,
    excursion: float | None = None,
) -> tuple[int, bool, dict[str, object]]:
    y = np.asarray(signal_smooth, dtype=np.float64)
    n = y.size
    lo = max(0, int(top_start))
    hi = min(n - 1, int(bottom))
    debug: dict[str, object] = {
        "method": "main_descent_lobe",
        "found": False,
        "selected_segment": None,
        "velocity_threshold": float("nan"),
        "candidate_segments": [],
    }
    if hi <= lo + 3 or not np.isfinite(y[hi]):
        return lo, False, debug
    exc = float(excursion) if excursion is not None and np.isfinite(excursion) else float(rom)
    if exc <= 0 or not np.isfinite(exc):
        exc = max(float(rom), 1.0)

    dy = np.diff(y)
    down = np.maximum(-dy, 0.0)
    down[~np.isfinite(down)] = 0.0
    vel_win = max(1, int(round(VELOCITY_SMOOTH_SECONDS * float(fps))))
    down_smooth = _moving_average(down, vel_win)
    interval_down = down_smooth[lo:hi]
    vel_thr = _velocity_threshold(interval_down, rom)
    debug["velocity_threshold"] = float(vel_thr)

    active_mask = interval_down >= vel_thr
    raw_segments = _segments_from_mask(active_mask, offset=lo)
    max_gap = max(1, int(round(MAIN_LOBE_MAX_GAP_SECONDS * float(fps))))
    min_block_len = max(2, int(round(MAIN_LOBE_MIN_BLOCK_SECONDS * float(fps))))
    segments = _merge_close_segments(raw_segments, max_gap=max_gap)

    min_block_drop = max(MAIN_LOBE_MIN_BLOCK_DROP_ABS, MAIN_LOBE_MIN_BLOCK_DROP_RATIO * max(exc, 0.0))
    min_future_drop = max(MAIN_LOBE_MIN_FUTURE_DROP_ABS, MAIN_LOBE_MIN_FUTURE_DROP_RATIO * max(exc, 0.0))
    candidates: list[dict[str, object]] = []
    for start_v, end_v in segments:
        frame_start = int(start_v)
        frame_end = int(min(end_v + 1, hi))
        if frame_end <= frame_start:
            continue
        length = int(end_v - start_v + 1)
        block_drop = _drop_between(y, frame_start, frame_end)
        future_drop = _drop_between(y, frame_start, hi)
        if not np.isfinite(block_drop) or not np.isfinite(future_drop):
            continue
        score = (1.00 * max(block_drop, 0.0) + 0.25 * max(future_drop, 0.0) - 0.015 * max(hi - frame_end, 0))
        accepted = bool(
            length >= min_block_len
            and block_drop >= min_block_drop
            and future_drop >= min_future_drop
        )
        candidates.append(
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "length": length,
                "block_drop": float(block_drop),
                "future_drop": float(future_drop),
                "score": float(score),
                "accepted": accepted,
            }
        )
    debug["candidate_segments"] = candidates
    accepted = [c for c in candidates if bool(c["accepted"])]
    if not accepted:
        total_down = float(np.sum(down[lo:hi]))
        if total_down > 1e-6:
            target = 0.12 * total_down
            cum = np.cumsum(down[lo:hi])
            idx = int(np.searchsorted(cum, target, side="left"))
            fallback = min(hi - 1, lo + max(0, idx))
            fallback = _refine_ecc_start_first_quarter(y, fallback, hi, excursion=exc)
            debug["method"] = "fallback_cumulative_descent"
            return int(fallback), False, debug
        return lo, False, debug

    best = max(accepted, key=lambda c: (float(c["score"]), float(c["block_drop"])))
    onset = int(best["frame_start"])
    low_thr = ECC_START_LOW_VEL_RATIO * vel_thr
    back_limit = max(lo, onset - int(round(ECC_START_BACKTRACK_SECONDS * float(fps))))
    while onset > back_limit:
        prev_slot = onset - 1
        if prev_slot < 0 or prev_slot >= down_smooth.size:
            break
        if down_smooth[prev_slot] < low_thr:
            break
        onset -= 1
    onset = _refine_ecc_start_first_quarter(y, onset, hi, excursion=exc)
    debug["found"] = True
    debug["selected_segment"] = best
    debug["onset_after_refine"] = int(onset)
    return int(onset), True, debug


def _has_sustained_trend(
    y: np.ndarray,
    start: int,
    *,
    direction: str,
    window: int,
    velocity_thr: float,
    min_ratio: float,
    min_trend: float,
) -> bool:
    end = min(y.size - 1, int(start) + max(2, int(window)))
    if end <= start:
        return False
    dy = np.diff(y[start : end + 1])
    dy = dy[np.isfinite(dy)]
    if dy.size == 0:
        return False
    ratio = float(np.mean(dy < -velocity_thr)) if direction == "down" else float(np.mean(dy > velocity_thr))
    return bool(ratio >= min_ratio or _trend_score(dy, direction) >= min_trend)


def find_concentric_start(y: np.ndarray, bottom: int, search_end: int, *, rom: float, fps: float) -> int:
    window = max(4, int(round(CON_START_WINDOW_SECONDS * float(fps))))
    velocity_thr = max(DESCENT_VEL_ABS_MIN, DESCENT_VEL_ROM_RATIO * float(max(rom, 0.0)))
    lo = max(0, int(bottom))
    hi = min(y.size - 2, int(search_end))
    for t in range(lo, hi + 1):
        if _has_sustained_trend(
            y,
            t,
            direction="up",
            window=window,
            velocity_thr=velocity_thr,
            min_ratio=CON_START_MIN_RATIO,
            min_trend=CON_START_MIN_TREND,
        ):
            return int(t)
    return int(bottom)


def find_concentric_end(
    y: np.ndarray,
    con_start: int,
    search_end: int,
    *,
    top_band: float,
    rom: float,
    fps: float,
) -> int:
    hold = max(4, int(round(CON_END_HOLD_SECONDS * float(fps))))
    tolerance = max(CON_END_TOP_TOLERANCE_ABS, CON_END_TOP_TOLERANCE_RATIO * float(max(rom, 0.0)))
    lo = max(0, int(con_start))
    hi = min(y.size - 1, int(search_end))
    first_crossing: int | None = None
    for t in range(lo, hi + 1):
        if not np.isfinite(y[t]) or y[t] < top_band:
            continue
        if first_crossing is None:
            first_crossing = int(t)
        window = y[t : min(y.size, t + hold)]
        valid = window[np.isfinite(window)]
        if valid.size >= max(2, hold // 2):
            near_top_ratio = float(np.mean(valid >= top_band - tolerance))
            if near_top_ratio >= 0.65:
                return int(t)
        local = y[t : min(y.size, t + max(3, hold // 2))]
        dy = np.diff(local)
        dy = dy[np.isfinite(dy)]
        if dy.size and _trend_score(dy, "up") < 0.55:
            return int(t)
    if first_crossing is not None:
        return int(first_crossing)
    return int(search_end)


def _normalize_phase(signal: np.ndarray, start: int, end: int, percentages: tuple[int, ...]) -> dict:
    start = int(start)
    end = int(end)
    seg = signal[start : end + 1]
    n = len(seg)
    if n < 2:
        val = float(seg[0]) if n == 1 else float("nan")
        curve = [val] * _NORM_POINTS
        return {"curve_101": curve, **{f"pct_{p}": val for p in percentages}}
    x_orig = np.linspace(0.0, 100.0, n)
    x_norm = np.linspace(0.0, 100.0, _NORM_POINTS)
    curve = np.interp(x_norm, x_orig, seg)
    result: dict = {"curve_101": [round(float(v), 3) for v in curve]}
    for p in percentages:
        idx = int(p * (_NORM_POINTS - 1) / 100)
        result[f"pct_{p}"] = round(float(curve[idx]), 3)
    return result


def _infer_top_before_bottom(y: np.ndarray, bottom_idx: int, *, min_excursion_abs: float) -> int | None:
    mn = int(bottom_idx)
    if mn <= 1 or mn >= y.size or not np.isfinite(y[mn]):
        return None
    region = y[:mn]
    finite = np.isfinite(region)
    if not finite.any():
        return None
    peak = int(np.nanargmax(region))
    if not np.isfinite(y[peak]):
        return None
    drop = float(y[peak] - y[mn])
    if drop < float(min_excursion_abs):
        return None
    return peak


def _finite_frames_in_phase(signal: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
    start = int(start_frame)
    end = int(end_frame)
    if start > end:
        start, end = end, start
    if start < 0:
        start = 0
    if end >= signal.size:
        end = signal.size - 1
    if end < start:
        return np.asarray([], dtype=int)
    frames = np.arange(start, end + 1, dtype=int)
    finite = np.isfinite(np.asarray(signal[frames], dtype=np.float64))
    return frames[finite].astype(int)


def _nearest_valid_frame(target_frame: int, valid_frames: np.ndarray) -> int | None:
    frames = np.asarray(valid_frames, dtype=int)
    if frames.size == 0:
        return None
    target = int(target_frame)
    distances = np.abs(frames - target)
    min_dist = int(np.min(distances))
    tied = frames[distances == min_dist]
    return int(np.min(tied))


def _invalid_anchor(
    *,
    pct: int,
    method: str,
    temporal_hint: int,
    warning: str,
    target_signal: float | None = None,
    nearest_shift: int | None = None,
) -> dict:
    return {
        "frame": None,
        "pct": int(pct),
        "method": method,
        "valid": False,
        "signal": None,
        "target_signal": float(target_signal) if target_signal is not None and np.isfinite(target_signal) else None,
        "temporal_hint": int(temporal_hint),
        "nearest_shift": int(nearest_shift) if nearest_shift is not None else None,
        "warning": warning,
    }


def _select_phase_anchor_by_signal_progress(
    *,
    signal: np.ndarray,
    phase_start: int,
    phase_end: int,
    pct: int,
    valid_frames: np.ndarray,
    previous_anchor_frame: int | None,
    min_phase_rom_deg: float,
) -> dict:
    temporal_hint = int(round(int(phase_start) + (int(phase_end) - int(phase_start)) * int(pct) / 100.0))
    if valid_frames.size < 2:
        return _invalid_anchor(
            pct=pct,
            method="signal_progress",
            temporal_hint=temporal_hint,
            warning="INSUFFICIENT_VALID_FRAMES_FOR_SIGNAL_PROGRESS",
        )

    start_valid = _nearest_valid_frame(int(phase_start), valid_frames)
    end_valid = _nearest_valid_frame(int(phase_end), valid_frames)
    if start_valid is None or end_valid is None:
        return _invalid_anchor(
            pct=pct,
            method="signal_progress",
            temporal_hint=temporal_hint,
            warning="MISSING_VALID_PHASE_BOUNDARY",
        )

    start_value = float(signal[start_valid])
    end_value = float(signal[end_valid])
    if not (np.isfinite(start_value) and np.isfinite(end_value)):
        return _invalid_anchor(
            pct=pct,
            method="signal_progress",
            temporal_hint=temporal_hint,
            warning="NON_FINITE_PHASE_BOUNDARY_SIGNAL",
        )

    phase_rom = abs(end_value - start_value)
    if phase_rom < float(min_phase_rom_deg):
        return _invalid_anchor(
            pct=pct,
            method="signal_progress",
            temporal_hint=temporal_hint,
            warning="PHASE_ROM_TOO_SMALL_FOR_SIGNAL_PROGRESS",
        )

    target_signal = float(start_value + (end_value - start_value) * (int(pct) / 100.0))
    warning: str | None = None
    candidate_frames = np.asarray(valid_frames, dtype=int)
    if previous_anchor_frame is not None:
        filtered = candidate_frames[candidate_frames >= int(previous_anchor_frame)]
        if filtered.size > 0:
            candidate_frames = filtered
        else:
            warning = "ORDER_FALLBACK_USED"

    if candidate_frames.size == 0:
        return _invalid_anchor(
            pct=pct,
            method="signal_progress",
            temporal_hint=temporal_hint,
            warning="NO_VALID_CANDIDATE_AFTER_ORDER_FILTER",
            target_signal=target_signal,
        )

    candidate_values = np.asarray(signal[candidate_frames], dtype=np.float64)
    finite_mask = np.isfinite(candidate_values)
    candidate_frames = candidate_frames[finite_mask]
    candidate_values = candidate_values[finite_mask]
    if candidate_frames.size == 0:
        return _invalid_anchor(
            pct=pct,
            method="signal_progress",
            temporal_hint=temporal_hint,
            warning="NO_FINITE_CANDIDATE_SIGNAL",
            target_signal=target_signal,
        )

    errors = np.abs(candidate_values - target_signal)
    temporal_dist = np.abs(candidate_frames - temporal_hint)
    order = np.lexsort((candidate_frames, temporal_dist, errors))
    selected_frame = int(candidate_frames[order[0]])
    if not np.isfinite(signal[selected_frame]):
        return _invalid_anchor(
            pct=pct,
            method="signal_progress",
            temporal_hint=temporal_hint,
            warning="SELECTED_ANCHOR_IS_NON_FINITE",
            target_signal=target_signal,
        )
    return {
        "frame": int(selected_frame),
        "pct": int(pct),
        "method": "signal_progress",
        "valid": True,
        "signal": float(signal[selected_frame]),
        "target_signal": float(target_signal),
        "temporal_hint": int(temporal_hint),
        "nearest_shift": int(selected_frame - temporal_hint),
        "warning": warning,
    }


def _select_phase_anchor_by_valid_frame_percent(
    *,
    signal: np.ndarray,
    phase_start: int,
    phase_end: int,
    pct: int,
    valid_frames: np.ndarray,
) -> dict:
    temporal_hint = int(round(int(phase_start) + (int(phase_end) - int(phase_start)) * int(pct) / 100.0))
    frames = np.asarray(valid_frames, dtype=int)
    if frames.size == 0:
        return _invalid_anchor(
            pct=pct,
            method="valid_frame_percent",
            temporal_hint=temporal_hint,
            warning="NO_VALID_FRAMES_IN_PHASE",
        )
    idx = int(round((frames.size - 1) * int(pct) / 100.0))
    selected_frame = int(frames[idx])
    if not np.isfinite(signal[selected_frame]):
        return _invalid_anchor(
            pct=pct,
            method="valid_frame_percent",
            temporal_hint=temporal_hint,
            warning="SELECTED_ANCHOR_IS_NON_FINITE",
        )
    return {
        "frame": int(selected_frame),
        "pct": int(pct),
        "method": "valid_frame_percent",
        "valid": True,
        "signal": float(signal[selected_frame]),
        "target_signal": None,
        "temporal_hint": int(temporal_hint),
        "nearest_shift": int(selected_frame - temporal_hint),
        "warning": None,
    }


def _select_phase_anchor_nearest_valid(
    *,
    signal: np.ndarray,
    phase_start: int,
    phase_end: int,
    pct: int,
    valid_frames: np.ndarray,
) -> dict:
    temporal_hint = int(round(int(phase_start) + (int(phase_end) - int(phase_start)) * int(pct) / 100.0))
    selected_frame = _nearest_valid_frame(temporal_hint, valid_frames)
    if selected_frame is None:
        return _invalid_anchor(
            pct=pct,
            method="nearest_valid_frame",
            temporal_hint=temporal_hint,
            warning="NO_NEAREST_VALID_FRAME_FOUND",
        )
    if not np.isfinite(signal[selected_frame]):
        return _invalid_anchor(
            pct=pct,
            method="nearest_valid_frame",
            temporal_hint=temporal_hint,
            warning="NEAREST_VALID_FRAME_IS_NON_FINITE",
        )
    return {
        "frame": int(selected_frame),
        "pct": int(pct),
        "method": "nearest_valid_frame",
        "valid": True,
        "signal": float(signal[selected_frame]),
        "target_signal": None,
        "temporal_hint": int(temporal_hint),
        "nearest_shift": int(selected_frame - temporal_hint),
        "warning": None,
    }


def _build_phase_anchors(
    *,
    signal: np.ndarray,
    phase_name: str,
    phase_start: int,
    phase_end: int,
    percentages: tuple[int, ...],
    min_valid_frames: int,
    min_phase_rom_deg: float,
    allow_nearest_valid_fallback: bool,
) -> dict:
    valid_frames = _finite_frames_in_phase(signal, phase_start, phase_end)
    total_frame_count = abs(int(phase_end) - int(phase_start)) + 1
    valid_frame_count = int(valid_frames.size)
    valid_ratio = float(valid_frame_count / total_frame_count) if total_frame_count > 0 else 0.0
    anchors: dict[str, dict] = {}

    if valid_frame_count < int(min_valid_frames):
        for pct in percentages:
            temporal_hint = int(round(int(phase_start) + (int(phase_end) - int(phase_start)) * int(pct) / 100.0))
            anchors[str(int(pct))] = _invalid_anchor(
                pct=int(pct),
                method="signal_progress",
                temporal_hint=temporal_hint,
                warning="INSUFFICIENT_VALID_FRAMES_IN_PHASE",
            )
        return {
            "phase": str(phase_name),
            "start": int(phase_start),
            "end": int(phase_end),
            "valid_frame_count": valid_frame_count,
            "total_frame_count": int(total_frame_count),
            "valid_ratio": float(valid_ratio),
            "anchors": anchors,
            "valid": False,
            "warning": "INSUFFICIENT_VALID_FRAMES_IN_PHASE",
        }

    previous_anchor_frame: int | None = None
    for pct in percentages:
        anchor = _select_phase_anchor_by_signal_progress(
            signal=signal,
            phase_start=phase_start,
            phase_end=phase_end,
            pct=int(pct),
            valid_frames=valid_frames,
            previous_anchor_frame=previous_anchor_frame,
            min_phase_rom_deg=float(min_phase_rom_deg),
        )
        if not bool(anchor["valid"]):
            anchor = _select_phase_anchor_by_valid_frame_percent(
                signal=signal,
                phase_start=phase_start,
                phase_end=phase_end,
                pct=int(pct),
                valid_frames=valid_frames,
            )
        if not bool(anchor["valid"]) and bool(allow_nearest_valid_fallback):
            anchor = _select_phase_anchor_nearest_valid(
                signal=signal,
                phase_start=phase_start,
                phase_end=phase_end,
                pct=int(pct),
                valid_frames=valid_frames,
            )

        anchors[str(int(pct))] = anchor
        if bool(anchor["valid"]) and anchor["frame"] is not None:
            previous_anchor_frame = int(anchor["frame"])

    phase_valid = all(bool(a.get("valid")) for a in anchors.values())
    return {
        "phase": str(phase_name),
        "start": int(phase_start),
        "end": int(phase_end),
        "valid_frame_count": valid_frame_count,
        "total_frame_count": int(total_frame_count),
        "valid_ratio": float(valid_ratio),
        "anchors": anchors,
        "valid": bool(phase_valid),
        "warning": None if phase_valid else "INVALID_ANCHORS_IN_PHASE",
    }


def _build_simple_anchor_map(
    *,
    bottom_frame: int,
    ecc_anchor_info: dict,
    con_anchor_info: dict,
    percentages: tuple[int, ...],
) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    ecc_anchors = ecc_anchor_info.get("anchors", {})
    con_anchors = con_anchor_info.get("anchors", {})
    for pct in percentages:
        k = str(int(pct))
        ecc_item = ecc_anchors.get(k, {})
        con_item = con_anchors.get(k, {})
        out[f"ecc_{k}"] = int(ecc_item["frame"]) if bool(ecc_item.get("valid")) and ecc_item.get("frame") is not None else None
        out[f"con_{k}"] = int(con_item["frame"]) if bool(con_item.get("valid")) and con_item.get("frame") is not None else None
    out["bottom"] = int(bottom_frame)
    return out


def rebuild_rdl_candidate_phase_and_anchors(
    rep: dict,
    signal_smooth: np.ndarray,
    *,
    fps: float,
    global_rom: float,
    config: "RDLSegmentationConfig",
) -> dict:
    y = _finite_valid_signal(signal_smooth)
    top_start = int(rep["top_start"])
    bottom = int(rep["bottom"])
    top_end_candidate = int(rep.get("candidate_top_end", rep.get("top_end", bottom)))
    if top_end_candidate < bottom:
        top_end_candidate = int(rep.get("top_end", bottom))
    top_end_candidate = min(max(top_end_candidate, bottom), y.size - 1)
    top_start = min(max(top_start, 0), max(0, bottom - 1))

    top_start_val = _safe_value(y, top_start)
    top_end_val = _safe_value(y, top_end_candidate)
    bottom_val = _safe_value(y, bottom)
    if not np.isfinite([top_start_val, top_end_val, bottom_val]).all():
        return dict(rep)
    top_ref = min(float(top_start_val), float(top_end_val))
    excursion = float(top_ref - float(bottom_val))
    if excursion <= 0.0 or not np.isfinite(excursion):
        return dict(rep)

    finite_vals = y[np.isfinite(y)]
    if finite_vals.size < 3:
        return dict(rep)
    p5 = float(np.percentile(finite_vals, 5.0))
    p95 = float(np.percentile(finite_vals, 95.0))
    top_band_global = float(p95 - float(config.phase_top_margin_ratio) * float(max(global_rom, 0.0)))
    bottom_band_global = float(p5 + float(config.phase_bottom_margin_ratio) * float(max(global_rom, 0.0)))
    top_margin = float(config.phase_top_margin_ratio) * excursion
    bottom_margin = float(config.phase_bottom_margin_ratio) * excursion
    top_band = max(top_band_global, top_ref - top_margin)
    bottom_band = min(bottom_band_global, float(bottom_val) + bottom_margin)

    ecc_start = int(rep.get("ecc_start", top_start))
    ecc_start = max(top_start, min(ecc_start, bottom))
    ecc_end = int(bottom)
    con_start = find_concentric_start(y, bottom, top_end_candidate, rom=float(max(global_rom, 0.0)), fps=fps)
    con_end = find_concentric_end(
        y,
        con_start,
        top_end_candidate,
        top_band=top_band,
        rom=float(max(global_rom, 0.0)),
        fps=fps,
    )
    if con_start > con_end:
        con_start = int(bottom)
        con_end = int(top_end_candidate)

    ecc_norm = _normalize_phase(y, ecc_start, ecc_end, config.anchor_percentages)
    con_norm = _normalize_phase(y, con_start, con_end, config.anchor_percentages)
    ecc_anchor_info = _build_phase_anchors(
        signal=y,
        phase_name="eccentric",
        phase_start=ecc_start,
        phase_end=ecc_end,
        percentages=config.anchor_percentages,
        min_valid_frames=config.anchor_min_valid_frames,
        min_phase_rom_deg=config.anchor_min_phase_rom_deg,
        allow_nearest_valid_fallback=config.anchor_allow_nearest_valid_fallback,
    )
    con_anchor_info = _build_phase_anchors(
        signal=y,
        phase_name="concentric",
        phase_start=con_start,
        phase_end=con_end,
        percentages=config.anchor_percentages,
        min_valid_frames=config.anchor_min_valid_frames,
        min_phase_rom_deg=config.anchor_min_phase_rom_deg,
        allow_nearest_valid_fallback=config.anchor_allow_nearest_valid_fallback,
    )
    anchor_details = {"eccentric": ecc_anchor_info, "concentric": con_anchor_info}
    anchors = _build_simple_anchor_map(
        bottom_frame=bottom,
        ecc_anchor_info=ecc_anchor_info,
        con_anchor_info=con_anchor_info,
        percentages=config.anchor_percentages,
    )
    anchor_valid = bool(ecc_anchor_info["valid"] and con_anchor_info["valid"])
    warnings: list[str] = []
    for phase in (ecc_anchor_info, con_anchor_info):
        if phase.get("warning"):
            warnings.append(str(phase["warning"]))
        for item in phase.get("anchors", {}).values():
            w = item.get("warning")
            if w:
                warnings.append(str(w))

    out = dict(rep)
    out.update(
        {
            "top_start": int(top_start),
            "top_end": int(con_end),
            "candidate_top_end": int(top_end_candidate),
            "ecc_start": int(ecc_start),
            "ecc_end": int(ecc_end),
            "con_start": int(con_start),
            "con_end": int(con_end),
            "excursion": round(float(excursion), 2),
            "top_margin": round(float(top_margin), 2),
            "bottom_margin": round(float(bottom_margin), 2),
            "top_band": round(float(top_band), 3),
            "bottom_band": round(float(bottom_band), 3),
            "pre_eccentric_hold_start": int(top_start),
            "pre_eccentric_hold_end": int(max(top_start, ecc_start - 1)),
            "pre_eccentric_hold_frames": int(max(0, ecc_start - top_start)),
            "ecc_frames": int(ecc_end - ecc_start),
            "con_frames": int(con_end - con_start),
            "ecc_norm": ecc_norm,
            "con_norm": con_norm,
            "anchors": anchors,
            "anchor_details": anchor_details,
            "anchor_valid": anchor_valid,
            "anchor_warnings": sorted(set(warnings)),
        }
    )
    return out


def build_rdl_repetitions(
    signal_smooth: np.ndarray,
    bottom_events_valid: list[int],
    top_events_valid: list[int],
    *,
    fps: float,
    config: "RDLSegmentationConfig",
) -> list[dict]:
    y = _finite_valid_signal(signal_smooth)
    tops = sorted(int(i) for i in top_events_valid)
    bottoms = sorted(int(i) for i in bottom_events_valid)
    if len(bottoms) < 1:
        return []

    effective_tops = list(tops)
    if bottoms and (not tops or bottoms[0] < tops[0]):
        inferred_start: int | None = None
        for btm in bottoms:
            peak = _infer_top_before_bottom(y, btm, min_excursion_abs=INFERRED_START_MIN_EXCURSION_ABS)
            if peak is not None:
                inferred_start = int(peak)
                break
        if inferred_start is not None:
            effective_tops.insert(0, inferred_start)
    if tops and bottoms[-1] > tops[-1]:
        region = y[bottoms[-1] :]
        finite = np.isfinite(region)
        if finite.any():
            peak = bottoms[-1] + int(np.nanargmax(region))
            if np.isfinite(y[peak]) and y[peak] > y[bottoms[-1]]:
                effective_tops.append(peak)

    effective_tops = sorted(set(effective_tops))
    if len(effective_tops) < 2:
        return []

    finite_vals = y[np.isfinite(y)]
    if finite_vals.size < 3:
        return []
    p5 = float(np.percentile(finite_vals, 5.0))
    p95 = float(np.percentile(finite_vals, 95.0))
    global_rom = float(max(0.0, p95 - p5))

    reps: list[dict] = []
    min_ptr = 0
    for i in range(len(effective_tops) - 1):
        top0 = int(effective_tops[i])
        top1 = int(effective_tops[i + 1])
        btm = None
        while min_ptr < len(bottoms):
            if bottoms[min_ptr] > top0:
                if bottoms[min_ptr] < top1:
                    btm = int(bottoms[min_ptr])
                break
            min_ptr += 1
        if btm is None:
            continue
        if not (0 <= top0 < y.size and 0 <= btm < y.size and 0 <= top1 < y.size):
            min_ptr += 1
            continue
        if not np.isfinite([y[top0], y[btm], y[top1]]).all():
            min_ptr += 1
            continue
        top_ref = min(float(y[top0]), float(y[top1]))
        excursion = float(top_ref - float(y[btm]))
        if excursion <= 0:
            min_ptr += 1
            continue

        ecc_start, active_ecc_found, ecc_debug = find_active_eccentric_start(
            y,
            top_start=top0,
            bottom=btm,
            rom=global_rom,
            fps=fps,
            excursion=excursion,
        )
        ecc_start_fallback_used = not bool(active_ecc_found)
        if ecc_start > btm:
            ecc_start = top0
            ecc_start_fallback_used = True

        base_rep = {
            "rep_id": len(reps),
            "top_start": int(top0),
            "bottom": int(btm),
            "candidate_top_end": int(top1),
            "ecc_start": int(ecc_start),
            "ecc_start_method": "main_descent_lobe" if active_ecc_found else "fallback",
            "ecc_start_fallback_used": bool(ecc_start_fallback_used),
            "phase_debug": {"ecc_start": ecc_debug, "top_ref": float(top_ref), "global_rom": float(global_rom)},
        }
        reps.append(
            rebuild_rdl_candidate_phase_and_anchors(
                base_rep,
                y,
                fps=fps,
                global_rom=global_rom,
                config=config,
            )
        )
        min_ptr += 1
    return reps

