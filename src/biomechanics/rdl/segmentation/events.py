
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_top_bottom_candidates(
    signal: np.ndarray,
    *,
    min_distance: int = 15,
    min_prominence_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(y)
    valid_vals = y[finite]
    if valid_vals.size < 3:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    p5 = float(np.percentile(valid_vals, 5.0))
    p95 = float(np.percentile(valid_vals, 95.0))
    rom = max(0.0, p95 - p5)
    min_prom = float(min_prominence_ratio) * rom
    top_idx, _ = find_peaks(y, distance=max(1, int(min_distance)), prominence=min_prom)
    bottom_idx, _ = find_peaks(-y, distance=max(1, int(min_distance)), prominence=min_prom)
    return (
        np.asarray(sorted(bottom_idx), dtype=np.int64),
        np.asarray(sorted(top_idx), dtype=np.int64),
    )


def enforce_top_bottom_alternation(
    signal: np.ndarray,
    bottoms: np.ndarray,
    tops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(signal, dtype=np.float64)
    events: list[tuple[str, int, float]] = []
    for idx_raw in bottoms:
        idx = int(idx_raw)
        if 0 <= idx < y.size and np.isfinite(y[idx]):
            events.append(("bottom", idx, float(y[idx])))
    for idx_raw in tops:
        idx = int(idx_raw)
        if 0 <= idx < y.size and np.isfinite(y[idx]):
            events.append(("top", idx, float(y[idx])))
    events.sort(key=lambda e: e[1])
    if not events:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    cleaned: list[tuple[str, int, float]] = [events[0]]
    for kind, idx, val in events[1:]:
        prev_kind, _prev_idx, prev_val = cleaned[-1]
        if kind == prev_kind:
            if kind == "bottom" and val < prev_val:
                cleaned[-1] = (kind, idx, val)
            elif kind == "top" and val > prev_val:
                cleaned[-1] = (kind, idx, val)
        else:
            cleaned.append((kind, idx, val))

    final_bottom = np.asarray(sorted(idx for k, idx, _ in cleaned if k == "bottom"), dtype=np.int64)
    final_top = np.asarray(sorted(idx for k, idx, _ in cleaned if k == "top"), dtype=np.int64)
    return final_bottom, final_top


def merge_micro_oscillations(
    signal: np.ndarray,
    bottoms: np.ndarray,
    tops: np.ndarray,
    *,
    thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(signal, dtype=np.float64)
    events: list[tuple[str, int, float]] = []
    for idx_raw in bottoms:
        idx = int(idx_raw)
        if 0 <= idx < y.size and np.isfinite(y[idx]):
            events.append(("bottom", idx, float(y[idx])))
    for idx_raw in tops:
        idx = int(idx_raw)
        if 0 <= idx < y.size and np.isfinite(y[idx]):
            events.append(("top", idx, float(y[idx])))
    events.sort(key=lambda e: e[1])

    changed = True
    while changed and len(events) >= 2:
        changed = False
        min_amp = float("inf")
        min_pos = -1
        for i in range(len(events) - 1):
            amp = abs(events[i][2] - events[i + 1][2])
            if amp < min_amp:
                min_amp = amp
                min_pos = i
        if min_amp < thr and min_pos >= 0:
            del events[min_pos : min_pos + 2]
            if events:
                re_cleaned: list[tuple[str, int, float]] = [events[0]]
                for kind, idx, val in events[1:]:
                    if kind == re_cleaned[-1][0]:
                        pk, _pi, pv = re_cleaned[-1]
                        if kind == "bottom" and val < pv:
                            re_cleaned[-1] = (kind, idx, val)
                        elif kind == "top" and val > pv:
                            re_cleaned[-1] = (kind, idx, val)
                    else:
                        re_cleaned.append((kind, idx, val))
                events = re_cleaned
            changed = True

    final_bottom = np.asarray(sorted(idx for k, idx, _ in events if k == "bottom"), dtype=np.int64)
    final_top = np.asarray(sorted(idx for k, idx, _ in events if k == "top"), dtype=np.int64)
    return final_bottom, final_top


def add_boundary_top_bottom_events(
    signal: np.ndarray,
    bottoms: np.ndarray,
    tops: np.ndarray,
    *,
    thr: float,
    allow_boundary_events: bool = True,
    boundary_hold_frames: int | None = None,
    boundary_requires_stable_top: bool = False,
    top_band: float | None = None,
    valid_mask: np.ndarray | None = None,
    min_window_valid_ratio: float = 0.85,
    max_window_p95_delta: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(signal, dtype=np.float64)
    if not allow_boundary_events:
        return bottoms, tops
    finite = np.isfinite(y)
    valid = finite if valid_mask is None else (np.asarray(valid_mask, dtype=bool) & finite)
    bottom_list = sorted(int(i) for i in bottoms)
    top_list = sorted(int(i) for i in tops)

    def _stable_top_window(start: int, end: int) -> bool:
        if not boundary_requires_stable_top:
            return True
        if top_band is None or not np.isfinite(top_band):
            return False
        hold = max(1, int(boundary_hold_frames or 1))
        start = max(0, int(start))
        end = min(y.size - 1, int(end))
        if end - start + 1 < hold:
            return False
        for ws in range(start, end - hold + 2):
            we = ws + hold
            window = y[ws:we]
            valid_w = valid[ws:we]
            if window.size < hold:
                continue
            if float(np.mean(valid_w)) < min_window_valid_ratio:
                continue
            valid_vals = window[valid_w]
            if valid_vals.size == 0:
                continue
            top_ratio = float(np.mean(valid_vals >= float(top_band) - 5.0))
            if top_ratio < 0.80:
                continue
            deltas = np.abs(np.diff(valid_vals))
            p95_delta = float(np.percentile(deltas, 95.0)) if deltas.size else 0.0
            if p95_delta <= max_window_p95_delta:
                return True
        return False

    all_events = [(int(i), "bottom") for i in bottoms] + [(int(i), "top") for i in tops]
    all_events.sort()
    if not all_events:
        return bottoms, tops

    first_idx, first_kind = all_events[0]
    last_idx, last_kind = all_events[-1]

    if first_kind == "bottom" and first_idx > 0:
        region = y[:first_idx]
        valid_region = valid[:first_idx]
        if np.any(valid_region) and _stable_top_window(0, first_idx - 1):
            candidate = int(np.nanargmax(region))
            if valid[candidate]:
                amp = float(y[candidate] - y[first_idx])
                if amp > thr:
                    top_list = sorted(set(top_list + [candidate]))
    elif first_kind == "top" and first_idx > 0:
        region = y[:first_idx]
        valid_region = valid[:first_idx]
        if np.any(valid_region):
            candidate = int(np.nanargmin(region))
            if valid[candidate]:
                amp = float(y[first_idx] - y[candidate])
                if amp > thr:
                    bottom_list = sorted(set(bottom_list + [candidate]))

    if last_kind == "bottom" and last_idx < y.size - 1:
        region = y[last_idx + 1 :]
        valid_region = valid[last_idx + 1 :]
        if np.any(valid_region) and _stable_top_window(last_idx + 1, y.size - 1):
            offset = last_idx + 1
            candidate = offset + int(np.nanargmax(region))
            if finite[candidate]:
                amp = float(y[candidate] - y[last_idx])
                if amp > thr:
                    top_list = sorted(set(top_list + [candidate]))
    elif last_kind == "top" and last_idx < y.size - 1:
        region = y[last_idx + 1 :]
        valid_region = valid[last_idx + 1 :]
        if np.any(valid_region):
            offset = last_idx + 1
            candidate = offset + int(np.nanargmin(region))
            if finite[candidate]:
                amp = float(y[last_idx] - y[candidate])
                if amp > thr:
                    bottom_list = sorted(set(bottom_list + [candidate]))

    return np.asarray(bottom_list, dtype=np.int64), np.asarray(top_list, dtype=np.int64)


def collapse_double_bottoms(
    signal: np.ndarray,
    bottoms: np.ndarray,
    tops: np.ndarray,
    *,
    min_separation_frames: int,
    bridge_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(signal, dtype=np.float64)
    min_sep = max(1, int(min_separation_frames))
    bridge_sep = max(min_sep, int(round(float(bridge_factor) * min_sep)))

    events: list[tuple[str, int, float]] = []
    for idx_raw in bottoms:
        idx = int(idx_raw)
        if 0 <= idx < y.size and np.isfinite(y[idx]):
            events.append(("bottom", idx, float(y[idx])))
    for idx_raw in tops:
        idx = int(idx_raw)
        if 0 <= idx < y.size and np.isfinite(y[idx]):
            events.append(("top", idx, float(y[idx])))
    events.sort(key=lambda e: e[1])

    changed = True
    while changed and len(events) >= 3:
        changed = False
        i = 0
        while i < len(events) - 2:
            k0, t0, v0 = events[i]
            k1, t1, _v1 = events[i + 1]
            k2, t2, v2 = events[i + 2]
            if k0 == "bottom" and k1 == "top" and k2 == "bottom":
                direct_close = (t2 - t0) <= min_sep
                bridge_close = min(t1 - t0, t2 - t1) <= bridge_sep
                if direct_close or bridge_close:
                    kept = events[i] if v0 <= v2 else events[i + 2]
                    events[i : i + 3] = [kept]
                    changed = True
                    break
            i += 1

    return enforce_top_bottom_alternation(
        y,
        np.asarray([t for k, t, _ in events if k == "bottom"], dtype=np.int64),
        np.asarray([t for k, t, _ in events if k == "top"], dtype=np.int64),
    )


def center_top_bottom_plateaus(
    signal: np.ndarray,
    bottoms: np.ndarray,
    tops: np.ndarray,
    *,
    rom: float,
    plateau_epsilon_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(signal, dtype=np.float64)
    epsilon = float(plateau_epsilon_ratio) * float(rom)
    bottom_list = sorted(int(i) for i in bottoms)
    top_list = sorted(int(i) for i in tops)

    def _centroid(idx: int, kind: str) -> int:
        opp = bottom_list if kind == "top" else top_list
        left_bound = 0
        for o in opp:
            if o < idx:
                left_bound = o
        right_bound = y.size - 1
        for o in opp:
            if o > idx:
                right_bound = o
                break
        region = y[left_bound : right_bound + 1]
        finite_mask = np.isfinite(region)
        if not np.any(finite_mask):
            return idx
        if kind == "top":
            ref_val = float(np.nanmax(region))
            mask = (region >= ref_val - epsilon) & finite_mask
        else:
            ref_val = float(np.nanmin(region))
            mask = (region <= ref_val + epsilon) & finite_mask
        indices_above = np.where(mask)[0]
        if indices_above.size == 0:
            return idx
        return int(np.round(np.mean(indices_above))) + left_bound

    new_top = np.asarray([_centroid(int(i), "top") for i in tops], dtype=np.int64)
    new_bottom = np.asarray([_centroid(int(i), "bottom") for i in bottoms], dtype=np.int64)
    return new_bottom, new_top


def _nearest(sorted_indices: list[int], target: int, direction: str) -> int | None:
    if direction == "left":
        for v in reversed(sorted_indices):
            if v < target:
                return int(v)
    else:
        for v in sorted_indices:
            if v > target:
                return int(v)
    return None


def _amplitude_label(amplitude: float, rom: float) -> str:
    if amplitude > (0.25 * rom):
        return "VALID"
    if amplitude > (0.10 * rom):
        return "LOW_AMPLITUDE"
    return "NOISE"


def _classify_one_kind(
    y: np.ndarray,
    indices: list[int],
    opposite_indices: list[int],
    kind: str,
    *,
    rom: float,
    thr: float,
) -> tuple[list[int], list[dict]]:
    valid: list[int] = []
    details: list[dict] = []
    for pos, idx in enumerate(indices):
        val = float(y[idx]) if np.isfinite(y[idx]) else float("nan")
        left_opp = _nearest(opposite_indices, idx, "left")
        right_opp = _nearest(opposite_indices, idx, "right")
        if kind == "bottom":
            left_amp = float(y[left_opp] - val) if left_opp is not None and np.isfinite(val) else float("nan")
            right_amp = float(y[right_opp] - val) if right_opp is not None and np.isfinite(val) else float("nan")
        else:
            left_amp = float(val - y[left_opp]) if left_opp is not None and np.isfinite(val) else float("nan")
            right_amp = float(val - y[right_opp]) if right_opp is not None and np.isfinite(val) else float("nan")
        positive_amps = [a for a in [left_amp, right_amp] if np.isfinite(a) and a > 0]
        is_valid = False
        reason = "no_valid_neighbor"
        local_amp = float("nan")
        is_border = pos == 0 or pos == len(indices) - 1
        if kind == "bottom" and len(positive_amps) != 2:
            if len(positive_amps) == 1 and is_border:
                local_amp = positive_amps[0]
                is_valid = local_amp > thr
                reason = "ok" if is_valid else "low_local_prominence"
            else:
                reason = "missing_two_sided_prominence"
        elif len(positive_amps) == 2:
            local_amp = float(min(positive_amps))
            is_valid = all(a > thr for a in positive_amps)
            reason = "ok" if is_valid else "low_local_prominence"
        elif len(positive_amps) == 1 and is_border:
            local_amp = positive_amps[0]
            is_valid = local_amp > thr
            reason = "ok" if is_valid else "low_local_prominence"
        if is_valid:
            valid.append(idx)
        details.append(
            {
                "kind": kind,
                "index": idx,
                "value": val,
                "left_neighbor_index": left_opp,
                "right_neighbor_index": right_opp,
                "left_amplitude": left_amp,
                "right_amplitude": right_amp,
                "local_amplitude": local_amp,
                "is_valid": bool(is_valid),
                "reason": reason,
                "label": _amplitude_label(local_amp, rom) if np.isfinite(local_amp) else "NOISE",
            }
        )
    return valid, details


def classify_top_bottom_events(
    signal: np.ndarray,
    bottoms: np.ndarray,
    tops: np.ndarray,
    *,
    rom: float,
    thr: float,
) -> tuple[list[int], list[dict], list[int], list[dict]]:
    y = np.asarray(signal, dtype=np.float64)
    top_list = sorted(int(i) for i in tops)
    bottom_list = sorted(int(i) for i in bottoms)
    bottom_valid, bottom_details = _classify_one_kind(
        y, bottom_list, top_list, "bottom", rom=rom, thr=thr
    )
    top_valid, top_details = _classify_one_kind(
        y, top_list, bottom_list, "top", rom=rom, thr=thr
    )
    return bottom_valid, bottom_details, top_valid, top_details

