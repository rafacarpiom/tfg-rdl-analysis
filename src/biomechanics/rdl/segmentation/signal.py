
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(slots=True)
class RDLSignalBundle:
    signal_raw: np.ndarray
    signal_smooth: np.ndarray
    valid_mask_raw: np.ndarray
    interp_mask: np.ndarray
    valid_mask: np.ndarray
    rom: float
    p5: float
    p95: float
    savgol_fallback_used: bool


def _compute_angle_signal(
    kps_xy: np.ndarray,
    kps_score: np.ndarray,
    *,
    kp_a: int,
    kp_b: int,
    kp_c: int,
    thr_conf: float,
) -> tuple[np.ndarray, np.ndarray]:
    a = kps_xy[:, kp_a, :]
    b = kps_xy[:, kp_b, :]
    c = kps_xy[:, kp_c, :]
    sa = kps_score[:, kp_a]
    sb = kps_score[:, kp_b]
    sc = kps_score[:, kp_c]

    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(c).all(axis=1)
    conf_ok = (sa >= thr_conf) & (sb >= thr_conf) & (sc >= thr_conf)
    valid_raw = finite & conf_ok

    signal_raw = np.full(kps_xy.shape[0], np.nan, dtype=np.float64)
    if not np.any(valid_raw):
        return signal_raw, valid_raw

    v1 = a[valid_raw] - b[valid_raw]
    v2 = c[valid_raw] - b[valid_raw]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    good = denom > 1e-8
    cosang = np.full(denom.shape, np.nan, dtype=np.float64)
    cosang[good] = np.sum(v1[good] * v2[good], axis=1) / denom[good]
    cosang = np.clip(cosang, -1.0, 1.0)
    signal_raw[valid_raw] = np.degrees(np.arccos(cosang))
    return signal_raw, valid_raw


def interpolate_short_gaps(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_interp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interp_signal = np.asarray(signal, dtype=np.float64).copy()
    valid = np.asarray(valid_mask, dtype=bool)
    interp_mask = np.zeros(interp_signal.shape[0], dtype=bool)
    n = interp_signal.shape[0]
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        start = i
        while i < n and not valid[i]:
            i += 1
        end = i - 1
        length = end - start + 1
        left = start - 1
        right = end + 1
        if length <= max_gap_interp and left >= 0 and right < n and valid[left] and valid[right]:
            interp_signal[start : end + 1] = np.linspace(
                interp_signal[left],
                interp_signal[right],
                num=length + 2,
            )[1:-1]
            interp_mask[start : end + 1] = True
    valid_after = valid | interp_mask
    return interp_signal, interp_mask, valid_after


def _moving_average(seq: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return seq.copy()
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(seq, kernel, mode="same")


def smooth_signal_savgol(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    window_length: int,
    polyorder: int,
) -> tuple[np.ndarray, bool]:
    smooth = np.full(signal.shape[0], np.nan, dtype=np.float64)
    idx = np.where(valid_mask)[0]
    if idx.size == 0:
        return smooth, False
    seq = signal[idx]
    max_odd = seq.size if seq.size % 2 == 1 else seq.size - 1
    wl = min(window_length, max_odd)
    po = min(polyorder, max(0, wl - 1))
    fallback_used = False
    try:
        if wl < 3 or po < 1:
            raise ValueError("Savitzky-Golay not applicable for sequence length.")
        smooth[idx] = savgol_filter(seq, window_length=wl, polyorder=po, mode="interp")
        return smooth, fallback_used
    except Exception:
        fallback_used = True
        max_ma_odd = seq.size if seq.size % 2 == 1 else seq.size - 1
        ma_win = min(5, max_ma_odd) if max_ma_odd >= 1 else 1
        smooth[idx] = _moving_average(seq, ma_win)
        return smooth, fallback_used


def build_rdl_signal(
    kps_xy: np.ndarray,
    kps_score: np.ndarray,
    *,
    thr_conf: float = 0.30,
    max_gap_interp: int = 5,
    savgol_window_length: int = 11,
    savgol_polyorder: int = 2,
) -> RDLSignalBundle:
    signal_raw, valid_raw = _compute_angle_signal(
        kps_xy,
        kps_score,
        kp_a=6,
        kp_b=12,
        kp_c=14,
        thr_conf=thr_conf,
    )
    interp_signal, interp_mask, valid_after = interpolate_short_gaps(signal_raw, valid_raw, max_gap_interp)
    signal_smooth, savgol_fallback_used = smooth_signal_savgol(
        interp_signal,
        valid_after,
        savgol_window_length,
        savgol_polyorder,
    )
    finite = valid_after & np.isfinite(signal_smooth)
    valid_vals = signal_smooth[finite]
    if valid_vals.size > 0:
        p5 = float(np.percentile(valid_vals, 5.0))
        p95 = float(np.percentile(valid_vals, 95.0))
        rom = float(max(0.0, p95 - p5))
    else:
        p5 = float("nan")
        p95 = float("nan")
        rom = 0.0
    return RDLSignalBundle(
        signal_raw=signal_raw,
        signal_smooth=signal_smooth,
        valid_mask_raw=valid_raw,
        interp_mask=interp_mask,
        valid_mask=valid_after,
        rom=rom,
        p5=p5,
        p95=p95,
        savgol_fallback_used=bool(savgol_fallback_used),
    )

