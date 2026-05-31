
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _scatter_events(ax: plt.Axes, x: np.ndarray, y: np.ndarray, details: list[dict[str, object]], *, kind: str) -> None:
    color_map = {"VALID": "tab:red" if kind == "bottom" else "tab:green", "LOW_AMPLITUDE": "tab:orange", "NOISE": "gray"}
    marker = "v" if kind == "bottom" else "^"
    label_base = "bottom" if kind == "bottom" else "top"
    used_labels: set[str] = set()
    for item in details:
        if str(item.get("kind")) != kind or not bool(item.get("is_valid", False)):
            continue
        idx = int(item["index"])
        if idx < 0 or idx >= x.size or not np.isfinite(y[idx]):
            continue
        label = str(item.get("label", "NOISE"))
        color = color_map.get(label, "gray")
        legend = f"{label_base}_{label.lower()}"
        if legend in used_labels:
            legend = None
        else:
            used_labels.add(legend)
        ax.scatter(x[idx], y[idx], color=color, marker=marker, s=60, label=legend, zorder=4)


def _scatter_candidates(ax: plt.Axes, x: np.ndarray, y: np.ndarray, candidate_indices: list[int], *, kind: str) -> None:
    marker = "v" if kind == "bottom" else "^"
    label = f"{kind}_candidate"
    first = True
    for idx in candidate_indices:
        if idx < 0 or idx >= x.size or not np.isfinite(y[idx]):
            continue
        ax.scatter(x[idx], y[idx], color="lightgray", marker=marker, s=22, alpha=0.6, zorder=2, label=label if first else None)
        first = False


def _nearest_finite_for_plot(signal: np.ndarray, frame: int) -> tuple[int | None, float | None]:
    if signal.size == 0:
        return None, None
    f = int(frame)
    if 0 <= f < signal.size and np.isfinite(signal[f]):
        return f, float(signal[f])
    finite_idx = np.flatnonzero(np.isfinite(signal))
    if finite_idx.size == 0:
        return None, None
    distances = np.abs(finite_idx - f)
    min_dist = int(np.min(distances))
    tied = finite_idx[distances == min_dist]
    selected = int(np.min(tied))
    return selected, float(signal[selected])


def _iter_anchor_points(rep: dict, phase_key: str, signal: np.ndarray) -> list[tuple[int, int, float, bool]]:
    phase = dict(rep.get("anchor_details", {}).get(phase_key, {}))
    anchors = dict(phase.get("anchors", {}))
    if not anchors:
        prefix = "ecc_" if phase_key == "eccentric" else "con_"
        legacy = {}
        for pct in (0, 25, 50, 75, 100):
            key = f"{prefix}{pct}"
            value = rep.get("anchors", {}).get(key)
            if value is None:
                continue
            legacy[str(pct)] = {"frame": int(value), "pct": int(pct), "valid": True, "signal": None}
        anchors = legacy
    out: list[tuple[int, int, float, bool]] = []
    for pct_key, item in sorted(anchors.items(), key=lambda kv: int(kv[0])):
        if not bool(item.get("valid", False)):
            continue
        frame = item.get("frame")
        if frame is None:
            continue
        frame_int = int(frame)
        y_val = item.get("signal")
        relocated = False
        if y_val is None or not np.isfinite(float(y_val)):
            if 0 <= frame_int < signal.size and np.isfinite(signal[frame_int]):
                y_val = float(signal[frame_int])
            else:
                nearest_frame, nearest_signal = _nearest_finite_for_plot(signal, frame_int)
                if nearest_frame is None or nearest_signal is None:
                    continue
                frame_int = nearest_frame
                y_val = nearest_signal
                relocated = True
        out.append((int(pct_key), frame_int, float(y_val), relocated))
    return out


def _draw_phase_spans(ax: plt.Axes, reps: list[dict], signal: np.ndarray) -> None:
    ecc_drawn = False
    con_drawn = False
    bottom_drawn = False
    for rep in reps:
        ecc_label = "eccentric" if not ecc_drawn else None
        con_label = "concentric" if not con_drawn else None
        bottom_label = "bottom" if not bottom_drawn else None
        es, ee = rep["ecc_start"], rep["ecc_end"]
        cs, ce = rep["con_start"], rep["con_end"]
        bottom_frame = int(rep.get("bottom", ee))
        warn = str(rep.get("validation_label")) == "VALID_WITH_WARNINGS"
        edge_color = "tab:orange" if warn else "none"
        ax.axvspan(es, ee, color="tab:red", alpha=0.10, zorder=0, label=ecc_label)
        ax.axvspan(cs, ce, color="tab:green", alpha=0.10, zorder=0, label=con_label)
        if warn:
            ax.axvspan(rep["top_start"], rep["top_end"], facecolor="none", edgecolor=edge_color, linewidth=1.0, linestyle="--", zorder=2, label="valid_with_warnings" if not ecc_drawn else None)
        for key in ("ecc_start", "ecc_end", "con_start", "con_end"):
            ax.axvline(rep[key], color="gray", linewidth=0.6, linestyle=":", alpha=0.5, zorder=1)
        for pct, frame, y_pos, relocated in _iter_anchor_points(rep, "eccentric", signal):
            ax.axvline(frame, color="tab:red", linewidth=0.5, linestyle="--", alpha=0.35, zorder=1)
            ax.annotate(f"{pct}%{'*' if relocated else ''}\nt={frame}", xy=(frame, y_pos), fontsize=5.0, color="tab:red", alpha=0.75, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
        for pct, frame, y_pos, relocated in _iter_anchor_points(rep, "concentric", signal):
            ax.axvline(frame, color="tab:green", linewidth=0.5, linestyle="--", alpha=0.35, zorder=1)
            ax.annotate(f"{pct}%{'*' if relocated else ''}\nt={frame}", xy=(frame, y_pos), fontsize=5.0, color="tab:green", alpha=0.75, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
        if 0 <= bottom_frame < signal.size:
            ax.axvline(bottom_frame, color="tab:purple", linewidth=0.9, linestyle="-", alpha=0.55, zorder=1, label=bottom_label)
        ecc_drawn = con_drawn = bottom_drawn = True


def _draw_discarded_candidates(ax: plt.Axes, candidates: list[dict], signal: np.ndarray) -> None:
    color_by_label = {"PARTIAL_START": "tab:orange", "PARTIAL_END": "tab:orange", "POST_EXERCISE_NOISE": "tab:brown", "UNSTABLE_SIGNAL": "tab:red", "OUTSIDE_CONSISTENT_BLOCK": "gray", "WRONG_MOVEMENT_OR_NOT_RDL": "tab:purple", "INVALID_CANDIDATE": "gray"}
    used_labels: set[str] = set()
    for cand in candidates:
        if str(cand.get("label")) in {"VALID", "VALID_WITH_WARNINGS"}:
            continue
        try:
            start = int(cand["top_start"]); end = int(cand["top_end"])
        except Exception:
            continue
        label = str(cand.get("label", "INVALID_CANDIDATE"))
        color = color_by_label.get(label, "gray")
        legend = label if label not in used_labels else None
        used_labels.add(label)
        ax.axvspan(start, end, color=color, alpha=0.07, zorder=0, label=legend)
        ax.axvline(start, color=color, linewidth=0.6, linestyle="--", alpha=0.35, zorder=1)
        ax.axvline(end, color=color, linewidth=0.6, linestyle="--", alpha=0.35, zorder=1)


def plot_rdl_segmentation_debug(result: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.asarray(result["signals"]["signal_smooth"], dtype=np.float64)
    x = np.arange(signal.size, dtype=np.int64)
    events = result["event_candidates"]
    bottom_details = list(events.get("bottom_details", []))
    top_details = list(events.get("top_details", []))
    bottom_candidates = list(events.get("bottom_indices_candidates", []))
    top_candidates = list(events.get("top_indices_candidates", []))
    reps = list(result.get("reps", []))
    rep_candidates = list(result.get("rep_candidates", []))
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    if rep_candidates:
        _draw_discarded_candidates(ax, rep_candidates, signal)
    if reps:
        _draw_phase_spans(ax, reps, signal)
    ax.plot(x, signal, color="tab:blue", linewidth=1.8, label="signal_smooth")
    _scatter_candidates(ax, x, signal, bottom_candidates, kind="bottom")
    _scatter_candidates(ax, x, signal, top_candidates, kind="top")
    _scatter_events(ax, x, signal, bottom_details, kind="bottom")
    _scatter_events(ax, x, signal, top_details, kind="top")
    status = result.get("segmentation_status", "OK")
    ax.set_title(f"RDL segmentation debug - {result['video_id']} - {status}")
    ax.set_xlabel("Frame t"); ax.set_ylabel("Signal angle (deg)"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def export_segmentation_debug(*, segmentation_result: dict, output_path: str | Path) -> None:
    plot_rdl_segmentation_debug(segmentation_result, Path(output_path))
