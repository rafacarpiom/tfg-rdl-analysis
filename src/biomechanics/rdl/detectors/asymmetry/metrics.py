
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Pares articulares bilaterales (COCO-17): articulación → (índice_L, índice_R) ──────────────

BILATERAL_PAIRS: dict[str, tuple[int, int]] = {
    "shoulder": (5, 6),
    "elbow":    (7, 8),
    "wrist":    (9, 10),
    "hip":      (11, 12),
    "knee":     (13, 14),
    "ankle":    (15, 16),
}

# Roles en vista sagital:
#   primary   → clasificación de severidad
#   secondary → refuerza confianza si co-asimétrico con su primary,
#               no dispara severidad solo.
# Hombros/caderas excluidos: en lateral el X refleja ancho/perspectiva, no asimetría útil.
ARM_PRIMARY_JOINTS:   tuple[str, ...] = ("wrist",)
ARM_SECONDARY_JOINTS: tuple[str, ...] = ("elbow",)
ARM_JOINTS:           tuple[str, ...] = ARM_PRIMARY_JOINTS + ARM_SECONDARY_JOINTS

LEG_PRIMARY_JOINTS:   tuple[str, ...] = ("ankle",)
LEG_SECONDARY_JOINTS: tuple[str, ...] = ("knee",)
LEG_JOINTS:           tuple[str, ...] = LEG_PRIMARY_JOINTS + LEG_SECONDARY_JOINTS

# Segmento de referencia para escala corporal: R_shoulder(6) → R_hip(12).
# Solo lado derecho en lateral: hombros/caderas proyectan en la misma banda X; longitud vertical estable.
_SCALE_REF = (6, 12)

ALL_BILATERAL_IDX: frozenset[int] = frozenset(
    idx for pair in BILATERAL_PAIRS.values() for idx in pair
)


# ── Clase de datos ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JointAsymmetry:

    joint: str
    diff_x: float           # L.x − R.x  (px con signo)
    diff_y: float           # L.y − R.y (px; referencia, no clasifica)
    signed_diff: float      # L.x − R.x (alias semántico de diff_x)
    forward_diff: float     # abs(signed_diff) en px
    forward_diff_norm: float  # forward_diff / body_scale (métrica principal)
    confident: bool         # True si L y R tienen score ≥ thr_conf
    body_scale: float       # longitud R_shoulder→R_hip en px; NaN si degenera


# ── Auxiliares ───────────────────────────────────────────────────────────────────

def body_scale(kps: np.ndarray) -> float:
    i, j = _SCALE_REF
    v = kps[j].astype(float) - kps[i].astype(float)
    n = float(np.linalg.norm(v))
    return n if n > 1e-6 else float("nan")


# ── API pública ────────────────────────────────────────────────────────────────

def frame_asymmetry(
    kps: np.ndarray,      # (17, 2)
    scores: np.ndarray,   # (17,)
    *,
    thr_conf: float = 0.3,
) -> dict[str, JointAsymmetry]:
    if kps.shape != (17, 2):
        raise ValueError(f"Expected kps shape (17, 2), got {kps.shape}.")
    scores_1d = np.asarray(scores, dtype=float).ravel()
    if scores_1d.shape[0] != 17:
        raise ValueError(f"Expected 17 scores, got {scores_1d.shape[0]}.")

    scale = body_scale(kps)
    result: dict[str, JointAsymmetry] = {}

    for joint, (li, ri) in BILATERAL_PAIRS.items():
        lp = kps[li].astype(float)
        rp = kps[ri].astype(float)
        if not (np.isfinite(lp).all() and np.isfinite(rp).all()):
            continue

        dx = float(lp[0] - rp[0])
        dy = float(lp[1] - rp[1])
        fwd = abs(dx)
        fwd_norm = fwd / scale if np.isfinite(scale) else float("nan")
        confident = bool(scores_1d[li] >= thr_conf and scores_1d[ri] >= thr_conf)

        result[joint] = JointAsymmetry(
            joint=joint,
            diff_x=dx,
            diff_y=dy,
            signed_diff=dx,
            forward_diff=fwd,
            forward_diff_norm=fwd_norm,
            confident=confident,
            body_scale=scale,
        )

    return result
