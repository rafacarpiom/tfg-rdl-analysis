
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.biomechanics.normalization import NormalizationConfig, normalize_pose_sequence
from src.biomechanics.rdl.analysis_context import RDL_ANCHOR_NAMES, resolve_rdl_anchor_frames
from src.biomechanics.rdl.detectors.knee_dominant import detect_knee_dominant_error
from src.utils.paths import OUTPUTS, ROOT
from src.utils.paths import rdl_reference_dir

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _ask_mode() -> int:
    print("\n1. Seleccionar video desde data/")
    print("2. Seleccionar bundle existente desde outputs/debug_runs/rdl/")
    print("3. Salir")
    while True:
        try:
            v = int(input("\nSelecciona una opcion: ").strip())
            if v in (1, 2, 3):
                return v
        except ValueError:
            pass
        print("Entrada invalida.")


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "s", "si"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Entrada invalida.")


def _list_videos() -> list[Path]:
    data = ROOT / "data"
    if not data.is_dir():
        return []
    return sorted([p for p in data.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])


def _ask_video() -> Path:
    videos = _list_videos()
    if not videos:
        raw = input("No hay videos en data/. Ruta manual: ").strip()
        p = Path(raw).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for i, path in enumerate(videos, start=1):
        print(f"{i}. {path.as_posix()}")
    while True:
        try:
            idx = int(input("Selecciona numero: ").strip())
            if 1 <= idx <= len(videos):
                return videos[idx - 1]
        except ValueError:
            pass
        print("Entrada invalida.")


def _list_bundles() -> list[Path]:
    root = OUTPUTS / "debug_runs" / "rdl"
    if not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _ask_bundle() -> Path:
    bundles = _list_bundles()
    if not bundles:
        raise FileNotFoundError("No hay bundles en outputs/debug_runs/rdl")
    for i, p in enumerate(bundles, start=1):
        print(f"{i}. {p.as_posix()}")
    while True:
        try:
            idx = int(input("Selecciona bundle: ").strip())
            if 1 <= idx <= len(bundles):
                return bundles[idx - 1]
        except ValueError:
            pass
        print("Entrada invalida.")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalido: {path}")
    return data


def _load_cache_analysis_context(path: Path) -> dict:
    with np.load(str(path), allow_pickle=True) as npz:
        if "analysis_context" not in npz.files:
            raise ValueError("analysis_context no encontrado en cache")
        raw = npz["analysis_context"]
        if raw.size == 0:
            raise ValueError("analysis_context vacio en cache")
        ctx = raw.ravel()[0]
        if not isinstance(ctx, dict):
            raise ValueError("analysis_context invalido en cache")
        return ctx


def _valid_rep_indices(segmentation_result: dict[str, Any]) -> list[int]:
    reps = segmentation_result.get("reps")
    if not isinstance(reps, list):
        return []
    return [i for i, rep in enumerate(reps) if isinstance(rep, dict) and rep.get("anchor_valid", True) is True]


def build_minimal_comparative_context_from_bundle_and_reference(
    bundle_dir: Path,
    reference_dir: Path,
    *,
    ideal_valid_rep_index: int = 1,
) -> dict:
    user_seg = _load_json(bundle_dir / "01_segmentation_result.json")
    with np.load(str(bundle_dir / "07_user_pose_clean.npz"), allow_pickle=True) as npz:
        if "kps_xy_clean" in npz.files:
            user_kps_clean = np.asarray(npz["kps_xy_clean"], dtype=np.float64)
        elif "kps_xy" in npz.files:
            user_kps_clean = np.asarray(npz["kps_xy"], dtype=np.float64)
        else:
            raise ValueError("07_user_pose_clean.npz no contiene kps_xy_clean ni kps_xy")
        if "kps_score_clean" in npz.files:
            user_kps_score = np.asarray(npz["kps_score_clean"], dtype=np.float64)
        elif "kps_score" in npz.files:
            user_kps_score = np.asarray(npz["kps_score"], dtype=np.float64)
        else:
            user_kps_score = np.full((user_kps_clean.shape[0], 17), np.nan, dtype=np.float64)

    norm_cfg = NormalizationConfig(method="pelvis_torso_scale", sequence_scale_mode="fixed_median", apply_rotation=False)
    user_norm = normalize_pose_sequence(user_kps_clean, config=norm_cfg)
    user_kps_normalized = np.asarray(user_norm["kps_xy_normalized"], dtype=np.float64)

    ideal_seg = _load_json(reference_dir / "ideal_segmentation_result.json")
    _ = _load_json(reference_dir / "ideal_pose_sequence_normalized_meta.json")
    with np.load(str(reference_dir / "ideal_pose_sequence_normalized.npz"), allow_pickle=True) as npz:
        if "kps_xy_normalized" not in npz.files:
            raise ValueError("ideal_pose_sequence_normalized.npz sin kps_xy_normalized")
        ideal_kps_normalized = np.asarray(npz["kps_xy_normalized"], dtype=np.float64)

    ideal_valid = _valid_rep_indices(ideal_seg)
    if not ideal_valid:
        raise ValueError("ideal segmentation sin reps validas")
    chosen_pos = min(max(0, ideal_valid_rep_index), len(ideal_valid) - 1)
    ideal_rep_raw_index = int(ideal_valid[chosen_pos])
    ideal_rep = ideal_seg["reps"][ideal_rep_raw_index]
    ideal_resolved = resolve_rdl_anchor_frames(ideal_rep)
    ideal_frames = ideal_resolved.get("frames", {}) if isinstance(ideal_resolved, dict) else {}
    ideal_valid_map = ideal_resolved.get("valid", {}) if isinstance(ideal_resolved, dict) else {}

    user_valid = _valid_rep_indices(user_seg)
    paired_repetitions: list[dict[str, Any]] = []
    warnings: list[str] = []
    user_reps = user_seg.get("reps") if isinstance(user_seg.get("reps"), list) else []
    for rep_order, user_rep_raw_index in enumerate(user_valid, start=1):
        user_rep = user_reps[user_rep_raw_index]
        user_resolved = resolve_rdl_anchor_frames(user_rep)
        user_frames = user_resolved.get("frames", {}) if isinstance(user_resolved, dict) else {}
        user_valid_map = user_resolved.get("valid", {}) if isinstance(user_resolved, dict) else {}
        anchors: dict[str, dict[str, Any]] = {}
        for anchor in RDL_ANCHOR_NAMES:
            uf = user_frames.get(anchor)
            inf = ideal_frames.get(anchor)
            pair_warnings: list[str] = []
            valid = True
            if not bool(user_valid_map.get(anchor, False)) or not isinstance(uf, int):
                valid = False
                pair_warnings.append(f"USER_ANCHOR_INVALID:{anchor}")
            if not bool(ideal_valid_map.get(anchor, False)) or not isinstance(inf, int):
                valid = False
                pair_warnings.append(f"IDEAL_ANCHOR_INVALID:{anchor}")
            if valid and not (0 <= uf < user_kps_normalized.shape[0]):
                valid = False
                pair_warnings.append(f"USER_FRAME_OOB:{anchor}:{uf}")
            if valid and not (0 <= inf < ideal_kps_normalized.shape[0]):
                valid = False
                pair_warnings.append(f"IDEAL_FRAME_OOB:{anchor}:{inf}")
            anchors[anchor] = {
                "anchor": anchor,
                "valid": valid,
                "user_frame": int(uf) if isinstance(uf, int) else None,
                "ideal_frame": int(inf) if isinstance(inf, int) else None,
                "user_kps_clean": np.asarray(user_kps_clean[uf], dtype=np.float64) if valid else None,
                "user_kps_normalized": np.asarray(user_kps_normalized[uf], dtype=np.float64) if valid else None,
                "ideal_kps_normalized": np.asarray(ideal_kps_normalized[inf], dtype=np.float64) if valid else None,
                "warnings": pair_warnings,
            }
            warnings.extend(pair_warnings)

        paired_repetitions.append(
            {
                "user_rep_raw_index": int(user_rep_raw_index),
                "user_rep_order": int(rep_order),
                "ideal_rep_raw_index": int(ideal_rep_raw_index),
                "ideal_valid_rep_index": int(chosen_pos),
                "anchors": anchors,
            }
        )

    return {
        "user": {
            "video_id": bundle_dir.name,
            "pose_clean": {"kps_xy_clean": user_kps_clean, "kps_score_clean": user_kps_score},
            "segmentation_result": user_seg,
        },
        "reference": {
            "reference_name": "PM-Ideal",
            "reference_dir": str(reference_dir),
            "segmentation_result": ideal_seg,
            "ideal_valid_rep_index": int(chosen_pos),
            "ideal_rep_raw_index": int(ideal_rep_raw_index),
        },
        "anchor_pairs": {"anchor_names": list(RDL_ANCHOR_NAMES), "paired_repetitions": paired_repetitions},
        "context_meta": {
            "source": "minimal_comparative_context_from_bundle_and_reference",
            "video_id": bundle_dir.name,
            "user_bundle_dir": str(bundle_dir),
            "reference_dir": str(reference_dir),
            "normalization_source": "computed_from_07_user_pose_clean",
        },
        "warnings": sorted(set(warnings)),
    }


def _run_from_context(ctx: dict, video_id: str) -> None:
    from src.visualization.rdl.detectors import export_knee_dominant_debug

    result = detect_knee_dominant_error(ctx)
    out_dir = OUTPUTS / "debug_detectors" / "rdl" / "repeticiones" / video_id / "knee_dominant"
    export_knee_dominant_debug(analysis_context=ctx, knee_dominant_result=result, output_dir=out_dir)
    print(f"\nDebug knee_dominant generado: {out_dir.as_posix()}")
    print(f"detected={result.get('detected', False)} severity={result.get('severity', 'none')} score={result.get('score', 0.0)}")


def _run_full(video_path: Path) -> None:
    from src.pipeline import run_full_analysis
    from src.visualization.rdl import export_rdl_debug_bundle

    artifacts = run_full_analysis(video_path)
    runtime = artifacts.to_runtime_dict()
    ctx = runtime.get("analysis_context") if isinstance(runtime.get("analysis_context"), dict) else {}
    video_id = str((ctx.get("user") or {}).get("video_id", video_path.stem)) if isinstance(ctx, dict) else video_path.stem
    bundle_dir = OUTPUTS / "debug_runs" / "rdl" / video_id
    export_rdl_debug_bundle(artifacts=artifacts, output_dir=bundle_dir)
    _run_from_context(ctx, video_id)


def _resolve_video_path_from_bundle(bundle_dir: Path) -> Path | None:
    summary_path = bundle_dir / "00_pipeline_summary.json"
    if summary_path.is_file():
        try:
            data = _load_json(summary_path)
            raw = data.get("video_path")
            if isinstance(raw, str) and raw.strip():
                p = Path(raw).expanduser()
                if not p.is_absolute():
                    p = (ROOT / p).resolve()
                if p.is_file():
                    return p
        except Exception:
            pass
    for ext in VIDEO_EXTENSIONS:
        candidate = ROOT / "data" / f"{bundle_dir.name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _run_bundle(bundle_dir: Path) -> None:
    for cache_name in ("08_analysis_context_anchor_cache.npz", "07_analysis_context_anchor_cache.npz"):
        cache = bundle_dir / cache_name
        if cache.is_file():
            ctx = _load_cache_analysis_context(cache)
            print(f"Contexto knee_dominant cargado desde cache: {cache_name}")
            _run_from_context(ctx, bundle_dir.name)
            return

    reference_dir = rdl_reference_dir("PM-Ideal")
    required = [
        bundle_dir / "01_segmentation_result.json",
        bundle_dir / "07_user_pose_clean.npz",
        reference_dir / "ideal_pose_sequence_normalized.npz",
        reference_dir / "ideal_segmentation_result.json",
        reference_dir / "ideal_pose_sequence_normalized_meta.json",
    ]
    missing = [p for p in required if not p.is_file()]
    if not missing:
        try:
            ctx = build_minimal_comparative_context_from_bundle_and_reference(bundle_dir, reference_dir, ideal_valid_rep_index=1)
            print("Contexto knee_dominant reconstruido desde bundle + referencia ideal. No se ejecuta RTMPose.")
            _run_from_context(ctx, bundle_dir.name)
            return
        except Exception as exc:
            print(f"[WARN] No se pudo reconstruir contexto minimo: {exc}")

    if missing:
        print("Faltan archivos para modo rápido knee_dominant:")
        for p in missing:
            print(f"- {p.as_posix()}")
    if not _ask_yes_no("¿Ejecutar pipeline completo ahora para reconstruir contexto?", default=False):
        return
    video_path = _resolve_video_path_from_bundle(bundle_dir)
    if video_path is None:
        print("No se pudo resolver video automáticamente. Usa opción 1.")
        return
    _run_full(video_path)


def main() -> None:
    print("\n=== Debug knee_dominant ===")
    mode = _ask_mode()
    if mode == 3:
        return
    if mode == 1:
        _run_full(_ask_video())
        return
    _run_bundle(_ask_bundle())


if __name__ == "__main__":
    main()
