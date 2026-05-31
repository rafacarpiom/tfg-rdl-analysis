
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.biomechanics.normalization import NormalizationConfig, normalize_pose_sequence
from src.biomechanics.rdl.analysis_context import RDL_ANCHOR_NAMES, resolve_rdl_anchor_frames
from src.biomechanics.rdl.detectors.bar_far import detect_bar_far
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
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for i, p in enumerate(videos, start=1):
        print(f"{i}. {p.as_posix()}")
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


def _load_analysis_context_from_anchor_cache_file(cache_path: Path) -> dict:
    with np.load(str(cache_path), allow_pickle=True) as npz:
        if "analysis_context" not in npz.files:
            raise ValueError("analysis_context no encontrado en cache")
        raw = npz["analysis_context"]
        if raw.size == 0:
            raise ValueError("analysis_context vacio en cache")
        ctx = raw.ravel()[0]
        if not isinstance(ctx, dict):
            raise ValueError("analysis_context invalido en cache")
        return ctx


def _load_analysis_context_from_anchor_cache(bundle_dir: Path) -> dict:
    cache = bundle_dir / "07_analysis_context_anchor_cache.npz"
    return _load_analysis_context_from_anchor_cache_file(cache)


def _resolve_video_path_from_bundle(bundle_dir: Path) -> Path | None:
    summary_path = bundle_dir / "00_pipeline_summary.json"
    if summary_path.is_file():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalido en {path.as_posix()}")
    return data


def _get_pose_arrays_from_clean_npz(clean_npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(clean_npz_path), allow_pickle=True) as npz:
        if "kps_xy_clean" in npz.files:
            kps_xy = np.asarray(npz["kps_xy_clean"], dtype=np.float64)
        elif "kps_xy" in npz.files:
            kps_xy = np.asarray(npz["kps_xy"], dtype=np.float64)
        else:
            raise ValueError(f"{clean_npz_path.name} no contiene kps_xy_clean ni kps_xy")

        if "kps_score_clean" in npz.files:
            kps_score = np.asarray(npz["kps_score_clean"], dtype=np.float64)
        elif "kps_score" in npz.files:
            kps_score = np.asarray(npz["kps_score"], dtype=np.float64)
        else:
            raise ValueError(f"{clean_npz_path.name} no contiene kps_score_clean ni kps_score")
    if kps_xy.ndim != 3 or kps_xy.shape[1:] != (17, 2):
        raise ValueError(f"kps_xy invalido: {kps_xy.shape}")
    if kps_score.ndim != 2 or kps_score.shape[0] != kps_xy.shape[0]:
        raise ValueError(f"kps_score invalido: {kps_score.shape}")
    return kps_xy, kps_score


def _valid_rep_indices(segmentation_result: dict[str, Any]) -> list[int]:
    reps = segmentation_result.get("reps")
    if not isinstance(reps, list):
        return []
    out: list[int] = []
    for idx, rep in enumerate(reps):
        if isinstance(rep, dict) and rep.get("anchor_valid", True) is True:
            out.append(idx)
    return out


def build_minimal_bar_far_context_from_bundle_and_reference(
    bundle_dir: Path,
    reference_dir: Path,
    *,
    ideal_valid_rep_index: int = 1,
) -> dict:
    user_seg_path = bundle_dir / "01_segmentation_result.json"
    user_clean_npz_path = bundle_dir / "07_user_pose_clean.npz"
    ideal_npz_path = reference_dir / "ideal_pose_sequence_normalized.npz"
    ideal_seg_path = reference_dir / "ideal_segmentation_result.json"
    ideal_meta_path = reference_dir / "ideal_pose_sequence_normalized_meta.json"

    user_segmentation = _load_json(user_seg_path)
    kps_xy_clean, kps_score_clean = _get_pose_arrays_from_clean_npz(user_clean_npz_path)
    norm_cfg = NormalizationConfig(method="pelvis_torso_scale", sequence_scale_mode="fixed_median", apply_rotation=False)
    user_norm = normalize_pose_sequence(kps_xy_clean, config=norm_cfg)
    user_kps_normalized = np.asarray(user_norm.get("kps_xy_normalized"), dtype=np.float64)

    with np.load(str(ideal_npz_path), allow_pickle=True) as ideal_npz:
        if "kps_xy_normalized" not in ideal_npz.files:
            raise ValueError(f"{ideal_npz_path.name} no contiene kps_xy_normalized")
        ideal_kps_normalized = np.asarray(ideal_npz["kps_xy_normalized"], dtype=np.float64)
    ideal_segmentation = _load_json(ideal_seg_path)
    _ = _load_json(ideal_meta_path)

    ideal_valid = _valid_rep_indices(ideal_segmentation)
    if not ideal_valid:
        raise ValueError("Referencia ideal sin reps validas")
    chosen_valid_pos = min(max(0, ideal_valid_rep_index), len(ideal_valid) - 1)
    ideal_rep_raw_index = int(ideal_valid[chosen_valid_pos])
    ideal_reps = ideal_segmentation.get("reps") if isinstance(ideal_segmentation.get("reps"), list) else []
    ideal_rep = ideal_reps[ideal_rep_raw_index]
    ideal_resolved = resolve_rdl_anchor_frames(ideal_rep)
    ideal_frames = ideal_resolved.get("frames", {}) if isinstance(ideal_resolved, dict) else {}
    ideal_valid_map = ideal_resolved.get("valid", {}) if isinstance(ideal_resolved, dict) else {}

    user_reps = user_segmentation.get("reps") if isinstance(user_segmentation.get("reps"), list) else []
    user_valid = _valid_rep_indices(user_segmentation)
    paired_repetitions: list[dict[str, Any]] = []
    ctx_warnings: list[str] = []

    for rep_order, user_rep_raw_index in enumerate(user_valid, start=1):
        user_rep = user_reps[user_rep_raw_index]
        user_resolved = resolve_rdl_anchor_frames(user_rep)
        user_frames = user_resolved.get("frames", {}) if isinstance(user_resolved, dict) else {}
        user_valid_map = user_resolved.get("valid", {}) if isinstance(user_resolved, dict) else {}
        anchors: dict[str, dict[str, Any]] = {}
        for anchor in RDL_ANCHOR_NAMES:
            u_ok = bool(user_valid_map.get(anchor, False))
            i_ok = bool(ideal_valid_map.get(anchor, False))
            uf = user_frames.get(anchor)
            inf = ideal_frames.get(anchor)
            pair_warnings: list[str] = []
            valid = True
            if not u_ok or not isinstance(uf, int):
                valid = False
                pair_warnings.append(f"USER_ANCHOR_INVALID:{anchor}")
            if not i_ok or not isinstance(inf, int):
                valid = False
                pair_warnings.append(f"IDEAL_ANCHOR_INVALID:{anchor}")
            if valid and not (0 <= uf < kps_xy_clean.shape[0]):
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
                "user_kps_clean": np.asarray(kps_xy_clean[uf], dtype=np.float64) if valid else None,
                "user_kps_normalized": np.asarray(user_kps_normalized[uf], dtype=np.float64) if valid else None,
                "ideal_kps_normalized": np.asarray(ideal_kps_normalized[inf], dtype=np.float64) if valid else None,
                "warnings": pair_warnings,
            }
            ctx_warnings.extend(pair_warnings)

        paired_repetitions.append(
            {
                "user_rep_raw_index": int(user_rep_raw_index),
                "user_rep_order": int(rep_order),
                "ideal_rep_raw_index": int(ideal_rep_raw_index),
                "ideal_valid_rep_index": int(chosen_valid_pos),
                "anchors": anchors,
            }
        )

    return {
        "user": {
            "video_id": bundle_dir.name,
            "pose_clean": {
                "kps_xy_clean": kps_xy_clean,
                "kps_score_clean": kps_score_clean,
            },
            "segmentation_result": user_segmentation,
        },
        "reference": {
            "reference_name": "PM-Ideal",
            "reference_dir": str(reference_dir),
            "ideal_valid_rep_index": int(chosen_valid_pos),
            "ideal_rep_raw_index": int(ideal_rep_raw_index),
        },
        "anchor_pairs": {
            "anchor_names": list(RDL_ANCHOR_NAMES),
            "paired_repetitions": paired_repetitions,
        },
        "context_meta": {
            "source": "minimal_bar_far_context_from_bundle_and_reference",
            "video_id": bundle_dir.name,
            "user_bundle_dir": str(bundle_dir),
            "reference_dir": str(reference_dir),
            "normalization_source": "computed_from_07_user_pose_clean",
        },
        "warnings": sorted(set(ctx_warnings)),
    }


def _run_from_context(ctx: dict, video_id: str) -> None:
    from src.visualization.rdl.detectors import export_bar_far_debug

    result = detect_bar_far(ctx)
    out_dir = OUTPUTS / "debug_detectors" / "rdl" / "repeticiones" / video_id / "bar_far"
    export_bar_far_debug(analysis_context=ctx, bar_far_result=result, output_dir=out_dir)
    print(f"\nDebug bar_far generado: {out_dir.as_posix()}")
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


def _run_bundle(bundle_dir: Path) -> None:
    cache08 = bundle_dir / "08_analysis_context_anchor_cache.npz"
    cache07 = bundle_dir / "07_analysis_context_anchor_cache.npz"
    for cache_file in (cache08, cache07):
        if cache_file.is_file():
            ctx = _load_analysis_context_from_anchor_cache_file(cache_file)
            print(f"Contexto bar_far cargado desde cache: {cache_file.name}")
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
            ctx = build_minimal_bar_far_context_from_bundle_and_reference(bundle_dir, reference_dir, ideal_valid_rep_index=1)
            print("Contexto bar_far reconstruido desde bundle + referencia ideal. No se ejecuta RTMPose.")
            _run_from_context(ctx, bundle_dir.name)
            return
        except Exception as exc:
            print(f"[WARN] No se pudo reconstruir contexto minimo: {exc}")

    if missing:
        print("Faltan archivos para modo rápido bar_far:")
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
    print("\n=== Debug bar_far ===")
    mode = _ask_mode()
    if mode == 3:
        return
    if mode == 1:
        _run_full(_ask_video())
        return
    _run_bundle(_ask_bundle())


if __name__ == "__main__":
    main()
