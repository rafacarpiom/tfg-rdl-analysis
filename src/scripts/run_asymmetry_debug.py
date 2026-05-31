
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.biomechanics.rdl.detectors.asymmetry import detect_asymmetry
from src.utils.paths import OUTPUTS, ROOT

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _ask_mode() -> int:
    print("\n1. Seleccionar video desde data/")
    print("2. Seleccionar bundle existente desde outputs/debug_runs/rdl/")
    print("3. Salir")
    while True:
        try:
            value = int(input("\nSelecciona una opcion: ").strip())
            if value in (1, 2, 3):
                return value
        except ValueError:
            pass
        print("Entrada invalida.")


def _ask_yes_no(question: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = input(question + suffix).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "s", "si"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Entrada invalida.")


def _list_videos(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        return []
    return sorted([p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])


def _ask_video_path() -> Path:
    videos = _list_videos(ROOT / "data")
    if not videos:
        raw = input("No se encontraron videos en data/. Ruta manual: ").strip()
        p = Path(raw).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Ruta inexistente: {p}")
        return p
    for idx, path in enumerate(videos, start=1):
        print(f"{idx}. {path.as_posix()}")
    while True:
        try:
            choice = int(input("Selecciona numero: ").strip())
            if 1 <= choice <= len(videos):
                return videos[choice - 1]
        except ValueError:
            pass
        print("Entrada invalida.")


def _list_bundles() -> list[Path]:
    root = OUTPUTS / "debug_runs" / "rdl"
    if not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _ask_bundle_dir() -> Path:
    bundles = _list_bundles()
    if not bundles:
        raise FileNotFoundError("No hay bundles en outputs/debug_runs/rdl")
    for idx, path in enumerate(bundles, start=1):
        print(f"{idx}. {path.as_posix()}")
    while True:
        try:
            choice = int(input("Selecciona numero de bundle: ").strip())
            if 1 <= choice <= len(bundles):
                return bundles[choice - 1]
        except ValueError:
            pass
        print("Entrada invalida.")


def build_minimal_asymmetry_context_from_bundle(bundle_dir: Path) -> dict:
    seg_path = bundle_dir / "01_segmentation_result.json"
    clean_npz = bundle_dir / "07_user_pose_clean.npz"
    with seg_path.open("r", encoding="utf-8") as f:
        segmentation_result = json.load(f)
    with np.load(str(clean_npz), allow_pickle=True) as npz:
        kps_xy_clean = np.asarray(npz["kps_xy_clean"] if "kps_xy_clean" in npz.files else npz["kps_xy"], dtype=np.float64)
        kps_score_clean = np.asarray(npz["kps_score_clean"] if "kps_score_clean" in npz.files else npz["kps_score"], dtype=np.float64)
        kps_xy = np.asarray(npz["kps_xy"], dtype=np.float64) if "kps_xy" in npz.files else None
        kps_score = np.asarray(npz["kps_score"], dtype=np.float64) if "kps_score" in npz.files else None
    pose_clean = {"kps_xy_clean": kps_xy_clean, "kps_score_clean": kps_score_clean}
    if kps_xy is not None:
        pose_clean["kps_xy"] = kps_xy
    if kps_score is not None:
        pose_clean["kps_score"] = kps_score
    return {
        "user": {
            "video_id": bundle_dir.name,
            "pose_clean": pose_clean,
            "segmentation_result": segmentation_result,
        },
        "context_meta": {"source": "debug_bundle_clean_npz", "video_id": bundle_dir.name},
        "warnings": [],
    }


def _run_from_context(ctx: dict, video_id: str) -> None:
    from src.visualization.rdl.detectors import export_asymmetry_debug

    result = detect_asymmetry(ctx)
    out_dir = OUTPUTS / "debug_detectors" / "rdl" / "repeticiones" / video_id / "asymmetry"
    export_asymmetry_debug(analysis_context=ctx, asymmetry_result=result, output_dir=out_dir)
    print(f"\nDebug asymmetry generado en: {out_dir.as_posix()}")
    print(f"detected={result.get('detected', False)} severity={result.get('severity', 'none')} score={result.get('score', 0.0)}")


def _run_from_video(video_path: Path) -> None:
    from src.pipeline import run_full_analysis

    artifacts = run_full_analysis(video_path)
    runtime = artifacts.to_runtime_dict()
    analysis_context = runtime.get("analysis_context", {}) if isinstance(runtime, dict) else {}
    video_id = analysis_context.get("user", {}).get("video_id", video_path.stem) if isinstance(analysis_context, dict) else video_path.stem
    _run_from_context(analysis_context, str(video_id))


def _run_from_bundle(bundle_dir: Path) -> None:
    seg_path = bundle_dir / "01_segmentation_result.json"
    clean_npz = bundle_dir / "07_user_pose_clean.npz"
    if not seg_path.is_file():
        print(f"[ERROR] Falta {seg_path.name} en {bundle_dir.as_posix()}")
        return
    if not clean_npz.is_file():
        print("[WARN] Bundle antiguo sin 07_user_pose_clean.npz")
        if not _ask_yes_no("¿Regenerar ejecutando pipeline completo?", default=False):
            return
        video_guess = ROOT / "data" / f"{bundle_dir.name}.mp4"
        if not video_guess.is_file():
            print("No se pudo resolver video automáticamente. Usa modo 1.")
            return
        _run_from_video(video_guess)
        return
    ctx = build_minimal_asymmetry_context_from_bundle(bundle_dir)
    _run_from_context(ctx, bundle_dir.name)


def main() -> None:
    print("\n=== Debug asymmetry ===")
    mode = _ask_mode()
    if mode == 3:
        return
    if mode == 1:
        video = _ask_video_path()
        _run_from_video(video)
        return
    bundle = _ask_bundle_dir()
    _run_from_bundle(bundle)


if __name__ == "__main__":
    main()
