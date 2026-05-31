
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.biomechanics.rdl.analysis_context import resolve_rdl_anchor_frames
from src.biomechanics.rdl.detectors.arms import detect_bent_arms
from src.utils.paths import OUTPUTS, ROOT

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
_REQUIRED_GENERAL_BUNDLE_FILES = (
    "00_pipeline_summary.json",
    "01_segmentation_result.json",
    "02_segmentation_debug.png",
    "03_normalization_summary.json",
    "04_normalization_debug.png",
    "05_analysis_context_summary.json",
    "06_analysis_context_debug.png",
)


def _ask_mode() -> int:
    print("\n1. Seleccionar video desde data/")
    print("2. Seleccionar bundle existente desde outputs/debug_runs/rdl/")
    print("3. Salir")
    while True:
        raw = input("\nSelecciona una opcion: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Entrada invalida. Introduce un numero.")
            continue
        if choice in (1, 2, 3):
            return choice
        print("Numero fuera de rango (1..3).")


def _list_videos(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        return []
    videos = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def _ask_video_path() -> Path:
    videos = _list_videos(Path("data"))
    if not videos:
        raw = input("\nNo se encontraron videos en data/. Introduce ruta manual: ").strip()
        video_path = Path(raw).expanduser()
        if not video_path.is_file():
            raise FileNotFoundError(f"Ruta de video inexistente: {video_path}")
        return video_path

    print("\nVideos disponibles (data/):")
    for idx, path in enumerate(videos, start=1):
        print(f"{idx}. {path.as_posix()}")

    while True:
        raw = input("\nSelecciona numero de video: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Entrada invalida. Introduce un numero.")
            continue
        if 1 <= choice <= len(videos):
            return videos[choice - 1]
        print(f"Numero fuera de rango (1..{len(videos)}).")


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
        print("Entrada invalida. Responde y/n.")


def _get_video_id(video_path: Path, runtime: dict | None = None) -> str:
    if not isinstance(runtime, dict):
        return video_path.stem
    analysis_context = runtime.get("analysis_context") if isinstance(runtime.get("analysis_context"), dict) else {}
    ctx_user = analysis_context.get("user") if isinstance(analysis_context, dict) else {}
    segmentation = runtime.get("segmentation_result") if isinstance(runtime.get("segmentation_result"), dict) else {}
    if isinstance(ctx_user, dict) and ctx_user.get("video_id"):
        return str(ctx_user.get("video_id"))
    if isinstance(segmentation, dict) and segmentation.get("video_id"):
        return str(segmentation.get("video_id"))
    return video_path.stem


def _missing_general_bundle_files(bundle_dir: Path) -> list[str]:
    if not bundle_dir.is_dir():
        return list(_REQUIRED_GENERAL_BUNDLE_FILES)
    missing: list[str] = []
    for name in _REQUIRED_GENERAL_BUNDLE_FILES:
        if not (bundle_dir / name).exists():
            missing.append(name)
    return missing


def _has_general_debug_bundle(bundle_dir: Path) -> bool:
    return len(_missing_general_bundle_files(bundle_dir)) == 0


def _list_debug_bundles(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    candidates = [p for p in root.iterdir() if p.is_dir()]
    complete = [p for p in candidates if _has_general_debug_bundle(p)]
    return sorted(complete)


def _ask_debug_bundle_dir() -> Path:
    bundles_root = OUTPUTS / "debug_runs" / "rdl"
    bundles = _list_debug_bundles(bundles_root)
    if not bundles:
        raise FileNotFoundError(f"No hay bundles completos en: {bundles_root}")

    print(f"\nBundles disponibles ({bundles_root.as_posix()}):")
    for idx, path in enumerate(bundles, start=1):
        print(f"{idx}. {path.as_posix()}")

    while True:
        raw = input("\nSelecciona numero de bundle: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Entrada invalida. Introduce un numero.")
            continue
        if 1 <= choice <= len(bundles):
            return bundles[choice - 1]
        print(f"Numero fuera de rango (1..{len(bundles)}).")


def _extract_video_path_candidates(data: dict) -> list[str]:
    candidates: list[str] = []
    paths = (
        ("video_path",),
        ("input", "video_path"),
        ("pipeline", "video_path"),
        ("summary", "video_path"),
    )
    for key_path in paths:
        obj = data
        ok = True
        for key in key_path:
            if not isinstance(obj, dict) or key not in obj:
                ok = False
                break
            obj = obj[key]
        if ok and isinstance(obj, str) and obj.strip():
            candidates.append(obj.strip())
    return candidates


def _resolve_video_path_from_bundle(bundle_dir: Path) -> Path:
    summary_path = bundle_dir / "00_pipeline_summary.json"
    if summary_path.is_file():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for raw_path in _extract_video_path_candidates(data if isinstance(data, dict) else {}):
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = (ROOT / candidate).resolve()
                if candidate.is_file():
                    return candidate
        except Exception:
            pass

    video_id = bundle_dir.name
    for ext in VIDEO_EXTENSIONS:
        candidate = ROOT / "data" / f"{video_id}{ext}"
        if candidate.is_file():
            return candidate

    while True:
        raw = input("\nNo se pudo resolver el video original para este bundle. Introduce ruta del video: ").strip()
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        if candidate.is_file():
            return candidate
        print(f"Ruta invalida: {candidate}")


def _run_bent_arms_for_video(
    video_path: Path,
    *,
    expected_bundle_dir: Path | None = None,
) -> None:
    from src.pipeline import run_full_analysis
    from src.visualization.rdl import export_rdl_debug_bundle
    from src.visualization.rdl.detectors import export_bent_arms_debug

    try:
        result = run_full_analysis(video_path)
        runtime = result.to_runtime_dict()
    except Exception as exc:
        print(f"\n[ERROR] Fallo durante run_full_analysis: {exc}")
        return

    analysis_context = runtime.get("analysis_context") if isinstance(runtime.get("analysis_context"), dict) else {}
    video_id = _get_video_id(video_path, runtime)
    general_bundle_dir = OUTPUTS / "debug_runs" / "rdl" / video_id
    if expected_bundle_dir is not None and expected_bundle_dir.name != video_id:
        print(f"[WARN] El video_id reconstruido ({video_id}) no coincide con el folder seleccionado ({expected_bundle_dir.name}).")

    if not _has_general_debug_bundle(general_bundle_dir):
        export_rdl_debug_bundle(artifacts=result, output_dir=general_bundle_dir)
        print(f"Bundle general generado/actualizado: {general_bundle_dir.as_posix()}")

    bent = detect_bent_arms(analysis_context)
    out_dir = OUTPUTS / "debug_detectors" / "rdl" / "repeticiones" / video_id / "bent_arms"
    export_bent_arms_debug(
        analysis_context=analysis_context,
        bent_arms_result=bent,
        output_dir=out_dir,
    )
    print("\nDebug bent_arms generado:")
    print(f"carpeta: {out_dir.as_posix()}")
    print(f"detected: {bent.get('detected', False)}")
    print(f"severity: {bent.get('severity', 'none')}")
    print(f"num_reps_analyzed: {bent.get('num_reps_analyzed', 0)}")
    print(f"num_reps_detected: {bent.get('num_reps_detected', 0)}")


def _build_minimal_bent_arms_context_from_bundle(bundle_dir: Path) -> dict:
    seg_path = bundle_dir / "01_segmentation_result.json"
    clean_npz_path = bundle_dir / "07_user_pose_clean.npz"
    with seg_path.open("r", encoding="utf-8") as f:
        segmentation_result = json.load(f)
    with np.load(str(clean_npz_path), allow_pickle=True) as npz:
        if "kps_xy_clean" in npz.files:
            kps = np.asarray(npz["kps_xy_clean"], dtype=np.float64)
        elif "kps_xy" in npz.files:
            kps = np.asarray(npz["kps_xy"], dtype=np.float64)
        else:
            raise ValueError("07_user_pose_clean.npz no contiene kps_xy_clean ni kps_xy")
    if kps.ndim != 3 or kps.shape[1:] != (17, 2):
        raise ValueError(f"Forma invalida para keypoints clean: {kps.shape}")

    reps = segmentation_result.get("reps") if isinstance(segmentation_result, dict) else []
    if not isinstance(reps, list):
        reps = []
    paired_repetitions: list[dict] = []
    for rep_idx, rep in enumerate(reps):
        if not isinstance(rep, dict):
            continue
        anchor_resolution = resolve_rdl_anchor_frames(rep)
        frames = anchor_resolution.get("frames", {}) if isinstance(anchor_resolution, dict) else {}
        valid = anchor_resolution.get("valid", {}) if isinstance(anchor_resolution, dict) else {}
        anchor_warnings = anchor_resolution.get("warnings", []) if isinstance(anchor_resolution.get("warnings"), list) else []
        anchors: dict[str, dict] = {}
        for anchor_name, frame_val in frames.items():
            frame = int(frame_val) if isinstance(frame_val, int) and 0 <= frame_val < kps.shape[0] else None
            is_valid = bool(valid.get(anchor_name, False)) and frame is not None
            user_kps_clean = np.asarray(kps[frame], dtype=np.float64) if is_valid else None
            anchors[anchor_name] = {
                "anchor": anchor_name,
                "valid": is_valid,
                "user_frame": frame,
                "ideal_frame": None,
                "user_kps_clean": user_kps_clean,
                "user_kps_normalized": None,
                "ideal_kps_normalized": None,
                "warnings": list(anchor_warnings) if not is_valid else [],
            }
        paired_repetitions.append(
            {
                "user_rep_raw_index": int(rep_idx),
                "user_rep_order": int(rep_idx + 1),
                "ideal_rep_raw_index": -1,
                "ideal_valid_rep_index": -1,
                "anchors": anchors,
            }
        )

    return {
        "user": {"video_id": bundle_dir.name},
        "anchor_pairs": {"paired_repetitions": paired_repetitions},
        "context_meta": {
            "source": "debug_bundle_clean_npz",
            "video_id": bundle_dir.name,
            "pose_clean_path": str(clean_npz_path),
            "segmentation_path": str(seg_path),
        },
        "warnings": [],
    }


def _run_bent_arms_for_bundle(bundle_dir: Path, video_path: Path) -> None:
    from src.visualization.rdl.detectors import export_bent_arms_debug

    seg_path = bundle_dir / "01_segmentation_result.json"
    clean_npz_path = bundle_dir / "07_user_pose_clean.npz"
    if not seg_path.is_file():
        print(f"[ERROR] Bundle invalido: falta {seg_path.name} en {bundle_dir.as_posix()}")
        return
    if not clean_npz_path.is_file():
        print("El bundle existe pero no contiene 07_user_pose_clean.npz. Fue generado con una version antigua.")
        if not _ask_yes_no("¿Regenerar bundle ejecutando pipeline completo ahora?", default=False):
            print("Cancelado.")
            return
        _run_bent_arms_for_video(video_path, expected_bundle_dir=bundle_dir)
        return

    try:
        ctx = _build_minimal_bent_arms_context_from_bundle(bundle_dir)
    except Exception as exc:
        print(f"[ERROR] No se pudo construir contexto minimo desde bundle: {exc}")
        return

    bent = detect_bent_arms(ctx)
    out_dir = OUTPUTS / "debug_detectors" / "rdl" / "repeticiones" / bundle_dir.name / "bent_arms"
    export_bent_arms_debug(
        analysis_context=ctx,
        bent_arms_result=bent,
        output_dir=out_dir,
    )
    print("\nDebug bent_arms generado (modo bundle):")
    print(f"bundle: {bundle_dir.as_posix()}")
    print(f"carpeta: {out_dir.as_posix()}")
    print(f"detected: {bent.get('detected', False)}")
    print(f"severity: {bent.get('severity', 'none')}")
    print(f"num_reps_analyzed: {bent.get('num_reps_analyzed', 0)}")
    print(f"num_reps_detected: {bent.get('num_reps_detected', 0)}")


def main() -> None:
    print("\n=== Debug bent_arms ===")
    mode = _ask_mode()
    if mode == 3:
        print("Saliendo.")
        return

    if mode == 1:
        try:
            video_path = _ask_video_path()
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            return

        preliminary_video_id = _get_video_id(video_path, runtime=None)
        general_bundle_dir = OUTPUTS / "debug_runs" / "rdl" / preliminary_video_id
        if _has_general_debug_bundle(general_bundle_dir):
            print(f"\nBundle general encontrado: {general_bundle_dir.as_posix()}")
        else:
            print(f"\nNo existe bundle general para este video: {general_bundle_dir.as_posix()}")
            missing = _missing_general_bundle_files(general_bundle_dir)
            if missing:
                print("Archivos faltantes del bundle general:")
                for item in missing:
                    print(f"- {item}")
            print("Se necesita ejecutar el pipeline completo para construir analysis_context.")
            if not _ask_yes_no("¿Ejecutar pipeline ahora y generar bundle general?", default=False):
                print("Cancelado. Ejecuta primero python -m src.scripts.run_pipeline o acepta generar el bundle.")
                return
        _run_bent_arms_for_video(video_path)
        return

    try:
        bundle_dir = _ask_debug_bundle_dir()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return

    video_path = _resolve_video_path_from_bundle(bundle_dir)
    print(f"\nBundle seleccionado: {bundle_dir.as_posix()}")
    print(f"Video asociado: {video_path.as_posix()}")
    _run_bent_arms_for_bundle(bundle_dir, video_path)


if __name__ == "__main__":
    main()
