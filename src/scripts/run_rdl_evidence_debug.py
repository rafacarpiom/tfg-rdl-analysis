
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.biomechanics.rdl.detectors import run_rdl_detectors
from src.biomechanics.rdl.feedback.evidence_normalizer import normalize_rdl_detector_evidence
from src.utils.paths import OUTPUTS, ROOT
from src.visualization.rdl import export_rdl_debug_bundle

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _ask_mode() -> int:
    print("\n=== Debug RDL evidence normalizer ===")
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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_readme(path: Path) -> None:
    text = (
        "RDL evidence normalizer\n"
        "=======================\n\n"
        "Este output contiene evidencias normalizadas de detectores RDL.\n"
        "No es feedback final ni recomendaciones.\n"
        "Agrupa por repeticion/fase/anchors/frames con severidad y score.\n"
        "Los anchors se usan como evidencia de localizacion, no como unidad primaria.\n"
    )
    path.write_text(text, encoding="utf-8")


def _load_cache_analysis_context(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as npz:
        raw = npz.get("analysis_context")
        if raw is None or raw.size == 0:
            raise ValueError("analysis_context no encontrado en cache")
        ctx = raw.ravel()[0]
        if not isinstance(ctx, dict):
            raise ValueError("analysis_context invalido en cache")
        return ctx


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


def _detector_results_summary(detector_results: dict[str, Any]) -> dict[str, Any]:
    detectors = detector_results.get("detectors") if isinstance(detector_results.get("detectors"), dict) else {}
    out: dict[str, Any] = {
        "summary": detector_results.get("summary", {}),
        "detectors": {},
        "warnings": detector_results.get("warnings", []),
    }
    for name, res in detectors.items():
        if not isinstance(res, dict):
            continue
        out["detectors"][name] = {
            "detected": res.get("detected"),
            "severity": res.get("severity"),
            "score": res.get("score"),
            "num_reps_analyzed": res.get("num_reps_analyzed"),
            "num_reps_detected": res.get("num_reps_detected"),
            "rep_results": [
                {
                    "rep_index": idx,
                    "user_rep_raw_index": (rep.get("user_rep_raw_index") if isinstance(rep, dict) else None),
                    "user_rep_order": (rep.get("user_rep_order") if isinstance(rep, dict) else None),
                    "detected": (rep.get("detected") if isinstance(rep, dict) else None),
                    "severity": (rep.get("severity") if isinstance(rep, dict) else None),
                }
                for idx, rep in enumerate(res.get("rep_results", []))
                if isinstance(res.get("rep_results"), list)
            ],
            "warnings": res.get("warnings", []),
        }
    return out


def _save_debug_outputs(video_id: str, feedback_evidence: dict[str, Any], detector_results: dict[str, Any]) -> Path:
    out_dir = OUTPUTS / "debug_feedback" / "rdl" / "evidence" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "feedback_evidence.json", feedback_evidence)
    _save_json(out_dir / "detector_results_summary.json", _detector_results_summary(detector_results))
    _write_readme(out_dir / "evidence_readme.txt")
    return out_dir


def _run_from_video(video_path: Path) -> None:
    from src.pipeline import run_full_analysis

    runtime = run_full_analysis(video_path)
    if not isinstance(runtime, dict):
        raise ValueError("run_full_analysis devolvio formato no esperado")
    analysis_context = runtime.get("analysis_context") if isinstance(runtime.get("analysis_context"), dict) else {}
    detector_results = runtime.get("detector_results") if isinstance(runtime.get("detector_results"), dict) else {}
    detector_map = detector_results.get("detectors") if isinstance(detector_results.get("detectors"), dict) else detector_results
    feedback_evidence = runtime.get("feedback_evidence")
    if not isinstance(feedback_evidence, dict):
        feedback_evidence = normalize_rdl_detector_evidence(
            detector_results=detector_map if isinstance(detector_map, dict) else {},
            analysis_context=analysis_context,
        )
    video_id = str((analysis_context.get("user") or {}).get("video_id", video_path.stem)) if isinstance(analysis_context, dict) else video_path.stem
    bundle_dir = OUTPUTS / "debug_runs" / "rdl" / video_id
    export_rdl_debug_bundle(artifacts=runtime, output_dir=bundle_dir)
    out_dir = _save_debug_outputs(video_id, feedback_evidence, detector_results)
    print(f"\nEvidence debug generado: {out_dir.as_posix()}")


def _run_from_bundle(bundle_dir: Path) -> None:
    existing = bundle_dir / "10_feedback_evidence.json"
    if existing.is_file():
        feedback_evidence = _load_json(existing)
        detector_results_summary = _load_json(bundle_dir / "08_detector_results_summary.json") if (bundle_dir / "08_detector_results_summary.json").is_file() else {"summary": {}, "detectors": {}, "warnings": []}
        detector_results = {
            "summary": detector_results_summary.get("summary", {}),
            "detectors": detector_results_summary.get("detectors", {}),
            "warnings": detector_results_summary.get("warnings", []),
        }
        out_dir = _save_debug_outputs(bundle_dir.name, feedback_evidence, detector_results)
        print(f"\nEvidence debug generado desde bundle existente: {out_dir.as_posix()}")
        return

    for cache_name in ("08_analysis_context_anchor_cache.npz", "07_analysis_context_anchor_cache.npz"):
        cache = bundle_dir / cache_name
        if cache.is_file():
            ctx = _load_cache_analysis_context(cache)
            detector_results = run_rdl_detectors(ctx)
            detector_map = detector_results.get("detectors") if isinstance(detector_results.get("detectors"), dict) else {}
            feedback_evidence = normalize_rdl_detector_evidence(detector_results=detector_map, analysis_context=ctx)
            out_dir = _save_debug_outputs(bundle_dir.name, feedback_evidence, detector_results)
            print(f"\nEvidence debug generado desde cache de bundle: {out_dir.as_posix()}")
            return

    print("No hay feedback_evidence ni cache de analysis_context en el bundle.")
    if not _ask_yes_no("¿Ejecutar run_full_analysis para regenerar?", default=False):
        return
    video_path = _resolve_video_path_from_bundle(bundle_dir)
    if video_path is None:
        print("No se pudo resolver video automáticamente. Usa opción 1.")
        return
    _run_from_video(video_path)


def main() -> None:
    mode = _ask_mode()
    if mode == 3:
        return
    if mode == 1:
        _run_from_video(_ask_video())
        return
    _run_from_bundle(_ask_bundle())


if __name__ == "__main__":
    main()
