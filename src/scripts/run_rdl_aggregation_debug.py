
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.biomechanics.rdl.feedback.aggregation import aggregate_rdl_feedback_evidence
from src.utils.paths import OUTPUTS, ROOT

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _ask_mode() -> int:
    print("\n=== Debug RDL feedback aggregation ===")
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


def _write_readme(path: Path) -> None:
    text = (
        "RDL feedback aggregation\n"
        "========================\n\n"
        "Este output agrega la evidencia normalizada de detectores RDL.\n"
        "No es feedback final ni recomendaciones.\n"
        "Agrupa por error, repeticion, fase y severidad con soporte de anchors/frames.\n"
        "primary_focus se basa en severidad observada real, no en jerarquia fija de detectores.\n"
    )
    path.write_text(text, encoding="utf-8")


def _save_debug_outputs(video_id: str, feedback_evidence: dict[str, Any], feedback_aggregation: dict[str, Any]) -> Path:
    out_dir = OUTPUTS / "debug_feedback" / "rdl" / "aggregation" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "feedback_aggregation.json", feedback_aggregation)
    _save_json(out_dir / "feedback_evidence.json", feedback_evidence)
    _write_readme(out_dir / "aggregation_readme.txt")
    return out_dir


def _run_from_video(video_path: Path) -> None:
    from src.pipeline import run_full_analysis

    runtime = run_full_analysis(video_path)
    if not isinstance(runtime, dict):
        raise ValueError("run_full_analysis devolvio formato no esperado")
    feedback_evidence = runtime.get("feedback_evidence") if isinstance(runtime.get("feedback_evidence"), dict) else {}
    feedback_aggregation = runtime.get("feedback_aggregation")
    if not isinstance(feedback_aggregation, dict):
        feedback_aggregation = aggregate_rdl_feedback_evidence(feedback_evidence=feedback_evidence, analysis_context=runtime.get("analysis_context"))
    video_id = str((runtime.get("segmentation_result") or {}).get("video_id", video_path.stem))
    out_dir = _save_debug_outputs(video_id, feedback_evidence, feedback_aggregation)
    print(f"\nAggregation debug generado: {out_dir.as_posix()}")


def _run_from_bundle(bundle_dir: Path) -> None:
    p11 = bundle_dir / "11_feedback_aggregation.json"
    p10 = bundle_dir / "10_feedback_evidence.json"
    if p11.is_file():
        feedback_aggregation = _load_json(p11)
        feedback_evidence = _load_json(p10) if p10.is_file() else {}
        out_dir = _save_debug_outputs(bundle_dir.name, feedback_evidence, feedback_aggregation)
        print(f"\nAggregation debug generado desde bundle existente: {out_dir.as_posix()}")
        return
    if p10.is_file():
        feedback_evidence = _load_json(p10)
        feedback_aggregation = aggregate_rdl_feedback_evidence(feedback_evidence=feedback_evidence, analysis_context=None)
        out_dir = _save_debug_outputs(bundle_dir.name, feedback_evidence, feedback_aggregation)
        print(f"\nAggregation debug generado desde feedback_evidence del bundle: {out_dir.as_posix()}")
        return
    print("El bundle no contiene 10_feedback_evidence.json ni 11_feedback_aggregation.json.")
    if not _ask_yes_no("¿Ejecutar pipeline completo como fallback?", default=False):
        return
    from src.pipeline import run_full_analysis

    maybe_video = _resolve_video_path_from_bundle(bundle_dir)
    if maybe_video is None:
        print("No se pudo resolver video automaticamente. Usa opcion 1.")
        return
    runtime = run_full_analysis(maybe_video)
    feedback_evidence = runtime.get("feedback_evidence") if isinstance(runtime.get("feedback_evidence"), dict) else {}
    feedback_aggregation = runtime.get("feedback_aggregation") if isinstance(runtime.get("feedback_aggregation"), dict) else aggregate_rdl_feedback_evidence(feedback_evidence=feedback_evidence, analysis_context=runtime.get("analysis_context"))
    out_dir = _save_debug_outputs(bundle_dir.name, feedback_evidence, feedback_aggregation)
    print(f"\nAggregation debug generado via fallback: {out_dir.as_posix()}")


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
