
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.biomechanics.rdl.feedback.aggregation import aggregate_rdl_feedback_evidence
from src.biomechanics.rdl.feedback.builder import build_rdl_feedback_report
from src.utils.paths import OUTPUTS, ROOT

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _ask_mode() -> int:
    print("\n=== Debug RDL feedback report ===")
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
        "RDL feedback report\n"
        "===================\n\n"
        "Este output es el feedback humano final del RDL.\n"
        "Se genera desde feedback_aggregation y evidencia ya calculada.\n"
        "Incluye que ocurre, donde ocurre, por que importa y como corregirlo.\n"
        "No recalcula detectores ni keypoints.\n"
    )
    path.write_text(text, encoding="utf-8")


def _save_debug_outputs(
    *,
    video_id: str,
    feedback_report: dict[str, Any],
    feedback_aggregation: dict[str, Any],
) -> Path:
    out_dir = OUTPUTS / "debug_feedback" / "rdl" / "report" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "feedback_report.json", feedback_report)
    (out_dir / "feedback_report.txt").write_text(str(feedback_report.get("plain_text", "")), encoding="utf-8")
    _save_json(out_dir / "feedback_aggregation.json", feedback_aggregation)
    _write_readme(out_dir / "report_readme.txt")
    return out_dir


def _run_from_video(video_path: Path) -> None:
    from src.pipeline import run_full_analysis

    runtime = run_full_analysis(video_path)
    if not isinstance(runtime, dict):
        raise ValueError("run_full_analysis devolvio formato no esperado")
    report = runtime.get("feedback_report") if isinstance(runtime.get("feedback_report"), dict) else {}
    aggregation = runtime.get("feedback_aggregation") if isinstance(runtime.get("feedback_aggregation"), dict) else {}
    if not report and aggregation:
        report = build_rdl_feedback_report(
            feedback_aggregation=aggregation,
            feedback_evidence=runtime.get("feedback_evidence"),
            analysis_context=runtime.get("analysis_context"),
        )
    video_id = str((runtime.get("segmentation_result") or {}).get("video_id", video_path.stem))
    out_dir = _save_debug_outputs(video_id=video_id, feedback_report=report, feedback_aggregation=aggregation)
    print(f"\nFeedback report debug generado: {out_dir.as_posix()}")


def _run_from_bundle(bundle_dir: Path) -> None:
    p12 = bundle_dir / "12_feedback_report.json"
    p13 = bundle_dir / "13_feedback_report.txt"
    p11 = bundle_dir / "11_feedback_aggregation.json"
    p10 = bundle_dir / "10_feedback_evidence.json"

    if p12.is_file() and p13.is_file():
        report = _load_json(p12)
        aggregation = _load_json(p11) if p11.is_file() else {}
        out_dir = _save_debug_outputs(video_id=bundle_dir.name, feedback_report=report, feedback_aggregation=aggregation)
        print(f"\nFeedback report debug generado desde bundle existente: {out_dir.as_posix()}")
        return

    if p11.is_file():
        aggregation = _load_json(p11)
        evidence = _load_json(p10) if p10.is_file() else None
        report = build_rdl_feedback_report(feedback_aggregation=aggregation, feedback_evidence=evidence, analysis_context=None)
        out_dir = _save_debug_outputs(video_id=bundle_dir.name, feedback_report=report, feedback_aggregation=aggregation)
        print(f"\nFeedback report debug generado desde aggregation del bundle: {out_dir.as_posix()}")
        return

    if p10.is_file():
        evidence = _load_json(p10)
        aggregation = aggregate_rdl_feedback_evidence(feedback_evidence=evidence, analysis_context=None)
        report = build_rdl_feedback_report(feedback_aggregation=aggregation, feedback_evidence=evidence, analysis_context=None)
        out_dir = _save_debug_outputs(video_id=bundle_dir.name, feedback_report=report, feedback_aggregation=aggregation)
        print(f"\nFeedback report debug generado desde evidence del bundle: {out_dir.as_posix()}")
        return

    print("El bundle no contiene report/aggregation/evidence suficientes.")
    if not _ask_yes_no("¿Ejecutar pipeline completo como fallback?", default=False):
        return
    video_path = _resolve_video_path_from_bundle(bundle_dir)
    if video_path is None:
        print("No se pudo resolver video automaticamente. Usa opcion 1.")
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
