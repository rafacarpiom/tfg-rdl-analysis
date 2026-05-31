
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

SEVERITY_RANK: dict[str, int] = {"none": 0, "posible": 1, "leve": 2, "media": 3, "grave": 4}
PIPELINE_PYTHON = os.getenv("PIPELINE_PYTHON", ".venv/bin/python")

NPZ_DIR = Path("outputs") / "npz" / "clear"
NPZ_SUFFIX = "_rtmpose_clean.npz"

# Claves del JSON de config → subcarpetas reales en data/tests/
SECTION_DATA_DIRS: dict[str, str] = {
    "functional": "funcional",
    "baseline": "baseline",
    "robustness": "robustness",
}


def _data_subdir(section: str) -> str:
    return SECTION_DATA_DIRS.get(section, section)


def _find_npz(video_path: Path, case: dict[str, Any] | None = None) -> Path | None:
    if case and case.get("force_video") is True:
        return None
    candidate = NPZ_DIR / f"{video_path.stem}{NPZ_SUFFIX}"
    return candidate if candidate.is_file() else None


def run_pipeline(video_path: Path, case: dict[str, Any] | None = None) -> dict[str, Any]:
    npz = _find_npz(video_path, case)
    source_type = "npz" if npz else "video"
    source = str(npz) if npz else str(video_path)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = tmp.name

    cmd = [
        PIPELINE_PYTHON,
        "-u",
        "-m",
        "src.scripts.run_pipeline_json",
        "--npz" if source_type == "npz" else "--video",
        source,
        "--output",
        output_path,
    ]
    env = {**os.environ, "PYTHONPATH": str(Path.cwd()), "PYTHONUNBUFFERED": "1"}

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if not Path(output_path).is_file():
            return {
                "status": "failed",
                "ok": False,
                "user_message": f"Sin salida JSON. stderr: {proc.stderr[-500:]}",
                "_source_type": source_type,
                "_source": source,
            }
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        result["_source_type"] = source_type
        result["_source"] = source
        return result
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "ok": False,
            "user_message": "Timeout (>600s)",
            "_source_type": source_type,
            "_source": source,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "ok": False,
            "user_message": str(exc),
            "_source_type": source_type,
            "_source": source,
        }
    finally:
        Path(output_path).unlink(missing_ok=True)


def _upsert_detector(detectors: dict[str, str], detector: str, severity: str) -> None:
    det = str(detector or "").strip()
    if not det:
        return
    sev = str(severity or "none")
    if sev == "none":
        return
    prev = detectors.get(det, "none")
    if SEVERITY_RANK.get(sev, 0) > SEVERITY_RANK.get(prev, 0):
        detectors[det] = sev


def _collect_from_feedback_item(detectors: dict[str, str], item: Any) -> None:
    if not isinstance(item, dict):
        return
    _upsert_detector(detectors, str(item.get("detector", "")), str(item.get("severity", "none")))


def _get_detected_detectors(result: dict) -> dict[str, str]:
    detectors: dict[str, str] = {}
    feedback = result.get("feedback")
    if not isinstance(feedback, dict):
        return detectors

    rep_feedback = feedback.get("rep_feedback")
    if isinstance(rep_feedback, list):
        for rep in rep_feedback:
            if not isinstance(rep, dict):
                continue
            for bucket in ("primary_errors", "secondary_errors", "observations"):
                items = rep.get(bucket)
                if isinstance(items, list):
                    for item in items:
                        _collect_from_feedback_item(detectors, item)

    by_segment = feedback.get("by_segment")
    if isinstance(by_segment, dict):
        for rep_block in by_segment.values():
            if not isinstance(rep_block, dict):
                continue
            rep_level = rep_block.get("rep_level")
            if isinstance(rep_level, list):
                for item in rep_level:
                    _collect_from_feedback_item(detectors, item)
            segments = rep_block.get("segments")
            if isinstance(segments, dict):
                for items in segments.values():
                    if isinstance(items, list):
                        for item in items:
                            _collect_from_feedback_item(detectors, item)

    by_serie = feedback.get("by_serie")
    if isinstance(by_serie, dict):
        common = by_serie.get("common_errors")
        if isinstance(common, list):
            for item in common:
                if isinstance(item, dict):
                    sev = item.get("max_severity", item.get("severity", "none"))
                    _upsert_detector(detectors, str(item.get("detector", "")), str(sev))

    return detectors


def _get_max_severity(detectors: dict[str, str]) -> str:
    if not detectors:
        return "none"
    return max(detectors.values(), key=lambda s: SEVERITY_RANK.get(s, 0))


def _get_num_reps(result: dict) -> int:
    reps = result.get("repetitions") or []
    return len(reps) if isinstance(reps, list) else 0


def _get_valid_frame_ratio(result: dict) -> float | None:
    quality = result.get("quality") or {}
    val = quality.get("normalization_valid_frame_ratio")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _flip_applied(result: dict) -> bool:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    orientation = artifacts.get("orientation")
    if isinstance(orientation, dict) and "flip_applied" in orientation:
        return bool(orientation["flip_applied"])
    processed = str(artifacts.get("processed_video_path") or "")
    if processed:
        stem = Path(processed).stem
        if "_flipped" in stem or stem.endswith("_flipped"):
            return True
    return False


def evaluate_functional(result: dict, config: dict) -> dict[str, str]:
    status = result.get("status", "failed")
    if status not in {"ok", "partial_analysis"}:
        return {"resultado": "FALLO", "motivo": f"Pipeline terminó con status: {status}"}

    detected = _get_detected_detectors(result)
    primary = config["primary_detector"]
    secondary = config.get("secondary_detectors", [])

    if primary in detected:
        sev = detected[primary]
        others = [d for d in secondary if d in detected]
        msg = f"Detector primario '{primary}' activado (severidad: {sev})"
        if others:
            msg += f"; secundarios también detectados: {others}"
        return {"resultado": "SUPERADO", "motivo": msg}

    if any(d in detected for d in secondary):
        found = [d for d in secondary if d in detected]
        return {
            "resultado": "PARCIAL",
            "motivo": f"Detector primario '{primary}' no activado; secundarios detectados: {found}",
        }

    active = list(detected.keys()) or ["ninguno"]
    return {
        "resultado": "FALLO",
        "motivo": f"Detector primario '{primary}' no activado. Detectores activos: {active}",
    }


def evaluate_baseline(result: dict, config: dict) -> dict[str, str]:
    status = result.get("status", "failed")
    if status not in {"ok", "partial_analysis"}:
        return {"resultado": "FALLO", "motivo": f"Pipeline terminó con status: {status}"}

    detected = _get_detected_detectors(result)
    max_sev = _get_max_severity(detected)
    max_allowed = config.get("max_allowed_severity", "leve")

    rank_actual = SEVERITY_RANK.get(max_sev, 0)
    rank_allowed = SEVERITY_RANK.get(max_allowed, 0)
    rank_grave = SEVERITY_RANK["grave"]

    det_info = f"Detectores: {list(detected.keys())}" if detected else "Sin detecciones"

    if rank_actual <= rank_allowed:
        return {
            "resultado": "SUPERADO",
            "motivo": f"Severidad máxima '{max_sev}' dentro del límite '{max_allowed}'. {det_info}",
        }
    if rank_actual < rank_grave:
        return {
            "resultado": "PARCIAL",
            "motivo": f"Severidad '{max_sev}' supera límite '{max_allowed}' sin llegar a grave. {det_info}",
        }
    return {"resultado": "FALLO", "motivo": f"Severidad grave en vídeo baseline. {det_info}"}


def evaluate_robustness(result: dict, config: dict) -> dict[str, str]:
    status = result.get("status", "failed")
    expected_statuses = config.get("expected_status", ["ok", "partial_analysis"])
    specific_check = config.get("specific_check", "pipeline_completes")

    if status not in expected_statuses:
        return {
            "resultado": "FALLO",
            "motivo": f"Pipeline terminó con status inesperado: '{status}'",
        }

    if specific_check == "flip_applied":
        if _flip_applied(result):
            return {
                "resultado": "SUPERADO",
                "motivo": "Flip horizontal aplicado correctamente (orientación izquierda detectada y corregida)",
            }
        return {
            "resultado": "PARCIAL",
            "motivo": "Pipeline completó pero no se confirmó aplicación de flip en los artifacts",
        }

    if specific_check == "person_detected":
        num_reps = _get_num_reps(result)
        if num_reps > 0:
            return {
                "resultado": "SUPERADO",
                "motivo": f"Sujeto principal detectado correctamente ({num_reps} repeticiones segmentadas)",
            }
        return {
            "resultado": "PARCIAL",
            "motivo": "Pipeline completó pero sin repeticiones segmentadas (posible confusión multi-persona)",
        }

    num_reps = _get_num_reps(result)
    return {
        "resultado": "SUPERADO",
        "motivo": f"Pipeline completó con status '{status}' ({num_reps} repeticiones segmentadas)",
    }


def run_section(
    section: str,
    cases: list[dict],
    data_dir: Path,
    evaluate_fn: Callable[[dict, dict], dict[str, str]],
) -> list[dict]:
    subdir = _data_subdir(section)
    results: list[dict] = []
    for case in cases:
        video_name = case["video"]
        video_path = data_dir / subdir / video_name
        npz = _find_npz(video_path, case)
        fuente = "video" if case.get("force_video") else ("npz" if npz else "video")

        print(f"  [{subdir}] {video_name} (fuente: {fuente}) ... ", end="", flush=True)

        if not video_path.is_file() and npz is None:
            print("FICHERO NO ENCONTRADO")
            results.append(
                {
                    "seccion": subdir,
                    "video": video_name,
                    "status_pipeline": "not_found",
                    "reps_esperadas": case.get("expected_reps", "?"),
                    "reps_detectadas": "-",
                    "detectores_activados": "-",
                    "severidad_maxima": "-",
                    "valid_frame_ratio": "-",
                    "fuente": "-",
                    "resultado": "FALLO",
                    "motivo": f"No existe en data/tests/{subdir}/ ni NPZ en {NPZ_DIR}",
                    "notas": case.get("notes", ""),
                }
            )
            continue

        run_path = video_path if video_path.is_file() else npz
        assert run_path is not None
        pipeline_result = run_pipeline(Path(run_path), case)
        evaluation = evaluate_fn(pipeline_result, case)

        detected = _get_detected_detectors(pipeline_result)
        detected_str = (
            "; ".join(f"{d}:{s}" for d, s in sorted(detected.items())) if detected else "ninguno"
        )
        vfr = _get_valid_frame_ratio(pipeline_result)

        row = {
            "seccion": subdir,
            "video": video_name,
            "status_pipeline": pipeline_result.get("status", "unknown"),
            "reps_esperadas": case.get("expected_reps", "?"),
            "reps_detectadas": _get_num_reps(pipeline_result),
            "detectores_activados": detected_str,
            "severidad_maxima": _get_max_severity(detected),
            "valid_frame_ratio": round(vfr, 3) if vfr is not None else "-",
            "fuente": fuente,
            "resultado": evaluation["resultado"],
            "motivo": evaluation["motivo"],
            "notas": case.get("notes", ""),
        }
        print(evaluation["resultado"])
        results.append(row)
    return results


def _summary_line(rows: list[dict]) -> str:
    total = len(rows)
    superados = sum(1 for r in rows if r["resultado"] == "SUPERADO")
    parciales = sum(1 for r in rows if r["resultado"] == "PARCIAL")
    fallos = sum(1 for r in rows if r["resultado"] == "FALLO")
    return f"SUPERADO: {superados}/{total}  |  PARCIAL: {parciales}/{total}  |  FALLO: {fallos}/{total}"


def _expected_criterion(config_section: str, case: dict[str, Any]) -> str:
    if config_section == "functional":
        line = f"Detector primario: {case.get('primary_detector', '?')}"
        secondary = case.get("secondary_detectors") or []
        if secondary:
            line += f"; secundarios: {', '.join(secondary)}"
        return line
    if config_section == "baseline":
        return f"Severidad máxima permitida: {case.get('max_allowed_severity', 'leve')}"
    condition = str(case.get("condition", "")).strip()
    check = str(case.get("specific_check", "pipeline_completes"))
    check_labels = {
        "flip_applied": "Comprobar flip horizontal aplicado",
        "person_detected": "Sujeto principal detectado (reps segmentadas > 0)",
        "pipeline_completes": "Pipeline completa sin error terminal",
    }
    parts = [p for p in (condition, check_labels.get(check, check)) if p]
    return "; ".join(parts)


def _result_symbol(resultado: str) -> str:
    return {"SUPERADO": "✓", "FALLO": "✗", "PARCIAL": "⚠"}.get(resultado, "?")


def _system_obtained_line(row: dict[str, Any]) -> str:
    if row.get("status_pipeline") == "not_found":
        return "Sin ejecución (fichero no encontrado)"
    parts = [
        f"Status pipeline: {row.get('status_pipeline', '?')}",
        f"Reps: {row.get('reps_detectadas')} detectadas (esperadas: {row.get('reps_esperadas')})",
        f"Detectores activos: {row.get('detectores_activados', 'ninguno')}",
        f"Severidad máxima: {row.get('severidad_maxima', '-')}",
    ]
    vfr = row.get("valid_frame_ratio")
    if vfr not in (None, "-", ""):
        parts.append(f"Valid frame ratio: {vfr}")
    parts.append(f"Fuente: {row.get('fuente', '?')}")
    return "\n  ".join(parts)


def _build_summary_txt(all_rows: list[dict], config: dict[str, Any]) -> str:
    func_rows = [r for r in all_rows if r["seccion"] == "funcional"]
    base_rows = [r for r in all_rows if r["seccion"] == "baseline"]
    rob_rows = [r for r in all_rows if r["seccion"] == "robustness"]

    lines: list[str] = [
        "=== RESUMEN DE EVALUACIÓN ===",
        "",
        f"Funcional  ({len(func_rows)} casos): {_summary_line(func_rows)}",
        f"Baseline   ({len(base_rows)} casos): {_summary_line(base_rows)}",
        f"Robustez   ({len(rob_rows)} casos):  {_summary_line(rob_rows)}",
        "",
        f"TOTAL ({len(all_rows)} casos): {_summary_line(all_rows)}",
        "",
        "--- Detalle de FALLOS y PARCIALES (resumen) ---",
    ]
    for r in all_rows:
        if r["resultado"] in {"FALLO", "PARCIAL"}:
            sym = _result_symbol(r["resultado"])
            lines.append(f"  {sym} [{r['resultado']}] {r['seccion']}/{r['video']}: {r['motivo']}")

    lines.extend(["", "=== DETALLE POR VÍDEO ===", ""])

    section_config_keys = {
        "funcional": "functional",
        "baseline": "baseline",
        "robustness": "robustness",
    }
    cases_by_video: dict[tuple[str, str], dict[str, Any]] = {}
    for cfg_key, subdir in SECTION_DATA_DIRS.items():
        for case in config.get(cfg_key, []):
            if isinstance(case, dict) and case.get("video"):
                cases_by_video[(subdir, str(case["video"]))] = case

    for r in all_rows:
        case = cases_by_video.get((r["seccion"], r["video"]), {})
        cfg_section = section_config_keys.get(r["seccion"], r["seccion"])
        sym = _result_symbol(r["resultado"])
        lines.append(f"[{r['seccion']}] {r['video']}")
        lines.append(f"  Criterio esperado: {_expected_criterion(cfg_section, case)}")
        lines.append(f"  Obtenido:\n  {_system_obtained_line(r)}")
        lines.append(f"  Resultado: {sym} {r['resultado']} — {r['motivo']}")
        if r.get("notas"):
            lines.append(f"  Notas: {r['notas']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del corpus de pruebas RDL")
    parser.add_argument("--data-dir", default="data/tests")
    parser.add_argument("--config", default="corpus_config.json")
    parser.add_argument("--output-csv", default="evaluation_report.csv")
    parser.add_argument("--output-txt", default="evaluation_summary.txt")
    parser.add_argument(
        "--skip-sections",
        default="",
        help="Comma-separated config sections to skip (e.g. robustness)",
    )
    args = parser.parse_args()
    skip_sections = {s.strip().lower() for s in args.skip_sections.split(",") if s.strip()}

    data_dir = Path(args.data_dir)
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: No se encuentra el config en: {config_path.resolve()}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    all_rows: list[dict] = []
    print("\n=== EVALUACIÓN DEL CORPUS DE PRUEBAS RDL ===\n")

    if "functional" not in skip_sections:
        print(">> Sección: funcional")
        all_rows += run_section("functional", config.get("functional", []), data_dir, evaluate_functional)
    else:
        print(">> Sección: funcional (omitida)")

    if "baseline" not in skip_sections:
        print("\n>> Sección: baseline")
        all_rows += run_section("baseline", config.get("baseline", []), data_dir, evaluate_baseline)
    else:
        print("\n>> Sección: baseline (omitida)")

    if "robustness" not in skip_sections:
        print("\n>> Sección: robustez")
        all_rows += run_section("robustness", config.get("robustness", []), data_dir, evaluate_robustness)
    else:
        print("\n>> Sección: robustez (omitida)")

    fieldnames = [
        "seccion",
        "video",
        "status_pipeline",
        "reps_esperadas",
        "reps_detectadas",
        "detectores_activados",
        "severidad_maxima",
        "valid_frame_ratio",
        "fuente",
        "resultado",
        "motivo",
        "notas",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = _build_summary_txt(all_rows, config)
    print(f"\n{summary}\n")
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"CSV guardado en:     {args.output_csv}")
    print(f"Resumen guardado en: {args.output_txt}")


if __name__ == "__main__":
    main()
