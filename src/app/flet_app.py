
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

import flet as ft

from .adapters import run_pipeline_via_subprocess
from .components import (
    app_header_gradient,
    error_card,
    error_message,
    feedback_card,
    feedback_section,
    file_list_card,
    level_tabs,
    loading_screen,
    project_title_screen,
    results_column,
    source_type_buttons,
    status_badge,
    summary_card,
    technical_details,
)
from .formatting import (
    extract_pose_quality,
    extract_prioritized_feedback_by_rep,
    extract_repetition_count,
    get_recommendation_for_status,
    get_status_color,
    get_status_label,
)
from . import theme
from .theme import BG, PRIMARY

AnalysisLevel = Literal["tramo", "repeticion", "serie"]

_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_NPZ_SUFFIX = "_rtmpose_clean.npz"


def main(page: ft.Page) -> None:
    page.title = "Technique Coach AI"
    page.bgcolor = BG
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.CrossAxisAlignment.START
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 0

    state: dict[str, Any] = {
        "screen": "welcome",
        "source_type": None,
        "selected_file": None,
        "exercise": "rdl",
        "is_loading": False,
        "result": None,
        "analysis_log": [],
        "analysis_started_at": None,
        "analysis_last_at": None,
        "current_level": "repeticion",
        "ui_error": "",
    }

    def _list_files(source_type: str) -> list[Path]:
        if source_type == "video":
            directory = Path.cwd() / "data"
            if not directory.exists():
                return []
            return sorted(
                f
                for f in directory.rglob("*")
                if f.is_file() and f.suffix.lower() in _VIDEO_EXT
            )
        directory = Path.cwd() / "outputs" / "npz"
        if not directory.exists():
            return []
        return sorted(f for f in directory.iterdir() if f.is_file() and f.name.endswith(_NPZ_SUFFIX))

    def _on_welcome_tap(_: ft.ControlEvent) -> None:
        state["screen"] = "source_select"
        _render()

    def _on_source_type_select(source_type: str) -> None:
        state["source_type"] = source_type
        state["screen"] = "file_browser"
        state["ui_error"] = ""
        _render()

    def _on_file_select(file_path: Path) -> None:
        state["selected_file"] = str(file_path)
        state["ui_error"] = ""
        _start_analysis()

    def _on_back_to_source_select(_: ft.ControlEvent) -> None:
        state["screen"] = "source_select"
        state["source_type"] = None
        state["selected_file"] = None
        state["ui_error"] = ""
        _render()

    def _on_level_change(level: str) -> None:
        if level in ("tramo", "repeticion", "serie"):
            state["current_level"] = level
            _render()

    def _on_reset(_: ft.ControlEvent) -> None:
        state.update(
            {
                "screen": "welcome",
                "source_type": None,
                "selected_file": None,
                "result": None,
                "ui_error": "",
                "is_loading": False,
                "current_level": "repeticion",
                "analysis_log": [],
            }
        )
        _render()

    def _start_analysis() -> None:
        if not state["selected_file"] or not state["source_type"]:
            return
        state["is_loading"] = True
        state["screen"] = "loading"
        state["result"] = None
        state["analysis_log"] = ["Iniciando analisis..."]
        state["analysis_started_at"] = time.monotonic()
        state["analysis_last_at"] = time.monotonic()
        _render()

        def _on_progress(message: str) -> None:
            state["analysis_log"].append(message)
            if len(state["analysis_log"]) > 120:
                del state["analysis_log"][: len(state["analysis_log"]) - 120]
            state["analysis_last_at"] = time.monotonic()
            _dispatch_ui_update(page, _safe_update)

        def worker() -> None:
            result = run_pipeline_via_subprocess(
                state["selected_file"],
                state["exercise"],
                source_type=str(state["source_type"]),
                on_progress=_on_progress,
            )
            state["result"] = result
            state["is_loading"] = False
            state["screen"] = "result"
            _dispatch_ui_update(page, _safe_update)

        threading.Thread(target=worker, daemon=True).start()

    def _safe_update() -> None:
        _render()

    def _render() -> None:
        page.controls.clear()

        if state["screen"] == "welcome":
            page.add(project_title_screen(on_tap=_on_welcome_tap))

        elif state["screen"] == "source_select":
            page.add(
                ft.Column(
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0,
                    controls=[
                        app_header_gradient("Technique Coach AI"),
                        ft.Container(height=24),
                        source_type_buttons(on_select=_on_source_type_select),
                    ],
                )
            )

        elif state["screen"] == "file_browser":
            st = state["source_type"] or "video"
            files = _list_files(st)
            label = "videos" if st == "video" else "archivos NPZ"
            page.add(
                ft.Column(
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0,
                    controls=[
                        app_header_gradient("Selecciona archivo"),
                        ft.Container(height=16),
                        file_list_card(
                            files=files,
                            file_type=label,
                            on_select=_on_file_select,
                            on_back=_on_back_to_source_select,
                        ),
                    ],
                )
            )
            if state["ui_error"]:
                page.add(ft.Container(height=12))
                page.add(error_message(state["ui_error"]))

        elif state["screen"] == "loading":
            page.add(
                loading_screen(
                    source_type=state.get("source_type"),
                    log_lines=state.get("analysis_log", []),
                    started_at=state.get("analysis_started_at"),
                    last_at=state.get("analysis_last_at"),
                )
            )

        elif state["screen"] == "result" and isinstance(state.get("result"), dict):
            result_controls = _build_result_for_level(
                state["result"],
                state["exercise"],
                state["current_level"],
            )
            page.add(
                ft.Column(
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0,
                    controls=[
                        app_header_gradient("Resultados"),
                        ft.Container(height=16),
                        level_tabs(current_level=state["current_level"], on_change=_on_level_change),
                        ft.Container(height=12),
                        results_column(result_controls),
                        ft.Container(height=16),
                        ft.ElevatedButton(
                            "Analizar otro",
                            on_click=_on_reset,
                            style=ft.ButtonStyle(
                                bgcolor=PRIMARY,
                                color="#FFFFFF",
                                padding=ft.padding.symmetric(horizontal=24, vertical=12),
                            ),
                        ),
                        ft.Container(height=32),
                    ],
                )
            )

        page.update()

    _render()


def _build_result_for_level(result: dict[str, Any], exercise: str, level: AnalysisLevel) -> list[ft.Control]:
    _ = exercise
    return [_build_result_content(result, level)]


def _feedback_payload(result: dict[str, Any]) -> dict[str, Any]:
    feedback = result.get("feedback")
    if isinstance(feedback, dict):
        return feedback
    return result


def _build_result_content(result: dict[str, Any], level: AnalysisLevel) -> ft.Control:
    if not isinstance(result, dict):
        return error_message("Error al procesar los resultados del análisis")

    status = result.get("status", "failed")

    if status not in ("ok", "partial_analysis"):
        user_message = result.get("user_message", "Error desconocido")
        return error_card("Error en el análisis", user_message)

    if level == "tramo":
        return _build_tramo_view(result)
    if level == "repeticion":
        return _build_repeticion_view(result)
    return _build_serie_view_ui(result)


def _build_tramo_view(result: dict[str, Any]) -> ft.Control:
    from .components import segment_view_card

    data = _feedback_payload(result)
    by_segment = data.get("by_segment", {})
    if not isinstance(by_segment, dict) or not by_segment:
        return ft.Container(
            width=480,
            padding=16,
            margin=ft.margin.symmetric(horizontal=16),
            content=ft.Text(
                "No hay datos de segmento disponibles",
                color=theme.TEXT_SECONDARY,
                text_align=ft.TextAlign.CENTER,
            ),
        )

    cards = []
    for rep_key in sorted(by_segment.keys()):
        rep_data = by_segment[rep_key]
        if isinstance(rep_data, dict):
            cards.append(segment_view_card(rep_key, rep_data))

    if not cards:
        return ft.Container(
            width=480,
            padding=16,
            margin=ft.margin.symmetric(horizontal=16),
            content=ft.Text(
                "No se detectaron errores por segmento",
                color=theme.TEXT_SECONDARY,
                text_align=ft.TextAlign.CENTER,
            ),
        )

    return ft.Column(spacing=16, controls=cards)


def _build_repeticion_view(result: dict[str, Any]) -> ft.Control:
    from .components import repeticion_view_card

    rep_feedback = _feedback_payload(result).get("rep_feedback", [])
    if not isinstance(rep_feedback, list) or not rep_feedback:
        return ft.Container(
            width=480,
            padding=16,
            margin=ft.margin.symmetric(horizontal=16),
            content=ft.Text(
                "No se detectaron errores en las repeticiones",
                color=theme.TEXT_SECONDARY,
                text_align=ft.TextAlign.CENTER,
            ),
        )

    cards = []
    for rep in rep_feedback:
        if isinstance(rep, dict):
            cards.append(repeticion_view_card(rep))

    if not cards:
        return ft.Container(
            width=480,
            padding=16,
            margin=ft.margin.symmetric(horizontal=16),
            content=ft.Text(
                "No se detectaron errores en las repeticiones",
                color=theme.TEXT_SECONDARY,
                text_align=ft.TextAlign.CENTER,
            ),
        )

    return ft.Column(spacing=16, controls=cards)


def _build_serie_view_ui(result: dict[str, Any]) -> ft.Control:
    from .components import serie_view_column

    data = _feedback_payload(result)
    by_serie = data.get("by_serie", {})
    if not isinstance(by_serie, dict):
        by_serie = {}

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    rep_feedback = data.get("rep_feedback", [])
    if not isinstance(rep_feedback, list):
        rep_feedback = []

    headline = str(data.get("headline", "") or "")
    return serie_view_column(rep_feedback, by_serie, summary, headline=headline)


def _build_result_controls(result: dict[str, Any], exercise: str) -> list[ft.Control]:
    status = str(result.get("status", "failed"))
    status_label = get_status_label(status)
    status_color = get_status_color(status)
    reps = extract_repetition_count(result)
    quality = extract_pose_quality(result)
    prioritized = extract_prioritized_feedback_by_rep(result)
    errors_prioritized = prioritized.get("errors", {})
    observations_prioritized = prioritized.get("observations", {})
    possible_prioritized = prioritized.get("possible_errors", {})
    potential_prioritized = prioritized.get("potential_errors", {})
    user_message = str(result.get("user_message", ""))

    out: list[ft.Control] = [
        status_badge(status_label, status_color),
        summary_card(
            {
                "Estado": status_label,
                "Ejercicio": "Peso muerto rumano" if exercise == "rdl" else "Remo con barra",
                "Repeticiones detectadas": str(reps),
                "Calidad de pose": _format_quality(quality),
            }
        ),
    ]

    if status == "ok":
        out.extend(_build_prioritized_rep_cards("Errores", errors_prioritized))
        out.extend(_build_prioritized_rep_cards("Observaciones", observations_prioritized))
        out.extend(_build_uncertain_feedback_sections(possible_prioritized, potential_prioritized))
    elif status == "partial_analysis":
        out.append(error_card("Analisis completado parcialmente", user_message, "#F59E0B"))
        out.extend(_build_prioritized_rep_cards("Errores", errors_prioritized))
        out.extend(_build_prioritized_rep_cards("Observaciones", observations_prioritized))
        out.extend(_build_uncertain_feedback_sections(possible_prioritized, potential_prioritized))
        out.append(technical_details(_build_technical_text(result)))
    elif status in {
        "invalid_input",
        "video_decode_error",
        "no_person_detected",
        "insufficient_pose_quality",
        "unknown_orientation",
        "wrong_exercise",
        "no_reps_detected",
        "invalid_segmentation",
        "invalid_anchors",
    }:
        out.append(error_card(status_label, user_message, "#EF4444"))
        out.append(
            feedback_card(
                "Como solucionarlo",
                [
                    {
                        "title": tip,
                        "severity": "",
                        "where": "",
                        "what_happens": "",
                        "recommendation": "",
                    }
                    for tip in get_recommendation_for_status(status)
                ],
                "#F59E0B",
            )
        )
        out.append(technical_details(_build_technical_text(result)))
    else:
        out.append(error_card("Ha ocurrido un error interno durante el analisis.", user_message, "#EF4444"))
        out.append(technical_details(_build_technical_text(result)))

    return out


def _build_uncertain_feedback_sections(
    possible_prioritized: dict[str, Any],
    potential_prioritized: dict[str, Any],
) -> list[ft.Control]:
    controls: list[ft.Control] = []
    possible_reps = possible_prioritized.get("reps", []) if isinstance(possible_prioritized, dict) else []
    if isinstance(possible_reps, list) and possible_reps:
        controls.append(status_badge("Posible error (priorizado por repeticion)", "#EAB308"))
        for rep in possible_reps:
            rep_label = rep.get("rep", "?")
            items = rep.get("items", [])
            if not isinstance(items, list) or not items:
                continue
            controls.append(
                feedback_section(
                    f"Posible error - Repeticion {rep_label}",
                    (
                        "Indicios no confirmados por otros errores activos en la misma zona. "
                        "Corrige primero los errores principales."
                    ),
                    items,
                    "#EAB308",
                )
            )

    potential_reps = potential_prioritized.get("reps", []) if isinstance(potential_prioritized, dict) else []
    if isinstance(potential_reps, list) and potential_reps:
        controls.append(status_badge("Error potencial (priorizado por repeticion)", "#F59E0B"))
        for rep in potential_reps:
            rep_label = rep.get("rep", "?")
            items = rep.get("items", [])
            if not isinstance(items, list) or not items:
                continue
            controls.append(
                feedback_section(
                    f"Error potencial - Repeticion {rep_label}",
                    "Anomalia con menor certeza; revisar aunque no sea error confirmado.",
                    items,
                    "#F59E0B",
                )
            )
    return controls


def _build_prioritized_rep_cards(section_title: str, prioritized: dict[str, Any]) -> list[ft.Control]:
    reps = prioritized.get("reps", []) if isinstance(prioritized, dict) else []
    if not isinstance(reps, list) or not reps:
        return []

    color = "#EF4444" if section_title == "Errores" else "#3B82F6"
    controls: list[ft.Control] = [
        status_badge(f"{section_title} (por repeticion)", color),
    ]
    for rep in reps:
        rep_label = rep.get("rep", "?")
        items = rep.get("items", [])
        if not isinstance(items, list) or not items:
            continue
        controls.append(
            feedback_section(f"{section_title} - Repeticion {rep_label}", "", items, color),
        )
    return controls


def _format_quality(quality: dict[str, Any]) -> str:
    if not isinstance(quality, dict):
        return "No disponible"
    score = quality.get("overall_score")
    if score is None:
        return "No disponible"
    try:
        return f"{float(score) * 100:.0f}%"
    except (ValueError, TypeError):
        return "No disponible"


def _build_technical_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    warnings = result.get("technical_warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("Avisos tecnicos:")
        for w in warnings:
            lines.append(f"- {w}")
    errors = result.get("errors", [])
    if isinstance(errors, list) and errors:
        lines.append("\nErrores:")
        for e in errors:
            if isinstance(e, dict):
                lines.append(f"- [{e.get('stage', '?')}] {e.get('message', e)}")
            else:
                lines.append(f"- {e}")
    subprocess_output = result.get("subprocess_output", [])
    if isinstance(subprocess_output, list) and subprocess_output:
        lines.append("\nSalida del proceso:")
        lines.extend(f"  {line}" for line in subprocess_output[-12:])
    return "\n".join(lines) if lines else "Sin informacion tecnica adicional."


def _dispatch_ui_update(page: ft.Page, callback: Any) -> None:
    try:
        if hasattr(page, "call_from_thread"):
            page.call_from_thread(callback)
            return
    except Exception:
        pass
    try:

        async def _wrap() -> None:
            callback()

        page.run_task(_wrap)
        return
    except Exception:
        pass
    try:
        callback()
    except Exception:
        pass


if __name__ == "__main__":
    ft.app(target=main)
