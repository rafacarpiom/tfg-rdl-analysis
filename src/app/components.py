
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import flet as ft

from . import theme

# Flet 0.80+ usa Alignment(x, y) en lugar de ft.alignment.top_center
_ALIGN_TOP = ft.Alignment(0, -1)
_ALIGN_BOTTOM = ft.Alignment(0, 1)
_ALIGN_CENTER = ft.Alignment(0, 0)


def _icons() -> Any:
    return getattr(ft, "Icons", None) or getattr(ft, "icons", None)


def _icon(name: str) -> Any:
    icons = _icons()
    if icons is None:
        return name
    return getattr(icons, name, name)


def _with_opacity(opacity: float, color: str) -> str:
    colors = getattr(ft, "colors", None) or getattr(ft, "Colors", None)
    if colors is not None and hasattr(colors, "with_opacity"):
        return colors.with_opacity(opacity, color)
    return color


def project_title_screen(on_tap: Callable[[ft.ControlEvent], None]) -> ft.Control:
    return ft.Container(
        expand=True,
        gradient=ft.LinearGradient(
            begin=_ALIGN_TOP,
            end=_ALIGN_BOTTOM,
            colors=[theme.BG_GRADIENT_START, theme.BG_GRADIENT_END],
        ),
        on_click=on_tap,
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=24,
            controls=[
                ft.Icon(_icon("FITNESS_CENTER"), size=80, color=theme.TEXT_ON_PRIMARY),
                ft.Text(
                    "Technique Coach AI",
                    size=32,
                    weight=ft.FontWeight.W_700,
                    color=theme.TEXT_ON_PRIMARY,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Text(
                    "Analisis biomecanico automatico",
                    size=16,
                    color=_with_opacity(0.9, theme.TEXT_ON_PRIMARY),
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Container(height=32),
                ft.Text(
                    "Toca para comenzar",
                    size=14,
                    color=_with_opacity(0.7, theme.TEXT_ON_PRIMARY),
                    italic=True,
                ),
            ],
        ),
    )


def app_header_gradient(title: str) -> ft.Control:
    return ft.Container(
        width=theme.APP_MAX_WIDTH,
        height=theme.HEADER_HEIGHT,
        gradient=ft.LinearGradient(
            begin=_ALIGN_TOP,
            end=_ALIGN_BOTTOM,
            colors=[theme.BG_GRADIENT_START, theme.BG_GRADIENT_END],
        ),
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Icon(_icon("FITNESS_CENTER"), size=48, color=theme.TEXT_ON_PRIMARY),
                ft.Container(height=8),
                ft.Text(
                    title,
                    size=24,
                    weight=ft.FontWeight.W_700,
                    color=theme.TEXT_ON_PRIMARY,
                ),
            ],
        ),
    )


def source_type_buttons(on_select: Callable[[str], None]) -> ft.Control:
    def _make_button(label: str, subtitle: str, icon_name: str, source_type: str) -> ft.Control:
        return ft.Container(
            width=200,
            height=160,
            border_radius=theme.RADIUS_L,
            bgcolor=theme.CARD,
            border=ft.border.all(2, theme.BORDER),
            on_click=lambda _: on_select(source_type),
            ink=True,
            shadow=ft.BoxShadow(
                blur_radius=12,
                spread_radius=0,
                color="#00000010",
                offset=ft.Offset(0, 4),
            ),
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=12,
                controls=[
                    ft.Icon(_icon(icon_name), size=56, color=theme.PRIMARY),
                    ft.Text(label, size=18, weight=ft.FontWeight.W_600, color=theme.TEXT_PRIMARY),
                    ft.Text(
                        subtitle,
                        size=12,
                        color=theme.TEXT_SECONDARY,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            ),
        )

    return ft.Container(
        width=theme.APP_MAX_WIDTH,
        padding=theme.SPACE_L,
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=24,
            controls=[
                ft.Text(
                    "Selecciona el tipo de entrada",
                    size=16,
                    weight=ft.FontWeight.W_600,
                    color=theme.TEXT_PRIMARY,
                ),
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=16,
                    controls=[
                        _make_button("Video", "Procesar desde\nvideo original", "VIDEO_FILE", "video"),
                        _make_button("NPZ", "Desde pose\nya procesado", "INSERT_DRIVE_FILE", "npz"),
                    ],
                ),
            ],
        ),
    )


def file_list_card(
    files: list[Path],
    file_type: str,
    on_select: Callable[[Path], None],
    on_back: Callable[[ft.ControlEvent], None],
) -> ft.Control:
    if not files:
        content = ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=16,
            controls=[
                ft.Icon(_icon("FOLDER_OFF"), size=48, color=theme.TEXT_MUTED),
                ft.Text(f"No hay {file_type} disponibles", color=theme.TEXT_SECONDARY),
                ft.ElevatedButton(
                    "Volver",
                    on_click=on_back,
                    style=ft.ButtonStyle(
                        bgcolor=_with_opacity(0.1, theme.PRIMARY),
                        color=theme.PRIMARY,
                    ),
                ),
            ],
        )
    else:
        is_video = file_type == "videos"
        file_buttons = [
            ft.Container(
                border_radius=theme.RADIUS_M,
                bgcolor=theme.CARD_ALT,
                padding=theme.SPACE_M,
                on_click=lambda _, f=file_path: on_select(f),
                ink=True,
                content=ft.Row(
                    spacing=12,
                    controls=[
                        ft.Icon(
                            _icon("VIDEO_FILE" if is_video else "INSERT_DRIVE_FILE"),
                            size=24,
                            color=theme.PRIMARY,
                        ),
                        ft.Text(
                            file_path.name,
                            size=14,
                            color=theme.TEXT_PRIMARY,
                            weight=ft.FontWeight.W_500,
                        ),
                    ],
                ),
            )
            for file_path in files
        ]
        content = ft.Column(
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
            height=400,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(f"{len(files)} {file_type} disponibles", size=14, color=theme.TEXT_SECONDARY),
                        ft.TextButton(
                            "Volver",
                            on_click=on_back,
                            style=ft.ButtonStyle(color=theme.PRIMARY),
                        ),
                    ],
                ),
                *file_buttons,
            ],
        )

    return ft.Container(
        width=theme.APP_MAX_WIDTH,
        padding=theme.SPACE_L,
        margin=ft.margin.symmetric(horizontal=16),
        border_radius=theme.RADIUS_L,
        bgcolor=theme.CARD,
        border=ft.border.all(1, theme.BORDER),
        shadow=ft.BoxShadow(blur_radius=16, spread_radius=0, color="#00000008", offset=ft.Offset(0, 4)),
        content=content,
    )


def loading_screen(
    source_type: str | None,
    log_lines: list[str],
    started_at: float | None,
    *,
    last_at: float | None = None,
) -> ft.Control:
    elapsed = ""
    stale = ""
    if started_at is not None:
        elapsed_sec = int(time.monotonic() - started_at)
        elapsed = f"{elapsed_sec}s transcurridos"
        ref = last_at if last_at is not None else started_at
        stale = f"ultima actividad hace {int(time.monotonic() - ref)}s"

    st = source_type or "video"
    title = "Analizando video..." if st == "video" else "Analizando NPZ..."
    lines = log_lines[-18:] if log_lines else ["Esperando salida del pipeline..."]

    return ft.Container(
        expand=True,
        gradient=ft.LinearGradient(
            begin=_ALIGN_TOP,
            end=_ALIGN_BOTTOM,
            colors=[theme.BG_GRADIENT_START, theme.BG_GRADIENT_END],
        ),
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=16,
            controls=[
                ft.ProgressRing(color=theme.TEXT_ON_PRIMARY, width=64, height=64),
                ft.Text(title, size=20, weight=ft.FontWeight.W_600, color=theme.TEXT_ON_PRIMARY),
                ft.Text(elapsed, size=14, color=_with_opacity(0.85, theme.TEXT_ON_PRIMARY)),
                ft.Text(stale, size=12, color=_with_opacity(0.7, theme.TEXT_ON_PRIMARY)),
                ft.Container(
                    width=400,
                    height=220,
                    border_radius=theme.RADIUS_M,
                    bgcolor=_with_opacity(0.2, theme.TEXT_ON_PRIMARY),
                    padding=theme.SPACE_M,
                    content=ft.Column(
                        spacing=4,
                        scroll=ft.ScrollMode.AUTO,
                        controls=[
                            ft.Text(
                                line,
                                size=11,
                                color=_with_opacity(0.95, theme.TEXT_ON_PRIMARY),
                                font_family="monospace",
                            )
                            for line in lines
                        ],
                    ),
                ),
            ],
        ),
    )


def level_tabs(current_level: str, on_change: Callable[[str], None]) -> ft.Control:
    def _make_tab(label: str, level: str) -> ft.Control:
        is_active = current_level == level
        return ft.Container(
            expand=True,
            height=theme.TAB_HEIGHT,
            border_radius=theme.RADIUS_M,
            bgcolor=theme.PRIMARY if is_active else theme.CARD_ALT,
            on_click=lambda _: on_change(level),
            ink=True,
            alignment=_ALIGN_CENTER,
            content=ft.Text(
                label,
                size=14,
                weight=ft.FontWeight.W_600 if is_active else ft.FontWeight.W_500,
                color=theme.TEXT_ON_PRIMARY if is_active else theme.TEXT_PRIMARY,
                text_align=ft.TextAlign.CENTER,
            ),
        )

    return ft.Container(
        width=theme.APP_MAX_WIDTH,
        padding=ft.padding.symmetric(horizontal=16),
        content=ft.Container(
            padding=4,
            border_radius=theme.RADIUS_M,
            bgcolor=theme.CARD,
            border=ft.border.all(1, theme.BORDER),
            content=ft.Row(
                spacing=4,
                controls=[
                    _make_tab("Tramo", "tramo"),
                    _make_tab("Repeticion", "repeticion"),
                    _make_tab("Serie", "serie"),
                ],
            ),
        ),
    )


def error_message(message: str) -> ft.Control:
    return ft.Container(
        width=theme.APP_MAX_WIDTH,
        padding=theme.SPACE_M,
        margin=ft.margin.symmetric(horizontal=16),
        border_radius=theme.RADIUS_M,
        bgcolor=_with_opacity(0.1, theme.ERROR),
        border=ft.border.all(1, theme.ERROR),
        content=ft.Row(
            spacing=12,
            controls=[
                ft.Icon(_icon("ERROR_OUTLINE"), size=24, color=theme.ERROR),
                ft.Text(message, size=14, color=theme.ERROR, expand=True),
            ],
        ),
    )


def results_column(controls: list[ft.Control]) -> ft.Control:
    return ft.Container(
        width=theme.APP_MAX_WIDTH,
        padding=ft.padding.symmetric(horizontal=16),
        content=ft.Column(spacing=12, controls=controls),
    )


# --- Bloques de feedback/resultado (flet_app) ---


def _card(content: ft.Control) -> ft.Container:
    return ft.Container(
        padding=ft.padding.all(theme.SPACE_M),
        border_radius=theme.RADIUS_L,
        bgcolor=theme.CARD,
        border=ft.border.all(1, theme.BORDER),
        shadow=ft.BoxShadow(blur_radius=12, spread_radius=0, color="#00000008", offset=ft.Offset(0, 4)),
        content=content,
    )


def summary_card(items: dict[str, str]) -> ft.Control:
    rows = [
        ft.Container(
            padding=ft.padding.symmetric(horizontal=12, vertical=10),
            border_radius=theme.RADIUS_M,
            bgcolor=theme.CARD_ALT,
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Text(k, color=theme.TEXT_SECONDARY, size=13),
                    ft.Text(v, color=theme.TEXT_PRIMARY, size=13, weight=ft.FontWeight.W_600),
                ],
            ),
        )
        for k, v in items.items()
    ]
    return _card(
        ft.Column(
            spacing=theme.SPACE_S,
            controls=[ft.Text("Resumen", weight=ft.FontWeight.W_600, color=theme.TEXT_PRIMARY), *rows],
        )
    )


def status_badge(label: str, color: str) -> ft.Control:
    return ft.Container(
        padding=ft.padding.symmetric(horizontal=10, vertical=6),
        border_radius=theme.RADIUS_FULL,
        bgcolor=color,
        content=ft.Text(label, color=theme.TEXT_ON_PRIMARY, size=11, weight=ft.FontWeight.W_600),
    )


def feedback_section(
    title: str,
    description: str,
    items: list[dict[str, Any]],
    accent_color: str,
) -> ft.Control:
    body: list[ft.Control] = [
        ft.Text(title, weight=ft.FontWeight.W_600, color=theme.TEXT_PRIMARY),
    ]
    if description:
        body.append(ft.Text(description, color=theme.TEXT_SECONDARY, size=12))
    if not items:
        body.append(ft.Text("Sin elementos para mostrar.", color=theme.TEXT_SECONDARY))
    else:
        for item in items:
            subtitle = " | ".join(x for x in [item.get("severity", ""), item.get("where", "")] if x)
            body.append(
                ft.Container(
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, accent_color),
                    border_radius=theme.RADIUS_M,
                    content=ft.Column(
                        spacing=6,
                        controls=[
                            ft.Text(str(item.get("title", "Observacion")), color=theme.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                            ft.Text(subtitle, color=theme.TEXT_SECONDARY, size=12),
                            ft.Text(str(item.get("what_happens", "")), color=theme.TEXT_SECONDARY, size=12),
                            ft.Text("Como corregirlo:", color=theme.TEXT_PRIMARY, size=12, weight=ft.FontWeight.W_600),
                            ft.Text(str(item.get("recommendation", "")), color=theme.TEXT_SECONDARY, size=12),
                        ],
                    ),
                )
            )
    return _card(ft.Column(spacing=theme.SPACE_S, controls=body))


def feedback_card(title: str, items: list[dict[str, Any]], accent_color: str) -> ft.Control:
    return feedback_section(title, "", items, accent_color)


def error_card(title: str, message: str, color: str | None = None) -> ft.Control:
    if color is not None:
        return _card(
            ft.Column(
                spacing=theme.SPACE_S,
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(_icon("WARNING_AMBER_ROUNDED"), color=color),
                            ft.Text(title, color=theme.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                        ]
                    ),
                    ft.Text(message, color=theme.TEXT_SECONDARY),
                ],
            )
        )

    return ft.Container(
        width=480,
        padding=theme.SPACE_L,
        margin=ft.margin.symmetric(horizontal=16),
        border_radius=theme.RADIUS_L,
        bgcolor=_with_opacity(0.1, theme.ERROR),
        border=ft.border.all(2, theme.ERROR),
        content=ft.Column(
            spacing=12,
            controls=[
                ft.Row(
                    spacing=12,
                    controls=[
                        ft.Icon(_icon("ERROR_OUTLINE"), size=32, color=theme.ERROR),
                        ft.Text(
                            title,
                            size=16,
                            weight=ft.FontWeight.W_700,
                            color=theme.ERROR,
                        ),
                    ],
                ),
                ft.Text(
                    message,
                    size=14,
                    color=theme.TEXT_PRIMARY,
                ),
            ],
        ),
    )


def technical_details(content: str) -> ft.Control:
    return ft.ExpansionTile(
        title=ft.Text("Detalles tecnicos", color=theme.TEXT_SECONDARY, size=12),
        controls=[
            ft.Container(
                bgcolor=theme.CARD_ALT,
                border_radius=theme.RADIUS_M,
                padding=10,
                content=_safe_selectable_text(content or "Sin detalles.", theme.TEXT_SECONDARY, 12),
            )
        ],
    )


def _safe_selectable_text(value: str, color: str, size: int) -> ft.Control:
    selectable = getattr(ft, "SelectableText", None)
    if selectable is not None:
        return selectable(value, color=color, size=size)
    return ft.Text(value, color=color, size=size)


def _dedupe_segment_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("error_code", "")),
            str(item.get("detector", "")),
            str(item.get("kind", "error")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _parse_tramo_rep_payload(rep_data: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if "segments" in rep_data:
        segments = rep_data.get("segments") if isinstance(rep_data.get("segments"), dict) else {}
        rep_level = rep_data.get("rep_level") if isinstance(rep_data.get("rep_level"), list) else []
        return segments, rep_level
    return rep_data, []


def _bucket_rep_segments(rep_data: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    from .segment_keys import CANONICAL_SEGMENT_IDS, to_canonical_segment

    bucketed: dict[str, list[dict[str, Any]]] = {seg_id: [] for seg_id in CANONICAL_SEGMENT_IDS}
    for segment, items in rep_data.items():
        if not isinstance(items, list):
            continue
        if str(segment).strip().replace(" ", "_").lower() == "bottom":
            continue
        canonical = to_canonical_segment(str(segment))
        if canonical is None:
            continue
        for item in items:
            if isinstance(item, dict):
                bucketed[canonical].append(item)
    for seg_id in CANONICAL_SEGMENT_IDS:
        bucketed[seg_id] = _dedupe_segment_items(bucketed[seg_id])
    return bucketed


def _tramo_item_tile(item: dict[str, Any], *, observation: bool = False) -> ft.Control:
    severity = str(item.get("severity", "leve"))
    severity_color = _get_severity_color(severity)
    if observation:
        severity_label = f"OBSERVACIÓN · {severity.upper()}"
    else:
        severity_label = severity.upper()

    return ft.Container(
        padding=10,
        margin=ft.margin.only(bottom=8),
        border_radius=theme.RADIUS_M,
        bgcolor=_with_opacity(0.08, severity_color),
        border=ft.border.all(1.5, _with_opacity(0.3, severity_color)),
        content=ft.Column(
            spacing=6,
            controls=[
                ft.Row(
                    spacing=8,
                    controls=[
                        ft.Container(
                            width=10,
                            height=10,
                            border_radius=5,
                            bgcolor=severity_color,
                        ),
                        ft.Text(
                            str(item.get("title", "Error técnico")),
                            size=13,
                            weight=ft.FontWeight.W_600,
                            color=theme.TEXT_PRIMARY,
                            expand=True,
                        ),
                        ft.Container(
                            padding=ft.padding.symmetric(horizontal=6, vertical=2),
                            border_radius=4,
                            bgcolor=_with_opacity(0.2, severity_color),
                            content=ft.Text(
                                severity_label,
                                size=10,
                                weight=ft.FontWeight.W_700,
                                color=severity_color,
                            ),
                        ),
                    ],
                ),
                ft.Text(
                    str(item.get("what_happens", "")),
                    size=12,
                    color=theme.TEXT_SECONDARY,
                ),
            ],
        ),
    )


def segment_view_card(rep_key: str, rep_data: dict[str, Any]) -> ft.Control:
    from .segment_keys import CANONICAL_SEGMENT_IDS, CANONICAL_SEGMENT_LABELS

    rep_num = rep_key.replace("rep_", "")
    raw_segments, rep_level_items = _parse_tramo_rep_payload(rep_data if isinstance(rep_data, dict) else {})
    segments = _bucket_rep_segments(raw_segments)
    segment_controls: list[ft.Control] = []

    if rep_level_items:
        segment_controls.append(
            ft.Container(
                padding=ft.padding.only(bottom=8),
                content=ft.Text(
                    "Toda la repetición",
                    size=15,
                    weight=ft.FontWeight.W_700,
                    color=theme.ACCENT,
                ),
            )
        )
        for item in rep_level_items:
            if isinstance(item, dict):
                segment_controls.append(_tramo_item_tile(item, observation=False))
        segment_controls.append(ft.Container(height=8))

    for seg_id in CANONICAL_SEGMENT_IDS:
        items = segments.get(seg_id, [])
        segment_label = CANONICAL_SEGMENT_LABELS.get(seg_id, seg_id)
        errors = [i for i in items if isinstance(i, dict) and str(i.get("kind", "error")) != "observation"]
        observations = [i for i in items if isinstance(i, dict) and str(i.get("kind")) == "observation"]

        segment_controls.append(
            ft.Container(
                padding=ft.padding.only(top=12, bottom=8),
                content=ft.Text(
                    segment_label,
                    size=15,
                    weight=ft.FontWeight.W_700,
                    color=theme.PRIMARY,
                ),
            )
        )

        if not errors and not observations:
            segment_controls.append(
                ft.Text(
                    "Sin errores detectados en este tramo",
                    size=12,
                    color=theme.TEXT_MUTED,
                    italic=True,
                )
            )
            continue

        for item in errors:
            segment_controls.append(_tramo_item_tile(item, observation=False))

        if observations:
            segment_controls.append(
                ft.Container(
                    padding=ft.padding.only(top=6, bottom=4),
                    content=ft.Text(
                        "Observaciones",
                        size=13,
                        weight=ft.FontWeight.W_600,
                        color=theme.INFO,
                    ),
                )
            )
            for item in observations:
                segment_controls.append(_tramo_item_tile(item, observation=True))

    return ft.Container(
        width=480,
        padding=theme.SPACE_M,
        margin=ft.margin.symmetric(horizontal=16),
        border_radius=theme.RADIUS_L,
        bgcolor=theme.CARD,
        border=ft.border.all(1, theme.BORDER),
        shadow=ft.BoxShadow(
            blur_radius=8,
            spread_radius=0,
            color="#00000008",
            offset=ft.Offset(0, 2),
        ),
        content=ft.Column(
            spacing=4,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(
                            f"Repetición {rep_num}",
                            size=16,
                            weight=ft.FontWeight.W_700,
                            color=theme.TEXT_PRIMARY,
                        ),
                        _severity_legend_compact(),
                    ],
                ),
                ft.Container(height=4),
                *segment_controls,
            ],
        ),
    )


def _severity_legend_compact() -> ft.Control:
    return ft.Row(
        spacing=8,
        controls=[
            ft.Row(
                spacing=4,
                controls=[
                    ft.Container(width=8, height=8, border_radius=4, bgcolor=theme.SEVERITY_GRAVE),
                    ft.Text("Grave", size=10, color=theme.TEXT_SECONDARY),
                ],
            ),
            ft.Row(
                spacing=4,
                controls=[
                    ft.Container(width=8, height=8, border_radius=4, bgcolor=theme.SEVERITY_MEDIA),
                    ft.Text("Media", size=10, color=theme.TEXT_SECONDARY),
                ],
            ),
            ft.Row(
                spacing=4,
                controls=[
                    ft.Container(width=8, height=8, border_radius=4, bgcolor=theme.SEVERITY_LEVE),
                    ft.Text("Leve", size=10, color=theme.TEXT_SECONDARY),
                ],
            ),
            ft.Row(
                spacing=4,
                controls=[
                    ft.Container(width=8, height=8, border_radius=4, bgcolor=theme.WARNING),
                    ft.Text("Posible", size=10, color=theme.TEXT_SECONDARY),
                ],
            ),
        ],
    )


def repeticion_view_card(rep: dict[str, Any]) -> ft.Control:
    rep_order = rep.get("user_rep_order", "?")
    max_severity = rep.get("max_severity", "none")
    primary_errors = rep.get("primary_errors", [])
    secondary_errors = rep.get("secondary_errors", [])
    observations = rep.get("observations", [])

    severity_color = _get_severity_color(str(max_severity))
    error_controls: list[ft.Control] = []

    if isinstance(primary_errors, list) and primary_errors:
        error_controls.append(
            ft.Text(
                "Errores principales",
                size=13,
                weight=ft.FontWeight.W_600,
                color=theme.TEXT_PRIMARY,
            )
        )
        for item in primary_errors:
            if isinstance(item, dict):
                error_controls.append(_build_error_item(item))
        error_controls.append(ft.Container(height=8))

    if isinstance(secondary_errors, list) and secondary_errors:
        error_controls.append(
            ft.Text(
                "Errores secundarios",
                size=13,
                weight=ft.FontWeight.W_600,
                color=theme.TEXT_SECONDARY,
            )
        )
        for item in secondary_errors:
            if isinstance(item, dict):
                error_controls.append(_build_error_item(item))
        error_controls.append(ft.Container(height=8))

    if isinstance(observations, list) and observations:
        error_controls.append(
            ft.Text(
                "Observaciones",
                size=13,
                weight=ft.FontWeight.W_600,
                color=theme.INFO,
            )
        )
        for item in observations:
            if isinstance(item, dict):
                error_controls.append(_build_error_item(item))

    if not error_controls:
        error_controls.append(
            ft.Text(
                "No se detectaron errores en esta repetición",
                size=12,
                color=theme.TEXT_SECONDARY,
                italic=True,
            )
        )

    return ft.Container(
        width=480,
        padding=theme.SPACE_M,
        margin=ft.margin.symmetric(horizontal=16),
        border_radius=theme.RADIUS_L,
        bgcolor=theme.CARD,
        border=ft.border.all(2, _with_opacity(0.3, severity_color)),
        content=ft.Column(
            spacing=8,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(
                            f"Repetición {rep_order}",
                            size=16,
                            weight=ft.FontWeight.W_700,
                            color=theme.TEXT_PRIMARY,
                        ),
                        ft.Container(
                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                            border_radius=theme.RADIUS_S,
                            bgcolor=_with_opacity(0.15, severity_color),
                            content=ft.Text(
                                str(max_severity).upper(),
                                size=11,
                                weight=ft.FontWeight.W_600,
                                color=severity_color,
                            ),
                        ),
                    ],
                ),
                ft.Container(height=4),
                *error_controls,
            ],
        ),
    )


def serie_summary_card(by_serie: dict[str, Any], summary: dict[str, Any]) -> ft.Control:
    total_reps = by_serie.get("total_reps_analyzed", summary.get("num_reps_with_issues", 0))
    reps_graves = by_serie.get("reps_with_graves", 0)
    reps_medias = by_serie.get("reps_with_medias", 0)
    reps_leves = by_serie.get("reps_with_leves", 0)
    global_max_sev = by_serie.get("global_max_severity", summary.get("max_severity", "none"))
    common_errors = by_serie.get("common_errors", [])

    stats_controls: list[ft.Control] = [
        _build_stat_row("Total repeticiones analizadas", str(total_reps)),
    ]

    if reps_graves > 0:
        stats_controls.append(
            _build_stat_row("Repeticiones con errores graves", str(reps_graves), theme.SEVERITY_GRAVE)
        )
    if reps_medias > 0:
        stats_controls.append(
            _build_stat_row("Repeticiones con errores medios", str(reps_medias), theme.SEVERITY_MEDIA)
        )
    if reps_leves > 0:
        stats_controls.append(
            _build_stat_row("Repeticiones con errores leves", str(reps_leves), theme.SEVERITY_LEVE)
        )

    common_error_controls: list[ft.Control] = []
    if isinstance(common_errors, list) and common_errors:
        common_error_controls.append(
            ft.Container(
                padding=ft.padding.only(top=16, bottom=8),
                content=ft.Text(
                    "Errores más frecuentes",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=theme.TEXT_PRIMARY,
                ),
            )
        )

        for error in common_errors[:3]:
            if isinstance(error, dict):
                title = error.get("title", error.get("error_code", "Error"))
                severity = error.get("max_severity", "leve")
                affected = len(error.get("affected_reps", []))
                severity_color = _get_severity_color(str(severity))

                common_error_controls.append(
                    ft.Container(
                        padding=8,
                        margin=ft.margin.only(bottom=8),
                        border_radius=theme.RADIUS_M,
                        bgcolor=_with_opacity(0.05, severity_color),
                        border=ft.border.all(1, _with_opacity(0.2, severity_color)),
                        content=ft.Row(
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            controls=[
                                ft.Text(
                                    str(title),
                                    size=13,
                                    weight=ft.FontWeight.W_500,
                                    color=theme.TEXT_PRIMARY,
                                    expand=True,
                                ),
                                ft.Text(
                                    f"{affected}/{total_reps} reps",
                                    size=11,
                                    color=theme.TEXT_SECONDARY,
                                ),
                            ],
                        ),
                    )
                )

    global_severity_color = _get_severity_color(str(global_max_sev))

    return ft.Container(
        width=480,
        padding=theme.SPACE_L,
        margin=ft.margin.symmetric(horizontal=16),
        border_radius=theme.RADIUS_L,
        bgcolor=theme.CARD,
        border=ft.border.all(2, _with_opacity(0.3, global_severity_color)),
        content=ft.Column(
            spacing=8,
            controls=[
                ft.Text(
                    "Resumen de la serie",
                    size=18,
                    weight=ft.FontWeight.W_700,
                    color=theme.TEXT_PRIMARY,
                ),
                ft.Container(
                    padding=ft.padding.symmetric(horizontal=12, vertical=8),
                    border_radius=theme.RADIUS_M,
                    bgcolor=_with_opacity(0.1, global_severity_color),
                    content=ft.Text(
                        f"Severidad máxima detectada: {str(global_max_sev).upper()}",
                        size=13,
                        weight=ft.FontWeight.W_600,
                        color=global_severity_color,
                    ),
                ),
                ft.Container(height=8),
                *stats_controls,
                *common_error_controls,
            ],
        ),
    )


def _build_stat_row(label: str, value: str, color: str | None = None) -> ft.Control:
    return ft.Container(
        padding=ft.padding.symmetric(vertical=4),
        content=ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Text(
                    label,
                    size=13,
                    color=theme.TEXT_SECONDARY,
                ),
                ft.Text(
                    value,
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=color if color else theme.TEXT_PRIMARY,
                ),
            ],
        ),
    )


def _build_error_item(item: dict[str, Any]) -> ft.Control:
    title = item.get("title", "Error técnico")
    severity = item.get("severity", "leve")
    what_happens = item.get("what_happens", "")
    how_to_fix = item.get("how_to_fix", item.get("recommendation", ""))

    severity_color = _get_severity_color(str(severity))

    return ft.Container(
        padding=10,
        margin=ft.margin.only(bottom=8),
        border_radius=theme.RADIUS_M,
        bgcolor=_with_opacity(0.05, severity_color),
        border=ft.border.all(1, _with_opacity(0.2, severity_color)),
        content=ft.Column(
            spacing=6,
            controls=[
                ft.Row(
                    spacing=8,
                    controls=[
                        ft.Container(
                            width=8,
                            height=8,
                            border_radius=4,
                            bgcolor=severity_color,
                        ),
                        ft.Text(
                            str(title),
                            size=13,
                            weight=ft.FontWeight.W_600,
                            color=theme.TEXT_PRIMARY,
                        ),
                    ],
                ),
                ft.Text(
                    str(what_happens),
                    size=12,
                    color=theme.TEXT_SECONDARY,
                ),
                ft.Text(
                    f"💡 {how_to_fix}",
                    size=12,
                    color=theme.TEXT_PRIMARY,
                    italic=True,
                ),
            ],
        ),
    )


def _get_severity_color(severity: str) -> str:
    severity_lower = str(severity).lower()
    if severity_lower == "grave":
        return theme.SEVERITY_GRAVE
    if severity_lower == "media":
        return theme.SEVERITY_MEDIA
    if severity_lower == "leve":
        return theme.SEVERITY_LEVE
    if severity_lower == "posible":
        return theme.WARNING
    return theme.SEVERITY_NONE
