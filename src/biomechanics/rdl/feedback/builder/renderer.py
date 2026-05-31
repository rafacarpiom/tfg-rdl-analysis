
from __future__ import annotations

from typing import Any

from .labels import humanize_severity


def _render_feedback_bucket(title: str, items: list[dict[str, Any]]) -> list[str]:
    lines = [f"{title}:"]
    if not items:
        lines.append("- Ninguno.")
        return lines
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item.get('title', item.get('error_code', 'error'))}")
        lines.append(f"   - Severidad: {humanize_severity(str(item.get('severity', 'none')))}")
        lines.append(f"   - Dónde ocurre: {item.get('where', 'n/d')}")
        lines.append(f"   - Qué pasa: {item.get('what_happens', '')}")
        lines.append(f"   - Por qué importa: {item.get('why_it_matters', '')}")
        lines.append(f"   - Cómo corregirlo: {item.get('how_to_fix', '')}")
    return lines


def render_plain_text_report(report_dict: dict[str, Any]) -> str:
    if not isinstance(report_dict, dict):
        return "ANÁLISIS RDL\n\nResumen:\nNo hay datos suficientes para generar feedback."

    if report_dict.get("status") == "no_relevant_issues":
        return (
            "ANÁLISIS RDL\n\n"
            "Resumen:\n"
            "No se han detectado errores técnicos relevantes en las repeticiones analizadas.\n\n"
            "Mantén la técnica actual y revisa que el movimiento sea consistente en más series y cargas."
        )

    lines: list[str] = []
    lines.append("ANÁLISIS RDL")
    lines.append("")
    lines.append("Resumen:")
    lines.append(str(report_dict.get("headline", "")))
    lines.append("")
    rep_feedback = list(report_dict.get("rep_feedback", []))
    rep_feedback = [r for r in rep_feedback if isinstance(r, dict)]
    rep_feedback = [
        r
        for r in rep_feedback
        if list(r.get("primary_errors", [])) or list(r.get("secondary_errors", [])) or list(r.get("observations", []))
    ]
    for rep in rep_feedback:
        lines.append(f"Repetición {rep.get('user_rep_order', 'n/d')}")
        lines.append("")
        lines.extend(_render_feedback_bucket("Errores principales", list(rep.get("primary_errors", []))))
        lines.append("")
        lines.extend(_render_feedback_bucket("Errores secundarios", list(rep.get("secondary_errors", []))))
        lines.append("")
        lines.extend(_render_feedback_bucket("Observaciones", list(rep.get("observations", []))))
        lines.append("")
    return "\n".join(lines)
