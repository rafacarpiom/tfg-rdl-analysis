
from __future__ import annotations

from typing import Any


def get_status_label(status: str) -> str:
    labels = {
        "ok": "Analisis completado",
        "partial_analysis": "Analisis completado parcialmente",
        "invalid_input": "Entrada invalida",
        "video_decode_error": "Error al leer el video",
        "no_person_detected": "Persona no detectada",
        "insufficient_pose_quality": "Calidad de pose insuficiente",
        "unknown_orientation": "Orientacion no determinada",
        "wrong_exercise": "Ejercicio no valido",
        "no_reps_detected": "No se detectaron repeticiones",
        "invalid_segmentation": "Segmentacion invalida",
        "invalid_anchors": "Anchors invalidos",
        "failed": "Error interno",
    }
    return labels.get(str(status), "Estado desconocido")


def get_status_color(status: str) -> str:
    s = str(status)
    if s == "ok":
        return "#22C55E"
    if s == "partial_analysis":
        return "#F59E0B"
    if s in {
        "invalid_input",
        "video_decode_error",
        "no_person_detected",
        "insufficient_pose_quality",
        "unknown_orientation",
        "wrong_exercise",
        "no_reps_detected",
        "invalid_segmentation",
        "invalid_anchors",
        "failed",
    }:
        return "#EF4444"
    return "#9CA3AF"


def get_recommendation_for_status(status: str) -> list[str]:
    base = {
        "invalid_input": [
            "Sube un video valido (MP4, MOV, AVI o MKV).",
            "Comprueba que el archivo no este danado.",
        ],
        "video_decode_error": [
            "Reexporta el video en un formato estandar.",
            "Prueba con otro archivo de la misma grabacion.",
        ],
        "no_person_detected": [
            "Asegura cuerpo completo visible en todo el set.",
            "Evita fondos con mucha distraccion.",
        ],
        "insufficient_pose_quality": [
            "Mejora iluminacion y evita contraluz.",
            "Manten camara fija y vista lateral.",
        ],
        "unknown_orientation": [
            "Graba claramente desde perfil lateral.",
            "Evita angulos diagonales.",
        ],
        "wrong_exercise": [
            "Selecciona el ejercicio correcto antes de analizar.",
            "Usa un video que corresponda al ejercicio elegido.",
        ],
        "no_reps_detected": [
            "Graba 5-6 repeticiones completas.",
            "Evita pausas largas entre repeticiones.",
        ],
        "invalid_segmentation": [
            "Manten ritmo estable durante la serie.",
            "Evita recortes de video justo en mitad del movimiento.",
        ],
        "invalid_anchors": [
            "Asegura visibilidad en inicio, fondo y subida.",
            "Evita oclusiones de cadera, rodilla y tobillo.",
        ],
        "failed": [
            "Reintenta con otro video para descartar archivo corrupto.",
            "Si persiste, comparte los detalles tecnicos.",
        ],
    }
    return base.get(str(status), ["Verifica video, encuadre lateral y repeticiones completas."])


def extract_repetition_count(result: dict[str, Any]) -> int:
    reps = result.get("repetitions", [])
    return len(reps) if isinstance(reps, list) else 0


def extract_pose_quality(result: dict[str, Any]) -> dict[str, Any]:
    quality = result.get("quality", {})
    return quality if isinstance(quality, dict) else {}


def extract_feedback_sections(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    feedback = result.get("feedback", {})
    if not isinstance(feedback, dict):
        return {
            "primary": [],
            "secondary": [],
            "observations": [],
            "possible_errors": [],
            "potential_errors": [],
        }

    rep_feedback = feedback.get("rep_feedback")
    if isinstance(rep_feedback, list):
        primary: list[dict[str, Any]] = []
        secondary: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        possible_errors: list[dict[str, Any]] = []
        potential_errors: list[dict[str, Any]] = []
        for rep in rep_feedback:
            if not isinstance(rep, dict):
                continue
            rep_label = rep.get("user_rep_order", rep.get("rep"))
            primary.extend(_ensure_items(rep.get("primary_errors"), rep_label))
            secondary.extend(_ensure_items(rep.get("secondary_errors"), rep_label))
            observations.extend(_ensure_items(rep.get("observations"), rep_label))
            possible_errors.extend(_ensure_items(rep.get("possible_errors"), rep_label))
            potential_errors.extend(_ensure_items(rep.get("potential_errors"), rep_label))
        return {
            "primary": primary,
            "secondary": secondary,
            "observations": observations,
            "possible_errors": possible_errors,
            "potential_errors": potential_errors,
        }

    return {
        "primary": _ensure_items(feedback.get("main_feedback"), None),
        "secondary": _ensure_items(feedback.get("secondary_feedback"), None),
        "observations": _ensure_items(feedback.get("observations"), None),
        "possible_errors": _ensure_items(feedback.get("possible_errors"), None),
        "potential_errors": _ensure_items(feedback.get("potential_errors"), None),
    }


def extract_prioritized_feedback_by_rep(result: dict[str, Any]) -> dict[str, Any]:
    feedback = result.get("feedback", {})
    if not isinstance(feedback, dict):
        return {
            "errors": {"target_severity": "none", "reps": []},
            "observations": {"target_severity": "none", "reps": []},
            "possible_errors": {"target_severity": "none", "reps": []},
            "potential_errors": {"target_severity": "none", "reps": []},
        }

    rep_feedback = feedback.get("rep_feedback")
    if not isinstance(rep_feedback, list):
        sections = extract_feedback_sections(result)
        errors_flat = sections["primary"] + sections["secondary"]
        obs_flat = sections["observations"]
        return {
            "errors": _prioritize_flat(errors_flat),
            "observations": _prioritize_flat(obs_flat),
            "possible_errors": _prioritize_uncertain_flat(sections["possible_errors"]),
            "potential_errors": _prioritize_uncertain_flat(sections["potential_errors"]),
        }

    errors_by_rep: list[dict[str, Any]] = []
    observations_by_rep: list[dict[str, Any]] = []
    possible_by_rep: list[dict[str, Any]] = []
    potential_by_rep: list[dict[str, Any]] = []
    all_error_items: list[dict[str, Any]] = []
    all_observation_items: list[dict[str, Any]] = []
    all_possible_items: list[dict[str, Any]] = []
    all_potential_items: list[dict[str, Any]] = []
    for rep in rep_feedback:
        if not isinstance(rep, dict):
            continue
        rep_label = rep.get("user_rep_order", rep.get("rep", "?"))
        error_items = _ensure_items(rep.get("primary_errors"), rep_label) + _ensure_items(rep.get("secondary_errors"), rep_label)
        obs_items = _ensure_items(rep.get("observations"), rep_label)
        possible_items = _ensure_items(rep.get("possible_errors"), rep_label)
        potential_items = _ensure_items(rep.get("potential_errors"), rep_label)
        all_error_items.extend(error_items)
        all_observation_items.extend(obs_items)
        all_possible_items.extend(possible_items)
        all_potential_items.extend(potential_items)
        errors_by_rep.append({"rep": rep_label, "items": error_items})
        observations_by_rep.append({"rep": rep_label, "items": obs_items})
        possible_by_rep.append({"rep": rep_label, "items": possible_items})
        potential_by_rep.append({"rep": rep_label, "items": potential_items})

    return {
        "errors": _prioritize_by_rep(all_error_items, errors_by_rep),
        "observations": _prioritize_by_rep(all_observation_items, observations_by_rep),
        "possible_errors": _prioritize_uncertain_by_rep(all_possible_items, possible_by_rep),
        "potential_errors": _prioritize_uncertain_by_rep(all_potential_items, potential_by_rep),
    }


def _ensure_items(raw: Any, rep: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        clean = {
            "title": str(item.get("title", "Observacion tecnica")),
            "severity": str(item.get("severity", "none")),
            "bucket": str(item.get("bucket", "")),
            "error_code": str(item.get("error_code", "")),
            "where": str(item.get("where", "")),
            "what_happens": str(item.get("what_happens", "")),
            "why_it_matters": str(item.get("why_it_matters", "")),
            "recommendation": str(item.get("recommendation", item.get("how_to_fix", ""))),
        }
        if rep is not None:
            clean["rep"] = rep
        out.append(clean)
    return out


def _choose_target_severity(items: list[dict[str, Any]]) -> str:
    if any(_severity_rank(i.get("severity")) == 4 for i in items):
        return "grave"
    if any(_severity_rank(i.get("severity")) == 3 for i in items):
        return "media"
    if any(_severity_rank(i.get("severity")) == 2 for i in items):
        return "leve"
    return "none"


def _severity_rank(value: Any) -> int:
    s = str(value or "").lower()
    if s == "grave":
        return 4
    if s in {"media", "medio"}:
        return 3
    if s == "leve":
        return 2
    if s in {"posible", "warning"}:
        return 1
    return 0


def _simplify_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        where = str(item.get("where", "")).strip()
        short_where = f"Momento: {where}" if where else "Momento: no especificado"
        detail = str(item.get("what_happens", "")).strip() or str(item.get("recommendation", "")).strip()
        recommendation = str(item.get("recommendation", "")).strip() or "Ajusta la tecnica y repite con control."
        out.append(
            {
                "title": str(item.get("title", "Observacion tecnica")),
                "severity": str(item.get("severity", "")),
                "where": short_where,
                "what_happens": detail,
                "recommendation": recommendation,
            }
        )
    return out


def _prioritize_flat(items: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = _choose_target_severity(items)
    if chosen == "none":
        return {"target_severity": "none", "reps": []}
    rank = _severity_rank(chosen)
    filtered = [i for i in items if _severity_rank(i.get("severity")) == rank]
    return {"target_severity": chosen, "reps": [{"rep": "General", "items": _simplify_items(filtered)}]}


def _prioritize_by_rep(all_items: list[dict[str, Any]], by_rep: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = _choose_target_severity(all_items)
    if chosen == "none":
        return {"target_severity": "none", "reps": []}
    out_reps: list[dict[str, Any]] = []
    for rep in by_rep:
        items = rep.get("items", [])
        if not isinstance(items, list):
            continue
        rep_target = _choose_target_severity(items)
        rep_rank = _severity_rank(rep_target)
        filtered = [i for i in items if _severity_rank(i.get("severity")) == rep_rank]
        if filtered:
            out_reps.append({"rep": rep.get("rep", "?"), "items": _simplify_items(filtered)})
    return {"target_severity": chosen, "reps": out_reps}


def _prioritize_uncertain_flat(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"target_severity": "none", "reps": []}
    return {"target_severity": "posible", "reps": [{"rep": "General", "items": _simplify_items(items)}]}


def _prioritize_uncertain_by_rep(all_items: list[dict[str, Any]], by_rep: list[dict[str, Any]]) -> dict[str, Any]:
    if not all_items:
        return {"target_severity": "none", "reps": []}
    out_reps: list[dict[str, Any]] = []
    for rep in by_rep:
        items = rep.get("items", [])
        if not isinstance(items, list) or not items:
            continue
        out_reps.append({"rep": rep.get("rep", "?"), "items": _simplify_items(items)})
    if not out_reps:
        return {"target_severity": "none", "reps": []}
    return {"target_severity": "posible", "reps": out_reps}

