
from __future__ import annotations

from typing import Any


_ANCHOR_WHERE_TEXT: dict[str, str] = {
    "ecc_0": "al comenzar la bajada",
    "ecc_25": "al inicio de la bajada",
    "ecc_50": "a mitad de la bajada",
    "ecc_75": "al final de la bajada",
    "ecc_100": "en la posicion baja",
    "bottom": "en la posicion baja",
    "con_0": "en la posicion baja",
    "con_25": "al inicio de la subida",
    "con_50": "a mitad de la subida",
    "con_75": "al final de la subida",
    "con_100": "al finalizar arriba",
    "end": "al finalizar arriba",
}

_ANCHOR_ORDER: dict[str, int] = {
    "ecc_0": 0,
    "ecc_25": 1,
    "ecc_50": 2,
    "ecc_75": 3,
    "ecc_100": 4,
    "bottom": 4,
    "con_0": 4,
    "con_25": 5,
    "con_50": 6,
    "con_75": 7,
    "con_100": 8,
    "end": 8,
}

_FAILED_SEGMENT_TEXT: dict[str, str] = {
    "ecc_0_to_ecc_25": "al inicio de la bajada",
    "ecc_25_to_ecc_50": "a mitad de la bajada",
    "ecc_50_to_ecc_75": "entre la mitad y el final de la bajada",
    "ecc_75_to_ecc_100": "al final de la bajada, cerca de la posicion baja",
}


def humanize_severity(severity: str) -> str:
    mapping = {
        "grave": "grave",
        "media": "moderado",
        "leve": "leve",
        "posible": "posible",
        "none": "sin error",
    }
    return mapping.get(str(severity or "none").lower(), "sin error")


def humanize_phase(phase: str) -> str:
    mapping = {
        "eccentric": "durante la bajada",
        "bottom": "en la posicion baja",
        "concentric": "durante la subida",
        "lockout": "al finalizar la repeticion",
        "full_rep": "durante la repeticion",
        "unknown": "en un momento no determinado",
    }
    return mapping.get(str(phase or "unknown").lower(), "en un momento no determinado")


def humanize_location_label(location_label: str) -> str:
    mapping = {
        "eccentric": "durante la bajada",
        "concentric": "durante la subida",
        "lockout": "al finalizar arriba",
        "full_rep": "durante la repeticion",
        "eccentric_start": "al comenzar la bajada",
        "eccentric_early": "al inicio de la bajada",
        "eccentric_mid": "a mitad de la bajada",
        "eccentric_mid_late": "entre la mitad y el final de la bajada",
        "eccentric_late": "al final de la bajada",
        "eccentric_late_bottom": "al final de la bajada, cerca de la posicion baja",
        "eccentric_range": "durante el rango de bajada",
        "bottom": "en la posicion baja",
        "concentric_early": "al inicio de la subida",
        "concentric_mid": "a mitad de la subida",
        "concentric_late": "al final de la subida",
        "concentric_late_lockout": "al final de la subida, cerca del cierre",
        "lockout": "al finalizar arriba",
        "full_rep": "durante la repeticion",
        "unknown": "en un momento no determinado",
    }
    return mapping.get(str(location_label or "unknown").lower(), "en un momento no determinado")


def humanize_reps(reps: list[int]) -> str:
    vals = sorted({int(r) for r in reps if isinstance(r, int)})
    if not vals:
        return "repeticiones no determinadas"
    if len(vals) == 1:
        return f"repeticion {vals[0]}"
    if len(vals) == 2:
        return f"repeticiones {vals[0]} y {vals[1]}"
    return f"repeticiones {', '.join(str(x) for x in vals[:-1])} y {vals[-1]}"


def _where_from_anchors(anchors: list[str]) -> str:
    clean = [str(a).strip().lower() for a in anchors if str(a).strip()]
    if not clean:
        return "en un momento no determinado"
    uniq = sorted(set(clean), key=lambda a: (_ANCHOR_ORDER.get(a, 999), a))
    parts: list[str] = []
    for a in uniq:
        text = _ANCHOR_WHERE_TEXT.get(a)
        if text and text not in parts:
            parts.append(text)
    if not parts:
        return "en un momento no determinado"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} y {parts[1]}"
    return f"{', '.join(parts[:-1])} y {parts[-1]}"


def _where_from_failed_segments(segments: list[str]) -> str:
    clean = [str(s).strip().lower() for s in segments if str(s).strip()]
    if not clean:
        return ""
    parts: list[str] = []
    for seg in clean:
        txt = _FAILED_SEGMENT_TEXT.get(seg)
        if txt and txt not in parts:
            parts.append(txt)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} y {parts[1]}"
    return f"{', '.join(parts[:-1])} y {parts[-1]}"


def build_where_text(issue: dict[str, Any], *, include_reps: bool = True) -> str:
    detector = str(issue.get("detector", "")).lower()
    location_labels = issue.get("location_labels", [])
    phases = issue.get("phases", [])
    anchors = issue.get("anchors", [])
    clean_labels = (
        [str(x).lower() for x in location_labels if str(x) and str(x).lower() != "unknown"]
        if isinstance(location_labels, list)
        else []
    )
    if detector == "asymmetry":
        phase_set = {str(p).lower() for p in phases} if isinstance(phases, list) else set()
        if "full_rep" in phase_set:
            loc_text = "durante la repeticion"
        elif {"eccentric", "concentric"}.issubset(phase_set):
            loc_text = "durante la repeticion"
        elif "eccentric" in phase_set:
            loc_text = "durante la bajada"
        elif "concentric" in phase_set:
            loc_text = "durante la subida"
        else:
            loc_text = "durante la repeticion"
    elif clean_labels:
        label_set = set(clean_labels)
        if {"eccentric_late", "eccentric_late_bottom"}.issubset(label_set):
            loc_text = "al final de la bajada, cerca de la posicion baja"
        elif {"eccentric_mid", "eccentric_late"}.issubset(label_set):
            loc_text = "entre la mitad y el final de la bajada"
        elif {"concentric_mid", "concentric_late"}.issubset(label_set):
            loc_text = "durante la segunda mitad de la subida"
        elif len(clean_labels) == 1:
            one = clean_labels[0]
            if one in {"eccentric", "concentric", "lockout", "full_rep"}:
                loc_text = humanize_phase(one)
            else:
                loc_text = humanize_location_label(one)
        else:
            human = []
            for l in clean_labels:
                txt = humanize_location_label(l)
                if txt not in human:
                    human.append(txt)
            if len(human) == 1:
                loc_text = human[0]
            elif len(human) == 2:
                loc_text = f"{human[0]} y {human[1]}"
            else:
                loc_text = f"{', '.join(human[:-1])} y {human[-1]}"
    elif isinstance(anchors, list) and anchors:
        loc_text = _where_from_anchors(anchors)
    elif isinstance(phases, list) and phases:
        clean_phases = [str(p).lower() for p in phases if str(p) and str(p).lower() != "unknown"]
        loc_text = humanize_phase(clean_phases[0]) if clean_phases else "durante la repeticion"
    else:
        metrics = issue.get("summary_metrics")
        failed_segments = metrics.get("failed_segments", []) if isinstance(metrics, dict) else []
        if isinstance(failed_segments, list) and failed_segments:
            loc_text = _where_from_failed_segments(failed_segments) or "durante la repeticion"
        elif detector in {"hip_hinge", "knee_dominant", "spine_flexion"}:
            loc_text = "durante la bajada"
        elif detector == "lockout":
            loc_text = "al finalizar arriba"
        else:
            loc_text = "durante la repeticion"
    if loc_text == "en un momento no determinado":
        if detector in {"hip_hinge", "knee_dominant", "spine_flexion", "neck_movement", "bar_far", "bent_arms", "short_rom"}:
            loc_text = "durante la bajada"
        elif detector == "lockout":
            loc_text = "al finalizar arriba"
        elif detector == "asymmetry":
            loc_text = "durante la repeticion"
        else:
            loc_text = "durante la repeticion"
    if not include_reps:
        return loc_text
    reps_text = humanize_reps(issue.get("reps", []))
    return f"{reps_text}, {loc_text}"
