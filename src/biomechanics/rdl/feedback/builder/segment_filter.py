
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.biomechanics.rdl.feedback.segment_keys import CANONICAL_SEGMENT_IDS

SEGMENTS = list(CANONICAL_SEGMENT_IDS)
MAX_ITEMS_PER_SEGMENT = 3
MAX_REP_ERROR_ITEMS = 3


def _sev(v: Any) -> str:
    s = str(v or "none").lower()
    return s if s in ("grave", "media", "leve", "posible", "none") else "none"


def filter_evidences_by_segment(evidences: list[dict[str, Any]]) -> dict[str, Any]:
    segment_evidences = [e for e in evidences if e.get("segment") is not None]
    rep_only_evidences = [e for e in evidences if e.get("segment") is None]

    by_rep_segment: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for evidence in segment_evidences:
        rep = evidence.get("rep_order")
        segment = evidence.get("segment")
        if isinstance(rep, int) and isinstance(segment, str):
            by_rep_segment[(rep, segment)].append(evidence)

    by_segment_filtered: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(dict)

    for (rep, segment), items in by_rep_segment.items():
        filtered = _filter_within_segment(items)
        if filtered:
            by_segment_filtered[rep][segment] = filtered

    by_rep_filtered: dict[int, list[dict[str, Any]]] = {}

    rep_orders_with_segments = {r for r, _ in by_rep_segment.keys()}
    rep_orders_rep_only = {
        e.get("rep_order") for e in rep_only_evidences if isinstance(e.get("rep_order"), int)
    }

    for rep in rep_orders_with_segments | rep_orders_rep_only:
        all_segment_items: list[dict[str, Any]] = []
        for segment in SEGMENTS:
            all_segment_items.extend(by_segment_filtered[rep].get(segment, []))

        rep_level_items = [e for e in rep_only_evidences if e.get("rep_order") == rep]
        filtered_rep = _filter_at_rep_level(all_segment_items, rep_level_items)
        posibles = [e for e in rep_level_items if _sev(e.get("severity")) == "posible"]
        if posibles:
            seen = {(str(i.get("error_code")), str(i.get("detector"))) for i in filtered_rep}
            for item in posibles:
                key = (str(item.get("error_code")), str(item.get("detector")))
                if key not in seen:
                    filtered_rep.append(item)
                    seen.add(key)
        if filtered_rep:
            by_rep_filtered[rep] = filtered_rep

    rep_level_only: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for evidence in rep_only_evidences:
        rep = evidence.get("rep_order")
        if isinstance(rep, int):
            rep_level_only[rep].append(evidence)

    return {
        "by_segment": dict(by_segment_filtered),
        "by_rep": by_rep_filtered,
        "rep_level_only": dict(rep_level_only),
    }


def _filter_within_segment(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []

    graves = [i for i in items if _sev(i.get("severity")) == "grave"]
    medias = [i for i in items if _sev(i.get("severity")) == "media"]
    leves = [i for i in items if _sev(i.get("severity")) == "leve"]

    graves = sorted(graves, key=lambda x: -float(x.get("score", 0.0) or 0.0))
    medias = sorted(medias, key=lambda x: -float(x.get("score", 0.0) or 0.0))
    leves = sorted(leves, key=lambda x: -float(x.get("score", 0.0) or 0.0))

    result: list[dict[str, Any]] = []

    if graves:
        result.extend(graves[:MAX_ITEMS_PER_SEGMENT])
        if len(result) < MAX_ITEMS_PER_SEGMENT:
            remaining = MAX_ITEMS_PER_SEGMENT - len(result)
            result.extend(medias[:remaining])
    elif medias:
        result.extend(medias[:MAX_ITEMS_PER_SEGMENT])
        if len(result) < MAX_ITEMS_PER_SEGMENT:
            remaining = MAX_ITEMS_PER_SEGMENT - len(result)
            result.extend(leves[:remaining])
    elif leves:
        result.extend(leves[:MAX_ITEMS_PER_SEGMENT])

    posibles = [i for i in items if _sev(i.get("severity")) == "posible"]
    seen = {(str(i.get("error_code")), str(i.get("detector")), str(i.get("segment"))) for i in result}
    for item in posibles:
        key = (str(item.get("error_code")), str(item.get("detector")), str(item.get("segment")))
        if key not in seen:
            result.append(item)
            seen.add(key)

    return result


def _item_score(item: dict[str, Any]) -> float:
    raw = item.get("score", item.get("max_score", 0.0))
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def select_rep_errors_by_severity(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []

    graves = [i for i in items if _sev(i.get("severity")) == "grave"]
    medias = [i for i in items if _sev(i.get("severity")) == "media"]

    graves = sorted(graves, key=_item_score, reverse=True)
    medias = sorted(medias, key=_item_score, reverse=True)

    if len(graves) >= 3:
        return graves

    if graves:
        result = list(graves)
        remaining = MAX_REP_ERROR_ITEMS - len(result)
        if remaining > 0:
            result.extend(medias[:remaining])
        return result

    if medias:
        return medias[:MAX_REP_ERROR_ITEMS]

    return []


def _filter_at_rep_level(
    segment_items: list[dict[str, Any]],
    rep_level_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return select_rep_errors_by_severity(segment_items + rep_level_items)
