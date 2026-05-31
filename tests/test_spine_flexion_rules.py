from __future__ import annotations

from src.biomechanics.rdl.detectors.spine_flexion.rules import (
    HIP_JUSTIFICATION_FACTOR,
    MIN_SEGMENTS_FOR_REP,
    detect_spine_flexion,
    rule_segment,
    _persistent_severity,
)


def _geom_segment(
    segment: str,
    *,
    torso_low: bool = True,
    torso_sev: str = "media",
    norm: float = 0.18,
    coherent: bool = True,
) -> dict:
    anchor = {
        "ecc_0_to_ecc_25": "ecc_25",
        "ecc_25_to_ecc_50": "ecc_50",
        "ecc_50_to_ecc_75": "ecc_75",
        "ecc_75_to_ecc_100": "ecc_100",
    }[segment]
    item = {
        "anchor": anchor,
        "status": "ok",
        "torso_low_failed": torso_low,
        "torso_low_severity": torso_sev,
        "shoulder_low_norm": norm,
        "shoulder_low_px": 12.0,
    }
    if coherent:
        item.update(
            user_shoulder_drop_from_top_norm=0.10,
            ideal_shoulder_drop_from_top_norm=0.12,
            user_hip_back_norm=0.20,
            ideal_hip_back_norm=0.18,
        )
    else:
        item.update(
            user_shoulder_drop_from_top_norm=0.55,
            ideal_shoulder_drop_from_top_norm=0.10,
            user_hip_back_norm=0.20,
            ideal_hip_back_norm=0.18,
        )
    return item


def _hip_rep(*, anchor: str, failed: bool, delta: float = 0.05) -> dict:
    return {
        "anchor_rulings": {
            anchor: {
                "failed": failed,
                "delta_hip_back": delta,
                "severity": "media" if failed else "none",
            }
        }
    }


def _knee_rep(*, anchor: str, failed: bool) -> dict:
    return {"anchor_rulings": {anchor: {"failed": failed}}}


def _neck_rep(*, segment: str, failed: bool) -> dict:
    return {"segment_results": {segment: {"failed": failed, "severity": "leve" if failed else "none"}}}


def test_knee_explained_sets_possible_not_triggered():
    seg = "ecc_50_to_ecc_75"
    r = rule_segment(
        seg,
        hip_result=_hip_rep(anchor="ecc_75", failed=True),
        knee_result=_knee_rep(anchor="ecc_75", failed=True),
        neck_result=_neck_rep(segment=seg, failed=False),
        spine_geometry={seg: _geom_segment(seg)},
    )
    assert r.possible is True
    assert r.triggered is False
    assert r.reason == "possible_spine_flexion_explained_by_knee_or_neck"


def test_neck_explained_sets_possible():
    seg = "ecc_50_to_ecc_75"
    r = rule_segment(
        seg,
        hip_result=_hip_rep(anchor="ecc_75", failed=True),
        knee_result=_knee_rep(anchor="ecc_75", failed=False),
        neck_result=_neck_rep(segment=seg, failed=True),
        spine_geometry={seg: _geom_segment(seg)},
    )
    assert r.possible is True
    assert r.triggered is False


def test_adequate_hip_hinge_does_not_trigger_spine():
    seg = "ecc_50_to_ecc_75"
    r = rule_segment(
        seg,
        hip_result=_hip_rep(anchor="ecc_75", failed=False),
        knee_result=_knee_rep(anchor="ecc_75", failed=False),
        neck_result=_neck_rep(segment=seg, failed=False),
        spine_geometry={seg: _geom_segment(seg, torso_sev="leve")},
    )
    assert r.triggered is False
    assert r.possible is False
    assert r.reason == "shoulder_drop_coherent_with_hip_pattern_vs_ideal"


def test_hip_ok_coherence_inconclusive_does_not_trigger():
    seg = "ecc_50_to_ecc_75"
    geom = {
        "anchor": "ecc_75",
        "status": "ok",
        "torso_low_failed": True,
        "torso_low_severity": "grave",
        "shoulder_low_norm": 0.25,
        "shoulder_low_px": 20.0,
    }
    r = rule_segment(
        seg,
        hip_result=_hip_rep(anchor="ecc_75", failed=False),
        knee_result=_knee_rep(anchor="ecc_75", failed=False),
        neck_result=_neck_rep(segment=seg, failed=False),
        spine_geometry={seg: geom},
    )
    assert r.triggered is False
    assert r.reason == "shoulder_hip_coherence_not_computed_no_spine_confirmation"


def test_hip_ok_incoherent_shoulder_triggers_spine():
    seg = "ecc_50_to_ecc_75"
    r = rule_segment(
        seg,
        hip_result=_hip_rep(anchor="ecc_75", failed=False),
        knee_result=_knee_rep(anchor="ecc_75", failed=False),
        neck_result=_neck_rep(segment=seg, failed=False),
        spine_geometry={seg: _geom_segment(seg, coherent=False)},
    )
    assert r.triggered is True
    assert r.reason == "shoulder_drop_exceeds_hip_coherence_vs_ideal"


def test_insufficient_hip_hinge_always_triggers_spine():
    seg = "ecc_50_to_ecc_75"
    r = rule_segment(
        seg,
        hip_result=_hip_rep(anchor="ecc_75", failed=True, delta=-0.2),
        knee_result=_knee_rep(anchor="ecc_75", failed=False),
        neck_result=_neck_rep(segment=seg, failed=False),
        spine_geometry={seg: _geom_segment(seg, norm=0.08)},
    )
    assert r.triggered is True
    assert r.reason == "shoulder_low_with_insufficient_hip_hinge"


def test_insufficient_hip_hinge_triggers_even_when_delta_positive():
    seg = "ecc_50_to_ecc_75"
    r = rule_segment(
        seg,
        hip_result=_hip_rep(anchor="ecc_75", failed=True, delta=0.08),
        knee_result=_knee_rep(anchor="ecc_75", failed=False),
        neck_result=_neck_rep(segment=seg, failed=False),
        spine_geometry={seg: _geom_segment(seg)},
    )
    assert r.triggered is True
    assert r.reason == "shoulder_low_with_insufficient_hip_hinge"


def test_detect_rep_requires_min_segments_no_rescue():
    segs = ("ecc_0_to_ecc_25", "ecc_25_to_ecc_50", "ecc_50_to_ecc_75", "ecc_75_to_ecc_100")
    geom = {
        "ecc_50_to_ecc_75": _geom_segment("ecc_50_to_ecc_75"),
        "ecc_75_to_ecc_100": _geom_segment("ecc_75_to_ecc_100"),
    }
    hip = {
        "anchor_rulings": {
            "ecc_75": {"failed": True, "delta_hip_back": 0.1},
            "ecc_100": {"failed": True, "delta_hip_back": 0.1},
        }
    }
    knee = {"anchor_rulings": {"ecc_75": {"failed": False}, "ecc_100": {"failed": False}}}
    neck = {
        "segment_results": {
            "ecc_50_to_ecc_75": {"failed": False},
            "ecc_75_to_ecc_100": {"failed": False},
        }
    }

    v = detect_spine_flexion(hip_result=hip, knee_result=knee, neck_result=neck, spine_geometry=geom)
    assert v.n_segments_triggered == 2
    assert v.detected is True
    assert len(v.possible_segments) == 0

    v_one = detect_spine_flexion(
        hip_result=hip,
        knee_result=knee,
        neck_result=neck,
        spine_geometry={"ecc_50_to_ecc_75": _geom_segment("ecc_50_to_ecc_75")},
    )
    assert v_one.detected is False


def test_persistent_severity_rules():
    assert _persistent_severity(["grave", "grave"]) == "grave"
    assert _persistent_severity(["grave", "media"]) == "grave"
    assert _persistent_severity(["grave", "leve"]) == "media"
    assert _persistent_severity(["media"]) == "media"
    assert _persistent_severity(["leve", "leve"]) == "media"
    assert _persistent_severity(["leve"]) == "leve"
    assert _persistent_severity([]) == "none"
    assert MIN_SEGMENTS_FOR_REP == 2
    assert HIP_JUSTIFICATION_FACTOR == 0.6
