from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline import status
from src.pipeline.config import FullAnalysisConfig
from src.pipeline.contracts import (
    validate_anchors_contract,
    validate_clean_npz_contract,
    validate_raw_npz_contract,
    validate_segmentation_contract,
)
from src.pipeline.full_analysis import run_full_analysis
from src.pipeline.validation import validate_video_input


def _ok_video_validation(_path: str | Path):
    from src.pipeline.results import ValidationResult

    return ValidationResult(
        ok=True,
        status=status.OK,
        user_message="ok",
        details={"fps": 30.0, "frame_count": 60, "duration_s": 2.0},
    )


def _pose_raw_stub() -> dict:
    kps = np.ones((10, 17, 2), dtype=np.float64)
    score = np.ones((10, 17), dtype=np.float64)
    return {"kps_xy": kps, "kps_score": score, "fps": 30.0, "frame_idx": np.arange(10), "meta": {}}


def test_validate_video_input_missing_file():
    res = validate_video_input("/tmp/this_file_should_not_exist_123456789.mp4")
    assert res.ok is False
    assert res.status == status.INVALID_INPUT


def test_validate_video_input_non_video_file(tmp_path: Path):
    p = tmp_path / "not_video.txt"
    p.write_text("hello", encoding="utf-8")
    res = validate_video_input(p)
    assert res.ok is False
    assert res.status in {status.INVALID_INPUT, status.VIDEO_DECODE_ERROR}


def test_validate_raw_npz_contract_missing_keys(tmp_path: Path):
    p = tmp_path / "raw_bad.npz"
    np.savez(str(p), only_this=np.array([1, 2, 3]))
    res = validate_raw_npz_contract(p)
    assert res.ok is False


def test_validate_clean_npz_contract_missing_keys(tmp_path: Path):
    p = tmp_path / "clean_bad.npz"
    np.savez(str(p), foo=np.array([1, 2, 3]))
    res = validate_clean_npz_contract(p)
    assert res.ok is False


def test_validate_segmentation_contract_no_reps():
    res = validate_segmentation_contract({"reps": []})
    assert res.ok is False
    assert res.status == status.NO_REPS_DETECTED


def test_validate_anchors_contract_empty():
    res = validate_anchors_contract({"anchor_pairs": {"paired_repetitions": []}})
    assert res.ok is False
    assert res.status == status.INVALID_ANCHORS


def test_pipeline_returns_no_reps_detected(monkeypatch, tmp_path: Path):
    video = tmp_path / "fake.mp4"
    video.write_text("placeholder", encoding="utf-8")
    cfg = FullAnalysisConfig(video_path=video)

    monkeypatch.setattr("src.pipeline.full_analysis.validate_video_input", _ok_video_validation)
    monkeypatch.setattr("src.pipeline.full_analysis.extract_video_pose", lambda **_: _pose_raw_stub())
    monkeypatch.setattr("src.pipeline.full_analysis.estimate_subject_facing_from_pose", lambda *_args, **_kw: {"facing": "right"})
    monkeypatch.setattr("src.pipeline.full_analysis.ensure_video_facing", lambda *args, **kwargs: (args[0], False))
    monkeypatch.setattr("src.pipeline.full_analysis.clean_pose_data", lambda _raw: {"kps_xy_clean": _pose_raw_stub()["kps_xy"], "fps": 30.0})
    monkeypatch.setattr("src.pipeline.full_analysis.run_rdl_segmentation_from_pose", lambda *_args, **_kw: {"video_id": "x", "reps": []})

    runtime = run_full_analysis(cfg)
    assert runtime["pipeline_result"]["status"] == status.NO_REPS_DETECTED
    assert runtime["pipeline_result"]["ok"] is False


def test_pipeline_returns_insufficient_pose_quality(monkeypatch, tmp_path: Path):
    video = tmp_path / "fake.mp4"
    video.write_text("placeholder", encoding="utf-8")
    cfg = FullAnalysisConfig(video_path=video)

    monkeypatch.setattr("src.pipeline.full_analysis.validate_video_input", _ok_video_validation)
    monkeypatch.setattr("src.pipeline.full_analysis.extract_video_pose", lambda **_: _pose_raw_stub())
    monkeypatch.setattr("src.pipeline.full_analysis.estimate_subject_facing_from_pose", lambda *_args, **_kw: {"facing": "right"})
    monkeypatch.setattr("src.pipeline.full_analysis.ensure_video_facing", lambda *args, **kwargs: (args[0], False))
    monkeypatch.setattr("src.pipeline.full_analysis.clean_pose_data", lambda _raw: {"kps_xy_clean": np.full((10, 17, 2), np.nan), "fps": 30.0})

    runtime = run_full_analysis(cfg)
    assert runtime["pipeline_result"]["status"] == status.INSUFFICIENT_POSE_QUALITY
    assert runtime["pipeline_result"]["ok"] is False

