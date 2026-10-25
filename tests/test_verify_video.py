from __future__ import annotations

from pathlib import Path

import pytest

from tools.verify_video import build_summary, inspect_video


def test_verify_video_summary_does_not_claim_future_forecast():
    result = {
        "status": "success",
        "pipeline": "AICC video inference",
        "model_version": "v1.0-car-only",
        "results_count": 2,
        "video": {"detector_sample_fps": 3.0},
        "results": [
            {"congestion_class": "Thong_thoang"},
            {"congestion_class": "Dong_duc"},
        ],
    }
    summary = build_summary(
        Path("clip.mp4"),
        {"size_bytes": 100, "fps": 30, "frame_count": 300, "width": 640, "height": 360, "duration_seconds": 10},
        result,
    )
    assert summary["class_counts"] == {"Thong_thoang": 1, "Dong_duc": 1}
    assert "+5/+10/+15" in summary["interpretation_limit"]
    assert summary["input_video"] == "clip.mp4"


def test_inspect_video_rejects_missing_path(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        inspect_video(tmp_path / "missing.mp4", max_duration_seconds=1200, max_bytes=1000000)


def test_inspect_video_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "clip.txt"
    path.write_text("not a video")
    with pytest.raises(ValueError, match="Unsupported"):
        inspect_video(path, max_duration_seconds=1200, max_bytes=1000000)
