import sys
from pathlib import Path

import cv2
import numpy as np

from traffic_brain.detector import Detection
from traffic_brain.history import History

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import replay_video  # noqa: E402


class FakeDetector:
    def detect(self, frame, class_ids=None, min_score=0.25):
        return [Detection(2, (10, 10, 40, 30), 0.9), Detection(3, (60, 50, 80, 70), 0.8)]


def _video(path, seconds, fps=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (160, 90))
    rng = np.random.default_rng(0)
    for _ in range(seconds * fps):
        writer.write(rng.integers(0, 255, (90, 160, 3), dtype=np.uint8))
    writer.release()


def test_start_time_is_hanoi_local_or_an_epoch():
    assert replay_video.parse_start("2026-03-02 07:00") == 1772409600.0  # 00:00 UTC
    assert replay_video.parse_start("1772409600") == 1772409600.0


def test_parts_are_laid_end_to_end_on_one_clock(tmp_path):
    a, b = tmp_path / "a.avi", tmp_path / "b.avi"
    _video(a, 20)
    _video(b, 20)
    history = History(tmp_path / "h.sqlite")
    result = replay_video.replay([a, b], FakeDetector(), history, "sim-1", 1_800_000_000.0, name="Mô phỏng")
    assert 36 <= result["written"] <= 40
    rows = [r for r in history.recent("sim-1", 3600, 2e9)]
    times = [r["t"] for r in rows]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert times[0] >= 1_800_000_000.0 and times[-1] <= 1_800_000_040.5
    assert any(t > 1_800_000_020 for t in times)  # the second part continues after the first
    assert result["end_epoch"] > 1_800_000_039
    assert history.cameras()[0]["name"] == "Mô phỏng"
