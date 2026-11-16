import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("model_lab", Path(__file__).resolve().parents[1] / "tools" / "model_lab.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


def test_bootstrap_adopts_only_a_gain_that_holds_across_videos():
    rng = np.random.default_rng(0)
    steady = {f"v{i}": (10.0, 9.0, 100) for i in range(7)}           # 10% better in every video
    point, lo, hi = ml.bootstrap_gain(steady, rng)
    assert point == pytest.approx(0.1) and lo > 0
    mixed = {"a": (10.0, 5.0, 100), "b": (10.0, 14.0, 100), "c": (10.0, 10.0, 100), "d": (10.0, 13.0, 100)}
    _, lo, _ = ml.bootstrap_gain(mixed, rng)
    assert lo < 0  # better in one video, worse in two: not adopted


def test_calibration_is_nested_by_video_and_reports_brier():
    rng = np.random.default_rng(1)
    rows = []
    for src in "ABCD":
        for _ in range(300):
            p = float(rng.uniform(0, 1))
            rows.append({"src": src, "h": 5, "p": p, "y": int(rng.uniform() < p * p), "now": 0.3, "pred": 0.3})
    out = ml.calibration(rows)["+5"]
    assert out["forecasts"] == 1200
    assert out["brier_calibrated"] <= out["brier_raw"] + 0.01  # p was overconfident (true rate p^2): calibration should not hurt
