from __future__ import annotations

import numpy as np
import pytest

from traffic_brain.hanoi_video_forecaster import HanoiVideoForecaster


class IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


class FixedSession:
    def run(self, _outputs: list[str], _inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.asarray([[-1.0, 1.0, -1.0]], dtype=np.float32)]


def test_hanoi_forecaster_maps_three_logits_to_seconds_and_probabilities(monkeypatch):
    forecaster = HanoiVideoForecaster()
    monkeypatch.setattr(
        forecaster,
        "load",
        lambda: {
            "scaler": IdentityScaler(),
            "session": FixedSession(),
            "input_name": "features",
            "output_name": "logits",
            "manifest": {"version": "v2.0-hanoi-forecast"},
        },
    )

    result = forecaster.predict(np.zeros((30, 9), dtype=np.float32))

    assert [row["horizon_seconds"] for row in result["predictions"]] == [5, 10, 15]
    assert [row["predicted_class"] for row in result["predictions"]] == [
        "Thong_thoang",
        "Tac_duong",
        "Thong_thoang",
    ]
    assert result["probability_calibrated"] is False


def test_hanoi_forecaster_rejects_wrong_sequence_shape():
    forecaster = HanoiVideoForecaster()
    with pytest.raises(ValueError, match="shape"):
        forecaster.predict(np.zeros((29, 9), dtype=np.float32))
