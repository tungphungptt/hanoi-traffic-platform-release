from pathlib import Path

from traffic_brain import b2b

ART = Path(__file__).resolve().parents[1] / "artifacts"


def test_overview_covers_every_demo():
    data = b2b.overview(ART)
    assert len(data["cameras"]) >= 9
    assert data["totals"]["cameras"] == len(data["cameras"])
    for cam in data["cameras"]:
        assert 0 <= cam["now_ci"] <= 1 and len(cam["series"]) <= b2b.SERIES_POINTS
        assert [h["minutes"] for h in cam["horizons"]] == [5, 10, 15]
        assert cam["level"] in {"free", "busy", "gridlock"}
    assert b2b.overview(ART) is data  # cached


def test_downsample_keeps_short_series():
    assert b2b._downsample([{"t": 0, "ci": 0.1}, {"t": 1, "ci": None}]) == [0.1]
