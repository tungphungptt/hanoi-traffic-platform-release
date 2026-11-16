import csv
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("context_lab", Path(__file__).resolve().parents[1] / "tools" / "context_lab.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)

DAY0 = 1_767_225_600  # 2026-01-01 00:00 UTC; Hanoi local = +7 h


def _export(path, days, rush, rainy_days=()):
    """One camera, 4 daily hours (06-08 and 16-18 local) at 0.2 Hz. With ``rush`` the CI jumps at 07:00 and 17:00 every day."""
    rng = np.random.default_rng(1)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["camera_id", "t_epoch", "t_local", "ci", "precipitation_mm"])
        for d in range(days):
            for hour in (6, 7, 16, 17):
                start = DAY0 + d * 86400 + hour * 3600 - 7 * 3600
                for s in range(0, 3600, 5):
                    # Rain starts 30 minutes into each hour of a rainy day; congestion follows 5 minutes later.
                    rain = d in rainy_days and s >= 1800
                    base = 0.25 + (0.35 if rush and hour in (7, 17) else 0.0) + (0.2 if d in rainy_days and s >= 2100 else 0.0)
                    w.writerow(["cam", start + s, "", float(np.clip(base + rng.normal(0, 0.03), 0, 1)), 1.5 if rain else 0.0])


def test_too_few_days_is_refused_not_guessed(tmp_path):
    _export(tmp_path / "a.csv", 5, True)
    result = cl.evaluate(cl.read_export(tmp_path / "a.csv"), min_days=14)
    assert "Chưa đủ dữ liệu" in result["verdict"] and "+5" not in result


def test_a_daily_pattern_is_found_and_no_pattern_is_not_invented(tmp_path):
    _export(tmp_path / "rush.csv", 16, True)
    found = cl.evaluate(cl.read_export(tmp_path / "rush.csv"), min_days=14)
    assert found["days"] == 16 and found["+15"]["adopt"] and found["+15"]["gain_vs_shipped_formulas"] > 0.02
    _export(tmp_path / "flat.csv", 16, False)
    flat = cl.evaluate(cl.read_export(tmp_path / "flat.csv"), min_days=14)
    assert not any(flat[f"+{h}"]["adopt"] for h in (5, 10, 15))


def test_rain_that_raises_congestion_is_found(tmp_path):
    _export(tmp_path / "rain.csv", 16, False, rainy_days=set(range(0, 16, 2)))
    result = cl.evaluate(cl.read_export(tmp_path / "rain.csv"), min_days=14)
    assert result["+5"]["adopt"] and result["+5"]["gain_vs_shipped_formulas"] > 0.05
