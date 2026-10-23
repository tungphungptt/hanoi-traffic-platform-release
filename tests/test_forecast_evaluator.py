from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_forecast", ROOT / "tools" / "evaluate_forecast.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_evaluator_reports_mae_rmse_per_horizon(tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[f"{kind}_{h}" for h in (5, 10, 15) for kind in ("y_true", "y_pred")])
        writer.writeheader()
        writer.writerow({"y_true_5": .5, "y_pred_5": .4, "y_true_10": .6, "y_pred_10": .6, "y_true_15": .7, "y_pred_15": .9})
        writer.writerow({"y_true_5": .7, "y_pred_5": .7, "y_true_10": .8, "y_pred_10": .7, "y_true_15": .9, "y_pred_15": .8})
    result = MODULE._read_pairs(path)
    m5 = MODULE._metrics(result[5])
    assert m5["n"] == 2
    assert m5["mae"] == pytest.approx(.05)
    assert m5["rmse"] == pytest.approx((.005) ** .5, abs=1e-6)


def test_evaluator_rejects_nonfinite_or_out_of_range_ci(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("y_true_5,y_pred_5,y_true_10,y_pred_10,y_true_15,y_pred_15\n0.5,NaN,0.5,0.5,0.5,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hữu hạn"):
        MODULE._read_pairs(path)


def test_future_data_schemas_are_shipped_and_explicitly_versioned():
    import json
    observation = json.loads((ROOT / "data/observations/observation.schema.json").read_text(encoding="utf-8"))
    target = json.loads((ROOT / "data/labels/forecast-target.schema.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / "data/camera_calibrations/camera-calibration.schema.json").read_text(encoding="utf-8"))
    assert observation["properties"]["schema_version"]["const"] == "hanoi-traffic-observation/1.0"
    assert target["properties"]["ci_t_plus_15m"]["maximum"] == 1
    assert "split_group" in target["required"]
    assert calibration["properties"]["homography_px_to_ground"]["oneOf"][0]["type"] == "null"
    assert "calibration_date" in calibration["required"]
