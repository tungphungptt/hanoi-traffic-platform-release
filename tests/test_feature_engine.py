from __future__ import annotations

import pytest
from traffic_brain.feature_engine import FeatureContractError, aggregate_traffic_features, estimate_speed_kmh, project_pixel_to_ground


def manifest():
    return {
        "detector_config": {"class_mapping": {
            "0": {"name": "motorcycle", "pcu_weight": 0.3, "avg_area_m2": 1.6},
            "1": {"name": "car", "pcu_weight": 1.0, "avg_area_m2": 8.5},
            "2": {"name": "bus", "pcu_weight": 2.5, "avg_area_m2": 24.0},
        }},
        "camera_calibration": {"effective_road_length_m": 50.0, "effective_road_area_m2": 300.0},
        "feature_schema": {
            "ordered_input_keys": ["flow.car_count", "flow.motorcycle_count", "flow.pcu_equivalent", "speed.mean_kmh", "speed.cv_ratio", "density.areal_occupancy", "temporal.pcu_lag_1", "temporal.speed_delta_1"],
            "default_fallback_values": {"speed.mean_kmh": 42.0, "speed.cv_ratio": 0.0, "density.areal_occupancy": 0.0, "temporal.pcu_lag_1": 0.0, "temporal.speed_delta_1": 0.0}
        }
    }


def test_homography_projection_and_speed_are_ground_calibrated():
    h = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]]
    assert project_pixel_to_ground((20, 10), h) == (10.0, 5.0)
    # Moves five ground-plane metres in one second: 18 km/h.
    assert estimate_speed_kmh((10, 10), (20, 10), 1.0, h) == 18.0
    assert estimate_speed_kmh((10, 10), (20, 10), 0.0, h) is None


def test_feature_aggregator_maps_classes_pcu_occupancy_and_speed_quality():
    data = manifest()
    observations = [
        {"track_id": "m1", "class_id": 0, "speed_kmh": 12.0},
        {"track_id": "c1", "class_id": 1, "speed_kmh": 24.0},
        {"track_id": "b1", "class_id": 2, "speed_kmh": 18.0},
    ]
    result = aggregate_traffic_features(observations, data, previous_features={"flow.pcu_equivalent": 2.0, "speed.mean_kmh": 20.0})
    features = result["features"]
    assert features["flow.car_count"] == 1.0
    assert features["flow.motorcycle_count"] == 1.0
    assert features["flow.pcu_equivalent"] == pytest.approx(3.8)
    assert features["speed.mean_kmh"] == pytest.approx(18.0)
    assert features["temporal.pcu_lag_1"] == 2.0
    assert features["temporal.speed_delta_1"] == pytest.approx(-2.0)
    assert result["quality"]["ready_for_real_forecast"] is True


def test_feature_engine_reports_fallbacks_instead_of_hiding_missing_calibration():
    result = aggregate_traffic_features([{"track_id": "c1", "class_id": 1}], manifest())
    assert result["features"]["speed.mean_kmh"] == 42.0
    assert "speed.mean_kmh" in result["quality"]["used_fallback_keys"]
    assert result["quality"]["ready_for_real_forecast"] is False


def test_feature_engine_requires_calibration():
    config = manifest()
    config["camera_calibration"]["effective_road_area_m2"] = 0
    with pytest.raises(FeatureContractError):
        aggregate_traffic_features([], config)


def test_homography_rejects_singular_calibration():
    with pytest.raises(FeatureContractError, match="suy biến"):
        project_pixel_to_ground((10, 10), [[1, 2, 3], [2, 4, 6], [0, 0, 1]])


def test_cv_requires_at_least_two_calibrated_speed_samples():
    observations = [{"track_id": "c1", "class_id": 1, "speed_kmh": 20.0}]
    result = aggregate_traffic_features(
        observations,
        manifest(),
        previous_features={"flow.pcu_equivalent": 1.0, "speed.mean_kmh": 20.0},
    )
    assert result["quality"]["ready_for_real_forecast"] is False
    assert "speed.cv_ratio" in result["quality"]["used_fallback_keys"]
