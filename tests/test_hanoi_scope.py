from app.hanoi_scope import is_hanoi_coordinate
from app.weather_service import get_weather_report
import pytest


def test_hanoi_centre_is_inside_boundary():
    assert is_hanoi_coordinate(21.0285, 105.8542)


@pytest.mark.parametrize("latitude, longitude", [
    (21.50, 105.80),
    (21.00, 106.15),
    (21.00, 105.05),
    (10.70, 106.70),
    (float("nan"), 105.85),
])
def test_non_hanoi_coordinates_are_outside(latitude, longitude):
    assert not is_hanoi_coordinate(latitude, longitude)


def test_weather_endpoint_service_rejects_outside_hanoi_before_network():
    with pytest.raises(ValueError, match="outside_hanoi_boundary"):
        get_weather_report(21.50, 105.80)
