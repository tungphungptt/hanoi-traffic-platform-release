import math
import sys
from pathlib import Path

import numpy as np
import pytest

from traffic_brain.detector import Detection
from traffic_brain.speed import CalibrationError, GroundPlane, SpeedTracker, summarise, track_speed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

# A road 7 m wide and 40 m long seen in perspective (nearer end wider and lower in the image).
IMAGE = [[300, 600], [700, 600], [560, 200], [440, 200]]
WORLD = [[0, 0], [7, 0], [7, 40], [0, 40]]


def _plane():
    return GroundPlane(IMAGE, WORLD)


def _box_at(plane, gx, gy, w=60):
    """Box whose bottom centre is the pixel of ground point (gx, gy)."""
    px, py = plane.to_pixels(np.array([[gx, gy]]))[0]
    return Detection(2, (px - w / 2, py - w, px + w / 2, py))


def _drive(plane, speed_mps, seconds=6.0, fps=4.0, noise_px=0.0, x=3.5, start=2.0, seed=0):
    rng = np.random.default_rng(seed)
    tracker = SpeedTracker(plane)
    for i in range(int(seconds * fps)):
        t = i / fps
        det = _box_at(plane, x, start + speed_mps * t)
        if noise_px:
            b = np.array(det.box) + rng.normal(0, noise_px, 4)
            det = Detection(2, tuple(b))
        tracker.update(t, [det])
    return tracker.finish()


def test_calibration_round_trip_and_rejects_bad_points():
    plane = _plane()
    assert plane.to_ground(np.array(IMAGE, dtype=float)) == pytest.approx(np.array(WORLD, dtype=float), abs=1e-3)
    assert plane.to_pixels(plane.to_ground(np.array([[500.0, 400.0]]))) == pytest.approx(np.array([[500.0, 400.0]]), abs=1e-2)
    with pytest.raises(CalibrationError):
        GroundPlane(IMAGE[:3], WORLD[:3])
    with pytest.raises(CalibrationError):
        GroundPlane([[0, 0], [1, 1], [2, 2], [3, 3]], WORLD)


def test_known_speed_is_recovered_in_km_per_hour():
    tracks = _drive(_plane(), speed_mps=10.0, seconds=3.5, noise_px=0.0)  # 36 km/h, stays inside the 40 m of road
    assert len(tracks) == 1
    assert tracks[0].speed_kmh == pytest.approx(36.0, rel=0.03)
    assert tracks[0].distance_m == pytest.approx(10.0 * (3.5 - 0.25), rel=0.05)


def test_pixel_noise_does_not_break_the_estimate_and_a_parked_car_reads_near_zero():
    plane = _plane()
    moving = _drive(plane, 8.0, noise_px=1.0)
    assert moving[0].speed_kmh == pytest.approx(28.8, rel=0.15)
    parked = _drive(plane, 0.0, noise_px=1.0, seconds=8.0)
    assert parked[0].speed_kmh < 4.0


def test_far_and_near_vehicles_get_the_same_speed_despite_perspective():
    plane = _plane()
    near = _drive(plane, 6.0, start=1.0, seconds=3.0)[0].speed_kmh
    far = _drive(plane, 6.0, start=22.0, seconds=3.0)[0].speed_kmh
    assert near == pytest.approx(21.6, rel=0.05) and far == pytest.approx(21.6, rel=0.08)


def test_two_vehicles_keep_their_own_speeds_and_outside_the_quadrilateral_is_ignored():
    plane = _plane()
    tracker = SpeedTracker(plane)
    for i in range(24):
        t = i / 4
        tracker.update(t, [_box_at(plane, 1.5, 2 + 12.0 * t), _box_at(plane, 5.5, 2 + 6.0 * t),
                           Detection(2, (0, 0, 50, 50))])  # top-left of the image: outside the road quad
    speeds = sorted(tr.speed_kmh for tr in tracker.finish())
    assert len(speeds) == 2
    assert speeds[0] == pytest.approx(21.6, rel=0.05) and speeds[1] == pytest.approx(43.2, rel=0.05)


def test_short_tracks_are_not_reported_and_windows_summarise_per_vehicle():
    assert track_speed([0, 0.25, 0.5], [[0, 0], [1, 0], [2, 0]]) is None
    plane = _plane()
    tracks = _drive(plane, 10.0, seconds=3.5) + _drive(plane, 0.0, seconds=8.0)
    rows = summarise(tracks, window_seconds=10.0, duration=10.0)
    assert rows[0]["vehicles"] == 2 and rows[0]["stopped_share"] == 0.5
    assert rows[0]["p85_kmh"] > rows[0]["median_kmh"] > 0
