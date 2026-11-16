import numpy as np
import pytest

from traffic_brain import video_analysis as va
from traffic_brain.detector import Detection, parse_detections, update_tracks


def test_visual_ci_is_bounded_and_monotone():
    assert va.visual_ci(0, 0.0, None) == 0.0
    assert va.visual_ci(40, 0.5, 0.0) == 1.0
    assert va.visual_ci(10, 0.10, None) < va.visual_ci(15, 0.10, None)
    # Stopped traffic is more congested than moving traffic at the same density.
    assert va.visual_ci(10, 0.10, 0.0) > va.visual_ci(10, 0.10, 0.05)


def _sig(seed):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=32 * 18)
    return (s - s.mean()) / s.std()


def test_single_camera_is_one_scene():
    base = _sig(1)
    rng = np.random.default_rng(0)
    sigs = [base + rng.normal(scale=0.2, size=base.size) for _ in range(600)]
    labels = va.segment_scenes([(s - s.mean()) / s.std() for s in sigs], list(range(600)))
    assert set(labels) == {0}


def test_broadcast_is_split_into_cameras_with_gaps():
    a, b = _sig(1), _sig(2)
    sigs, truth = [], []
    for block, (sig, cam) in enumerate([(a, 0), (None, -1), (b, 1), (a, 0), (b, 1)]):
        for _ in range(120 if sig is not None else 30):
            sigs.append(None if sig is None else sig.copy())
            truth.append(cam)
    # A 2-second cut between scenes is a transition, not a camera.
    sigs[150:152] = [_sig(9), _sig(9)]
    labels = va.segment_scenes(sigs, list(range(len(sigs))))
    assert labels[150] == -1 and labels[151] == -1
    assert all(l == -1 for l, t in zip(labels, truth) if t == -1)
    # Same camera keeps the same id after it comes back on air.
    assert labels[10] == labels[300] and labels[200] == labels[400] and labels[10] != labels[200]
    assert sorted(set(labels) - {-1}) == [0, 1]


def test_detector_output_decoding_and_nms():
    # YOLOv8 layout [1, 4 + 80, N]: two overlapping cars and one person (ignored).
    out = np.zeros((84, 200), dtype=np.float32)  # real model: (84, 8400)
    out[:4, 0] = [100, 100, 40, 20]; out[4 + 2, 0] = 0.9
    out[:4, 1] = [102, 101, 40, 20]; out[4 + 2, 1] = 0.8
    out[:4, 2] = [300, 300, 20, 40]; out[4 + 0, 2] = 0.95
    dets = parse_detections(out[None], (640, 640), (320, 640))
    assert len(dets) == 1 and dets[0].class_id == 2
    x1, y1, x2, y2 = dets[0].box
    assert (x1, x2) == pytest.approx((80, 120)) and (y1, y2) == pytest.approx((45, 55))


def test_tracker_reports_motion_of_matched_vehicles_only():
    tracks, next_id, moves = update_tracks([Detection(2, (0, 0, 10, 10))], {}, 1, 0)
    assert moves == [] and next_id == 2
    tracks, next_id, moves = update_tracks([Detection(2, (3, 4, 13, 14)), Detection(3, (500, 500, 510, 510))], tracks, next_id, 1)
    assert moves == [(5.0, 1)] and next_id == 3 and len(tracks) == 2


def test_whole_pipeline_runs_on_a_real_video_file(tmp_path):
    """End to end with the shipped ONNX detector: a synthetic 4-minute street scene at 2 fps."""
    import cv2
    pytest.importorskip("onnxruntime")
    from traffic_brain.detector import Detector
    path = tmp_path / "scene.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (320, 180))
    rng = np.random.default_rng(0)
    background = rng.integers(60, 120, (180, 320, 3), dtype=np.uint8)
    for i in range(480):
        frame = background.copy()
        cv2.rectangle(frame, (10 + i % 200, 80), (60 + i % 200, 110), (30, 30, 200), -1)
        writer.write(frame)
    writer.release()
    progress = []
    result = va.analyze_video(path, Detector(), max_duration_seconds=600, timing_confirmed=True,
                              progress=lambda p, d, t: progress.append(p))
    assert result["mode"] == "single_camera" and len(result["cameras"]) == 1
    assert result["video"]["duration_seconds"] == pytest.approx(240, abs=1)
    assert len(result["timeline"]) == pytest.approx(240, abs=2)  # 1 sample per second
    assert all(0 <= row["raw_ci"] <= 1 for row in result["timeline"])
    assert progress and progress[-1] >= 99
    forecast = result["cameras"][0]["minute_forecast"]
    assert forecast["available"] is False and forecast["reason"] == "video_too_short"
    payload = va.result_payload(result, "scene.avi")
    assert payload["minute_forecast"] is forecast and "series" not in payload["cameras"][0]
    frames = va.capture_evidence(path, Detector(), result["cameras"], tmp_path / "evidence")
    assert 2 <= len(frames) <= 8
    for f in frames:
        image = cv2.imread(str(tmp_path / "evidence" / f["file"]))
        assert image is not None and image.shape[1] <= va.EVIDENCE_WIDTH and f["reason_label"]
    with pytest.raises(va.VideoAnalysisError):
        va.analyze_video(path, Detector(), max_duration_seconds=60)


def test_evidence_picks_level_changes_peak_and_stay_within_limits():
    t = np.arange(0, 1800, 15.0)
    ci = np.clip(0.2 + 0.5 * np.sin(t / 300) ** 2, 0, 1)
    cams = [{"id": 0, "series": {"t": t, "ci": ci}}, {"id": 1, "series": {"t": t, "ci": np.full_like(t, 0.1)}}]
    picks = va.pick_evidence(cams, limit=10, per_camera=6)
    assert len(picks) <= 10 and {p["camera"] for p in picks} == {0, 1}
    assert any(p["reason"] == "up_gridlock" for p in picks if p["camera"] == 0)
    for cam in (0, 1):
        times = sorted(p["t"] for p in picks if p["camera"] == cam)
        assert all(b - a >= 60 for a, b in zip(times, times[1:]))
    assert va.pick_evidence([{"id": 0, "series": {"t": np.array([0.0]), "ci": np.array([np.nan])}}]) == []
