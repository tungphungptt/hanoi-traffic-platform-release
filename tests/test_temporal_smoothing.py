from __future__ import annotations

import csv
import io
import json
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

import app.main as main
from aicc.pipeline import AICCSequenceStabilizer
from traffic_brain.video_review import (
    DEFAULT_THRESHOLDS,
    STATE_LABELS,
    TemporalStateStabilizer,
    VideoReviewError,
    analyze_video,
    classify_visual_state,
    export_evidence_csv,
)

from traffic_brain.traffic_levels import level_for_ci

client = TestClient(main.app)


def test_stabilizer_strictly_causal_no_future_lookahead():
    """Verify stabilizer processes observations causally step-by-step."""
    stabilizer = TemporalStateStabilizer(hysteresis_margin=0.03, min_state_duration_seconds=2.0)
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
    raw_cis = [0.15, 0.18, 0.20, 0.22, 0.25]
    
    results = []
    for t, ci in zip(timestamps, raw_cis):
        smoothed_ci, state_key, state_label = stabilizer.update(ci, t)
        results.append((smoothed_ci, state_key, state_label))
        # Ensure CI is always within [0, 1] unit
        assert 0.0 <= smoothed_ci <= 1.0
        assert state_key == "free"
        assert state_label == STATE_LABELS["free"]
    
    assert len(results) == 5
    assert stabilizer.last_timestamp == 4.0


def test_stabilizer_suppresses_boundary_jitter_between_seconds():
    """Hysteresis metadata stays stable while official labels follow the EMA CI."""
    stabilizer = TemporalStateStabilizer(
        thresholds={"busy": 0.30, "heavy": 0.55, "gridlock": 0.55},
        hysteresis_margin=0.03,
        min_state_duration_seconds=2.0,
        ema_alpha=0.35,
    )
    # Oscillating around 0.30 boundary: 0.28, 0.32, 0.28, 0.32, 0.29, 0.31, 0.27, 0.31
    noisy_stream = [0.28, 0.32, 0.28, 0.32, 0.29, 0.31, 0.27, 0.31, 0.28, 0.30]
    
    states = []
    for idx, raw_ci in enumerate(noisy_stream):
        smoothed_ci, key, label = stabilizer.update(raw_ci, float(idx))
        states.append(stabilizer.current_state_key)
        assert (key, label) == level_for_ci(smoothed_ci)
        assert 0.0 <= smoothed_ci <= 1.0
    
    # Must remain stable in "free", zero flipping to "busy"
    assert all(k == "free" for k in states), f"States flipped unexpectedly: {states}"


def test_stabilizer_hysteresis_upward_and_downward_thresholds():
    """Verify upward transition requires threshold + margin, and downward requires threshold - margin."""
    stabilizer = TemporalStateStabilizer(
        thresholds={"busy": 0.30, "heavy": 0.55, "gridlock": 0.55},
        hysteresis_margin=0.04,
        min_state_duration_seconds=1.0,
        ema_alpha=1.0,  # disable EMA for direct threshold test
    )
    def stabilized_at(ci, timestamp):
        smoothed_ci, key, label = stabilizer.update(ci, timestamp)
        assert (key, label) == level_for_ci(smoothed_ci)
        return stabilizer.current_state_key

    # Start at free
    k = stabilized_at(0.20, 0.0)
    assert k == "free"
    
    # 0.32 is above 0.30 but below 0.30 + 0.04 (0.34) -> stays free
    k = stabilized_at(0.32, 1.0)
    assert k == "free"
    k = stabilized_at(0.32, 2.5)
    assert k == "free"
    
    # 0.36 is above 0.34 -> candidate becomes busy, holds for duration
    k = stabilized_at(0.36, 3.0)
    assert k == "free"  # first sample of candidate
    k = stabilized_at(0.36, 4.2)
    assert k == "busy"  # sustained past 1.0s -> commits to busy
    
    # Now in busy: drops to 0.28 (below 0.30, but above 0.30 - 0.04 = 0.26) -> stays busy!
    k = stabilized_at(0.28, 5.0)
    assert k == "busy"
    k = stabilized_at(0.28, 6.5)
    assert k == "busy"
    
    # Drops to 0.24 (below 0.26) and sustained -> transitions back to free
    k = stabilized_at(0.24, 7.0)
    assert k == "busy"  # candidate free, not sustained yet
    k = stabilized_at(0.24, 8.2)
    assert k == "free"  # sustained -> commits to free


def test_stabilizer_minimum_state_duration_rejects_single_second_spikes():
    """A transient spike cannot change hysteresis metadata or override the official CI label."""
    stabilizer = TemporalStateStabilizer(
        min_state_duration_seconds=2.0,
        ema_alpha=0.5,
    )
    # Steady free
    for t in range(5):
        stabilizer.update(0.15, float(t))
    
    # Spike at t=5.0
    _, key_spike, label_spike = stabilizer.update(0.50, 5.0)
    assert key_spike == "busy"  # EMA CI is 0.325; official classification must follow it.
    assert label_spike == "Đông đúc"
    assert stabilizer.current_state_key == "free"  # Separate hysteresis rejects the spike.
    
    # Immediately returns to low congestion at t=6.0
    _, key_next, _ = stabilizer.update(0.15, 6.0)
    assert key_next == "free"
    assert stabilizer.current_state_key == "free"


def test_classify_visual_state_distinguishes_raw_and_smoothed_fields():
    """Verify classify_visual_state returns both raw and smoothed fields with proper units."""
    res = classify_visual_state(vehicle_count=5, occupancy_ratio=0.08)
    assert "raw_ci" in res
    assert "smoothed_ci" in res
    assert "raw_state_key" in res
    assert "smoothed_state_key" in res
    assert "raw_state_label" in res
    assert "smoothed_state_label" in res
    assert res["ci"] == res["smoothed_ci"]
    assert res["state_key"] == res["smoothed_state_key"]
    assert res["state_label"] == res["smoothed_state_label"]
    assert 0.0 <= res["ci"] <= 1.0
    assert 0.0 <= res["raw_ci"] <= 1.0
    assert res["vehicle_count"] == 5
    assert res["occupancy_ratio"] == 0.08


def test_aicc_sequence_stabilizer_eliminates_boundary_toggle():
    """Verify AICC sequence stabilizer handles boundary probabilities stably."""
    stabilizer = AICCSequenceStabilizer(hysteresis_margin=0.08, min_consecutive_windows=2, ema_alpha=0.40)
    # Alternating probabilities around 0.50:
    # probs are [P(Thong_thoang), P(Dong_duc)]
    toggles = [
        (0, np.array([0.55, 0.45])),
        (1, np.array([0.48, 0.52])),
        (0, np.array([0.51, 0.49])),
        (1, np.array([0.47, 0.53])),
        (0, np.array([0.52, 0.48])),
    ]
    outputs = []
    for raw_id, probs in toggles:
        smoothed_id, prob_busy = stabilizer.update(raw_id, probs)
        outputs.append(smoothed_id)
        assert 0.0 <= prob_busy <= 1.0
    
    # Should not toggle back and forth
    assert all(out == 0 for out in outputs), f"AICC stabilizer flipped prematurely: {outputs}"


def test_evidence_csv_export_format_and_encoding(tmp_path):
    """Verify evidence CSV has expected headers, UTF-8 BOM, and valid values."""
    csv_file = tmp_path / "evidence.csv"
    snapshots = [
        {
            "filename": "frame_001.jpg",
            "timestamp_seconds": 0.0,
            "state_label": "Thông thoáng",
            "raw_state_label": "Thông thoáng",
            "state_key": "free",
            "raw_state_key": "free",
            "ci": 0.1500,
            "raw_ci": 0.1500,
            "vehicle_count": 3,
            "occupancy_ratio": 0.04500,
            "model_confidence": 0.8520,
            "aicc_sequence_label": "Thong_thoang",
        },
        {
            "filename": "frame_002.jpg",
            "timestamp_seconds": 5.0,
            "state_label": "Đông đúc",
            "raw_state_label": "Đông đúc",
            "state_key": "busy",
            "raw_state_key": "busy",
            "ci": 0.4200,
            "raw_ci": 0.4500,
            "vehicle_count": 8,
            "occupancy_ratio": 0.12000,
            "model_confidence": 0.7800,
            "aicc_sequence_label": "Dong_duc",
        },
    ]
    export_evidence_csv(csv_file, snapshots)
    assert csv_file.is_file()
    raw_bytes = csv_file.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf"), "CSV must have UTF-8 BOM for Excel"
    
    content = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["filename"] == "frame_001.jpg"
    assert rows[0]["state_label"] == "Thông thoáng"
    assert float(rows[0]["ci"]) == 0.15
    assert int(rows[0]["vehicle_count"]) == 3
    assert float(rows[1]["ci"]) == 0.42
    assert rows[1]["state_label"] == "Đông đúc"


def test_video_review_csv_endpoint_serves_generated_csv(monkeypatch, tmp_path):
    """Verify the API CSV download endpoint correctly returns the generated evidence CSV."""
    job_id = "e" * 32
    monkeypatch.setattr(main, "VIDEO_REVIEW_ROOT", tmp_path)
    monkeypatch.setattr(main, "VIDEO_REVIEW_JOBS", {job_id: {"status": "succeeded", "finished_epoch": __import__("time").time()}})
    job_dir = main.VIDEO_REVIEW_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    evidence_csv = job_dir / "evidence.csv"
    evidence_csv.write_text('\ufeff"filename","state_label"\r\n"frame_001.jpg","Thông thoáng"\r\n', encoding="utf-8")
    
    try:
        response = client.get(f"/api/scenarios/video-reviews/{job_id}/csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "Thông thoáng" in response.text
    finally:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)


def test_window_accounting_and_sequence_warmup():
    """Verify that a sequence model of length K emits exactly N - (K - 1) predictions for N windows."""
    from collections import deque
    seq_len = 5
    buffer: deque[int] = deque(maxlen=seq_len)
    
    total_windows = 300
    emitted = []
    for w in range(1, total_windows + 1):
        buffer.append(w)
        if len(buffer) == seq_len:
            emitted.append(w)
            
    # For 300 windows and sequence_length 5, exactly 300 - 4 = 296 predictions are emitted.
    # The first 4 windows (w=1, 2, 3, 4) do not emit because length < 5.
    assert len(emitted) == 296
    assert emitted[0] == 5
    assert emitted[-1] == 300


def test_stabilizer_metrics_and_end_of_video_stability():
    """Verify confidence, probability bounds, and stable state retention at sequence end."""
    stabilizer = AICCSequenceStabilizer(hysteresis_margin=0.08, min_consecutive_windows=2, ema_alpha=0.40)
    
    # Run 50 windows of low-congestion probabilities (prob_busy ~ 0.20-0.30)
    for i in range(50):
        prob = 0.22 + 0.05 * (i % 3)
        probs = np.array([1.0 - prob, prob])
        smoothed_id, smoothed_prob = stabilizer.update(0, probs)
        assert smoothed_id == 0
        assert 0.0 <= smoothed_prob <= 1.0
        
    # At the end of video, candidate count should be 0 because candidate matches current state
    assert stabilizer.candidate_count == 0
    assert stabilizer.current_label_id == 0


def test_cadence_accounting_and_residual_tail_window():
    """Verify exact timestamp cadence and residual tail window accounting.

    For 3000 frames at 10 FPS with stride 3 (sample_fps=3.333):
      - 1000 samples are taken (frames 0, 3, ..., 2997).
      - window_samples = round(3.333 * 1.0) = 3.
      - Full windows = 1000 // 3 = 333 (each 3 samples = 9 frames = 0.9s cadence).
      - Window 333 contains samples [996, 997, 998] (source frames 2988, 2991, 2994)
        and seals at boundary frame 2997 (timestamp 299.7s).
      - Residual sample count = 1000 % 3 = 1 sample (sample 999 = source frame 2997).
      - Window 334 (tail) contains sample 999 and seals at end-of-video frame 2999 (timestamp 299.9s).
      - Interval between window 333 boundary (299.7s) and tail window boundary (299.9s) is exactly 0.2s.
      - With sequence_length = 5, total predictions emitted is (333 + 1) - 4 = 330.
    """
    total_frames = 3000
    fps = 10.0
    stride = 3
    window_samples = 3
    seq_len = 5

    sampled_frames = list(range(0, total_frames, stride))
    assert len(sampled_frames) == 1000

    from collections import deque
    feature_seq: deque[int] = deque(maxlen=seq_len)
    windows_emitted = []

    window_start = 0
    for sample_idx, frame_idx in enumerate(sampled_frames):
        if sample_idx > window_start and sample_idx - window_start >= window_samples:
            feature_seq.append(sample_idx)
            if len(feature_seq) == seq_len:
                windows_emitted.append(round(frame_idx / fps, 3))
            window_start = sample_idx

    remaining = len(sampled_frames) - window_start
    assert remaining == 1
    if remaining > 0:
        feature_seq.append(len(sampled_frames))
        if len(feature_seq) == seq_len:
            windows_emitted.append(round((total_frames - 1) / fps, 3))

    assert len(windows_emitted) == 330
    assert windows_emitted[0] == 4.5
    assert windows_emitted[-2] == 299.7
    assert windows_emitted[-1] == 299.9

    cadences = [round(windows_emitted[i] - windows_emitted[i - 1], 3) for i in range(1, len(windows_emitted))]
    from collections import Counter
    counts = Counter(cadences)
    assert counts[0.9] == 328
    assert counts[0.2] == 1


def test_window_source_frame_indices_and_boundary_mapping():
    """Verify exact source frame indices contained in windows vs trigger boundaries.
    
    Distinguishes:
      - Frame timestamp: t_frame = frame_index / fps.
      - Window boundary timestamp: t_boundary = boundary_frame / fps when window finishes.
      - End-of-video timestamp: t_eof_frame = (total_frames - 1) / fps vs duration = total_frames / fps.
    """
    total_frames = 3000
    fps = 10.0
    stride = 3
    window_samples = 3

    source_frame_index = 0
    sample_index = 0
    window_start_sample = 0

    windows = []
    current_frames = []

    for f in range(total_frames):
        if f % stride == 0:
            if sample_index > window_start_sample and sample_index - window_start_sample >= window_samples:
                windows.append({
                    "window_index": len(windows) + 1,
                    "contained_source_frames": list(current_frames),
                    "boundary_frame_index": f,
                    "boundary_timestamp_seconds": round(f / fps, 3),
                })
                window_start_sample += (sample_index - window_start_sample)
                current_frames = []
            current_frames.append(f)
            sample_index += 1
        source_frame_index += 1

    remaining = sample_index - window_start_sample
    if remaining > 0:
        windows.append({
            "window_index": len(windows) + 1,
            "contained_source_frames": list(current_frames),
            "boundary_frame_index": source_frame_index - 1,
            "boundary_timestamp_seconds": round((source_frame_index - 1) / fps, 3),
        })

    assert len(windows) == 334

    # Window 1: contains frames 0, 3, 6; boundary is frame 9 (0.9s)
    assert windows[0]["contained_source_frames"] == [0, 3, 6]
    assert windows[0]["boundary_frame_index"] == 9
    assert windows[0]["boundary_timestamp_seconds"] == 0.9

    # Window 333 (last regular full window):
    # Contains sampled frames 2988, 2991, 2994 (sample indices 996, 997, 998).
    # Boundary frame that seals this window is frame 2997 (299.7s).
    assert windows[332]["contained_source_frames"] == [2988, 2991, 2994]
    assert windows[332]["boundary_frame_index"] == 2997
    assert windows[332]["boundary_timestamp_seconds"] == 299.7

    # Window 334 (residual tail window):
    # Contains sampled frame 2997 (sample index 999).
    # Sealed at EOF frame 2999 (299.9s).
    assert windows[333]["contained_source_frames"] == [2997]
    assert windows[333]["boundary_frame_index"] == 2999
    assert windows[333]["boundary_timestamp_seconds"] == 299.9

    # Distinction check:
    # Frame 2997 is NOT inside window 333; it is inside the tail window 334.
    assert 2997 not in windows[332]["contained_source_frames"]
    assert 2997 in windows[333]["contained_source_frames"]

    # End-of-video timestamp distinction:
    # Last frame index is 2999 -> 299.9s.
    # Total video duration is 3000 / 10.0 = 300.0s.
    assert round(2999 / fps, 3) == 299.9
    assert round(total_frames / fps, 3) == 300.0


def test_verify_video_summary_captures_stride_and_sampling():
    from tools.verify_video import build_summary
    result = {
        "status": "success",
        "pipeline": "AICC video inference",
        "model_version": "v1.0-car-only",
        "results_count": 1,
        "video": {
            "detector_sample_fps": 3.333,
            "frame_stride": 3,
            "frames_sampled": 1000,
            "frames_processed": 3000,
        },
        "results": [{"congestion_class": "Thong_thoang"}],
    }
    meta = {"size_bytes": 100, "fps": 10.0, "frame_count": 3000, "width": 1280, "height": 960, "duration_seconds": 300.0}
    summary = build_summary(Path("test.mp4"), meta, result)
    assert summary["sample_fps"] == 3.333
    assert summary["frame_stride"] == 3
    assert summary["frames_sampled"] == 1000


