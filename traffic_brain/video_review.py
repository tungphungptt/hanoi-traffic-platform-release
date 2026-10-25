"""Frame-based clip review. Heuristic state labels are not a calibrated traffic forecast."""
from __future__ import annotations
import math
from collections import deque, Counter
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
from aicc.core.preprocessing import preprocess_frame
from aicc.pipeline import _detect, _update_tracks, _predict_window
from traffic_brain.hanoi_video_forecaster import (
    FEATURES as HANOI_FEATURES,
    SEQUENCE_LENGTH as HANOI_SEQUENCE_LENGTH,
    hanoi_video_forecaster,
)

from traffic_brain.traffic_levels import (
    DEFAULT_THRESHOLDS, STATE_LABELS, level_for_ci, validate_thresholds,
)
MAX_DURATION_SECONDS = 20 * 60
MAX_SNAPSHOTS = 36

class VideoReviewError(ValueError):
    pass

def classify_visual_state(vehicle_count: int, occupancy_ratio: float, normalized_speed: float | None = None,
                          thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Explainable fixed-camera baseline using detections, box occupancy and optional motion."""
    if isinstance(vehicle_count, bool) or int(vehicle_count) < 0:
        raise VideoReviewError("vehicle_count phải là số nguyên không âm")
    if not math.isfinite(float(occupancy_ratio)) or not 0 <= float(occupancy_ratio) <= 1:
        raise VideoReviewError("occupancy_ratio phải hữu hạn trong [0,1]")
    if normalized_speed is not None and (not math.isfinite(float(normalized_speed)) or normalized_speed < 0):
        raise VideoReviewError("normalized_speed phải hữu hạn và không âm")
    occ = min(float(occupancy_ratio) / 0.22, 1.0)
    count = min(int(vehicle_count) / 20.0, 1.0)
    if normalized_speed is None:
        ci = 0.62 * occ + 0.38 * count
    else:
        stopped = 1.0 - min(float(normalized_speed) / 0.045, 1.0)
        ci = 0.50 * occ + 0.32 * count + 0.18 * stopped
    ci = max(0.0, min(1.0, ci))
    validate_thresholds(thresholds)
    key, _ = level_for_ci(ci)
    return {
        "ci": ci,
        "raw_ci": ci,
        "state_key": key,
        "state_label": STATE_LABELS[key],
        "raw_state_key": key,
        "raw_state_label": STATE_LABELS[key],
        "smoothed_state_key": key,
        "smoothed_state_label": STATE_LABELS[key],
        "smoothed_ci": ci,
        "vehicle_count": int(vehicle_count),
        "occupancy_ratio": round(float(occupancy_ratio), 5),
        "normalized_speed": None if normalized_speed is None else round(float(normalized_speed), 5),
        "method": "explainable_visual_rule_baseline",
    }


class TemporalStateStabilizer:
    """Causal CI smoother with separate hysteresis metadata for video review.

    Combines:
    1. Causal exponential smoothing of CI (never accesses future samples).
    2. Hysteresis thresholds around state boundaries to eliminate chatter.
    3. Minimum duration for transitions of the separate hysteresis state.

    update() always classifies its numeric EMA output using the official CI
    thresholds. current_state_key is the independent stabilized metadata state.
    """
    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        hysteresis_margin: float = 0.03,
        min_state_duration_seconds: float = 2.0,
        ema_alpha: float = 0.35,
    ) -> None:
        self.thresholds = validate_thresholds(thresholds)
        self.hysteresis = float(hysteresis_margin)
        self.min_state_duration = float(min_state_duration_seconds)
        self.ema_alpha = float(ema_alpha)

        self.smoothed_ci: float | None = None
        self.current_state_key: str | None = None
        self.candidate_state_key: str | None = None
        self.candidate_since: float | None = None
        self.state_since: float | None = None
        self.last_timestamp: float | None = None

    def _determine_candidate_state(self, ci: float) -> str:
        t_busy = self.thresholds["busy"]
        t_severe = self.thresholds["gridlock"]
        h = self.hysteresis
        curr = self.current_state_key
        if curr is None or curr == "free":
            if ci >= t_severe + h:
                return "gridlock"
            return "busy" if ci >= t_busy + h else "free"
        if curr == "busy":
            if ci >= t_severe + h:
                return "gridlock"
            return "free" if ci < t_busy - h else "busy"
        if ci < t_busy - h:
            return "free"
        return "busy" if ci < t_severe - h else "gridlock"

    def update(self, raw_ci: float, timestamp: float) -> tuple[float, str, str]:
        """Return full-precision EMA CI and its official threshold label.

        current_state_key retains the separate hysteresis state for metadata;
        it must never override the classification of the returned numeric CI.
        """
        raw_ci = max(0.0, min(1.0, float(raw_ci)))
        timestamp = float(timestamp)

        if self.smoothed_ci is None:
            self.smoothed_ci = raw_ci
            # First observation: immediate baseline assignment
            cand, _ = level_for_ci(raw_ci)
            self.current_state_key = cand
            self.candidate_state_key = cand
            self.candidate_since = timestamp
            self.state_since = timestamp
            self.last_timestamp = timestamp
            return self.smoothed_ci, cand, STATE_LABELS[cand]

        # Strictly causal EMA smoothing
        self.smoothed_ci = self.ema_alpha * raw_ci + (1.0 - self.ema_alpha) * self.smoothed_ci
        self.smoothed_ci = max(0.0, min(1.0, self.smoothed_ci))

        candidate = self._determine_candidate_state(self.smoothed_ci)

        if candidate == self.current_state_key:
            self.candidate_state_key = candidate
            self.candidate_since = timestamp
        else:
            if candidate != self.candidate_state_key:
                self.candidate_state_key = candidate
                self.candidate_since = timestamp

            sustained = timestamp - (self.candidate_since if self.candidate_since is not None else timestamp)
            if sustained >= self.min_state_duration:
                self.current_state_key = candidate
                self.state_since = timestamp
                self.candidate_since = timestamp

        self.last_timestamp = timestamp
        assert self.current_state_key is not None
        key, label = level_for_ci(self.smoothed_ci)
        return self.smoothed_ci, key, label


def export_evidence_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    """Export evidence or timeline rows to CSV with UTF-8 BOM for Excel compatibility."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "filename",
        "timestamp_seconds",
        "state_label",
        "raw_state_label",
        "state_key",
        "raw_state_key",
        "ci",
        "raw_ci",
        "vehicle_count",
        "occupancy_ratio",
        "model_confidence",
        "aicc_sequence_label",
    ]
    lines = [",".join(f'"{h}"' for h in headers)]
    for row in rows:
        vals = [
            str(row.get("filename", "")),
            f"{float(row.get('timestamp_seconds', 0.0)):.3f}",
            str(row.get("state_label", "")),
            str(row.get("raw_state_label", row.get("state_label", ""))),
            str(row.get("state_key", "")),
            str(row.get("raw_state_key", row.get("state_key", ""))),
            repr(float(row.get('ci', 0.0))),
            repr(float(row.get('raw_ci', row.get('ci', 0.0)))),
            str(int(row.get("vehicle_count", 0))),
            f"{float(row.get('occupancy_ratio', 0.0)):.5f}",
            "" if row.get("model_confidence") is None else f"{float(row['model_confidence']):.4f}",
            str(row.get("aicc_sequence_label") or ""),
        ]
        lines.append(",".join(f'"{v.replace(chr(34), chr(34)+chr(34))}"' for v in vals))
    content = "\ufeff" + "\r\n".join(lines) + "\r\n"
    csv_path.write_text(content, encoding="utf-8")

def project_trend_baseline(timeline: list[dict[str, Any]], *, source_duration_seconds: float,
                           preserve_real_timing: bool) -> dict[str, Any]:
    """Optional trend extrapolation, gated by caller confirmation; never a trained-model forecast."""
    clean = [r for r in timeline if isinstance(r, dict) and isinstance(r.get("timestamp_seconds"), (int,float))
             and isinstance(r.get("ci"), (int,float)) and math.isfinite(float(r["ci"]))]
    clean.sort(key=lambda r: float(r["timestamp_seconds"]))
    unavailable = {"available": False, "mode": "not_available", "confidence": "none", "predictions": []}
    if not preserve_real_timing:
        return {**unavailable, "message": "Chưa xác nhận video giữ đúng thời gian quay; không ngoại suy +5/+10 phút."}
    if source_duration_seconds < 60 or len(clean) < 20:
        return {**unavailable, "mode": "insufficient_history", "message": "Cần ít nhất 60 giây và 20 mẫu để tính trend baseline."}
    end = float(clean[-1]["timestamp_seconds"])
    recent = [r for r in clean if end - float(r["timestamp_seconds"]) <= 120.0]
    if len(recent) < 20:
        return {**unavailable, "mode": "insufficient_history", "message": "Không đủ mẫu gần cuối clip."}
    x = np.array([float(r["timestamp_seconds"]) for r in recent], dtype=np.float64)
    y = np.array([float(r["ci"]) for r in recent], dtype=np.float64)
    x = x - x.mean(); denom = float(np.dot(x, x))
    slope = float(np.dot(x, y - y.mean()) / denom) if denom > 1e-9 else 0.0
    slope = max(-0.25/60.0, min(0.25/60.0, slope))
    predictions = []
    for horizon in (5, 10):
        ci = max(0.0, min(1.0, float(y[-1]) + slope * horizon * 60))
        key, _ = level_for_ci(ci)
        predictions.append({"horizon_minutes": horizon, "ci": ci, "level_key": key, "level": STATE_LABELS[key]})
    return {"available": True, "mode": "explainable_trend_baseline",
            "confidence": "low-to-medium" if source_duration_seconds >= 180 and len(clean) >= 90 else "low",
            "slope_ci_per_minute": round(slope*60, 5), "predictions": predictions,
            "message": "Ngoại suy tuyến tính từ clip, không phải mô hình đã huấn luyện/kiểm định; không dùng để điều hành thực tế."}

def _draw_snapshot(frame: np.ndarray, detections: list[Any], state: dict[str, Any], timestamp: float,
                   model_label: str | None = None, model_confidence: float | None = None) -> np.ndarray:
    img = frame.copy(); h, w = img.shape[:2]
    for det in detections:
        x1,y1,x2,y2 = [int(round(v)) for v in det.box]
        x1=max(0,min(w-1,x1)); x2=max(0,min(w-1,x2)); y1=max(0,min(h-1,y1)); y2=max(0,min(h-1,y2))
        cv2.rectangle(img,(x1,y1),(x2,y2),(53,144,91),2)
    panel = img.copy()
    cv2.rectangle(panel, (0,0), (w, min(h,96)), (18,35,27), -1)
    img = cv2.addWeighted(panel, .72, img, .28, 0)
    
    first_line = f"{state['state_label']} | CI {state['ci']:.2f} | xe: {state['vehicle_count']}"
    if state.get("raw_state_label") and state["raw_state_label"] != state["state_label"]:
        first_line += f" (tho: {state['raw_state_label']})"
        
    second_line = f"t={timestamp:.1f}s | chiem dung: {state['occupancy_ratio']*100:.1f}%"
    conf = model_confidence if model_confidence is not None else state.get("model_confidence")
    if conf is not None:
        second_line += f" | tin cay: {conf*100:.0f}%"
        
    lines = [first_line, second_line]
    if model_label:
        lines.append(f"AICC sequence class (tham khao): {model_label}")
    y = 24
    for line in lines:
        cv2.putText(img, line, (12,y), cv2.FONT_HERSHEY_SIMPLEX, .52, (245,248,245), 2, cv2.LINE_AA); y += 24
    return img

def analyze_video(video_path: Path, bundle: dict[str, Any], output_dir: Path, *,
                  preserve_real_timing: bool = False, sample_fps: float = 1.0,
                  progress_callback: Callable[[int, int, int], None] | None = None,
                  stabilizer: TemporalStateStabilizer | None = None) -> dict[str, Any]:
    """Analyze an uploaded/local clip and save key annotated frames under output_dir/frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release(); raise VideoReviewError("Không mở được video; kiểm tra codec/định dạng.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total / fps if fps > 0 and total > 0 else 0.0
    if not math.isfinite(fps) or fps <= 0 or total <= 0 or duration <= 0:
        cap.release(); raise VideoReviewError("Video cần FPS, số frame và thời lượng hợp lệ.")
    if duration > MAX_DURATION_SECONDS:
        cap.release(); raise VideoReviewError(f"Video dài hơn giới hạn {MAX_DURATION_SECONDS//60} phút.")
    sample_fps = max(.25, min(2.0, float(sample_fps)))
    stride = max(1, int(round(fps / sample_fps)))
    effective_sample_fps = fps / stride
    sequence_len = int(bundle["manifest"].get("sequence_length", 5))
    feature_seq: deque[np.ndarray] = deque(maxlen=sequence_len)
    hanoi_feature_seq: deque[np.ndarray] = deque(maxlen=HANOI_SEQUENCE_LENGTH)
    hanoi_forecast: dict[str, Any] | None = None
    hanoi_rate_supported = math.isclose(effective_sample_fps, 1.0, abs_tol=0.05)
    tracks: dict[int, Any] = {}; next_track_id = 1; sample_index = 0; source_index = 0
    last_snap_time = -1e9; prev_centers: dict[int, tuple[float,float]] = {}; prev_sample_time: float | None = None
    timeline: list[dict[str, Any]] = []; snapshots: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True); frames_dir = output_dir / "frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    class_ids = set(bundle["class_ids"]); last_progress = -1
    state_smoother = stabilizer if stabilizer is not None else TemporalStateStabilizer()
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if source_index % stride == 0:
                timestamp = source_index / fps
                processed = preprocess_frame(frame)
                detections = _detect(bundle["detector"], bundle["detector_input_name"], processed, frame.shape[:2], class_ids)
                old_tracks = tracks
                tracks, next_track_id, associations = _update_tracks(detections, tracks, next_track_id, sample_index)
                speeds = []
                for tid, det, distance, elapsed in associations:
                    if elapsed > 0:
                        speeds.append(distance * effective_sample_fps / elapsed)
                average_speed = float(np.mean(speeds)) if speeds else 0.0
                speed_cv = float(np.std(speeds) / average_speed) if len(speeds) > 1 and average_speed > 0 else 0.0
                area = max(1, frame.shape[0] * frame.shape[1])
                occupancy = min(1.0, sum(max(0.0, (d.box[2]-d.box[0])*(d.box[3]-d.box[1])) for d in detections) / area)
                diagonal = math.hypot(frame.shape[1], frame.shape[0]) or 1.0
                normalized_speed = (float(np.median(speeds)) / diagonal) if len(speeds) >= 2 else None
                raw_state = classify_visual_state(len(detections), occupancy, normalized_speed)
                smoothed_ci, official_key, official_label = state_smoother.update(raw_state["ci"], timestamp)
                stabilized_key = state_smoother.current_state_key

                det_scores = [getattr(d, "score", None) for d in detections if getattr(d, "score", None) is not None]
                mean_det_conf = round(float(np.mean(det_scores)), 4) if det_scores else None

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                class_counts = Counter(detection.class_id for detection in detections)
                hanoi_features = np.asarray(
                    [
                        class_counts.get(2, 0),
                        class_counts.get(3, 0),
                        sum(class_counts.get(class_id, 0) for class_id in (2, 3, 5, 7)),
                        average_speed,
                        occupancy,
                        speed_cv,
                        float(np.mean(gray)),
                        float(np.std(gray)),
                        float(np.mean(hsv[:, :, 1])),
                    ],
                    dtype=np.float32,
                )
                if hanoi_features.shape != (len(HANOI_FEATURES),):
                    raise VideoReviewError("Hanoi feature vector không đúng 9 features.")
                if hanoi_rate_supported:
                    hanoi_feature_seq.append(hanoi_features)
                    if len(hanoi_feature_seq) == HANOI_SEQUENCE_LENGTH:
                        hanoi_forecast = hanoi_video_forecaster.predict(
                            np.stack(hanoi_feature_seq)
                        )

                model_label = None
                model_confidence = None
                feature_seq.append(np.asarray([len(detections), average_speed, speed_cv], dtype=np.float32))
                if len(feature_seq) == sequence_len:
                    try:
                        raw = np.stack(feature_seq).astype(np.float32)
                        scaled = bundle["scaler"].transform(raw).astype(np.float32)
                        tensor = np.ascontiguousarray(scaled[np.newaxis, ...])
                        logits = bundle["predictor"].run([bundle["predictor_output_name"]], {bundle["predictor_input_name"]: tensor})[0]
                        logits_arr = np.asarray(logits, dtype=np.float32).reshape(-1)
                        shifted = logits_arr - np.max(logits_arr)
                        exp_l = np.exp(shifted)
                        denom = float(np.sum(exp_l))
                        probs = exp_l / (denom if denom > 1e-9 else 1.0)
                        label_idx = int(np.argmax(probs))
                        model_confidence = round(float(probs[label_idx]), 4)
                        classes = bundle["manifest"].get("classes", [])
                        model_label = classes[label_idx] if 0 <= label_idx < len(classes) else f"class_{label_idx}"
                    except Exception:
                        model_label = None
                        model_confidence = None

                state = {
                    "ci": smoothed_ci,
                    "state_key": official_key,
                    "state_label": official_label,
                    "raw_ci": raw_state["ci"],
                    "raw_state_key": raw_state["state_key"],
                    "raw_state_label": raw_state["state_label"],
                    "smoothed_ci": smoothed_ci,
                    "smoothed_state_key": stabilized_key,
                    "smoothed_state_label": STATE_LABELS[stabilized_key],
                    "vehicle_count": len(detections),
                    "occupancy_ratio": round(float(occupancy), 5),
                    "normalized_speed": raw_state.get("normalized_speed"),
                    "detector_confidence": mean_det_conf,
                    "model_confidence": model_confidence,
                    "method": "stabilized_visual_rule_baseline",
                }
                row = {
                    "timestamp_seconds": round(timestamp, 3),
                    **state,
                    "avg_speed_pixels_per_second": round(average_speed, 3),
                    "speed_cv": round(speed_cv, 4),
                    "aicc_sequence_label": model_label,
                }
                timeline.append(row)
                # Emit on a stabilized state transition, every 5 seconds, and at the final sampled frame.
                state_changed = len(timeline) == 1 or row["state_key"] != timeline[-2]["state_key"]
                save_snapshot = (timestamp - last_snap_time >= 5.0) or state_changed
                if save_snapshot and len(snapshots) < MAX_SNAPSHOTS:
                    annotated = _draw_snapshot(frame, detections, state, timestamp, model_label, model_confidence)
                    max_width = 1100
                    if annotated.shape[1] > max_width:
                        scale = max_width / annotated.shape[1]
                        annotated = cv2.resize(annotated, (max_width, max(1, int(annotated.shape[0]*scale))), interpolation=cv2.INTER_AREA)
                    name = f"frame_{len(snapshots)+1:03d}.jpg"; destination = frames_dir / name
                    ok_encode, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 84])
                    if ok_encode:
                        destination.write_bytes(encoded.tobytes())
                        snapshots.append({
                            "filename": name,
                            "timestamp_seconds": round(timestamp, 3),
                            "state_key": state["state_key"],
                            "state_label": state["state_label"],
                            "raw_state_key": state["raw_state_key"],
                            "raw_state_label": state["raw_state_label"],
                            "ci": state["ci"],
                            "raw_ci": state["raw_ci"],
                            "vehicle_count": state["vehicle_count"],
                            "occupancy_ratio": state["occupancy_ratio"],
                            "model_confidence": model_confidence,
                            "detector_confidence": mean_det_conf,
                            "aicc_sequence_label": model_label,
                        })
                        last_snap_time = timestamp
                sample_index += 1
            source_index += 1
            pct = min(99, int(source_index*100/max(1,total)))
            if progress_callback and pct != last_progress:
                last_progress = pct; progress_callback(pct, source_index, total)
    finally:
        cap.release()
    if not timeline:
        raise VideoReviewError("Không có frame mẫu để phân tích.")
    states = Counter(row["state_key"] for row in timeline)
    raw_states = Counter(row["raw_state_key"] for row in timeline)
    dominant = states.most_common(1)[0][0]
    forecast = project_trend_baseline(timeline, source_duration_seconds=duration, preserve_real_timing=preserve_real_timing)
    if hanoi_forecast is None:
        if not hanoi_rate_supported:
            hanoi_forecast = {
                "available": False,
                "mode": "unsupported_sampling_rate",
                "predictions": [],
                "message": (
                    "Hanoi LSTM cần video được lấy mẫu ở 1 frame/giây; "
                    f"tốc độ hiện tại là {effective_sample_fps:.2f} frame/giây."
                ),
            }
        else:
            hanoi_forecast = {
                "available": False,
                "mode": "insufficient_history",
                "predictions": [],
                "message": (
                    "Cần ít nhất 30 giây video lấy mẫu 1 frame/giây "
                    "để tạo dự báo LSTM."
                ),
            }
    export_evidence_csv(output_dir / "evidence.csv", snapshots)
    export_evidence_csv(output_dir / "timeline.csv", timeline)
    result = {
        "status": "succeeded", "pipeline": "ONNX detector + Hanoi multi-horizon LSTM + visual rules", "video": {
            "duration_seconds": round(duration,3), "fps": round(fps,3), "width": width, "height": height,
            "frames_total": total, "frames_sampled": sample_index, "sample_fps": round(effective_sample_fps,3),
        },
        "summary": {
            "sampled_frames": len(timeline),
            "saved_frames": len(snapshots),
            "dominant_state": dominant,
            "dominant_state_label": STATE_LABELS[dominant],
            "state_counts": dict(states),
            "raw_state_counts": dict(raw_states),
            "mean_ci": float(np.mean([r["ci"] for r in timeline])),
            "mean_raw_ci": float(np.mean([r["raw_ci"] for r in timeline])),
            "current_state": timeline[-1]["state_label"],
            "current_ci": timeline[-1]["ci"],
            "current_raw_state": timeline[-1]["raw_state_label"],
            "current_raw_ci": timeline[-1]["raw_ci"],
        },
        "frames": snapshots, "timeline": timeline, "trend_forecast": forecast,
        "hanoi_forecast": hanoi_forecast,
        "evidence_csv": "evidence.csv", "timeline_csv": "timeline.csv",
        "limitations": [
            "Khung ảnh được phân loại theo CI sau EMA bằng ba ngưỡng chuẩn; smoothed_state là trạng thái hysteresis riêng, không ghi đè nhãn CI chính thức.",
            "Dự báo LSTM +5/+10/+15 giây là kết quả thử nghiệm từ clip, chưa được hiệu chuẩn hoặc xác nhận trên camera trực tiếp.",
            "Mô hình chỉ hỗ trợ horizon tính bằng giây; không thể diễn giải thành dự báo +5/+10/+15 phút.",
            "Tốc độ trong ảnh là pixel/giây, chưa phải km/h; chưa hiệu chuẩn homography/ROI.",
            "CI và nhãn thị giác là baseline giải thích được, không phải CI dự báo đã hiệu chuẩn.",
            "Một clip đơn lẻ không cho biết mức ùn tắc của tuyến thay thế B; cần telemetry mới ở các tuyến đó.",
        ],
    }
    (output_dir / "result.json").write_text(__import__("json").dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback: progress_callback(100, total, total)
    return result
