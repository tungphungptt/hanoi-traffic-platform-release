#!/usr/bin/env python3
"""Run Track 1 Vehicle Counting with MLP Movement Classifier and instrument fallback statistics."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicc.track1_counting import (
    CAM5_MOVEMENTS,
    Track1CountingEvent,
    Track1CountingPipeline,
    Track1MovementMatcher,
    Track1VehicleTracker,
    export_track1_csv,
    extract_trajectory_features,
    load_roi_polygon,
)
from tools.evaluate_heldout_track1 import evaluate_heldout_slice, load_csv_rows


class InstrumentedMLPMatcher(Track1MovementMatcher):
    def __init__(self, movements, ml_model_path):
        super().__init__(movements, ml_model_path=ml_model_path, classifier_type="mlp")
        self.stats = {
            "total_calls": 0,
            "mlp_accepted": 0,
            "fallback_short_track": 0,
            "fallback_low_confidence": 0,
            "fallback_out_of_range": 0,
            "fallback_exception": 0,
            "call_details": [],
        }

    def match(self, track: Track1VehicleTracker) -> int:
        self.stats["total_calls"] += 1
        heuristic_res = self.match_heuristic(track)

        if len(track.points) < 2:
            self.stats["fallback_short_track"] += 1
            self.stats["call_details"].append({
                "track_id": track.track_id,
                "points_len": len(track.points),
                "decision": "fallback_short_track",
                "final_movement": heuristic_res,
                "mlp_pred": None,
                "mlp_conf": None,
                "heuristic_pred": heuristic_res,
            })
            return heuristic_res

        try:
            feats = extract_trajectory_features(track)
            x_arr = np.array([feats], dtype=np.float32)
            if self.ml_scaler is not None:
                x_arr = self.ml_scaler.transform(x_arr)

            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(x_arr)[0]
                c_idx = int(np.argmax(probs))
                conf = float(probs[c_idx])
                pred_cls = int(self.ml_model.classes_[c_idx])

                if conf < 0.35:
                    self.stats["fallback_low_confidence"] += 1
                    self.stats["call_details"].append({
                        "track_id": track.track_id,
                        "points_len": len(track.points),
                        "decision": "fallback_low_confidence",
                        "final_movement": heuristic_res,
                        "mlp_pred": pred_cls,
                        "mlp_conf": round(conf, 4),
                        "heuristic_pred": heuristic_res,
                    })
                    return heuristic_res

                if not (1 <= pred_cls <= 12):
                    self.stats["fallback_out_of_range"] += 1
                    self.stats["call_details"].append({
                        "track_id": track.track_id,
                        "points_len": len(track.points),
                        "decision": "fallback_out_of_range",
                        "final_movement": heuristic_res,
                        "mlp_pred": pred_cls,
                        "mlp_conf": round(conf, 4),
                        "heuristic_pred": heuristic_res,
                    })
                    return heuristic_res

                # Accepted by MLP
                self.stats["mlp_accepted"] += 1
                self.stats["call_details"].append({
                    "track_id": track.track_id,
                    "points_len": len(track.points),
                    "decision": "mlp_accepted",
                    "final_movement": pred_cls,
                    "mlp_pred": pred_cls,
                    "mlp_conf": round(conf, 4),
                    "heuristic_pred": heuristic_res,
                })
                return pred_cls
            else:
                pred_cls = int(self.ml_model.predict(x_arr)[0])
                if 1 <= pred_cls <= 12:
                    self.stats["mlp_accepted"] += 1
                    return pred_cls
        except Exception as exc:
            self.stats["fallback_exception"] += 1
            self.stats["call_details"].append({
                "track_id": track.track_id,
                "points_len": len(track.points),
                "decision": f"fallback_exception: {exc}",
                "final_movement": heuristic_res,
                "mlp_pred": None,
                "mlp_conf": None,
                "heuristic_pred": heuristic_res,
            })
            return heuristic_res

        return heuristic_res


def main():
    video_path = ROOT / "AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4"
    roi_path = ROOT / "AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt"
    detector_path = ROOT / "models/detector.onnx"
    mlp_model_path = ROOT / "artifacts/track1_final/models/movement_classifier_train_only.pkl"
    gt_path = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"

    out_csv = ROOT / "artifacts/deployment_validation/mlp_cam5_600frames_phase1_run.csv"
    out_full_json = ROOT / "artifacts/deployment_validation/mlp_phase1_full_metrics.json"
    out_heldout_json = ROOT / "artifacts/deployment_validation/mlp_phase1_heldout_metrics.json"
    out_log_json = ROOT / "artifacts/deployment_validation/mlp_phase1_fallback_log.json"

    print("=" * 70)
    print("PHASE 1: INSTRUMENTED MLP INFERENCE AND FALLBACK AUDIT")
    print("=" * 70)
    print("Video:", video_path)
    print("ROI:", roi_path)
    print("Detector:", detector_path)
    print("MLP Model:", mlp_model_path)

    roi_poly = load_roi_polygon(roi_path)
    detector_session = ort.InferenceSession(str(detector_path), providers=["CPUExecutionProvider"])
    matcher = InstrumentedMLPMatcher(CAM5_MOVEMENTS, ml_model_path=mlp_model_path)

    pipeline = Track1CountingPipeline(
        detector_session=detector_session,
        roi_polygon=roi_poly,
        movements=CAM5_MOVEMENTS,
        video_clip_id=10,
        conf_threshold=0.35,
    )
    pipeline.matcher = matcher

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open {video_path}"
    total_frames = 600
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    start_t = time.time()
    for f_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        pipeline.process_frame(frame, f_idx)
        if (f_idx + 1) % 100 == 0:
            print(f"Processed {f_idx + 1}/{total_frames} frames...")
    cap.release()

    events = pipeline.finalize(total_frames - 1)
    runtime = time.time() - start_t
    print(f"Completed in {runtime:.2f}s ({total_frames / runtime:.1f} fps)")
    print(f"Total counting events emitted: {len(events)}")

    # Export CSV
    export_track1_csv(events, out_csv)
    print(f"Saved CSV: {out_csv}")

    # Fallback statistics
    st = matcher.stats
    print("\n--- MATCHER INFERENCE BREAKDOWN ---")
    print(f"Total movement match calls:     {st['total_calls']}")
    print(f"MLP accepted decisions:         {st['mlp_accepted']} ({st['mlp_accepted'] * 100 / max(1, st['total_calls']):.1f}%)")
    print(f"Fallback short track (<2 pts):  {st['fallback_short_track']}")
    print(f"Fallback low confidence (<0.35):{st['fallback_low_confidence']}")
    print(f"Fallback out of range:          {st['fallback_out_of_range']}")
    print(f"Fallback exception:             {st['fallback_exception']}")

    # Check how many mlp_accepted agreed with heuristic vs differed
    agreed = sum(1 for d in st["call_details"] if d["decision"] == "mlp_accepted" and d["mlp_pred"] == d["heuristic_pred"])
    differed = sum(1 for d in st["call_details"] if d["decision"] == "mlp_accepted" and d["mlp_pred"] != d["heuristic_pred"])
    print(f"Among MLP-accepted: {agreed} agreed with heuristic, {differed} differed from heuristic")

    st_summary = {
        "total_calls": st["total_calls"],
        "mlp_accepted_count": st["mlp_accepted"],
        "mlp_accepted_pct": round(st["mlp_accepted"] * 100 / max(1, st["total_calls"]), 2),
        "fallback_short_track": st["fallback_short_track"],
        "fallback_low_confidence": st["fallback_low_confidence"],
        "fallback_out_of_range": st["fallback_out_of_range"],
        "fallback_exception": st["fallback_exception"],
        "agreed_with_heuristic": agreed,
        "differed_from_heuristic": differed,
        "details_sample": st["call_details"][:20],
    }
    out_log_json.write_text(json.dumps(st_summary, indent=2), encoding="utf-8")
    print(f"Saved log summary: {out_log_json}")

    # Evaluate full 600 frames
    gt_rows = load_csv_rows(gt_path)
    pred_rows = load_csv_rows(out_csv)

    res_full = evaluate_heldout_slice(pred_rows, gt_rows, min_frame=1, max_frame=600)
    out_full_json.write_text(json.dumps(res_full, indent=2), encoding="utf-8")
    print(f"\n--- FULL 600-FRAME EVALUATION ---")
    print(f"Strict F1: {res_full['strict_matching']['f1_score']:.4f} (TP={res_full['strict_matching']['tp']}, FP={res_full['strict_matching']['fp']}, FN={res_full['strict_matching']['fn']})")
    print(f"Count-Only F1: {res_full['count_only_matching']['f1_score']:.4f} (TP={res_full['count_only_matching']['tp']}, FP={res_full['count_only_matching']['fp']}, FN={res_full['count_only_matching']['fn']})")

    # Evaluate held-out 351 to 600
    res_heldout = evaluate_heldout_slice(pred_rows, gt_rows, min_frame=351, max_frame=600)
    out_heldout_json.write_text(json.dumps(res_heldout, indent=2), encoding="utf-8")
    print(f"\n--- HISTORICAL HELD-OUT (351-600) EVALUATION ---")
    print(f"Strict F1: {res_heldout['strict_matching']['f1_score']:.4f} (TP={res_heldout['strict_matching']['tp']}, FP={res_heldout['strict_matching']['fp']}, FN={res_heldout['strict_matching']['fn']})")
    print(f"Count-Only F1: {res_heldout['count_only_matching']['f1_score']:.4f} (TP={res_heldout['count_only_matching']['tp']}, FP={res_heldout['count_only_matching']['fp']}, FN={res_heldout['count_only_matching']['fn']})")


if __name__ == "__main__":
    main()
