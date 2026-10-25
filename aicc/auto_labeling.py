"""Auto-labeling and pseudo-label curation pipeline for AI City Challenge Track 1.

Generates machine-proposed bounding box pseudo-labels (Car, Truck) from existing ONNX detectors,
exports standard YOLO-format annotations with full provenance metadata, and provides review/filtering
gates to curate verified training subsets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from aicc.track1_counting import COCO_TO_TRACK1, Track1VehicleClass


@dataclass
class AutoLabelBox:
    """Bounding box detection proposed as a pseudo-label."""
    track1_class_id: int           # 1 = Car, 2 = Truck
    coco_class_id: int             # 2 = COCO car, 5 = COCO bus, 7 = COCO truck
    class_name: str                # 'car' or 'truck'
    confidence: float              # Detector confidence score in [0.0, 1.0]
    bbox_xyxy: Tuple[float, float, float, float]       # (x1, y1, x2, y2) in pixels
    bbox_yolo_norm: Tuple[float, float, float, float]  # (x_center, y_center, width, height) in [0, 1]
    in_roi: bool                   # Whether box center is within ROI


@dataclass
class AutoLabelFrame:
    """Annotations for a single video frame."""
    video_id: str
    frame_idx: int
    timestamp_s: float
    image_width: int
    image_height: int
    boxes: List[AutoLabelBox]


class AutoLabelPipeline:
    """Inference and export engine generating pseudo-labels from detector.onnx."""

    def __init__(
        self,
        detector_session: ort.InferenceSession,
        roi_polygon: Optional[np.ndarray] = None,
        conf_threshold: float = 0.35,
        model_version: str = "yolov8n-coco-v1",
    ):
        self.session = detector_session
        self.input_name = detector_session.get_inputs()[0].name
        self.roi = roi_polygon
        self.conf_threshold = conf_threshold
        self.model_version = model_version

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        video_id: str,
        timestamp_s: float,
    ) -> AutoLabelFrame:
        """Run detector on a single frame and produce typed pseudo-label boxes."""
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (640, 640))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        out = self.session.run(None, {self.input_name: blob})[0].squeeze()
        boxes = out[:4, :].T
        scores = out[4:, :].T
        classes = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        valid_idx = np.where((max_scores >= self.conf_threshold) & np.isin(classes, list(COCO_TO_TRACK1.keys())))[0]

        result_boxes: List[AutoLabelBox] = []
        if len(valid_idx) > 0:
            v_boxes = boxes[valid_idx]
            v_scores = max_scores[valid_idx]
            v_classes = classes[valid_idx]

            # Convert from center-w-h in 640x640 space to pixels (x1, y1, w, h)
            x_scale = orig_w / 640.0
            y_scale = orig_h / 640.0
            boxes_xywh = []
            for b in v_boxes:
                cx, cy, w, h = b
                x1 = (cx - w / 2.0) * x_scale
                y1 = (cy - h / 2.0) * y_scale
                boxes_xywh.append([int(x1), int(y1), int(w * x_scale), int(h * y_scale)])

            indices = cv2.dnn.NMSBoxes(boxes_xywh, v_scores.tolist(), self.conf_threshold, 0.45)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x1, y1, w, h = boxes_xywh[idx]
                    x2 = x1 + w
                    y2 = y1 + h
                    # Clip coordinates to image boundary
                    x1_c = max(0.0, min(float(orig_w), float(x1)))
                    y1_c = max(0.0, min(float(orig_h), float(y1)))
                    x2_c = max(0.0, min(float(orig_w), float(x2)))
                    y2_c = max(0.0, min(float(orig_h), float(y2)))

                    box_w = max(1.0, x2_c - x1_c)
                    box_h = max(1.0, y2_c - y1_c)
                    cx_norm = (x1_c + box_w / 2.0) / orig_w
                    cy_norm = (y1_c + box_h / 2.0) / orig_h
                    w_norm = box_w / orig_w
                    h_norm = box_h / orig_h

                    coco_cls = int(v_classes[idx])
                    track1_cls = int(COCO_TO_TRACK1[coco_cls])
                    cls_name = "car" if track1_cls == Track1VehicleClass.CAR else "truck"

                    # Check ROI containment
                    center_pt = (x1_c + box_w / 2.0, y1_c + box_h / 2.0)
                    in_roi = True
                    if self.roi is not None:
                        in_roi = cv2.pointPolygonTest(self.roi, center_pt, False) >= 0

                    result_boxes.append(
                        AutoLabelBox(
                            track1_class_id=track1_cls,
                            coco_class_id=coco_cls,
                            class_name=cls_name,
                            confidence=round(float(v_scores[idx]), 4),
                            bbox_xyxy=(round(x1_c, 1), round(y1_c, 1), round(x2_c, 1), round(y2_c, 1)),
                            bbox_yolo_norm=(round(cx_norm, 6), round(cy_norm, 6), round(w_norm, 6), round(h_norm, 6)),
                            in_roi=bool(in_roi),
                        )
                    )

        return AutoLabelFrame(
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_s=round(timestamp_s, 3),
            image_width=orig_w,
            image_height=orig_h,
            boxes=result_boxes,
        )

    def process_video(
        self,
        video_path: Path | str,
        output_dir: Path | str,
        sample_stride: int = 10,
        max_frames: Optional[int] = None,
        save_images: bool = False,
    ) -> Dict[str, Any]:
        """Sample frames from a video, run detector, and export YOLO-format txt files and manifest."""
        v_path = Path(video_path)
        out_root = Path(output_dir)
        labels_dir = out_root / "labels"
        images_dir = out_root / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            images_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {v_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_id = v_path.stem

        limit = min(max_frames, total_frames) if max_frames and max_frames > 0 else total_frames

        frames_data: List[Dict[str, Any]] = []
        total_boxes = 0
        total_car = 0
        total_truck = 0
        total_in_roi = 0

        frame_idx = 0
        sampled_count = 0

        try:
            while frame_idx < limit:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_stride == 0:
                    ts = frame_idx / fps
                    frame_res = self.detect_frame(frame, frame_idx, video_id, ts)
                    sampled_count += 1

                    # Write YOLO format txt: class_index x_center y_center width height (0-indexed for YOLO: 0=car, 1=truck)
                    txt_name = f"{video_id}_frame_{frame_idx:06d}.txt"
                    txt_path = labels_dir / txt_name
                    with open(txt_path, "w", encoding="utf-8") as f:
                        for box in frame_res.boxes:
                            # In Track 1: CAR=1, TRUCK=2 -> YOLO 0-indexed: car=0, truck=1
                            yolo_cls = box.track1_class_id - 1
                            cx, cy, bw, bh = box.bbox_yolo_norm
                            f.write(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {box.confidence:.4f}\n")

                    if save_images:
                        img_name = f"{video_id}_frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(images_dir / img_name), frame)

                    # Accumulate stats
                    b_list = []
                    for b in frame_res.boxes:
                        total_boxes += 1
                        if b.track1_class_id == 1:
                            total_car += 1
                        else:
                            total_truck += 1
                        if b.in_roi:
                            total_in_roi += 1
                        b_list.append(asdict(b))

                    frames_data.append({
                        "frame_idx": frame_idx,
                        "timestamp_s": frame_res.timestamp_s,
                        "label_file": txt_name,
                        "boxes_count": len(b_list),
                        "boxes": b_list,
                    })

                frame_idx += 1
        finally:
            cap.release()

        manifest = {
            "dataset_provenance": {
                "source_video": str(v_path.name),
                "total_video_frames": total_frames,
                "sampled_frames_count": sampled_count,
                "sample_stride": sample_stride,
                "fps": round(fps, 2),
                "model_version": self.model_version,
                "confidence_threshold": self.conf_threshold,
                "annotation_status": "PROPOSED_PSEUDO_LABEL",
                "notes": "Generated by detector.onnx pseudo-labeler. Requires review before training.",
            },
            "summary_statistics": {
                "total_pseudo_boxes": total_boxes,
                "total_cars": total_car,
                "total_trucks": total_truck,
                "total_in_roi": total_in_roi,
                "average_boxes_per_frame": round(total_boxes / max(1, sampled_count), 2),
            },
            "frames": frames_data,
        }

        manifest_path = out_root / "auto_labels_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return manifest


def filter_and_curate_pseudo_labels(
    manifest_path: Path | str,
    output_dir: Path | str,
    min_confidence: float = 0.60,
    require_roi: bool = True,
) -> Dict[str, Any]:
    """Review and curate pseudo-labels by filtering on confidence threshold and ROI boundary."""
    m_path = Path(manifest_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {m_path}")

    with open(m_path, "r", encoding="utf-8") as f:
        src_manifest = json.load(f)

    out_root = Path(output_dir)
    curated_labels_dir = out_root / "curated_labels"
    curated_labels_dir.mkdir(parents=True, exist_ok=True)

    curated_frames = []
    total_curated_boxes = 0
    curated_cars = 0
    curated_trucks = 0

    for f_info in src_manifest.get("frames", []):
        f_idx = f_info["frame_idx"]
        ts = f_info["timestamp_s"]
        valid_boxes = []

        for b in f_info.get("boxes", []):
            if b["confidence"] < min_confidence:
                continue
            if require_roi and not b.get("in_roi", False):
                continue
            valid_boxes.append(b)

        if valid_boxes:
            v_id = src_manifest["dataset_provenance"]["source_video"].replace(".mp4", "")
            txt_name = f"{v_id}_frame_{f_idx:06d}.txt"
            with open(curated_labels_dir / txt_name, "w", encoding="utf-8") as out_f:
                for b in valid_boxes:
                    yolo_cls = b["track1_class_id"] - 1
                    cx, cy, bw, bh = b["bbox_yolo_norm"]
                    out_f.write(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {b['confidence']:.4f}\n")
                    total_curated_boxes += 1
                    if b["track1_class_id"] == 1:
                        curated_cars += 1
                    else:
                        curated_trucks += 1

            curated_frames.append({
                "frame_idx": f_idx,
                "timestamp_s": ts,
                "curated_boxes_count": len(valid_boxes),
                "boxes": valid_boxes,
            })

    curated_manifest = {
        "curation_provenance": {
            "source_manifest": str(m_path),
            "original_pseudo_boxes": src_manifest["summary_statistics"]["total_pseudo_boxes"],
            "filter_min_confidence": min_confidence,
            "filter_require_roi": require_roi,
            "curated_subset_status": "VERIFIED_SUBSET",
            "acceptance_ratio": round(total_curated_boxes / max(1, src_manifest["summary_statistics"]["total_pseudo_boxes"]), 4),
        },
        "curated_statistics": {
            "total_curated_boxes": total_curated_boxes,
            "curated_cars": curated_cars,
            "curated_trucks": curated_trucks,
            "frames_with_detections": len(curated_frames),
        },
        "frames": curated_frames,
    }

    curated_manifest_path = out_root / "verified_subset_manifest.json"
    curated_manifest_path.write_text(json.dumps(curated_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return curated_manifest
