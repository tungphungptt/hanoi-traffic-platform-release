#!/usr/bin/env python3
"""Compare vehicle detectors on the same frames: speed, counts, and agreement with the others.

There are no human box labels yet, so this is a RELATIVE comparison, not accuracy. "Consensus" boxes
are the vehicles at least two models agree on (IoU >= 0.5, class-agnostic); each model's recall and
precision against them says how much it misses or invents compared with the rest. Replace the
consensus with human labels (``--export-coco`` writes pre-annotations to correct in CVAT or Label
Studio) before drawing conclusions.

    python tools/bakeoff_detectors.py --videos a.mp4 b.mp4 --frames-per-video 5 \\
        --yolox yolox_s.onnx yolox_m.onnx --out artifacts/detector_bakeoff.json

Licences: the shipped YOLOv8n is AGPL-3.0 (Ultralytics). YOLOX (Megvii) is Apache-2.0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from traffic_brain.detector import Detection, Detector, VEHICLE_CLASS_IDS  # noqa: E402

CONSENSUS_IOU = 0.5
MIN_TRAFFIC_BOXES = 3


def load_yolox(path: Path) -> Detector:
    detector = Detector(path)
    detector.load()
    if detector.kind != "yolox":
        raise SystemExit(f"{path} không phải mô hình YOLOX (đầu ra {detector.kind}).")
    return detector


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def matched(box, others) -> bool:
    return any(iou(box, o.box) >= CONSENSUS_IOU for o in others)


def consensus(per_model: dict[str, list[Detection]]) -> list[tuple]:
    """Boxes (from any model) that at least two models found; duplicates across models are merged."""
    names = list(per_model)
    boxes: list[tuple] = []
    for name in names:
        for d in per_model[name]:
            votes = 1 + sum(matched(d.box, per_model[o]) for o in names if o != name)
            if votes >= 2 and not any(iou(d.box, b) >= CONSENSUS_IOU for b in boxes):
                boxes.append(d.box)
    return boxes


def sample_frames(videos: list[Path], per_video: int, seed: int = 7) -> list[tuple[str, float, np.ndarray]]:
    rng = np.random.default_rng(seed)
    frames = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        for index in sorted(rng.choice(max(total - 1, 1), size=min(per_video, max(total, 1)), replace=False)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append((video.name, float(index / fps), frame))
        cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--yolox", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "detector_bakeoff.json")
    parser.add_argument("--export-coco", type=Path, help="save the frames and the consensus boxes as COCO pre-annotations")
    args = parser.parse_args()

    models: dict[str, object] = {"yolov8n (shipped, AGPL-3.0)": Detector()}
    models["yolov8n (shipped, AGPL-3.0)"].load()
    for path in args.yolox:
        models[f"{Path(path).stem} (Apache-2.0)"] = load_yolox(Path(path))
    frames = sample_frames(args.videos, args.frames_per_video)
    print(f"{len(frames)} frames from {len(args.videos)} videos, {len(models)} models")

    stats = {n: {"ms": [], "count": [], "bikes": [], "cars": [], "recall": [], "precision": []} for n in models}
    traffic_only = {n: {"count": [], "bikes": [], "recall": [], "precision": []} for n in models}
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "vehicle"}]}
    for fi, (name, t, frame) in enumerate(frames):
        per_model: dict[str, list[Detection]] = {}
        for label, model in models.items():
            started = time.perf_counter()
            per_model[label] = model.detect(frame)
            stats[label]["ms"].append((time.perf_counter() - started) * 1000)
        truth = consensus(per_model)
        is_traffic = len(truth) >= MIN_TRAFFIC_BOXES  # studio shots and empty roads would only dilute the averages
        for label, found in per_model.items():
            s = stats[label]
            s["count"].append(len(found))
            s["bikes"].append(sum(d.class_id == 3 for d in found))
            s["cars"].append(sum(d.class_id in (2, 5, 7) for d in found))
            if truth:
                s["recall"].append(sum(any(iou(b, d.box) >= CONSENSUS_IOU for d in found) for b in truth) / len(truth))
            if found:
                s["precision"].append(sum(matched(d.box, [type("D", (), {"box": b}) for b in truth]) for d in found) / len(found))
            if is_traffic:
                t = traffic_only[label]
                t["count"].append(len(found))
                t["bikes"].append(sum(d.class_id == 3 for d in found))
                t["recall"].append(sum(any(iou(b, d.box) >= CONSENSUS_IOU for d in found) for b in truth) / len(truth))
                t["precision"].append(sum(matched(d.box, [type("D", (), {"box": b}) for b in truth]) for d in found) / max(1, len(found)))
        if args.export_coco:
            args.export_coco.mkdir(parents=True, exist_ok=True)
            fname = f"{fi:03d}_{Path(name).stem}_{int(t)}s.jpg"
            cv2.imwrite(str(args.export_coco / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            coco["images"].append({"id": fi, "file_name": fname, "width": frame.shape[1], "height": frame.shape[0]})
            for b in truth:
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": fi, "category_id": 1,
                                            "bbox": [round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1)],
                                            "area": round((b[2] - b[0]) * (b[3] - b[1]), 1), "iscrowd": 0})
    if args.export_coco:
        (args.export_coco / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")

    traffic_frames = len(traffic_only[next(iter(traffic_only))]["count"])
    report = {"frames": len(frames), "traffic_frames": traffic_frames, "videos": [v.name for v in args.videos],
              "note": "Relative comparison against the consensus of the models; no human labels.", "models": {}}
    for label, s in stats.items():
        report["models"][label] = {
            "ms_per_frame_mean": round(float(np.mean(s["ms"])), 1), "ms_per_frame_p95": round(float(np.percentile(s["ms"], 95)), 1),
            "vehicles_per_frame_mean": round(float(np.mean(s["count"])), 2),
            "motorbikes_per_frame_mean": round(float(np.mean(s["bikes"])), 2),
            "cars_buses_trucks_per_frame_mean": round(float(np.mean(s["cars"])), 2),
            "recall_vs_consensus": round(float(np.mean(s["recall"])), 3) if s["recall"] else None,
            "precision_vs_consensus": round(float(np.mean(s["precision"])), 3) if s["precision"] else None,
            "traffic_frames_only": {
                "vehicles_per_frame_mean": round(float(np.mean(traffic_only[label]["count"])), 2),
                "motorbikes_per_frame_mean": round(float(np.mean(traffic_only[label]["bikes"])), 2),
                "recall_vs_consensus": round(float(np.mean(traffic_only[label]["recall"])), 3),
                "precision_vs_consensus": round(float(np.mean(traffic_only[label]["precision"])), 3)} if traffic_frames else None}
        print(label, report["models"][label])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
