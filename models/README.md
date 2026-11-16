# Models

- `detector.onnx` — YOLOv8n vehicle detector (COCO classes car, motorcycle, bus, truck), 640×640 input,
  run on CPU with ONNX Runtime by `traffic_brain/detector.py`. YOLOv8 is published by Ultralytics under
  AGPL-3.0; see `THIRD_PARTY_NOTICES.md`.
- `forecast_library/` — the forecaster's training data: congestion-index series (numbers only, no
  images) of long traffic videos. See `forecast_library/README.md`.

## Detector comparison (`tools/bakeoff_detectors.py`, `artifacts/detector_bakeoff.json`)

Same 48 frames (8 video parts, 4 Hanoi cameras; 39 contain traffic), CPU with 2 threads, COCO classes
car/motorcycle/bus/truck, score ≥ 0.25. **There are no human labels yet**: recall and precision are
measured against the boxes at least two models agree on, so they are relative, not accuracy.
Frames with traffic only:

| Model | Licence | ms/frame | Vehicles/frame | Motorbikes/frame | Recall vs consensus | Precision vs consensus |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n (shipped) | AGPL-3.0 | 77 | 4.8 | 0.03 | 0.30 | 0.89 |
| YOLOX-nano | Apache-2.0 | 24 | 6.8 | 0.28 | 0.39 | 0.96 |
| YOLOX-tiny | Apache-2.0 | 71 | 10.3 | 0.26 | 0.58 | 0.90 |
| YOLOX-s | Apache-2.0 | 143 | 16.9 | 2.5 | 0.95 | 0.94 |
| YOLOX-m | Apache-2.0 | 320 | 21.5 | 4.9 | 0.97 | 0.75 |

What it shows: on these cameras the shipped YOLOv8n finds about a quarter of the vehicles YOLOX-s finds,
nearly no motorbikes, and sometimes labels a car as a bus. Looking at boxes drawn on frames, YOLOX-s boxes
sit on real vehicles (more distant cars, more motorbikes); none of the models separate a dense crowd of
motorbikes, so CI stays understated there whatever the detector. YOLOX-m finds more but adds unconfirmed
boxes (precision 0.75).

Same frames through the whole CI pipeline (`FrameMeter`, 1 frame/s, two cameras, 268 samples), YOLOv8n vs YOLOX-s:

| Camera part | Vehicles/frame | Mean CI | Share of time CI ≥ 0.30 (busy) | Share CI ≥ 0.55 (gridlock) |
| --- | --- | --- | --- | --- |
| cam2_004 | 8.3 → 16.9 | 0.55 → 0.77 | 99% → 99% | 42% → 99% |
| cam3_006 | 2.7 → 10.8 | 0.19 → 0.51 | 24% → 96% | 1% → 34% |

The two CI series correlate only 0.56–0.65. So the current thresholds mean something different with
every detector, and with YOLOX-s unchanged they would call most of the day busy or gridlocked.

Using another detector (not the default, nothing is switched yet):

```bash
curl -L -o yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
# sha256 c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
DETECTOR_MODEL=/path/yolox_s.onnx uvicorn app.main:app
```

`traffic_brain/detector.py` picks the decoder from the model's output shape. **Switching the detector
changes the CI scale** (counts and occupancy go up, thresholds 0.30/0.55 and `count/20` were set with
YOLOv8n) so the forecast library must be rebuilt with the same detector and the CI normalisation and
thresholds re-checked against human labels before it becomes the default. Until then the shipped
library, thresholds and demo stay consistent with YOLOv8n.

`--export-coco DIR` also writes the frames and the consensus boxes as COCO pre-annotations, to correct
by hand in CVAT or Label Studio (faster than drawing from scratch).
