# Models

- `detector.onnx` — YOLOv8n vehicle detector (COCO classes car, motorcycle, bus, truck), 640×640 input,
  run on CPU with ONNX Runtime by `traffic_brain/detector.py`. YOLOv8 is published by Ultralytics under
  AGPL-3.0; see `THIRD_PARTY_NOTICES.md`.
- `forecast_library/` — the forecaster's training data: congestion-index series (numbers only, no
  images) of long traffic videos. See `forecast_library/README.md`.
