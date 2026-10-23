# Upstream attribution and integration notes

Upstream repository: https://github.com/dinoraptor999/AICC

This project integrates an AICC-compatible video inference path into the Hanoi Traffic Forecasting dashboard. The upstream interface/contract was inspected from `api/main.py`, `core/preprocessing.py`, `config.yaml`, and `models/manifest.json`. The adaptation is maintained in `aicc/pipeline.py` and `aicc/core/preprocessing.py`; this is not a copy of the full upstream repository.

Expected model contract from upstream `models/manifest.json`:

- detector ONNX for COCO vehicle classes car=2, bus=5, truck=7
- predictor ONNX receiving a sequence of 5 observations; each observation has `[car_count, avg_speed, speed_cv]`
- trusted scikit-learn `MinMaxScaler` serialized as `scaler.pkl`
- output class index maps to the `classes` list in the manifest (`Thong_thoang`, `Dong_duc` by default)

Important: the bundled `models/manifest.json` is configuration, not the model weights. Inference remains unavailable until all 3 referenced model artifacts are supplied. Review the upstream repository's applicable licence before redistributing its artifacts.
