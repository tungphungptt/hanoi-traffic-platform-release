# Forecast training library (+5/+10/+15 minute CI)

These files are what the minute forecaster (`traffic_brain/minute_forecaster.py`) learns
from. Each JSON is one video turned into an observed congestion series: the visual CI
(60 s mean, every 15 s) plus occupancy, vehicle count and motion. No images, frames or
detections are stored. The web app loads this folder automatically (`FORECAST_BUNDLED_LIBRARY=true`)
together with videos analysed locally (`data/forecast_library/`). A video is never used to
train its own forecasts (files are keyed by a content fingerprint).

## Contents

| File (video_key) | Source video | Length | CI mean / min / max |
| --- | --- | --- | --- |
| `14dceead19e719efebfe120a50fd442d` | AIC21 Track 1 Dataset A `cam_2.mp4` | 30 min | 0.15 / 0.03 / 0.37 |
| `ec265317b963df87a414dc281a2d50eb` | AIC21 Track 1 Dataset A `cam_3.mp4` | 30 min | 0.27 / 0.05 / 0.38 |
| `87034197f709848ba0c0264a2a96f326` | AIC21 Track 1 Dataset A `cam_6.mp4` | 30 min | 0.49 / 0.36 / 0.64 |

These were the only street-traffic videos of 30 minutes or more found on the developer
machine that fit the transfer limit (`cam_4`, `cam_5`, `cam_7` are over 400 MB and were not
processed; the other Dataset A clips are 5 minutes, too short for minute horizons).

**Licence:** derived from the AI City Challenge 2021 dataset, which is licensed for
non-commercial academic research only. Do not use these records commercially. They are
US road scenes, not Hanoi; add Hanoi videos to make the library representative.

## How it was built

```bash
python tools/build_forecast_library.py cam_2.mp4 cam_3.mp4 cam_6.mp4 \
  --source-note "AI City Challenge 2021 Track 1 Dataset A (academic, non-commercial licence)" \
  --demo cam_6.mp4
```

Detector: `models/detector.onnx` (YOLOv8n COCO), 1 frame per second, same code path as the web app.

## Leave-one-video-out check (forecast each video using only the other two)

| Horizon | Forecasts checked | MAE model | MAE "keep current level" | MAE linear trend | Improvement |
| --- | --- | --- | --- | --- | --- |
| +5 min | 264 | 0.075 | 0.086 | 0.194 | +13.5% |
| +10 min | 204 | 0.075 | 0.083 | 0.254 | +10.1% |
| +15 min | 144 | 0.078 | 0.089 | 0.369 | +11.9% |

Full report: `artifacts/evaluation/forecast_library_report.json`. Ground truth is the
same visual CI, not expert labels. These scenes rarely reach CI ≥ 0.55, so the
"congestion onset" warnings are barely tested by this data (cam_6: 1 correct warning,
3 missed, 2 false alarms). Three videos are a small sample, not a benchmark.

A ready-to-open demo session is in `artifacts/evaluation/demo_cam_6.hanoiflow.json`:
on the home page choose "Mở kết quả đã lưu", then select `cam_6.mp4` as the video.
