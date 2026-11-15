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

## Hanoi cameras (added from the "Hanoi Flow" Google Drive folder)

The folder holds 5-minute recordings of the Hanoi traffic-camera broadcast (FM90 / "Hà Nội LIVE").
Each recording cycles between street cameras and studio/banner shots, so one file is **not** one
camera. `hn-*` records were built like this:

1. Contiguous 5-minute parts of one folder were analysed in order (same detector, 1 fps).
2. Every second was assigned to a camera by an image signature of the frame (red banner / studio
   frames excluded, cuts shorter than 5 s dropped).
3. One series per camera per recording, with **gaps (NaN) while that camera is off air**. The
   forecaster only scores a forecast when the camera is visible at the target time.

Each record lists `source_video_keys` (fingerprints of the original files); the web app skips a
record when the uploaded video is one of its sources, so a video never trains its own forecasts.

| File | Camera | Recording | Time the camera is on air |
| --- | --- | --- | --- |
| `hn-cam1-000-scene0` | 216 Trần Duy Hưng | Tran Duy Hung 000-005.mp4 | 7.2 min |
| `hn-cam1-000-scene2` | 300 Nguyễn Xiển | Tran Duy Hung 000-005.mp4 | 4.0 min |
| `hn-cam1-000-scene7` | Xuân Diệu | Tran Duy Hung 000-005.mp4 | 6.0 min |
| `hn-cam1-010-scene2` | 300 Nguyễn Xiển | Tran Duy Hung 010-010.mp4 | 3.5 min |
| `hn-cam1-020-scene0` | 216 Trần Duy Hưng | Tran Duy Hung 020-020.mp4 | 4.0 min |
| `hn-cam2-000-scene16` | Nút giao (camera #16, Hà Nội LIVE) | Nguyen Xien 000-005.mp4 | 9.0 min |
| `hn-cam2-000-scene2` | 300 Nguyễn Xiển | Nguyen Xien 000-005.mp4 | 6.0 min |
| `hn-cam3-000-scene21` | Hồ Tùng Mậu | Ho Tung Mau 000-005.mp4 | 2.8 min |
| `hn-cam3-000-scene41` | Nút giao phố cổ (camera #41) | Ho Tung Mau 000-005.mp4 | 9.2 min |
| `hn-cam3-000-scene7` | Xuân Diệu | Ho Tung Mau 000-005.mp4 | 2.0 min |
| `hn-cam4-000-scene12` | Quang Trung – Lê Trọng Tấn | Le Hong Phong 000-005.mp4 | 9.5 min |
| `hn-cam4-000-scene41` | Nút giao phố cổ (camera #41) | Le Hong Phong 000-005.mp4 | 6.0 min |

**Limits.** No camera stays on air for more than ~9.5 minutes in a row, so +15 min forecasts
cannot be checked on this footage, and the +5/+10 checks are few:

| Horizon | Forecasts checked (Hanoi, other recordings + AIC21 as library) | MAE model | MAE "keep current level" | Improvement |
| --- | --- | --- | --- | --- |
| +5 min | 23 | 0.174 | 0.183 | +4.7% |
| +10 min | 39 | 0.093 | 0.102 | +9.2% |
| +15 min | 0 | – | – | not checkable |

Hanoi CI is much higher than the AIC21 scenes (mean 0.29, 14% of points ≥ 0.55 "congested"), which
is why these records matter for the library. To verify +15 min on Hanoi, record a single camera
for 20–30 minutes without switching (e.g. the camera page on the city portal) and upload it.

**Source / licence:** recordings of a public broadcast shared in a Google Drive folder; only the
derived numeric series are stored here, no video or frames. Use for research/demo only and
credit the broadcaster (VOV Giao thông FM90 / Hà Nội LIVE).
