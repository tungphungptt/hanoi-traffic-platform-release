# Forecast training library (+5/+10/+15 minute CI)

Each JSON is the congestion series of one camera in one video: visual CI (60 s mean every 15 s)
plus occupancy, vehicle count and motion. No images or frames are stored. The web app learns from
all of them; a record is never used to forecast the video it came from (`source_video_keys`).

Built with the same pipeline as the web app:

```bash
python tools/build_forecast_library.py VIDEO... --out models/forecast_library --source-note "..."
```

## Contents

| Records | Source | Notes |
| --- | --- | --- |
| `ec265317…`, `14dceead…`, `87034197…` | AI City Challenge 2021 Track 1 Dataset A `cam_2`, `cam_3`, `cam_6` (30 min each, one fixed camera) | US roads. Licence: non-commercial academic research only. |
| `93dfb747…-cam0/1`, `c5023274…-cam0/1`, `f33efea3…-cam0/1`, `db1fafdf…-cam0/1` | "Hanoi Flow" Google Drive folder: four 30-minute recordings of the Hanoi traffic-camera broadcast (VOV Giao thông FM90 / Hà Nội LIVE), parts 000–005 of each folder | The programme cuts between cameras and studio shots; each record is one camera, with gaps while it is off air (6–12 min on air per record). Research/demo use; credit the broadcaster. |

Held out (never in the library): parts 016–021 of the "Tran Duy Hung" folder, used for the Hanoi demo
(`artifacts/demo_hanoi_tdh_htm.hanoiflow.json`).

## Leave-one-video-out check

Every record is forecast using only records from other videos, then scored against what that video
later showed (`artifacts/forecast_library_report.json`). MAE on the 0–1 CI scale.

| Horizon | Forecasts checked | Model | Keep current level | Linear trend | Improvement |
| --- | --- | --- | --- | --- | --- |
| +5 min | 296 | 0.084 | 0.096 | 0.204 | +12.7% |
| +10 min | 243 | 0.077 | 0.089 | 0.290 | +12.8% |
| +15 min | 144 | 0.078 | 0.089 | 0.369 | +11.9% |

Hanoi cameras only:

| Horizon | Forecasts checked | Model | Keep current level | Improvement |
| --- | --- | --- | --- | --- |
| +5 min | 32 | 0.161 | 0.178 | +9.3% |
| +10 min | 39 | 0.092 | 0.120 | +23.7% |
| +15 min | 0 | – | – | not checkable: no camera in these recordings is on air 15 min after a forecast |

The Hanoi counts are small; read them as a first signal, not a benchmark. The ground truth is the
same visual CI the system measures, not expert labels.
