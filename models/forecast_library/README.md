# Forecast training library (+5/+10/+15 minute CI)

Each JSON is the congestion series of one camera in one video: visual CI (60 s mean every 15 s)
plus occupancy, vehicle count, motion and turnover. No images or frames are stored. The web app
learns from all of them; a record is never used to forecast the video it came from
(`source_video_keys`).

Built with the same pipeline as the web app:

```bash
python tools/build_forecast_library.py VIDEO... --out models/forecast_library --source-note "..."
```

## Contents (30 camera series, 15 videos, about 13 hours of video)

| Source | Videos | Camera series | Notes |
| --- | --- | --- | --- |
| AI City Challenge 2021 Track 1 Dataset A `cam_2`, `cam_3`, `cam_6` | 3 × 30 min, one fixed camera each | 3 | US roads. Licence: non-commercial academic research only. |
| "Hanoi Flow" Google Drive folder: recordings of the Hanoi traffic-camera broadcast (VOV Giao thông FM90 / Hà Nội LIVE) | Nguyen Xien, Ho Tung Mau, Le Hong Phong parts 000–021 (109 min each); Tran Duy Hung parts 000–015 (80 min) | 19 | The programme cuts between cameras and studio shots; each record is one camera, with gaps while it is off air (5–75 min on air per record). Research/demo use; credit the broadcaster. |

| Google Drive folder "Mới" (`vid 1`, `vid 2`): eight 30-minute single-camera clips, 960×540, camera name in the top-left corner | 8 × 30 min | 8 | Added 2026-11-18. Each file is one fixed camera (the pipeline saw `vid1/04_Bottom_Right` as two views, and only one had enough footage to keep). The 2×2 grid file `giao_thong_00_30.mp4` in `vid 2` is the same four cameras side by side and was not used. Camera names are not read yet: records are labelled "Camera cố định". Clips of one folder are probably recorded at the same time, so they are not fully independent. |

Held out (never in the library): Tran Duy Hung parts 016–021, used for the Hanoi demo
(`artifacts/demo_hanoi_tdh_htm.hanoiflow.json`). All 88 parts of the Drive folder were downloaded
and their order checked frame by frame (last frame of a part vs first frame of the next).

## Leave-one-video-out check

Every record is forecast using only records from *other* videos, then scored against what that
video later showed (`artifacts/forecast_library_report.json`). MAE on the 0–1 CI scale; the 95%
interval resamples whole videos (bootstrap).

| Horizon | Forecasts checked | Model | Keep current level | Linear trend | Improvement | 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| +5 min | 1521 | 0.084 | 0.109 | 0.208 | +22.4% | +13.4% … +29.9% |
| +10 min | 1150 | 0.079 | 0.097 | 0.266 | +17.8% | +6.3% … +27.5% |
| +15 min | 836 | 0.088 | 0.107 | 0.322 | +18.4% | +10.7% … +26.5% |

(15 videos. Before the "Mới" clips were added, 7 videos gave +18.3% / +16.0% / +11.1%.)

The onset counts at fixed 40% / 20% thresholds are no longer listed: those thresholds were set after
seeing the old results, so the table below (thresholds chosen without the scored video) is the one to read.

### Thresholds chosen without the scored video (`tools/honest_eval.py`)

For each held-out video the cut-off is picked again from the other videos only (warning: highest recall
with at least 50% of firings right; watch: best F1) and then applied to the held-out video.
Result in `artifacts/honest_eval.json`.

| Horizon | Onsets (videos) | Warning caught / false | Watch caught / false |
| --- | --- | --- | --- |
| +5 min | 104 (9) | 50 / 35 | 50 / 26 |
| +10 min | 90 (8) | 33 / 34 | 33 / 34 |
| +15 min | 70 (8) | 31 / 49 | 34 / 96 |

Reading: at +5 minutes about 48% of onsets are caught and 6 in 10 warnings are right, with thresholds that
were not tuned on the scored video. At +10 minutes warnings are right about half of the time and catch
37%; at +15 minutes false alarms outnumber hits (watch precision 26%). So +5 is usable as an early warning,
+10 is borderline and +15 is a CI forecast rather than an alarm. The onsets come from 8–9 videos and some
clips were recorded at the same time, so every figure here has a wide margin.

### Model history (same check)

| Model | +5 | +10 | +15 | Onsets warned at +5 |
| --- | --- | --- | --- | --- |
| Five forecasters (persistence, reversion, damped trend, ridge, analogs) weighted by each video's last 40 errors | +9.5% | +9.9% | +7.9% | 8 / 52 |
| Current: CI now + w1·(pull to camera mean) + w2·(pull to last-10-min mean) + w3·(damped trend), w ≥ 0 by least squares on library + watched part | +18.3% | +16.0% | +11.1% | 23 / 52 |

Why the old model rarely warned: the per-video error weighting was noisy and kept about a third of
the weight on "no change", so forecasts stayed close to the current level and the 40% jam
probability was almost never reached. Ridge regression on 8 features and nearest-neighbour analogs
added nothing over the two mean-reversion terms (their least-squares weight was 0 when tried), and
a separate logistic jam classifier ranked onsets worse (AUC 0.67–0.80) than the forecast CI itself
(0.80–0.85). The 60 s CI swings with signal cycles and platoons; the best guess for 5–15 minutes
ahead is mostly a blend of the recent and long-run means.

Tried and not kept: "turnover" (share of vehicles not seen one sample earlier, a flow proxy) as a
forecasting feature changed the old model's results to +9.4% / +10.1% / +7.1% — no clear gain, so it
is stored but not used.

The ground truth is the same visual CI the system measures, not expert labels; use
`tools/label_frames.py` to check the CI against people's labels.

## Heavier models, relative speed and calibrated probabilities (`tools/model_lab.py`)

Leave-one-video-out on the 22 series that existed before the "Mới" clips were added (these results were not recomputed). Rule fixed before running: a candidate replaces the shipped
formulas only if the 95% interval of its MAE gain (bootstrap over whole videos) is above zero. Result in
`artifacts/model_lab.json`.

| Candidate | +5 min | +10 min | +15 min | Adopted |
| --- | --- | --- | --- | --- |
| Gradient boosting on CI features (scikit-learn; LightGBM is not installed) | −13% [−28, −5] | −20% [−37, −6] | −1% [−11, +8] | no |
| Gradient boosting + relative speed, occupancy, count, turnover | −18% [−39, −10] | −18% [−37, −6] | −2% [−14, +9] | no |
| Shipped three formulas + a relative-speed formula | −0.6% [−1.7, +0.3] | +0.5% [−0.5, +1.4] | +0.7% [−0.3, +2.1] | no |

Boosting trained on a handful of videos is clearly worse than three formulas with fitted weights, and relative
speed adds nothing the CI does not already hold. The shipped weights fitted on the library alone match the
shipped model within 1%, so the loss is not caused by the shipped model's per-video adaptation. A GRU was not
tried: PyTorch is not installed and the boosting result gives no reason to expect a recurrent net to do better
on 7 videos.

Probability calibration (isotonic, calibrator and alert threshold chosen from the other videos only, alert level
= lowest probability whose precision on those videos reaches 50%):

| Horizon | Brier: shipped p | Brier: calibrated | Brier: constant base rate | Onsets caught at 50% precision |
| --- | --- | --- | --- | --- |
| +5 min | 0.0487 | 0.0470 | 0.0638 | 4 / 52 (3 false) |
| +10 min | 0.0568 | 0.0601 | 0.0620 | 0 / 39 |
| +15 min | 0.0497 | 0.0603 | 0.0575 | 10 / 28 (33 false) |

Calibration helps a little at +5 and hurts at +10/+15; the alert level it picks catches far fewer onsets at +5
than the shipped 40% rule (23 / 52). Nothing was changed: the shipped probabilities and thresholds stay.

Not tested, because no dated data exists yet: time of week and rain. `tools/context_lab.py` does that on the
CSV from `GET /api/live/export` (leave-one-day-out, at least 14 days, same adoption rule); its tests show it
finds a built-in rush-hour pattern and a rain-onset effect and does not invent one in flat data.
