# Forecast training library (+5/+10/+15 minute CI)

Each JSON is the congestion series of one camera in one video: visual CI (60 s mean every 15 s)
plus occupancy, vehicle count, motion and turnover. No images or frames are stored. The web app
learns from all of them; a record is never used to forecast the video it came from
(`source_video_keys`).

Built with the same pipeline as the web app:

```bash
python tools/build_forecast_library.py VIDEO... --out models/forecast_library --source-note "..."
```

## Contents (8 camera series, 8 videos, about 3.7 hours of video)

Eight fixed-camera clips of Hanoi (Drive folder "Mới", one road each; the first 5 minutes of two clips
were cut because the camera showed another road then). Detector: YOLOX-s on two halves, CI scale as in
`traffic_brain/video_analysis.py`. The earlier library (AIC21 and broadcast series) was measured with another
detector and was removed because its CI is not on the same scale. Rebuild: `tools/rebuild_library.py`.
