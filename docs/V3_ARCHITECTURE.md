# Final 3.0 — architecture and integration boundary

## Target flow from the Hanoi technical proposal

`Authorized camera/file -> frame sampler -> detector -> multi-object tracker -> per-camera calibration + feature mapping -> forecasting provider -> operator dashboard/alerts`

Final 3.0 improves the reusable foundations without claiming unimplemented sources are live:

1. `aicc/pipeline.py` samples detector frames to a configured target rate (default 6 FPS; cap 12 FPS) using the shared tested policy in `traffic_brain.analytics.adaptive_frame_stride`.
2. `traffic_brain.analytics.TrafficAnalyticsEngine` accepts normalized, already-tracked points and exposes count-line crossing, motion-gated unique counting, active counts by class, a bounded in-memory event queue, candidate volume-surge signals, candidate long-stationary track signals and a rolling confidence-drift indicator.
3. `POST /api/analytics/ingest-frame` is the stable seam for a future detector/ByteTrack/RTSP worker. It accepts no video and does not run detection. It validates the camera, normalized coordinates and bounded request size. Public-demo write mode disables it.
4. `GET /api/analytics/cameras/{camera_id}/events` reads recent in-memory candidate events. The API states `persistent: false`; this is not a durable audit/event database.
5. `traffic_brain.feature_engine` performs manifest-ordered feature mapping from observations and uses a per-camera homography for pixel-to-ground speed. If measurements or calibration are missing, the forecast endpoint reports `insufficient_features` instead of inventing a real forecast.
6. `traffic_brain.providers` dispatches the same forecast contract to local ONNX or a compatible remote HTTPS API. The browser never receives the remote API key.
7. Camera feed association is kept separate from model inference. A saved clip mapped to a location is a local replay, not a live RTSP source.

## Event semantics and safety

- A track crossing is counted at most once per `track_id` and `line_id` during the lifetime of that in-memory engine. The engine prunes stale track state and caps history; restart clears the in-memory counters.
- Surge, stationary-track and confidence-drift outputs are heuristic **candidate signals**. They require operator verification and must never directly control signal phases.
- Coordinates are normalized to [0,1]. A real worker must transform actual detector/tracker coordinates with the correct width/height and timestamp, and load the exact counting-line configuration for that camera.
- Speed in km/h must come from valid camera homography and elapsed timestamps. Pixel/s from the AICC Phase-1 model is not km/h.
- The current AICC model contract is a 5-step sequence of 3 features with a 2-class classification result. It is not the proposal's CI predictor for horizons +5/+10/+15.

## Data and model upgrade boundaries

- `data/observations/observation.schema.json`: normalized record contract for future per-window feature datasets.
- `data/labels/forecast-target.schema.json`: target schema for measured target CI at +5/+10/+15 minutes.
- `artifacts/current/manifest.json`: versioned forecaster contract, input key ordering, calibration and thresholds.
- `.env`: selects local ONNX or a compatible remote forecast provider; secrets stay server-side.

Data splits for research must be grouped by camera and/or held-out time range to avoid temporal leakage. Dataset licensing and permission to retain/share faces, plates, and camera video require separate review. Don't use the demo fixture rows as training labels.

## Explicitly not delivered as production fact

This ZIP does not include a completed RTSP reconnecting worker, a fully verified ByteTrack integration, Hanoi-calibrated homography, a Hanoi-trained forecaster, ground-truth metric report, persistent event database, public production authentication, or CCTV credentials. The new interface allows these components to be added without rewriting the dashboard/provider boundary, but it is not proof those components are already connected.
