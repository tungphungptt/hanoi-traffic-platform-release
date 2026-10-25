# Changelog

## 8.0.0 — Reliable Windows bootstrap and model preflight

- Replaced fragile Python version detection with `Get-PythonMinor`, capturing `$LASTEXITCODE` immediately and parsing a complete version line.
- Recreates only the project-owned `.venv` if the interpreter cannot be verified as supported Python 3.11/3.12; global Python installations and files outside `.venv` are untouched.
- Added `tools/model_smoke.py` to load all four trusted AICC artifacts and execute detector/predictor CPU inference before the server starts. It fails closed if a present bundle is corrupt or incompatible.
- Added `tools/verify_video.py` for end-to-end inference on a selected local clip, with bounded file size/duration, progress and a JSON diagnostic summary.
- Added per-file SHA-256 inventory support, verification in the release checker, and expanded regression tests.
- Kept AICC Phase 1 separate from the untrained multi-horizon Hanoi forecaster; no claim of live CCTV access or prediction accuracy is made.

## 5.0.0 — Scenario-first prediction and route alternatives

- Reorganized the core workflow around origin A → sampled frame evidence → optional timing-gated +5/+10 minute trend baseline → real road-network route alternatives to the destination. Kept legacy AICC diagnostics in an advanced collapsed section.
- Removed fabricated citywide forecast, trend and alert data from the default operational views. Unverified configured camera points show neutral reference status; operational counts stay blank or zero until fresh trusted telemetry is ingested.
- Added/strengthened `traffic_brain/video_review.py` usage in the API: annotated JPG snapshots, frame-state counts, progress polling, file-size/duration constraints and clear warnings that uncalibrated speed is pixel/second and AICC Phase 1 is not a Hanoi forecast.
- Integrated OSRM alternative route geometry and conservative scoring: fresh RTSP telemetry is used only when every candidate route is covered; otherwise ranking is based on reference road-network duration, never mislabeled as live traffic.
- Added a protected ingestion contract for timestamped telemetry and explicit live-telemetry status endpoint. The app still does not spawn an RTSP collector; documented the authorized integration boundary.
- Added tests for visual-state logic, trend preconditions, actual sampled frame output, upload/data path boundaries, frame URL traversal, Hanoi bounds, route-to-origin analysis matching and telemetry freshness/verified location conditions.
- Updated release/operator documentation, forecast-provider contract, map instructions, Hanoi official-source integration notes and GitHub/license audit. No comparison-repository source or weights were copied into this release.

## 4.0.0 — Real Hanoi basemap and map data refresh

- Replaced schematic map with interactive Leaflet/OpenStreetMap basemap, metadata refresh, camera marker/video synchronization and optional configured traffic overlay.
- Added transparent indication that OSM does not provide live congestion data and that bundled camera coordinates/metrics were fixtures.

## 3.0.0 — Analytics and extensibility foundations

- Added independent traffic analytics primitives, tracker-ingestion contract, observation/ground-truth schemas and MAE/RMSE evaluation CLI.
- Added upstream license audit and forecast-provider contract, with AICC/video/forecast limitations called out.

## 2.0.0

- See preceding source release documentation.
