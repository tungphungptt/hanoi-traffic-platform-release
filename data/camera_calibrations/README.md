# Per-camera calibration base

Create one JSON file per authorized camera ID before using physical speed, density, or lane-count metrics. Do not copy calibration from another camera or use the example dashboard lat/lon as a homography.

Required steps:

1. Define the ROI polygon in normalized image coordinates `[0,1]` with points in consistent order.
2. Measure known ground-plane points in metres and record a homography that maps image pixels to road-plane coordinates; validate it visually and against a measured segment.
3. Record effective road length/area, frame resolution, time source, calibration date and operator.
4. Define counting line endpoints with stable IDs in normalized coordinates.
5. Test speed against a measured reference. If calibration fails, set `calibrated: false`; forecasts must not treat missing speeds as measurements.

`camera-calibration.schema.json` defines the intended persisted format. The current tracker-ingestion endpoint takes counting lines in each request; this directory is a configuration base for a future worker to load and validate those settings. It is not wired to a real camera or a production calibration manager yet.
