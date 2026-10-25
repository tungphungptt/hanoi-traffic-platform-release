# Weather and segment forecast rollout

- Weather is fetched server-side from Open-Meteo for the requested coordinate; the default UI uses central Hanoi.
- The weather endpoint has a bounded in-memory cache and marks stale cached responses. Weather is not invented if no upstream response/cache exists.
- Open-Meteo free service terms describe free non-commercial use and CC BY 4.0 attribution. Contact Open-Meteo before commercial use.
- Traffic segment +5/+10 results are a transparent linear-trend baseline from authorized, fresh RTSP collector telemetry. They are not an evaluated ML model and are shown only after at least five observations spanning four minutes.
- The telemetry history and weather cache are in memory, not durable across restarts. A production deployment needs protected collector authentication, persistent storage, a sustained data feed, labelled outcomes, and independent forecast evaluation.
- Render remains in public demo/read-only mode by default; do not switch it off until write APIs are protected.

## Hanoi-only geography guard
- Weather coordinates are validated against the bundled Hanoi administrative boundary polygon; coordinates outside the polygon are rejected before an upstream request.
- Segment forecasts only include configured camera coordinates inside that same polygon. Cameras outside the city are excluded from this endpoint.
- Boundary file: `data/geo/hanoi_boundary.geojson`, fetched from one of the OpenStreetMap administrative relation endpoints (Nominatim GeoJSON, the OSM polygon export, or the official OSM relation API XML). The chosen endpoint is recorded in the GeoJSON properties. OpenStreetMap data is under ODbL and must retain attribution to OpenStreetMap.
- This geofence narrows the service area; it does not make the baseline accurate by itself. Existing traffic forecasts remain a non-validated linear baseline and require fresh authorized telemetry. Weather remains a grid/model forecast, not a street-level rain-gauge measurement.
