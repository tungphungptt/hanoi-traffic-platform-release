# Weather and segment forecast rollout

- Weather is fetched server-side from Open-Meteo for the requested coordinate; the default UI uses central Hanoi.
- The weather endpoint has a bounded in-memory cache and marks stale cached responses. Weather is not invented if no upstream response/cache exists.
- Open-Meteo free service terms describe free non-commercial use and CC BY 4.0 attribution. Contact Open-Meteo before commercial use.
- Traffic segment +5/+10 results are a transparent linear-trend baseline from authorized, fresh RTSP collector telemetry. They are not an evaluated ML model and are shown only after at least five observations spanning four minutes.
- The telemetry history and weather cache are in memory, not durable across restarts. A production deployment needs protected collector authentication, persistent storage, a sustained data feed, labelled outcomes, and independent forecast evaluation.
- Render remains in public demo/read-only mode by default; do not switch it off until write APIs are protected.
