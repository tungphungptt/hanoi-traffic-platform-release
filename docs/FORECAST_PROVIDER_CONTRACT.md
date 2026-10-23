# Forecast provider contract v2 — Final 8.0.0

The provider interface is intentionally separate from the dashboard and camera/video workflow, so a later trained Hanoi model can replace the provider without rewriting the web UI.

## Provider modes

- `local_onnx`: load `artifacts/current/forecaster.onnx` using the ordered features in `artifacts/current/manifest.json`. In this ZIP the manifest is a schema scaffold; the trained multi-horizon forecaster weight file is not included. Without it, runtime must state model unavailable/demo rather than presenting the illustrative baseline as a trained forecast.
- `remote_http`: submit a versioned feature vector to a configured prediction API. No silent fallback is allowed on timeout, non-2xx, invalid JSON, missing horizon or malformed CI.

## Configuration

```dotenv
FORECAST_PROVIDER=remote_http
FORECAST_API_URL=https://model.example/api/forecast
FORECAST_API_KEY=set-in-secret-manager
FORECAST_API_TIMEOUT_SECONDS=8
FORECAST_API_PUBLIC_ENABLED=false
```

Keep credentials server-side. Never put server secrets in browser JS, public tile URLs, manifest or logs. Public forecasting endpoints should remain blocked unless auth, rate limiting and cost quotas are ready.

## Request/response

Request contract identifier is `hanoi-traffic-forecast/2.0`, manifest-ordered keys/features, horizons `[5,10,15]`. The service must return exactly one finite CI in `[0,1]` for each horizon, in order:

```json
{
  "contract_version": "hanoi-traffic-forecast/2.0",
  "predictions": [
    {"horizon_minutes": 5, "ci": 0.41},
    {"horizon_minutes": 10, "ci": 0.49},
    {"horizon_minutes": 15, "ci": 0.58}
  ]
}
```

CI must have identical meaning and calibration between train and serve. A provider may not duplicate one number across horizons, translate raw logits to CI without explicit calibration, silently fill missing features, or pass uncalibrated pixel/s values as km/h.

## Feature and model responsibilities

`artifacts/current/manifest.json` declares 8 features: `flow.car_count`, `flow.motorcycle_count`, `flow.pcu_equivalent`, `speed.mean_kmh`, `speed.cv_ratio`, `density.areal_occupancy`, `temporal.pcu_lag_1`, `temporal.speed_delta_1`. These require a real, correctly timed detector/tracker stream, verified camera geometry/calibration, and identical feature engineering during training and inference. If observations are incomplete, `POST /api/forecast/from-observations` can return `insufficient_features` instead of making an unreliable prediction.

A new provider belongs in `traffic_brain/providers.py` behind the existing normalized contract, with tests for timeout/HTTP errors, oversized responses, missing features and schema validation. The current `project_trend_baseline` in `traffic_brain/video_review.py` is an explicitly labeled linear trend baseline for a clip; it is not this trained provider.
