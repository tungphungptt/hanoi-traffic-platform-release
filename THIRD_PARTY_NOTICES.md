# Third-party notices — Hanoi Traffic Platform Final 8.0.0

## Source and model provenance

No source files, weights or datasets/videos were copied from the comparison GitHub repositories listed in `docs/UPSTREAM_LICENSE_AUDIT.md`. The project independently implements selected public technical ideas (bounded video sampling, explainable visual rule labels, route alternatives, tracker-to-analytics interface and forecast-provider contracts). This is not a legal opinion and does not transfer rights to any external code or data.

The AICC-compatible adapter/weights originated from an operator-supplied earlier project and the local `dinoraptor999/AICC` checkout. Their exact upstream source/model/dataset license must be verified by the operator before redistribution. The release ZIP excludes `detector.onnx`, `predictor.onnx`, `scaler.pkl`, video files, and `.env` secrets. `deploy_demo.ps1` can copy these weights from `D:\AICC\models` / `AICC_MODELS_DIR` on the operator's machine; it does not validate license rights or training scope.

## Leaflet

Leaflet 1.9.4 is loaded from the official CDN at runtime using Subresource Integrity (SRI), not vendored into this ZIP. It is distributed under BSD 2-Clause. License: https://github.com/Leaflet/Leaflet/blob/main/LICENSE.

## OpenStreetMap

Raster tiles are requested at runtime from `https://tile.openstreetmap.org/{z}/{x}/{y}.png`; the user interface keeps attribution and the copyright link visible. OSM data is licensed under ODbL; public standard tile service is best-effort and governed by a separate tile use policy. Do not package or bulk-prefetch public OSM tiles. Policies: https://operations.osmfoundation.org/policies/tiles/ and https://www.openstreetmap.org/copyright.

## OSRM

The default route adapter requests road-network routes from `https://router.project-osrm.org`. It is a public demo service, not a licensed live-traffic service, not guaranteed to return multiple alternatives, and not supported by a production SLA in this project. For production, operate or procure a routing provider under suitable terms. Route geometry/time returned by this endpoint must not be labeled live congestion data.

## Python and other runtime dependencies

Each dependency maintains its own license. This file is not a complete SBOM and does not substitute package-level review for commercial/public deployment. Pin the release environment and generate a CycloneDX or equivalent inventory before distributing to a production customer.
