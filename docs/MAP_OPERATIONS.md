# Map and route operations — Final 8.0.0

## Base map

The main geographic view uses Leaflet + OpenStreetMap roads/places for Hanoi. This is a real basemap, not real-time traffic. Camera markers from the current `data/cameras.json` are illustrative reference coordinates and must be validated against actual camera metadata before attaching telemetry.

- Keep OSM attribution visible.
- Standard OSM tile service is best-effort. Do not bulk download/prefetch tiles; production/high volume should use a provider with a suitable SLA or a self-hosted tile stack.
- `MAP_TILE_URL_TEMPLATE`, `MAP_BASE_NAME`, `MAP_ATTRIBUTION` and `CAMERA_REFRESH_SECONDS` configure base map and metadata polling.

## Workflow in v5

1. Select origin A in the “Dự báo & phương án” page.
2. Use a local/uploaded clip to generate sampled annotated frames. The state rule is a visual baseline, not expert ground truth.
3. Select destination, then fetch route alternatives from OSRM. The map draws real road-network geometry when the routing API and Internet are available.
4. If telemetry status is not fresh and complete for all alternatives, route ranking is static/reference duration only. The UI/API response includes the ranking basis and disclaimer. Do not call it real-time congestion-aware routing.
5. Video context can only be associated with route origin if the analyzed clip's camera ID equals the selected origin camera ID.

## Live traffic overlay

Set `TRAFFIC_TILE_URL_TEMPLATE` and `TRAFFIC_TILE_ATTRIBUTION` only after selecting a traffic provider licensed for the deployment. Provider keys visible in browser requests must be restricted to the website/domain and appropriate operations; do not expose server secrets in this URL. Empty configuration keeps the live overlay off and does not synthesize traffic colors.

## Live telemetry contract

`POST /api/traffic/telemetry` receives a measurement that an authorized collector already computed. It does not open a stream. Requirements include:

- camera exists, `location_verified=true`, and coordinates present;
- source exactly `rtsp` (set only by your trusted collector); `observed_at` no older than 180 seconds and no more than 15 seconds in the future;
- CI in `[0,1]`, and optional speed_kmh/flow_pcu in allowed ranges;
- protected deployment with authentication/authorization, because the contract endpoint is not a substitute for an authenticated ingestion gateway.

Data is in-memory and disappears on server restart. No collector runs by default. The route scorer uses telemetry within 350 meters of geometry; only when all candidate routes have recent live observations does it mark the comparison `traffic_aware=true`.

## Sources and permissions

Do not guess camera URLs, scrape private app sessions, or bypass iHanoi/VMS login. Official Hanoi sources describe the city-managed camera ecosystem; request technical documentation/access rights from the data owner. See `docs/HANOI_LIVE_INTEGRATION.md` and `docs/UPSTREAM_LICENSE_AUDIT.md`.
