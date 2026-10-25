# GitHub technical-reference and license audit — Final 8.0.0

Audit date: 2026-10-23. Scope is repository-root/README information available during this build; this is engineering due diligence, not legal advice. A repository code license does not automatically license model weights, training datasets, sample videos, screenshots, bundled media or dependencies.

## Repositories and decisions

| Repo | Useful public-described concepts | License observation | Use in this release |
|---|---|---|---|
| [Traffic-Video-Analytics-Project](https://github.com/salvirezwan/Traffic-Video-Analytics-Project) | YOLO/ByteTrack, FastAPI/WebSockets, line counts, alerts/drift, live dashboard, demo mode | No root `LICENSE` was visible in the repository-root listing reviewed. Treat as all-rights-reserved unless the owner supplies terms. | Reference only; no source/weights/media copied. |
| [real-time-traffic-classifier](https://github.com/anne-malbog/real-time-traffic-classifier) | YOLO11/ByteTrack, line count, event logging, speed/calibration, CPU/ONNX evaluation, failure analysis | GitHub root listing exposes a `LICENSE` entry, but the full license text was not retrievable during the check; exact terms not verified. README lists example Wikimedia videos with separate CC BY terms. | Reference only; no source/weights/media copied. |
| [traffic-object-tracking](https://github.com/felipebridge/traffic-object-tracking) | Track-aware counting, congestion levels, output video/CSV/JSON, config and tests | Repository README states MIT and root listing includes `LICENSE`; retain attribution/notices if any code is reused in future. | No source/weights copied; concepts independently implemented. |
| [TheWatch](https://github.com/creweagle/TheWatch) | Motion-gated counting, duplicate suppression, CPU tuning, MJPEG dashboard, bounded snapshot history | README says MIT; root `LICENSE` not visible in the inspected listing at audit time. Treat as needing full-text confirmation for reuse. | No source/weights copied; reference only. |
| [Traffix.id](https://github.com/dewiazizah968/Traffix.id) | Detector→features→multi-horizon forecasts, weather/holiday feature enrichment, API/dashboard | No root `LICENSE` grant was visible in inspected listing/content. | No source/weights/data copied; reference only. |
| [traffic-dashboard](https://github.com/YachanaParmar/traffic-dashboard) | Geographic display, map layers, traffic risk and prediction visualization | README indicates MIT. | No code/artifacts copied; concepts only. |
| [dinoraptor999/AICC](https://github.com/dinoraptor999/AICC/tree/main/models) | Operator-requested Phase-1 ONNX detector + sequence predictor + scaler bundle | Applicable upstream source/model/dataset license could not be independently verified from accessible content during this build. Operator's local files are not redistributed in the release. | Existing adapter carried forward from operator's prior project; no upstream weights are bundled. Operator must verify before redistribution. |

GitHub's general licensing guidance explains that without a license, default copyright applies and other users are not automatically permitted to reproduce, distribute or create derivative works: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository.

## External live-camera/data access

Hanoi official sources describe a centralized camera-management initiative under Decision 319/QĐ-UBND (17-01-2025); that public announcement is not a public developer API contract. iHanoi's app store listing mentions traffic camera functionality but is also not developer API documentation. Therefore this project contains no scraping/login-bypass code and does not claim an iHanoi feed is connected.

- Hanoi official announcement: https://hanoi.gov.vn/chi-dao-cua-ubnd-thanh-pho-ha-noi/phe-duyet-de-an-tong-the-ve-quan-ly-he-thong-camera-giam-sat-tap-trung-4250117165836444.htm
- iHanoi listing: https://apps.apple.com/vn/app/ihanoi/id6470901306

## Bundled/runtime assets

- `static/street-traffic-demo.jpg`: illustrative fallback photo, explicitly labeled as not live CCTV.
- Leaflet 1.9.4: fetched from official CDN at runtime with SRI; BSD-2-Clause.
- OSM map tiles: requested from public tile service at runtime; attribution is visible. Follow the tile policy and use a contracted/hosted tile service for heavy/production traffic.
- OSRM: default public road-routing endpoint provides network geometry and reference duration; not live congestion and no project-level SLA.

Before public/commercial distribution, verify all model/data/media rights, include required notices, pin dependencies and generate a package SBOM. No model artifacts, real camera recordings, user `.env` or private camera credentials are included in this release ZIP.
