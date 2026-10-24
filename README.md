# Hanoi Traffic Platform — Final 8.0.0

**Mục tiêu vận hành:** đánh giá đoạn A từ clip được cấp quyền, tạo ảnh minh chứng theo thời điểm, hiển thị nguy cơ theo luật giải thích được khi đủ điều kiện, lấy các tuyến thay thế trên mạng đường phố Hà Nội và chọn phương án dựa trên nguồn dữ liệu có sẵn. Không coi việc “chạy phân tích video” là đích đến; đầu ra phục vụ quyết định lộ trình.

> **Ranh giới dữ liệu:** bản đồ nền OpenStreetMap là dữ liệu địa lý thật. Các camera/tọa độ trong `data/cameras.json` là điểm minh họa gần khu vực được nêu tên, chưa xác minh thiết bị CCTV tại vị trí đó. Nếu chưa có camera/traffic telemetry thật được cấp quyền, thời gian tuyến từ OSRM chỉ là thời gian tham chiếu của mạng đường, không phải ETA kẹt xe thời gian thực. Video upload không trở thành luồng CCTV trực tiếp.

## Luồng làm việc chính

1. **Nạp clip** từ máy tính để phân tích ngay; chọn đoạn A là tùy chọn và chỉ dùng để gắn clip với tuyến. Video trong `data/` vẫn dành cho vận hành local.
2. Hệ thống lấy mẫu frame, đánh dấu detection từ AICC Phase 1 và gán mức Thông thoáng/Đông đúc/Ùn ứ/Tắc nghẽn bằng luật giải thích được dựa trên số xe phát hiện, phần diện tích bounding box và chuyển động tương đối trong ảnh.
3. **Xem dự báo thử nghiệm**: Hanoi LSTM yêu cầu tối thiểu 30 giây clip lấy mẫu 1 Hz và hiển thị xác suất tắc +5/+10/+15 giây; điểm xác suất chưa hiệu chuẩn, không thay thế dự báo theo phút.
4. Baseline xu hướng +5/+10 phút vẫn là một luồng riêng, chỉ được bật khi người dùng xác nhận clip giữ thời gian thật và đủ mẫu; đây là ngoại suy tuyến tính, không phải LSTM đã huấn luyện.
5. **Chọn điểm đến và so sánh tuyến thay thế**. Hệ thống gọi OSRM lấy hình học tuyến đường từ mạng đường OSM. Nếu chưa có telemetry RTSP mới xác minh và phủ đủ mọi phương án, tuyến ưu tiên chọn theo thời gian tham chiếu, không tuyên bố là “ít tắc nhất”. Nếu các tuyến đều được phủ bởi telemetry RTSP mới, dùng heuristic có phạt CI và ghi rõ cơ sở tính điểm.

## Khởi chạy trên Windows

Giải nén ZIP; mở PowerShell tại thư mục gốc dự án và chạy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy_demo.ps1
```

Bootstrap sẽ:

- dùng Python 3.12 64-bit (fallback 3.11 nếu có), tạo `.venv`;
- cài dependency từ `requirements-dev.txt`;
- đồng bộ nguyên bộ `detector.onnx`, `predictor.onnx`, `scaler.pkl`, `manifest.json` từ `D:\AICC\models` hoặc `AICC_MODELS_DIR` nếu đủ cả 4 file; không trộn trọng số mới với manifest khác phiên bản;
- kiểm tra phiên bản Python, tính nhất quán package bằng `pip check`, kiểm tra release/manifest, cú pháp và chạy pytest;
- khởi động Uvicorn trên `127.0.0.1:8000` và đợi `/api/health` sẵn sàng.

Tắt server bằng `Ctrl+C` trong cửa sổ Uvicorn. Có thể đổi cổng bằng `-Port 8001`. Nếu cổng đang sử dụng, script dừng và báo cách khắc phục. Bản ZIP không kèm footage/video, `.env`, hoặc trọng số model upstream vì chưa đủ xác minh quyền phân phối. Bản local cần các artifact AICC trong đường dẫn nêu trên để chạy inference video.

## Phân biệt bản local và demo Render khi phân tích video

`https://hanoi-traffic-platform.onrender.com/` là một backend riêng. Cấu hình `render.yaml` giữ bản demo công khai ở chế độ chỉ đọc và tắt phân tích video. Khởi động server trên Windows không bật upload trên trang Render. Hãy mở địa chỉ `http://127.0.0.1:8000` (hoặc cổng đã chọn) do launcher local thông báo để phân tích clip.

Tại checkout Git trên Windows, dừng Uvicorn cũ bằng `Ctrl+C`, cập nhật code rồi chạy:

```powershell
git status --short
git switch main
git pull --ff-only origin main
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy_demo.ps1
```

Nếu có thay đổi local, giữ lại chúng và giải quyết trước khi chuyển branch; không reset hoặc ghi đè dữ liệu cá nhân. Nếu thư mục được giải nén từ ZIP và không có `.git`, cần cập nhật từ bản phát hành đã chứa bản sửa thay vì chạy `git pull`.

Launcher dùng `.venv\Scripts\python.exe` với Python 3.12/3.11, đặt `PUBLIC_DEMO_MODE=false` và `AICC_VIDEO_ENABLED=true` trước khi tạo process con, đồng thời in đường dẫn dự án/interpreter và các cờ không chứa secret. Nó kiểm tra cổng cả trước setup và trước khi chạy Uvicorn, xác minh `app.main` thuộc checkout hiện tại, rồi chỉ báo `DEMO READY` sau khi `tools/check_local_runtime.py` kiểm tra health, mã phiên khởi chạy được process con kế thừa, PID server thực tế, quyền phân tích thực tế và sự hiện diện của bộ model. Process environment được ưu tiên trước `.env`; không cần ghi đè `.env` cá nhân. Python 3.14 không thuộc bộ runtime được launcher hỗ trợ.

Có thể kiểm tra chế độ của server đang mở bằng PowerShell trong cửa sổ khác:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/system | Select-Object public_demo_mode,process_id,video_review
Invoke-RestMethod http://127.0.0.1:8000/api/aicc/status | Select-Object public_demo_mode,video_enabled,enabled,artifacts_present,blocked_reason
```

Ở bản local được phép phân tích, `public_demo_mode` phải là `False`, `video_enabled`, `enabled` và `artifacts_present` phải là `True`. Nút upload dùng file đã chọn từ máy tính, độc lập với danh sách `data/`; danh sách trống chỉ khóa nút phân tích từ `data/`. File được chọn chưa được upload cho đến khi bấm phân tích. Nếu khởi động lại backend với cấu hình khác, dùng nút làm mới để đọc lại trạng thái có thẩm quyền từ API.

Bộ kiểm thử có test process con/HTTP trên Linux và test tĩnh launcher. Các test đó không thay thế việc chạy Windows PowerShell thực tế trên máy Windows.

## Cấu trúc dự án

```text
app/main.py                         FastAPI endpoints, job queue, artifact/provider runtime
traffic_brain/video_review.py       Frame sampling, visual heuristic, experimental clip forecasts
traffic_brain/hanoi_video_forecaster.py  Hanoi LSTM ONNX runtime for +5/+10/+15 seconds
traffic_brain/routing.py            OSRM road routing + conservative live-telemetry scoring
traffic_brain/providers.py          Local ONNX / external HTTP forecast adapter
traffic_brain/feature_engine.py     Manifest-ordered features, PCU, homography/calibration primitives
traffic_brain/analytics.py          Tracker-to-analytics contract, line crossing, bounded candidate events
aicc/pipeline.py                    AICC Phase-1 detector + sequence classifier
static/                              Operational dashboard and route/frames UI
data/cameras.json                   Camera slots; coordinates are reference/demo until verified
data/camera_feeds/                  Per-camera clip assignment (local data; ignored by Git)
data/camera_calibrations/            Schema/base for per-camera ROI and homography
 data/observations/                  Versioned schema for measured features
 data/labels/                        Versioned schema for future +5/+10/+15 ground truth
models/manifest.json                 AICC v1.0-car-only contract
artifacts/current/manifest.json      Separate multi-horizon forecaster contract scaffold
```

## Important model distinction

Two model contracts are deliberately separate:

- **AICC Phase 1:** `models/manifest.json` plus `detector.onnx`, `predictor.onnx`, `scaler.pkl`. Current manifest is `v1.0-car-only`, sequence length 5, features `car_count`, `avg_speed`, `speed_cv`, and 2 output classes. Speed derived by this AICC pipeline is pixel/second. This classifier does **not** predict congestion +5/+10/+15 minutes and does not recognize Hanoi motorcycle flow accurately enough for production claims.
- **Hanoi multi-horizon forecaster:** `artifacts/current/manifest.json` specifies ordered features and horizons 5/10/15. The manifest is a contract/schema only; this release does **not** include a trained `forecaster.onnx` or a validated Hanoi training dataset. `FORECAST_PROVIDER=remote_http` can connect a separately trained compatible service; it must return one calibrated CI per requested horizon and must not be treated as live if the upstream data isn't live.

The AICC-compatible integration is retained from the operator's earlier project. Its source/weights provenance and licenses must be checked independently before public redistribution. `deploy_demo.ps1` copies local model artifacts when available; it does not assert that they are trained for Hanoi or that every model file came from a verified upstream release.

## Main APIs

| Endpoint | Purpose | Boundary/limitations |
|---|---|---|
| `GET /api/health` | Service health | Does not assert model/camera health |
| `GET /api/cameras` | Camera metadata and assigned local clips | Bundled camera status/metrics are fixtures unless replaced with verified ingestion |
| `POST /api/scenarios/video-review` | Upload and queue video review | Public mode requires `PUBLIC_VIDEO_REVIEW_ENABLED=true`; 100 MB / 5 minute cap |
| `POST /api/scenarios/video-review-from-data` | Queue a file under `data/` | Rejects path escape, unsupported file type, oversize input and public demo writes |
| `GET /api/scenarios/video-reviews/{job_id}` | Job progress/result | In-memory job state; lost on restart |
| `GET /api/scenarios/video-reviews/{job_id}/frames/{filename}` | View a generated JPEG | Constrained job ID and filename pattern |
| `POST /api/routes/alternatives` | Fetch road alternatives + score | OSRM reference duration unless all alternatives have recent RTSP telemetry |
| `POST /api/traffic/telemetry` | Ingestion contract for a trusted RTSP collector | Does not capture RTSP itself; requires verified camera coordinates and freshness checks; no persistent store |
| `GET /api/traffic/telemetry/status` | Number of recent telemetry cameras | In-memory status; no collector runs by default |
| `POST /api/analytics/ingest-frame` | Ingest already-tracked normalized points | Contract for a future detector/tracker worker, not an image detector |
| `POST /api/forecast/from-observations` | Build manifest-ordered features and pass them to provider | Refuses prediction when required measured features are missing |

## Real Hanoi traffic/CCTV integration

The app can ingest video streams only after the operator supplies a legal, reachable RTSP/HLS/WebRTC feed and connects a collector/decoder. No authenticated iHanoi/VMS stream is bundled and the application does not scrape, bypass logins, or invent camera feed URLs. Hanoi's official camera system is managed by public authorities; request access/technical documentation through the authorized data owner before connecting. See `docs/HANOI_LIVE_INTEGRATION.md`.

For live traffic overlays, set `TRAFFIC_TILE_URL_TEMPLATE` and attribution to a provider whose license/terms allow the use. An ordinary OSM basemap is not live traffic. The default OSRM public demo endpoint supplies road-network geometry/reference routing, not real-time traffic ETA and not a production SLA. Configure an authorized routing/traffic provider before operational use. OSM tile use must comply with https://operations.osmfoundation.org/policies/tiles/.

## Forecast model/API contract

`FORECAST_PROVIDER=local_onnx` keeps local inference. `FORECAST_PROVIDER=remote_http` uses `FORECAST_API_URL`, server-only `FORECAST_API_KEY`, and timeout in `.env`. The expected response is `hanoi-traffic-forecast/2.0`, with three ordered predictions for 5, 10 and 15 minutes and CI in `[0,1]`. See `docs/FORECAST_PROVIDER_CONTRACT.md`. Provider/API status does not reveal secrets. Do not enable an external forecast API on a public endpoint without authentication, quotas and cost monitoring.

## Testing and evaluation

```powershell
.\.venv\Scripts\python.exe -m compileall -q app aicc traffic_brain tests
.\.venv\Scripts\python.exe -m pytest -q
node --check static/app.js
node --test tests/video_review_frontend.test.cjs
```

`tools/evaluate_forecast.py` computes MAE/RMSE at +5/+10/+15 on a CSV with verified ground truth. The header-only `data/examples/forecast-evaluation-template.csv` is an example, not a benchmark. Required independent evaluation should split by time/camera, and include mAP, IDF1, counting error, CI MAE/RMSE, macro-F1 per horizon, inference latency and RAM.

## GitHub ideas/license review

Useful reference concepts were independently implemented where appropriate: sampled detection and counting-line analysis, duplicate suppression, bounded event history, dashboard/forecast separation and CPU-aware processing. No source code/model weights were copied from comparison repositories. License/provenance results and unresolved items are documented in `docs/UPSTREAM_LICENSE_AUDIT.md` and `THIRD_PARTY_NOTICES.md`. This is engineering documentation, not legal advice.

## Deployment boundaries

The local bootstrap binds only to loopback. `render.yaml` keeps `PUBLIC_DEMO_MODE=true` and `PUBLIC_VIDEO_REVIEW_ENABLED=false`, so the public deployment stays read-only by default. The dedicated upload opt-in is retained for an explicitly configured protected deployment; stored-file and other public writes remain disabled. Public uploads are limited to 100 MB and 5 minutes and consume shared CPU, so deployment should add authentication, rate limits, storage-retention rules, and abuse monitoring before wider use. Video LSTM outputs are experimental, second-based, and uncalibrated. This release is a research/operational prototype, not a certified traffic-control system.

## Final 8.0 release hardening

See [`docs/V8_IMPROVEMENT_MATRIX.md`](docs/V8_IMPROVEMENT_MATRIX.md) for the acceptance criteria and remaining gates. See also [`docs/RELEASE_QA.md`](docs/RELEASE_QA.md) for verified checks and limitations.

- `deploy_demo.ps1` is checked for exactly one UTF-8 BOM to avoid Windows PowerShell parser issues. Its Python gate captures the executable exit code immediately, parses a complete version output, and recreates only the project-owned `.venv` if its interpreter cannot be verified.
- Bootstrap checks Python 3.11/3.12, runs `pip check`, syncs all four AICC artifacts as one bundle, executes `tools/model_smoke.py` if weights are present, then runs `tools/release_check.py`, compileall and pytest before starting Uvicorn.
- For a real local clip, run `python tools/verify_video.py "data\input.mp4" --sample-fps 3 --output data\diagnostics\aicc-video-smoke.json`. This invokes the AICC video pipeline; output explicitly preserves the pixel/second and non-forecasting limitations.
- To verify an extracted bundle without starting the service: run `python tools/release_check.py`, `python -m pytest -q`, `python -m compileall -q app aicc traffic_brain tests tools`, and `node --check static/app.js`. `docs/RELEASE_MANIFEST.json` records SHA-256 for immutable release source. Mutable operator manifests (`models/manifest.json`, `artifacts/current/manifest.json`, `data/cameras.json`) and local feed indexes are intentionally excluded from that inventory; `tools/release_check.py` separately parses/validates their contracts.
- This is a hardened local release candidate, not a claim of defect-free production certification. Static checks do not verify licensed CCTV access, live traffic data availability, or real Hanoi model accuracy.
