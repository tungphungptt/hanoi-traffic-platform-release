# hanoiFlow — Hanoi Traffic Platform 8.0.0

**Mục tiêu vận hành:** đánh giá đoạn A từ clip được cấp quyền, tạo ảnh minh chứng theo thời điểm, hiển thị nguy cơ theo luật giải thích được khi đủ điều kiện, lấy các tuyến thay thế trên mạng đường phố Hà Nội và chọn phương án dựa trên nguồn dữ liệu có sẵn. Không coi việc “chạy phân tích video” là đích đến; đầu ra phục vụ quyết định lộ trình.

> **Ranh giới dữ liệu:** bản đồ nền OpenStreetMap là dữ liệu địa lý thật. Các camera/tọa độ trong `data/cameras.json` là điểm minh họa gần khu vực được nêu tên, chưa xác minh thiết bị CCTV tại vị trí đó. Nếu chưa có camera/traffic telemetry thật được cấp quyền, thời gian tuyến từ OSRM chỉ là thời gian tham chiếu của mạng đường, không phải ETA kẹt xe thời gian thực. Video upload không trở thành luồng CCTV trực tiếp.

## Luồng làm việc chính

1. **Nạp clip** từ máy tính để phân tích ngay; chọn đoạn A là tùy chọn và chỉ dùng để gắn clip với tuyến. Video trong `data/` vẫn dành cho vận hành local.
2. Hệ thống lấy mẫu frame, đánh dấu detection từ AICC Phase 1 và gán mức Thông thoáng/Đông đúc/Tắc nghẽn nghiêm trọng bằng luật giải thích được dựa trên số xe phát hiện, phần diện tích bounding box và chuyển động tương đối trong ảnh.
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

## Phân tích video trên website public

Trên trang “Tải video để bắt đầu phân tích”, chọn video từ máy tính rồi bấm **Phân tích video**. Camera là tùy chọn. Kết quả gồm nhãn trạng thái, CI, ảnh minh chứng, LSTM thử nghiệm theo giây, baseline xu hướng nếu đủ dữ liệu và xuất CSV. Các chỉ số từ clip chưa phải dữ liệu camera trực tiếp đã kiểm định.

Public video review được bật riêng bằng `PUBLIC_VIDEO_REVIEW_ENABLED=true`. `PUBLIC_DEMO_MODE=true` và `AICC_VIDEO_ENABLED=false` vẫn giữ các thao tác model, manifest, clip camera và AICC nội bộ bị khóa. Giới hạn ứng dụng là **500 MiB (524.288.000 byte) và 5 phút**. Trình duyệt kiểm tra theo `upload_limit_bytes` từ `/api/aicc/status`; `upload_limit_mb` được giữ để tương thích với client cũ. Model AICC và Hanoi phải có đủ artifact tương thích trong checkout; không tự tải model hoặc đổi thuật toán khi nhận upload.

Trình duyệt báo tiến độ từ số byte XHR đã truyền. Sau HTTP 202, giao diện theo dõi `job_id`; phần trăm phân tích lấy từ worker. Nếu mất kết nối, polling tự thử lại và nút **Theo dõi lại kết quả** tiếp tục cùng tác vụ, không upload lại. Chỉ job ID được giữ trong sessionStorage, không lưu video hoặc tên file vào trình duyệt. Lỗi 404 sau khi restart/hết hạn cho phép gửi lại clip.

Nạp model và phân tích chạy trong worker sau HTTP 202; kiểm tra codec/thời lượng chạy ngoài event loop. Chỉ một lượt nhận upload public hoặc tác vụ CPU được tiếp nhận cùng lúc. Có quota toàn dịch vụ 6 lượt nhận/phút và phản hồi 429/Retry-After; không dùng `X-Forwarded-For` hoặc IP giả mạo để cấp quota. Middleware chặn body vượt mức kể cả thiếu Content-Length, giới hạn đọc không hoạt động 60 giây và tổng thời gian nhận 30 phút; XHR cũng có deadline 30 phút. Đây là giới hạn ứng dụng, không phải tuyên bố về timeout của Render.

Kiểm thử hồi quy tái hiện được lỗi vòng đời upload gate: giữ cờ nhận upload đến khi toàn bộ ASGI app kết thúc khiến cleanup sau response vẫn chặn request tiếp theo bằng 409; timeout/cancellation của receive cũng phải nhả đúng lượt sở hữu. Hotfix nhả lease khi response kết thúc hoặc request lỗi/ngắt kết nối; finalizer cũ không được nhả lease của request mới. Tác vụ CPU còn chạy vẫn phải trả 409. Đây là nguyên nhân đã kiểm chứng trong mã nguồn; chưa đủ để khẳng định mọi 409 trên một deployment cụ thể có cùng nguyên nhân. Cần đối chiếu build và trạng thái thực tế sau deploy.

Endpoint chỉ nhận multipart với một file và tối đa hai trường metadata, mỗi trường tối đa 4 KiB, để tránh giữ nhiều phần nhỏ trong RAM. Upload được Starlette spool xuống đĩa sau 1 MiB rồi ứng dụng sao chép theo chunk 1 MiB vào file riêng; không đọc toàn bộ video vào RAM. Trong lúc sao chép file 500 MiB, hai bản tạm có thể chiếm khoảng **1 GiB đĩa**, chưa tính model và output. Kiểm tra dung lượng trống trước khi nhận body phải dành chỗ cho cả spool, bản sao và phần dự phòng; cleanup vẫn chạy khi thành công, lỗi hoặc hủy request.

Video tạm được xóa sau khi xử lý/lỗi. Kết quả tối đa 8 tác vụ và hết hạn sau 1 giờ; cleanup chạy định kỳ, bao gồm output/upload mồ côi sau restart. File nằm trên filesystem ephemeral của Render; restart làm mất trạng thái tác vụ. Render Free có thể spin down sau 15 phút không có lưu lượng và cần khởi động lại khi có request mới (https://render.com/docs/free); thời gian chờ khởi động không phải tiến độ phân tích clip. Không dùng kết quả như kho lưu trữ lâu dài.

`render.yaml` khai báo cấu hình mong muốn; biến môi trường trên Render Dashboard có thể override và cần được kiểm tra riêng. Sau deploy, đối chiếu `GET /api/health` trường `build_commit` với commit trên `main`, kiểm tra `/api/aicc/status` có `video_review_enabled=true`, và chạy smoke upload nhỏ. **500 MiB là giới hạn trong mã nguồn, chưa phải chứng nhận upload 500 MiB thành công qua Render.** [Render Free](https://render.com/docs/compute-plans) có 0,1 CPU và 512 MB RAM; filesystem ephemeral có giới hạn nhưng [tài liệu xử lý lỗi](https://render.com/docs/troubleshooting-deploys) không nêu quota số cụ thể. [Tài liệu HTTP](https://render.com/docs/render-vs-vercel-comparison) cho phép response tới 100 phút; không suy ra từ đó trần request body hoặc khả năng xử lý video trên gói hiện tại. Chưa xác minh được giới hạn body/proxy trên đường truyền của deployment; cần kiểm tra trước khi thử file lớn. Nếu xác nhận proxy có trần thấp hơn, dùng upload chia chunk/resumable hoặc object storage phù hợp; chỉ tăng timeout không vượt được trần đó.

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

## Thang chỉ số CI

CI liên tục trong [0,1] dùng đúng ba mức: `0 ≤ CI < 0.30` là **Thông thoáng**,
`0.30 ≤ CI < 0.55` là **Đông đúc**, `0.55 ≤ CI ≤ 1` là **Tắc nghẽn nghiêm trọng**.
Các khoảng trên giao diện được viết gọn với hai chữ số thập phân; không làm tròn CI trước khi so ngưỡng.
Khóa API tương ứng là `free`, `busy`, `gridlock`. Manifest giữ trường `heavy` để tương thích,
nhưng `heavy` và `gridlock` đều phải bằng `0.55`; không có mức thứ tư.
Quy tắc chuẩn nằm trong `traffic_brain/traffic_levels.py`; frontend có kiểm thử đối chiếu cùng thang.
API và CSV giữ độ chính xác của CI, chỉ làm tròn khi trình bày trên giao diện.
Video giữ EMA để làm mượt CI; `state_key`/`state_label` luôn phân loại đúng giá trị `ci` trả về.
Trạng thái hysteresis riêng được giữ trong `smoothed_state_key`/`smoothed_state_label` để tham khảo,
không ghi đè nhãn chính thức và không được dùng làm nhãn chính trên giao diện. `raw_state_*` phân loại `raw_ci`.
Các lớp nhị phân AICC/LSTM và trạng thái kết nối camera là những thang độc lập, giữ nguyên.

## Main APIs

| Endpoint | Purpose | Boundary/limitations |
|---|---|---|
| `GET /api/health` | Service health | Does not assert model/camera health |
| `GET /api/cameras` | Camera metadata and assigned local clips | Bundled camera status/metrics are fixtures unless replaced with verified ingestion |
| `POST /api/scenarios/video-review` | Upload and queue video review | Public mode requires `PUBLIC_VIDEO_REVIEW_ENABLED=true`; application cap 500 MiB / 5 minutes |
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

The local bootstrap binds only to loopback. `render.yaml` keeps `PUBLIC_DEMO_MODE=true` and enables only the dedicated public video-review upload with `PUBLIC_VIDEO_REVIEW_ENABLED=true`; stored-file and other public writes remain disabled. The application limits public uploads to 500 MiB and 5 minutes; actual Render capacity remains subject to deployment verification described above. Uploads consume shared CPU, so deployment should add authentication, rate limits, storage-retention rules, and abuse monitoring before wider use. Video LSTM outputs are experimental, second-based, and uncalibrated. This release is a research/operational prototype, not a certified traffic-control system.

## Final 8.0 release hardening

See [`docs/V8_IMPROVEMENT_MATRIX.md`](docs/V8_IMPROVEMENT_MATRIX.md) for the acceptance criteria and remaining gates. See also [`docs/RELEASE_QA.md`](docs/RELEASE_QA.md) for verified checks and limitations.

- `deploy_demo.ps1` is checked for exactly one UTF-8 BOM to avoid Windows PowerShell parser issues. Its Python gate captures the executable exit code immediately, parses a complete version output, and recreates only the project-owned `.venv` if its interpreter cannot be verified.
- Bootstrap checks Python 3.11/3.12, runs `pip check`, syncs all four AICC artifacts as one bundle, executes `tools/model_smoke.py` if weights are present, then runs `tools/release_check.py`, compileall and pytest before starting Uvicorn.
- For a real local clip, run `python tools/verify_video.py "data\input.mp4" --sample-fps 3 --output data\diagnostics\aicc-video-smoke.json`. This invokes the AICC video pipeline; output explicitly preserves the pixel/second and non-forecasting limitations.
- To verify an extracted bundle without starting the service: run `python tools/release_check.py`, `python -m pytest -q`, `python -m compileall -q app aicc traffic_brain tests tools`, and `node --check static/app.js`. `docs/RELEASE_MANIFEST.json` records SHA-256 for immutable release source. Mutable operator manifests (`models/manifest.json`, `artifacts/current/manifest.json`, `data/cameras.json`) and local feed indexes are intentionally excluded from that inventory; `tools/release_check.py` separately parses/validates their contracts.
- This is a hardened local release candidate, not a claim of defect-free production certification. Static checks do not verify licensed CCTV access, live traffic data availability, or real Hanoi model accuracy.
