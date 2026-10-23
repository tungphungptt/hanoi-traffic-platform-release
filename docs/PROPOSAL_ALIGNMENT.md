# Đối chiếu đề án HN-TRAFFIC-AI-2026 — Final 8.0.0

Đối chiếu với tài liệu người dùng cung cấp: *Từ “Nhận diện” đến “Dự đoán” và “Có thể triển khai”: Kiến trúc dự báo tắc đường ngắn hạn trên CPU cho đô thị Hà Nội*.

## Logic nghiệp vụ được ưu tiên trong v6.0

1. Chọn đoạn A/điểm đi và nạp clip nguồn có quyền sử dụng.
2. Sample frame, phát hiện phương tiện bằng artifact AICC hiện có, sinh ảnh minh chứng theo thời điểm chuyển trạng thái/chu kỳ lấy mẫu.
3. Gắn nhãn trực quan bằng baseline giải thích được: số xe phát hiện, occupancy của bounding box và chuyển động tương đối. Đây là phân loại hình ảnh, không phải ground truth ùn tắc.
4. Chỉ khi người dùng xác nhận video giữ thời gian thật, duration ≥ 60s và đủ ≥20 mẫu gần cuối clip mới cho phép ngoại suy trend tuyến tính +5/+10 phút; kết quả vẫn gắn nhãn baseline độ tin cậy thấp và không được dùng điều hành thực tế.
5. Lấy phương án đường đi từ mạng road-routing. Khi có telemetry RTSP mới đã xác minh gần tất cả phương án, dùng heuristic CI/time; nếu không, nêu rõ lựa chọn là thời gian mạng đường tham chiếu và chưa chứng minh tuyến ít tắc hơn.
6. Chuẩn bị contract để nối RTSP collector và forecast provider tương lai mà không trộn model AICC Phase 1 với mô hình forecaster nhiều chân trời.

## Ma trận đề án – trạng thái thật trong 6.0

| Hạng mục đề án | Trạng thái ở release | Giới hạn/việc phải làm |
|---|---|---|
| Bản đồ đường phố Hà Nội | Có | Leaflet + OpenStreetMap thật; vị trí marker mẫu không phải CCTV đã xác minh. |
| Video → frame sampling → ảnh minh chứng | Có | Detector AICC cần model artifact có sẵn; frame labels là heuristic theo hình ảnh. |
| AICC Phase 1 trên CPU ONNX | Có integration | Manifest `v1.0-car-only`, 2 lớp, 3 đặc trưng, sequence length 5; chủ yếu class car; pixel/s, không phải km/h. |
| +5/+10 min risk | Có baseline có điều kiện | Ngoại suy tuyến tính từ clip giữ thời gian thật; không có model dự báo đã huấn luyện/kiểm định trong release. |
| Lấy các tuyến thay thế thật | Có | OSRM trả geometry từ road network; duration chỉ tham chiếu, không có live traffic ETA mặc định. |
| Cải thiện route bằng telemetry | Có API contract và scorer | Cần collector RTSP hợp pháp thực sự, camera tọa độ đã xác minh, telemetry mới bao phủ mọi phương án. Không có collector chạy mặc định. |
| YOLO + ByteTrack end-to-end từ RTSP | Chưa hoàn thành | AICC local video processing và analytics ingest endpoint không phải RTSP/ByteTrack production worker. |
| ROI, homography, tốc độ km/h, PCU, occupancy | Schema/primitives có | Phải hiệu chuẩn riêng camera, nhãn phương tiện và geometry mặt đường; hiện không có calibration field site. |
| Forecast model ONNX/API +5/+10/+15 | Contract có, model thực chưa có | `artifacts/current/manifest.json` là schema scaffold; không có trained `forecaster.onnx` hoặc ground truth dataset Hà Nội trong ZIP. Cần huấn luyện/kiểm định và benchmark. |
| Event/line-crossing analytics | Có API/module cơ sở | Nhận observations đã tracking; event store in-memory, không tự chạy detector. |
| Gắn camera iHanoi/VMS thật | Chưa có quyền/source | Cần tài liệu và quyền truy cập từ data owner; không scrape/bypass login/đoán stream URL. |
| KPI đề án (mAP, IDF1, MAPE count, CI MAE/RMSE, Macro-F1, latency/RAM) | Chưa chứng minh | Công cụ MAE/RMSE là scaffold; phải chạy trên test set độc lập có ground truth. |

## Tách biệt hai hợp đồng model

- `models/manifest.json`: AICC Phase 1 video classification; `car_count`, `avg_speed`, `speed_cv`; hiện `avg_speed` là pixel/giây.
- `artifacts/current/manifest.json`: feature contract cho forecaster đa chân trời; ordered features gồm car/motorcycle counts, PCU, speed km/h, CV, occupancy và lag/delta. Đây là contract schema, không tự sinh ra detector hay forecaster weights.

Không feed các đặc trưng pixel/s như thể chúng là km/h, không điền fallback thành số đo thật để ép mô hình dự báo và không đổi manifest nếu chưa khớp artefact huấn luyện.

## Kiểm thử release

- 83 automated tests pass trên source tree trong môi trường build; test tích hợp thêm cover image-frame creation, input validation, timing gating, path traversal, Hanoi coordinate bounds, route geometry/scoring disclaimer and route-to-origin clip binding.
- `compileall`, `node --check` và parse JSON được chạy trong QA.
- Chạy local Uvicorn + health check và API smoke tests trước khi đóng gói ZIP.
- Không có truy cập live CCTV/API chính thức hoặc benchmark model Hà Nội thực tế trong release build.

## Gate trước vận hành công khai/thực tế

- Yêu cầu access chính thức cho feed; xây RTSP/HLS/WebRTC collector có credential secret store, reconnect, health, buffer và retention.
- Nối detector/tracker thực với line/ROI/calibration per-camera.
- Tạo dataset theo timestamp/camera và target CI tương lai; split độc lập, đo KPI, uncertainty/calibration và drift.
- Thêm AuthN/AuthZ, rate limits, persistent audited storage, TLS, privacy & retention controls; kiểm tra license model, data và API provider.
