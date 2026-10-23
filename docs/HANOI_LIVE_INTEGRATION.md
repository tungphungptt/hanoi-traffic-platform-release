# Tích hợp camera/giao thông Hà Nội — hướng dẫn có kiểm soát

## Những nguồn có thể dùng/đề nghị quyền truy cập

- **Bản đồ nền:** OpenStreetMap dùng để vẽ đường phố/địa danh; không cung cấp traffic congestion live. Tôn trọng attribution và tile-use policy.
- **OSRM public routing demo:** cung cấp các tuyến trên road network và thời gian mạng tham chiếu; không được diễn giải thành live traffic. Endpoint public là best-effort, không có SLA sản xuất.
- **iHanoi:** thông tin cửa hàng ứng dụng công khai mô tả chức năng camera giao thông, nhưng đây không phải tài liệu developer API. Không có API key/contract công khai đã được xác minh trong release build này.
- **Hệ thống camera tập trung Hà Nội:** thành phố đã phê duyệt đề án quản lý camera tập trung (Quyết định 319/QĐ-UBND ngày 17/01/2025) và công bố kế hoạch triển khai/điều hành. Việc tồn tại hệ thống camera không có nghĩa một ứng dụng bên thứ ba tự động được quyền lấy stream/API. Hãy xin tài liệu interface, điều kiện sử dụng, account/role, API quota và phạm vi dữ liệu từ cơ quan/chủ nguồn dữ liệu.

Tham khảo nguồn công khai (truy cập ngày 23-10-2026):

- UBND Hà Nội — phê duyệt đề án camera tập trung: https://hanoi.gov.vn/chi-dao-cua-ubnd-thanh-pho-ha-noi/phe-duyet-de-an-tong-the-ve-quan-ly-he-thong-camera-giam-sat-tap-trung-4250117165836444.htm
- UBND Hà Nội — thông tin triển khai hệ thống camera tập trung: https://hanoi.gov.vn/tin-tuc-su-kien-noi-bat/thong-tin-chi-dao-dieu-hanh-cua-ubnd-thanh-pho-ha-noi-ngay-18-3-2025-4250318222558782.htm
- iHanoi App Store listing (feature descriptions only, not API docs): https://apps.apple.com/vn/app/ihanoi/id6470901306

## Adapter boundary the release provides

1. Camera metadata lives in `data/cameras.json`; set actual coordinates only after verifying site/asset identity. Use `location_verified=true` only after those coordinates are validated by the operator.
2. A collector must connect to an authorized RTSP/HLS/WebRTC source using credentials from an environment/secret store, then run a detector/tracker and report timestamped measurements. The app does not connect to RTSP by merely receiving `POST /api/traffic/telemetry`.
3. Send measurements to `POST /api/traffic/telemetry` using the contract in the OpenAPI docs. It requires a configured camera with verified location, `source=rtsp`, CI in [0,1], timezone-aware `observed_at`, and freshness <=180 seconds. This receiver currently stores only in RAM. Protect the service with authentication/authorization before exposing it beyond localhost.
4. The route scorer only uses fresh source-tagged RTSP telemetry near each candidate route. To declare traffic-aware comparison, every candidate route needs coverage under the configured 350 m proximity rule. Otherwise recommendation is static OSRM duration only.
5. For real traffic tile overlay, configure `TRAFFIC_TILE_URL_TEMPLATE` only with an authorized provider and its required attribution. Browser tile keys must be restricted/public keys, not backend secrets.

## Collector implementation not included

A real RTSP collector requires reconnection/backoff, per-source clock, decoding, FPS/frame skipping, bounded queue, detector + tracker, ROI/calibration, health metrics, persistent observations, privacy/retention policy, and observability. These are deployment components, not an API contract alone. The release intentionally does not claim they are complete.
