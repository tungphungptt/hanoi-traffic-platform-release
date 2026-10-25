# Data base

- `cameras.json`: danh sách camera minh họa và tọa độ marker tương đối trên sơ đồ.
- `input.mp4` hoặc thư mục con: video dùng cho AICC hoặc gán clip cho một camera từ giao diện.
- `camera_feeds/`: vùng lưu clip đã gán theo camera ID; index ánh xạ nằm tại `camera_feeds/index.json`.

Không commit video dữ liệu, footage riêng tư hay credential camera lên GitHub. Chỉ dùng video/RTSP có quyền truy cập. Video trong `data/` không tự động đại diện cho địa điểm thực cho đến khi được người vận hành gán rõ ràng.
