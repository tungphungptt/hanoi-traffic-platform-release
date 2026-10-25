# Raw Hanoi Traffic Video Data (`data/raw/hanoi_traffic/`)

Thư mục lưu trữ các video giao thông Hà Nội nguyên bản (Raw Data).

## Nguyên tắc bảo vệ dữ liệu (Data Invariance):
1. **Dữ liệu bất biến (Read-Only)**: Toàn bộ video trong thư mục này là nguồn gốc bất biến. Không một tiến trình xử lý, trích xuất hay huấn luyện nào được phép ghi đè hay sửa đổi nội dung video tại đây.
2. **Quy ước đặt tên file video khuyến nghị**:
   `HN_<tên_tuyến_đường>_<YYYYMMDD>_<HHMM>_<mã_camera_nếu_có>.mp4`
   * Ví dụ: `HN_nguyen_trai_20261023_0830_cam01.mp4`
   * Nếu file tải từ dataset chuẩn có sẵn mã (ví dụ: `cam_1.mp4`), có thể giữ nguyên tên file gốc và mô tả chi tiết trong `data/manifests/hanoi_videos_manifest.json`.
3. **Cách đưa video vào sau khi hoàn tất tải xuống**:
   * Khi tiến trình tải từ Google Drive hoặc nguồn ngoài hoàn tất (ví dụ: giải nén file `aicc-data.zip`), sao chép các tệp `.mp4` vào thư mục này:
     `D:\hanoi-traffic-platform-final-8.0\data\raw\hanoi_traffic\`
   * Sau đó khai báo bản ghi vào file `data/manifests/hanoi_videos_manifest.json`.
4. **Không commit video lên Git**:
   * Các tệp video dung lượng lớn được loại trừ qua `.gitignore`.
