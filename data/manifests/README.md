# Data Manifests

Thư mục chứa schema và catalog quản lý tập dữ liệu video giao thông Hà Nội.

## Các tệp chính:
- `dataset_manifest.schema.json`: JSON Schema chuẩn định nghĩa cấu trúc của bản ghi video (video ID, đường dẫn tương đối, tên đường, nguồn, timestamp, FPS, độ phân giải, trạng thái nhãn, split).
- `hanoi_videos_manifest.json`: File danh mục video thực tế của dự án.

## Quy tắc cập nhật Manifest khi đưa video mới vào:
1. Sau khi tải video hoàn tất vào thư mục `data/raw/hanoi_traffic/`, thêm bản ghi mới vào mảng `videos`.
2. Không tự ý bịa đặt tên đường (`road_name`) hoặc thời điểm ghi hình (`recorded_timestamp`); để `null` nếu chưa biết.
3. Khi mới thêm, đặt `label_status = "unlabeled"` và `split = "unassigned"`.
4. Việc phân chia `split` (`train`, `val`, `test`) phải thực hiện theo từng video độc lập (Video-level split), **tuyệt đối không xáo trộn ngẫu nhiên các khung hình trong cùng một video** để tránh data leakage.
