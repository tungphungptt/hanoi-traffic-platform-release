# Data Staging Area (`data/staging/`)

Khu vực cách ly và kiểm định chất lượng video trước khi đưa vào tập dữ liệu chính thức.

## Quy trình kiểm định (Staging & Quarantine):
1. **Kiểm tra tính toàn vẹn (Integrity Check)**: Kiểm tra file tải về có bị đứt gãy, lỗi header moov/mdat, hoặc hỏng khung hình hay không.
2. **Trích xuất thông số kỹ thuật (Probe Metadata)**: Xác định chính xác codec (H.264/H.265), FPS thực tế, độ phân giải (width x height) và tổng số khung hình.
3. Sau khi video vượt qua kiểm tra, chuyển video vào `data/raw/hanoi_traffic/` và đăng ký vào manifest.
