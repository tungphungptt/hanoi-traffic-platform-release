# Processed Traffic Data (`data/processed/`)

Khu vực lưu trữ các kết quả trung gian đã qua xử lý từ video gốc:
- Đặc trưng trích xuất theo từng khung hình (bounding boxes, vehicle tracks, tracker IDs, vận tốc ước tính pixel/s).
- Đặc trưng chuỗi thời gian gom theo cửa sổ trượt (sliding window sequences, feature vectors).
- Không sửa đổi dữ liệu gốc trong `data/raw/`.
