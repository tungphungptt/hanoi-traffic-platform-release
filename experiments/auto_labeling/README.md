# Auto-Labeling Experiments (`experiments/auto_labeling/`)

Chu trình tạo nhãn tự động / bán tự động cho video giao thông Hà Nội từ mô hình detector và sequence classifier.

## Quy trình gán nhãn:
1. **Lọc theo độ tin cậy**: Chỉ các khung hình hoặc cửa sổ có `model_confidence >= confidence_threshold` mới được sinh pseudo-label.
2. **Làm mượt nhân quả**: Áp dụng bộ lọc EMA và trễ hysteresis (`AICCSequenceStabilizer`) để triệt tiêu nhấp nháy nhãn giữa các giây liền kề.
3. **Phân loại**:
   - Nếu độ tin cậy cao: sinh nhãn `auto_labeled`.
   - Nếu độ tin cậy thấp hoặc gần ngưỡng phân định: đánh dấu `needs_human_review`.
4. Xem `config.example.yaml` để tham khảo cấu hình mẫu.
