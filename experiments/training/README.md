# Model Training Experiments (`experiments/training/`)

Chu trình huấn luyện và tinh chỉnh các mô hình nhận diện và dự báo giao thông.

## Các loại mô hình:
1. **Object Detector**: Phát hiện phương tiện (car, motorcycle, bus, truck) tối ưu cho giao thông hỗn hợp tại Hà Nội.
2. **Sequence Classifier**: Mô hình GRU/LSTM phân loại trạng thái Thông thoáng / Đông đúc từ chuỗi đặc trưng cửa sổ.
3. **Forecasting Model**: Mô hình hồi quy dự báo chỉ số tắc nghẽn liên tục $CI \in [0.0, 1.0]$.

## Nguyên tắc:
- Mọi lần huấn luyện đều tham chiếu đến file manifest cụ thể trong `data/manifests/`.
- Phân chia `train`, `val`, `test` phải theo video hoặc camera để kiểm thử khả năng tổng quát hóa trên các nút giao chưa từng thấy.
- Checkpoints sinh ra được lưu tại `models/experiments/<experiment_id>/`.
