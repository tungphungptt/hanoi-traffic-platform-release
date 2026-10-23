# Route-Level Traffic Forecasting Experiments (`experiments/route_forecasting/`)

Chu trình nghiên cứu, hiệu chuẩn và đánh giá mô hình dự báo tắc nghẽn cấp tuyến đường đa chân trời (+5, +10, +15 phút).

## Mục tiêu:
1. Dự báo chỉ số tắc nghẽn liên tục $CI \in [0.0, 1.0]$ trên từng phân đoạn tuyến đường mạng lưới Hà Nội.
2. Tích hợp trọng số quy đổi PCU (xe máy = 0.3, ô tô = 1.0, xe buýt = 2.5, xe tải = 2.0).
3. Đánh giá sai số MAE/RMSE theo từng chân trời (+5 phút, +10 phút, +15 phút) trên các tập dữ liệu thực nghiệm.
4. Liên kết với công cụ định tuyến thay thế tránh điểm nghẽn (`traffic_brain/routing.py`).
