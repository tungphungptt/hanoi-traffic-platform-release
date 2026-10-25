# Traffic Annotations & Ground Truth (`data/annotations/`)

Thư mục lưu trữ các bộ nhãn gán cho dữ liệu video giao thông Hà Nội:
- `detector/`: Nhãn phát hiện phương tiện (YOLO format txt/json: class, x_center, y_center, width, height) phân theo car, motorcycle, bus, truck.
- `congestion/`: Nhãn phân loại mức độ đông đúc theo chuỗi thời gian (Thong_thoang / Dong_duc).
- `route_ci/`: Nhãn chỉ số tắc nghẽn liên tục $CI \in [0.0, 1.0]$ cho từng phân đoạn tuyến đường phục vụ mô hình dự báo đa chân trời (+5/+10/+15 phút).

## Phân định mức độ tin cậy của nhãn:
- `auto_labeled`: Nhãn sinh tự động từ mô hình hoặc pipeline pseudo-labeling; có độ tin cậy kèm theo.
- `verified_ground_truth`: Nhãn đã được con người (chuyên viên phân tích) rà soát và xác nhận.
