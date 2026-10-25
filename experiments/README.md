# Experiments Architecture & Management (`experiments/`)

Thư mục tổ chức và quản lý các chu trình thực nghiệm của Hanoi Traffic Platform:
1. `experiments/auto_labeling/`: Gán nhãn tự động / bán tự động cho video giao thông.
2. `experiments/training/`: Huấn luyện và tinh chỉnh mô hình nhận diện và chuỗi thời gian.
3. `experiments/route_forecasting/`: Đánh giá và hiệu chuẩn dự báo tắc nghẽn cấp tuyến đường đa chân trời (+5/+10/+15 phút).

## Nguyên tắc quản lý thực nghiệm chuẩn:
1. **Experiment ID**: Mỗi lần chạy thực nghiệm có định danh duy nhất:
   `EXP-<MÃ_LOẠI>-<YYYYMMDD>-<SỐ_THỨ_TỰ>`
   * Ví dụ: `EXP-AUTOLABEL-20261023-01`, `EXP-TRAIN-20261023-01`, `EXP-ROUTE-20261023-01`.
2. **Khả năng tái lập (Reproducibility)**:
   * Mỗi thực nghiệm phải cố định `seed` ngẫu nhiên (ví dụ: `seed: 42`).
   * Sử dụng cấu hình khai báo (`config.yaml`) chỉ định rõ ràng đường dẫn manifest dữ liệu đầu vào.
   * Kết quả đánh giá được xuất ra `metrics.json` hoặc lưu vào `artifacts/evaluation/`.
3. **Phòng chống rò rỉ dữ liệu (No Data Leakage)**:
   * **Phân chia theo Video Clip độc lập (Clip-level split)**: Tuyệt đối không chia train/val/test ngẫu nhiên theo từng frame hay từng cửa sổ 1 giây trong cùng một video clip.
   * Toàn bộ các khung hình của một video chỉ được nằm trong Train, hoặc Val, hoặc Test.
4. **Cô lập dữ liệu chạy**:
   * Các thư mục con `runs/`, `checkpoints/`, `outputs/` bên trong các thư mục experiment được `.gitignore` loại trừ tự động.
