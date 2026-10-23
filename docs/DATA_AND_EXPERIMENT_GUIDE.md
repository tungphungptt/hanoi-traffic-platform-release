# Hướng Dẫn Quản Lý Dữ Liệu & Thực Nghiệm (Data & Experiment Architecture Guide)
*Hanoi Traffic Platform 8.0 — Chuẩn Hóa Chu Trình Auto-Labeling, Training & Route Forecasting*

---

## 1. Tổng Quan Cấu Trúc Thư Mục Chuẩn Hóa

Hệ thống được thiết kế theo nguyên tắc: **Tách biệt dữ liệu gốc bất biến khỏi dữ liệu đã xử lý, mã nguồn chạy thực và kết quả thực nghiệm**. Cấu trúc này mở rộng an toàn mà không làm xáo trộn bất kỳ API hay chức năng sẵn có của ứng dụng.

```text
D:\hanoi-traffic-platform-final-8.0\
│
├── aicc/                          # Pipeline suy luận video AICC (YOLO + GRU Stabilizer)
├── app/                           # FastAPI Backend application & Web endpoints
├── traffic_brain/                 # Module nghiệp vụ: video review, routing, multi-horizon forecasters
├── static/                        # Frontend UI Dashboard (HTML5, Leaflet Map, CSS, JS)
├── tools/                         # Các công cụ CLI (model_smoke, verify_video, release_check, ...)
├── tests/                         # Bộ kiểm thử tự động pytest
├── docs/                          # Tài liệu kiến trúc, vận hành, kiểm định chất lượng
│
├── data/                          # Vùng quản lý dữ liệu giao thông
│   ├── cameras.json               # Danh sách camera hiện hành và tọa độ marker trên bản đồ
│   ├── input.mp4                  # Video baseline 5 phút dùng cho kiểm tra AICC
│   ├── aicc_cam1_smoke.json       # Kết quả kiểm thử smoke test (giữ nguyên để bảo toàn manifest)
│   ├── camera_calibrations/       # Schema hiệu chuẩn góc nhìn camera
│   ├── camera_feeds/              # Video gán động cho camera ID
│   ├── examples/                  # File mẫu đánh giá dự báo
│   ├── labels/                    # Schema target dự báo
│   ├── observations/              # Schema quan trắc giao thông
│   ├── video_reviews/             # Khung hình và kết quả trích xuất từ UI Review Video
│   │
│   ├── raw/                       # [MỚI] Vùng chứa dữ liệu gốc bất biến (Immutable Data)
│   │   └── hanoi_traffic/         # Thư mục đích nhận các video giao thông Hà Nội tải về
│   ├── staging/                   # [MỚI] Vùng đệm cách ly kiểm tra tính toàn vẹn (Integrity check / probe)
│   ├── processed/                 # [MỚI] Dữ liệu đặc trưng trích xuất (tracks, sliding window features)
│   ├── annotations/               # [MỚI] Bounding boxes (YOLO), nhãn chuỗi thời gian, nhãn CI
│   └── manifests/                 # [MỚI] Danh mục quản lý tập dữ liệu và định nghĩa Train/Val/Test Split
│       ├── dataset_manifest.schema.json  # JSON Schema chuẩn kiểm soát metadata video
│       └── hanoi_videos_manifest.json    # Danh mục catalog video của dự án
│
├── experiments/                   # [MỚI] Vùng quản lý chu trình thực nghiệm
│   ├── auto_labeling/             # Thí nghiệm gán nhãn tự động / bán tự động
│   ├── training/                  # Thí nghiệm huấn luyện mô hình (detector / sequence predictor)
│   └── route_forecasting/         # Thí nghiệm dự báo tắc nghẽn cấp tuyến đường (+5/+10/+15 phút)
│
├── models/                        # Trọng số mô hình
│   ├── detector.onnx              # Mô hình Production detector
│   ├── predictor.onnx             # Mô hình Production GRU sequence
│   ├── scaler.pkl                 # StandardScaler Production
│   ├── manifest.json              # Manifest khai báo mô hình Production
│   └── experiments/               # [MỚI] Thư mục chứa checkpoints và mô hình thử nghiệm
│
└── artifacts/                     # Kết quả phân tích và báo cáo
    ├── current/                   # Manifest hợp đồng forecaster đa chân trời
    ├── examples/                  # Template đánh giá
    ├── reports/                   # [MỚI] Báo cáo thẩm định dữ liệu & video
    ├── evaluation/                # [MỚI] Kết quả đánh giá mô hình, ma trận nhầm lẫn, biểu đồ
    └── audit/                     # [MỚI] Nhật ký kiểm định thuật toán và tính toàn vẹn
```

---

## 2. Hướng Dẫn Đưa Video Đã Tải Hoàn Tất Vào Dataset

Khi tiến trình tải video từ Google Drive (ví dụ: `aicc-data.zip` trong Downloads hoặc nguồn khác) đã hoàn thành, thực hiện theo 3 bước sau:

### Bước 1: Sao chép video vào thư mục Raw
1. Giải nén hoặc di chuyển các file `.mp4` vào thư mục:
   ```text
   D:\hanoi-traffic-platform-final-8.0\data\raw\hanoi_traffic\
   ```
2. **Quy ước đặt tên file chuẩn**:
   * Nếu dữ liệu có mã cố định từ benchmark (ví dụ: `cam_1.mp4`, `cam_2.mp4`): có thể giữ nguyên.
   * Nếu thu thập thực địa: đặt theo mẫu `HN_<tên_tuyến_đường>_<YYYYMMDD>_<HHMM>_<mã_cam>.mp4` (ví dụ: `HN_nguyen_trai_20261023_0830_cam01.mp4`).

### Bước 2: Kiểm tra tính toàn vẹn tại Staging (Tùy chọn)
Trước khi đưa vào xử lý hàng loạt, có thể dùng `ffmpeg` hoặc OpenCV probe để kiểm tra codec và số khung hình không bị lỗi:
```powershell
.\.venv\Scripts\python -c "import cv2; cap = cv2.VideoCapture('data/raw/hanoi_traffic/cam_1.mp4'); print('OK:', cap.isOpened(), 'Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 'FPS:', cap.get(cv2.CAP_PROP_FPS)); cap.release()"
```

### Bước 3: Đăng ký video vào Manifest
Mở file `data/manifests/hanoi_videos_manifest.json` và bổ sung bản ghi cho video mới vào mảng `videos`:
```json
{
  "video_id": "HN-CAM01-001",
  "relative_path": "data/raw/hanoi_traffic/cam_1.mp4",
  "road_name": "Nguyễn Trãi",
  "camera_id": "cam_01",
  "source": "aicc-data.zip",
  "recorded_timestamp": null,
  "fps": 10.0,
  "resolution": {
    "width": 1280,
    "height": 960
  },
  "duration_seconds": 300.0,
  "frame_count": 3000,
  "label_status": "unlabeled",
  "split": "train",
  "notes": "Clip 5 phút quan trắc giờ cao điểm sáng"
}
```
*Lưu ý: Nếu thông số nào chưa rõ (ví dụ ngày giờ ghi hình chính xác), điền `null`, tuyệt đối không tự bịa đặt metadata.*

---

## 3. Quy Ước Quản Lý Thực Nghiệm (Experiment Standard)

Mỗi thực nghiệm (Auto-labeling, Training, Route-forecasting) cần tuân thủ các nguyên tắc:
1. **Experiment ID**: Đặt tên có cấu trúc: `EXP-<LOẠI>-<YYYYMMDD>-<SỐ_THỨ_TỰ>`.
   * Ví dụ: `EXP-AUTOLABEL-20261023-01`, `EXP-TRAIN-20261023-01`.
2. **Cố định Seed**: Mọi script phải cố định `seed: 42` (hoặc seed được cấu hình trong `config.yaml`).
3. **Phòng chống Rò rỉ Dữ liệu (No Data Leakage)**:
   * **Phân chia theo Video Clip độc lập (Clip-level splitting)**: Khi chia `train`, `val`, `test`, **toàn bộ** các khung hình của một clip video chỉ được phép thuộc về một tập duy nhất.
   * **Cấm tuyệt đối**: Lấy các cửa sổ 1 giây chẵn làm train và cửa sổ 1 giây lẻ làm test trong cùng một video clip, vì tính tương quan chuỗi thời gian liền kề sẽ làm sai lệch hoàn toàn kết quả đánh giá thực tế.
4. **Cô lập Checkpoints & Metrics**:
   * Checkpoints lưu tại: `models/experiments/<experiment_id>/`.
   * Đánh giá và báo cáo lưu tại: `artifacts/evaluation/<experiment_id>_metrics.json`.

---

## 4. Chuẩn Bị Cho Dự Báo Tắc Nghẽn Cấp Tuyến Đường (+5/+10/+15 Phút)

Dự báo cấp tuyến đường (Route-level forecasting) kế thừa mô hình dự báo đã được định nghĩa tại `traffic_brain/providers.py` và `traffic_brain/routing.py`:
1. **Quy đổi trọng số PCU (Passenger Car Unit)**:
   * Xe máy = $0.3$
   * Xe ô tô con = $1.0$
   * Xe buýt = $2.5$
   * Xe tải = $2.0$
2. **Chỉ số Tắc nghẽn ($CI \in [0.0, 1.0]$)**:
   * $CI < 0.30$: Thông thoáng (Free flow).
   * $0.30 \le CI < 0.55$: Bình thường (Moderate).
   * $0.55 \le CI < 0.78$: Đông đúc (Busy).
   * $CI \ge 0.78$: Ùn tắc nghiêm trọng (Severe Congestion).
3. **Ba chân trời dự báo**: $+5$ phút, $+10$ phút, $+15$ phút hỗ trợ điều hướng tuyến đường thay thế tránh điểm nghẽn trước khi người tham gia giao thông tới nút giao.
