# AI City Challenge 2021 Track 1 - Vehicle Counting (Bản Phát Hành FINAL)
## Geometric Baseline, CNN–LSTM Sequence Modeling, Training, Evaluation & Portable Package

> **Dự án**: Hanoi Traffic Platform 8.0  
> **Workspace**: `D:\hanoi-traffic-platform-final-8.0`  
> **Định danh lần chạy (Run ID)**: `RUN_TRACK1_RETRAIN_FINAL_20261023_220650`  
> **Phiên bản**: `8.0.0-final`  
> **Gói phát hành chính thức (Official Primary Release Archive)**:  
> [`dist/track1_vehicle_counting_final.zip`](file:///D:/hanoi-traffic-platform-final-8.0/dist/track1_vehicle_counting_final.zip)  
> *(Gói phụ / Alias: `dist/track1_cnn_lstm_final.zip`)*  
> *(Mã băm SHA-256 được đối soát độc lập tại `FINAL_DELIVERY_REPORT.md` và `docs/RELEASE_MANIFEST.json`)*  
> **Phân loại gói**: **Source & Portable Release Package (Cần weights detector bên ngoài; không phải fully standalone binary)**  
> **Trạng thái phát hành (Release Status)**:  
> - Phân phối thương mại / Public rộng rãi: **RELEASE BLOCKED FOR GENERAL PUBLIC / COMMERCIAL DISTRIBUTION** (do ràng buộc giấy phép dữ liệu AIC21 và AGPL-3.0 của YOLOv8).  
> - Đánh giá học thuật nội bộ: **RELEASE READY (Non-Commercial Academic Evaluation Only)**.  

---

## 1. Tổng Quan Kiến Trúc & Quyết Định Tuyển Chọn Mô Hình

Pipeline AI City Challenge Track 1 Vehicle Counting thực hiện chuỗi xử lý khép kín:
$$\text{Video Frame Input} \longrightarrow \text{YOLOv8n Detector (External)} \longrightarrow \text{Multi-Object Tracker} \longrightarrow \text{Tracklet Extraction} \longrightarrow \text{Movement Classification} \longrightarrow \text{ROI Exit Gate} \longrightarrow \text{Deduplication} \longrightarrow \text{Track 1 CSV Export}$$

### 1.1 Quyết định chọn mô hình cho Production
- **MẶC ĐỊNH SẢN XUẤT (Production Default)**: **Calibrated Geometric Heuristic Baseline**.
- **TÙY CHỌN THỬ NGHIỆM (Experimental Options)**:
  - MLP Trajectory Classifier (`--movement-classifier-type mlp`)
  - CNN–LSTM Sequence Classifier (`--movement-classifier-type cnn_lstm`)
- **Bằng chứng thực nghiệm trên phân vùng held-out phát triển (frames 351–600 của `cam_5.mp4`, 47 sự kiện Ground Truth)**:
  - **Standalone Movement Accuracy ($N=46$ mẫu test)**:
    - Geometric Heuristic Baseline đạt **89.13%** (41/46 mẫu đúng).
    - CNN–LSTM (với heuristic fallback) đạt **86.96%** (40/46 mẫu đúng).
    - MLP Classifier (Train-Only) đạt **86.96%** (40/46 mẫu đúng).
    - CNN–LSTM (Standalone không fallback) đạt **84.78%** (39/46 mẫu đúng).
  - **End-to-End Strict Counting F1 (yêu cầu đúng cả Class, Frame +-15 và Movement ID)**:
    - Geometric Heuristic Baseline đạt **0.8200** ($\text{TP}=41, \text{FP}=12, \text{FN}=6, \text{Wrong Movement}=5$).
    - MLP Train-Only đạt **0.8200** ($\text{TP}=41, \text{FP}=12, \text{FN}=6, \text{Wrong Movement}=5$).
    - CNN–LSTM Train-Only đạt **0.4800** ($\text{TP}=24, \text{FP}=29, \text{FN}=23, \text{Wrong Movement}=22$).
  - **Count-Only F1 Score (bỏ qua Movement ID, chỉ khớp Class và Frame +-15)**:
    - Mọi phương pháp đều đạt **0.9200** ($\text{TP}=46, \text{FP}=7, \text{FN}=1$).
- **Nguyên nhân CNN–LSTM đạt Strict F1 thấp hơn Heuristic trên held-out**:
  - Trong tập train (frames 0–350), Movement 5 chỉ có đúng **2 mẫu** (trong 46 mẫu), trong khi Movement 3 có 12 mẫu.
  - Trong tập test held-out (frames 351–600), lưu lượng Movement 5 tăng vọt lên **23 sự kiện**, còn Movement 7 và 10 xuất hiện lần đầu (hoàn toàn vắng bóng ở tập train).
  - Khi xe di chuyển qua Approach 2 hướng về Exit 4 (Movement 5), mô hình LSTM bị lệch xác suất tiên nghiệm (prior bias) sang Movement 3.
  - Ngược lại, Geometric Heuristic Baseline dựa trên vector định hướng hình học camera, nhận diện chính xác 20/23 xe Movement 5 mà không bị phụ thuộc vào phân phối lưu lượng quá khứ.
  - **Do đó, Baseline Heuristic bắt buộc phải là Production Default.**

### 1.2 Kiến trúc Thực tế của CNN–LSTM: Frozen Spatial CNN + Trained LSTM
- **Spatial CNN Feature Extractor**:
  - Sử dụng bộ lọc spatial convolution & quadrant pooling đa kênh (16 chiều) cố định, tất định khởi tạo với seed 42 (`SpatialCNNFeatureExtractor`). Trích xuất đặc trưng hình thái phương tiện on-the-fly từ vehicle patch $(T, 3, 64, 64)$ với RAM tiêu thụ $<100$ MB.
  - Ghép với 9 chiều kinematic tracking $[cx, cy, bw, bh, dx, dy, \text{speed}, \cos\theta, \sin\theta]$ thành vector 25 chiều mỗi timestep.
- **Thành phần Tối ưu hóa (Optimized in 70 Epochs)**:
  - Toàn bộ **7,622 tham số** của mạng LSTM ($W \in \mathbb{R}^{128 \times 25}, U \in \mathbb{R}^{128 \times 32}, b \in \mathbb{R}^{128}$) và output classification head ($W_{\text{out}} \in \mathbb{R}^{6 \times 32}, b_{\text{out}} \in \mathbb{R}^6$) được tối ưu hóa trực tiếp qua thuật toán Backpropagation Through Time (BPTT) và Adam optimizer strictly trên tập train ($\le 350$ frames).

---

## 2. Minh Bạch Về Giao Thức Đánh Giá & Giới Hạn Nghiên Cứu

1. **Phân Vùng Đánh Giá (Historical Held-Out / Development Partition)**:
   - Khoảng khung hình 351–600 trên `cam_5.mp4` được định danh là **Historical Held-out / Development Evaluation**, **KHÔNG PHẢI là một tập kiểm thử độc lập hoàn toàn (independent test set)**.
   - **Lý do**: Các tham số hình học của camera (tọa độ điểm thoát Approach 2 `(45.0, 370.0)`, ngưỡng khoảng cách thoát ROI 90px, quãng đường tối thiểu 35px) đã từng được khảo sát và tinh chỉnh trên các khung hình 339, 343, 359, 375, 425 của clip này trong các giai đoạn phát triển trước.
2. **Stratified 3-Fold Cross-Validation Giới Hạn Trên Tập Train của `cam_5`**:
   - Quá trình Cross-Validation được thực hiện chỉ trên 46 mẫu của phân vùng train ($\le 350$ frames).
   - **Cảnh báo Scikit-Learn đã ghi nhận**: `UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3`. Do Class 12 chỉ có 1 mẫu và Class 5 chỉ có 2 mẫu trong tập train, việc phân chia fold không thể cân bằng hoàn hảo cho các lớp thiểu số.
   - Đánh giá này chỉ phản ánh độ biến thiên phát triển nội bộ trên clip `cam_5`, **tuyệt đối không khẳng định khả năng tổng quát hóa sang camera khác** (`cam_1` đến `cam_4`, v.v.).
3. **Diễn Giải Đúng Về Count-Only F1 = 0.9200**:
   - Đây là metric kỹ thuật của bộ đánh giá khi **bỏ qua hoàn toàn thuộc tính Movement ID**, chỉ ghép các dự đoán với Ground Truth theo loại phương tiện (Car/Truck) trong cửa sổ thời gian sai số $\pm 15$ frames (1.5 giây).
   - Metric này **không thể dùng để khẳng định bộ đếm không gian thời gian "ổn định tuyệt đối"**, vì nó vẫn ghi nhận 7 False Positives và 1 False Negative trên đoạn clip 25 giây này, đồng thời bỏ qua toàn bộ 22 lỗi gán sai luồng di chuyển của mô hình CNN-LSTM.

---

## 3. Cấu Trúc Gói Phát Hành (`artifacts/track1_final/` & `dist/`)

```text
artifacts/track1_final/
├── README_FINAL.md                          # Tài liệu kỹ thuật chi tiết
├── FINAL_DELIVERY_REPORT.md                 # Báo cáo nghiệm thu master
├── provenance.json                          # Metadata phần cứng, commit và provenance
├── configs/
│   ├── data_manifest.json                   # Khảo sát 31 video, 20 camera scenes
│   ├── pipeline_config.json                 # Tham số detector, tracker, exit gates
│   └── camera_5_roi_moi.json                # Tọa độ ROI polygon và 12 MOI của cam_5
├── models/
│   ├── cnn_lstm_movement_experimental.pkl   # Checkpoint CNN-LSTM huấn luyện trên frames <= 350
│   ├── movement_classifier_train_only.pkl   # Checkpoint MLP huấn luyện trên frames <= 350
│   ├── counting_calibration_train_only.json # Hiệu chuẩn fit strictly trên train
│   └── movement_classifier_metadata.json    # Metadata phân phối lớp
├── outputs/
│   ├── output_counting_cam5_1min.csv               # CSV baseline heuristic
│   ├── output_counting_ml_train_only_cam5_1min.csv # CSV MLP train-only
│   └── output_counting_cnn_lstm_cam5_1min.csv      # CSV CNN-LSTM experimental
└── evaluation/
    ├── evaluation_metrics.json              # Metrics baseline production
    ├── evaluation_metrics_cnn_lstm.json     # Metrics CNN-LSTM full clip
    ├── heldout_evaluation_metrics.json      # Metrics temporal holdout (frames 351-600)
    ├── cnn_lstm_training_summary.json       # Lịch sử 70 epochs loss và accuracy
    └── RECONCILIATION_TABLE.md              # Bảng đối chiếu sự kiện chi tiết
```

---

## 4. Gói Phân Phối Portable Chính Thức (`dist/track1_vehicle_counting_final.zip`)

- **Tên gói chính thức**: [`dist/track1_vehicle_counting_final.zip`](file:///D:/hanoi-traffic-platform-final-8.0/dist/track1_vehicle_counting_final.zip)
- **Kích thước file**: ~124 KB (26 tệp tin và thư mục con).
- **Phân loại**: **Source / Portable Release Package**.
- **Xác thực toàn vẹn**: Đối soát mã băm SHA-256 trực tiếp bằng PowerShell `Get-FileHash dist\track1_vehicle_counting_final.zip` so với `FINAL_DELIVERY_REPORT.md` và `checksums.sha256`.
- **Thành phần người dùng cần cung cấp thêm khi vận hành**:
  1. `models/detector.onnx` (YOLOv8n ONNX weights từ Ultralytics).
  2. File video đầu vào (ví dụ `cam_5.mp4` từ AI City Challenge).
- **Loại trừ nghiêm ngặt**:
  - Không chứa video gốc `Dataset_A` (*.mp4).
  - Không chứa file trọng số `detector.onnx`.
  - Không chứa virtual environment (`.venv`) hoặc file cache `.pyc`.

---

## 5. Hướng Dẫn Vận Hành & Lệnh CLI Thực Tế

Môi trường yêu cầu: Python $\ge 3.10$, `numpy`, `scipy`, `scikit-learn`, `opencv-python`, `onnxruntime`.

### 5.1 Chạy Baseline Heuristic (Production Default)
```powershell
.venv\Scripts\python.exe tools/run_track1_counting.py `
  --video AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4 `
  --roi AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt `
  --detector models/detector.onnx `
  --movement-classifier-type heuristic `
  --output-csv artifacts/track1_final/outputs/output_counting_cam5_1min.csv
```

### 5.2 Chạy CNN–LSTM Experimental Option
```powershell
.venv\Scripts\python.exe tools/run_track1_counting.py `
  --video AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4 `
  --roi AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt `
  --detector models/detector.onnx `
  --movement-classifier-type cnn_lstm `
  --ml-movement-classifier artifacts/track1_final/models/cnn_lstm_movement_experimental.pkl `
  --output-csv artifacts/track1_final/outputs/output_counting_cnn_lstm_cam5_1min.csv
```

### 5.3 Chạy MLP Experimental Option
```powershell
.venv\Scripts\python.exe tools/run_track1_counting.py `
  --video AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4 `
  --roi AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt `
  --detector models/detector.onnx `
  --movement-classifier-type mlp `
  --ml-movement-classifier artifacts/track1_final/models/movement_classifier_train_only.pkl `
  --output-csv artifacts/track1_final/outputs/output_counting_ml_train_only_cam5_1min.csv
```

### 5.4 Chạy Đánh Giá So Sánh Held-Out (Frames 351–600)
```powershell
.venv\Scripts\python.exe tools/evaluate_heldout_track1.py
```

### 5.5 Chạy Toàn Bộ 22 Bước Master Build
```powershell
powershell -ExecutionPolicy Bypass -File tools/build_track1_cnn_lstm_final.ps1
```

---

## 6. Kiểm Định Bản Quyền, Nguồn Gốc & Ràng Buộc Pháp Lý (License & Provenance)

| Thành Phần | Nguồn Gốc / Tác Quyền | Giấy Phép Áp Dụng | Ràng Buộc Phân Phối Thực Tế |
|:---|:---|:---|:---|
| **Dữ liệu AIC21 Track 1** | NVIDIA & AI City Committee | Academic Research Only | Phân phối raw MP4 bị nghiêm cấm hoàn toàn. |
| **Checkpoints Movement ML/LSTM** | Huấn luyện từ quỹ đạo AIC21 `cam_5` | **Non-Commercial Academic Use Only** (AIC21 License Mục 2/3) | **KHÔNG THỂ cấp phép MIT tự do**. Bị ràng buộc bởi điều khoản sử dụng phi thương mại học thuật của dữ liệu gốc. |
| **YOLOv8n Detector** | Ultralytics LLC | **AGPL-3.0** / Commercial | Trọng số `detector.onnx` bị loại khỏi ZIP để tránh lây nhiễm copyleft; người dùng tự chịu trách nhiệm AGPL-3.0 khi tích hợp. |
| **Mã Nguồn Nền Tảng (Platform Code)** | Hanoi Traffic Platform | **MIT License** | Mã nguồn kịch bản, pipeline và công cụ được cấp phép MIT cho mục đích tái lập khoa học. |
| **Thư Viện Phụ Thuộc** | ONNX Runtime, OpenCV, Scikit-learn | MIT, Apache 2.0, BSD-3 | Phụ thuộc runtime tiêu chuẩn. |

**Kết luận phân loại trạng thái phát hành**:
- **Phân phối công khai rộng rãi / Thương mại**: **`RELEASE BLOCKED FOR GENERAL PUBLIC / COMMERCIAL DISTRIBUTION`** (Do các checkpoint nạp trực tiếp tri thức dữ liệu AIC21 học thuật và YOLOv8 AGPL-3.0).
- **Phát hành đánh giá nghiên cứu khoa học nội bộ**: **`RELEASE READY (Non-Commercial Academic Research Evaluation Only)`**.
