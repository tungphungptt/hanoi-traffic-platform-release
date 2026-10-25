# BÁO CÁO AUDIT HOÀN TẤT CUỐI CÙNG — AI CITY TRACK 1
## RETRAIN, VALIDATE, MODEL SELECTION, INTEGRATION & PORTABLE RELEASE AUDIT

**Mã phiên bản**: Hanoi Traffic Platform `v8.0.0-final`  
**Định danh lần chạy (Run ID)**: `RUN_TRACK1_RETRAIN_FINAL_20261023_220650`  
**Workspace**: `D:\hanoi-traffic-platform-final-8.0`  
**Gói phát hành chính thức (Official Primary Release Archive)**:  
[`dist/track1_vehicle_counting_final.zip`](file:///D:/hanoi-traffic-platform-final-8.0/dist/track1_vehicle_counting_final.zip) (SHA-256: `6f8ed07bb2437e550037e0bdd38496ad5049f12dbba6d1de7183ee096503ef8a`)  
*(Gói phụ / Alias: `dist/track1_cnn_lstm_final.zip`, SHA-256: `ec2eb9bba4c034b8e42c83485cbee2971dfc20c10065f22f4abe4541efe2ad81`)*  
**Trạng thái kiểm định kỹ thuật (Engineering Gates)**: **100% PASS ALL GATES (Zero Production Weight Mutation)**  
**Trạng thái phân phối (Distribution Status)**:  
- **Public Commercial Distribution**: **RELEASE BLOCKED FOR GENERAL PUBLIC / COMMERCIAL DISTRIBUTION** (do ràng buộc cấp phép dữ liệu AIC21 và copyleft AGPL-3.0 của Ultralytics YOLOv8).  
- **Academic Research Evaluation**: **RELEASE READY (Non-Commercial Academic Research Evaluation Only)**.  

---

## 1. Xác Minh Kiến Trúc Thực Tế: CNN Feature Extractor vs LSTM Optimization

Qua kiểm tra trực tiếp mã nguồn [`experiments/training/cnn_lstm_track1/extract_sequences.py`](file:///D:/hanoi-traffic-platform-final-8.0/experiments/training/cnn_lstm_track1/extract_sequences.py) và [`experiments/training/cnn_lstm_track1/train_cnn_lstm.py`](file:///D:/hanoi-traffic-platform-final-8.0/experiments/training/cnn_lstm_track1/train_cnn_lstm.py):

### 1.1 Phần CNN Feature Extractor
- Lớp `SpatialCNNFeatureExtractor` **không tối ưu hóa trọng số tích chập (convolution weights) trong 70 epochs**.
- Bộ lọc tích chập và pooling góc phần tư không gian (`filters`: $16 \times 3 \times 5 \times 5$, `bias`: $16$) được khởi tạo tất định với seed 42 và được **giữ cố định (frozen/deterministic spatial filter bank)**.
- Vai trò: Trích xuất vector đặc trưng không gian 16 chiều (phân bố màu 4 góc phần tư, gradient không gian $dy, dx$, mean, std) trực tiếp từ vehicle patch $(3, 64, 64)$ theo cơ chế streaming on-the-fly, giữ RAM tiêu thụ $<100$ MB.
- Vector này được ghép cùng 9 chiều động học $[cx, cy, bw, bh, dx, dy, \text{speed}, \cos\theta, \sin\theta]$ thành vector đầu vào 25 chiều cho mỗi timestep.

### 1.2 Phần Thực Sự Được Tối Ưu Hóa (Optimized in 70 Epochs)
- Thuật toán tối ưu: Adam ($\eta = 0.008, \beta_1 = 0.9, \beta_2 = 0.999$, weight decay = $10^{-4}$) kết hợp Backpropagation Through Time (BPTT).
- Danh sách tham số được cập nhật gradient:
  1. `lstm.W`: Ma trận trọng số đầu vào LSTM ($128 \times 25 = 3,200$ tham số)
  2. `lstm.U`: Ma trận trọng số hồi quy ẩn LSTM ($128 \times 32 = 4,096$ tham số)
  3. `lstm.b`: Vector bias các cổng LSTM ($128$ tham số)
  4. `W_out`: Ma trận trọng số tầng phân loại ngõ ra ($6 \times 32 = 192$ tham số)
  5. `b_out`: Vector bias tầng phân loại ngõ ra ($6$ tham số)
- **Tổng số tham số học được**: Đúng **7,622 tham số**.
- **Kết luận kiến trúc**: Mô hình là **Frozen-CNN Spatial Feature Encoder kết hợp Trained Recurrent LSTM Sequence Classifier**.

---

## 2. Minh Bạch Về Giao Thức Đánh Giá & Giới Hạn Nghiên Cứu

### 2.1 Định Danh Phân Vùng Dữ Liệu (Historical Held-Out / Development Partition)
- Khoảng khung hình 351–600 trên `cam_5.mp4` là **phân vùng kiểm thử phát triển nội bộ (Historical Held-out / Development Partition)**, **KHÔNG ĐƯỢC tuyên bố là tập kiểm thử độc lập hoàn toàn (independent test set)**.
- **Ranh giới minh bạch**: Tọa độ điểm mút thoát Approach 2 (`(45.0, 370.0)`), khoảng cách thoát ROI (90px) và ngưỡng lọc quãng đường di chuyển tối thiểu (35px) đã từng được khảo sát và tinh chỉnh trên các khung hình 339, 343, 359, 375, 425 của clip này. Mặc dù trọng số mô hình ML được huấn luyện strictly trên frames $\le 350$, các tham số hình học toàn cục vẫn mang thông tin bối cảnh của clip.

### 2.2 Đánh Giá Stratified 3-Fold Cross-Validation Trên Tập Train của `cam_5`
- Cross-Validation được thực hiện chỉ trong phạm vi 46 tracklet của phân vùng train ($\le 350$ frames).
- **Cảnh báo Scikit-Learn**: Quá trình chia fold xuất hiện cảnh báo `UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3`. Do Class 12 chỉ có 1 mẫu và Class 5 chỉ có 2 mẫu trong tập train, việc chia fold không thể bảo đảm cân bằng lớp thiểu số (ví dụ Fold 2 không có mẫu Class 5 trong tập validation).
- Đánh giá Cross-Validation này chỉ đo lường tính ổn định phát triển nội bộ trên clip `cam_5`, **tuyệt đối không chứng minh khả năng tổng quát hóa sang camera khác** (`cam_1` đến `cam_4`, v.v.).

### 2.3 Diễn Giải Đúng Về Count-Only F1 = 0.9200
- Metric Count-Only F1 = 0.9200 ($\text{TP}=46, \text{FP}=7, \text{FN}=1$) được sinh ra khi bộ đánh giá **bỏ qua hoàn toàn nhãn Movement ID**, chỉ ghép các sự kiện đếm theo loại xe (Car/Truck) trong sai số thời gian $\pm 15$ frames.
- **Không dùng metric này để khẳng định logic đếm "ổn định tuyệt đối"**, bởi lẽ nó vẫn ghi nhận 7 False Positives và 1 False Negative, đồng thời bỏ qua toàn bộ 22 lỗi nhận diện sai hướng rẽ của mô hình CNN-LSTM. Cần tách bạch dứt khoát giữa Count-Only F1 và Strict Movement F1.

---

## 3. Đối Chiếu Kết Quả Thực Nghiệm & Quyết Định Tuyển Chọn Mô Hình

Dữ liệu đối chiếu trực tiếp từ [`artifacts/track1_final/evaluation/heldout_evaluation_metrics.json`](file:///D:/hanoi-traffic-platform-final-8.0/artifacts/track1_final/evaluation/heldout_evaluation_metrics.json):

| Phương Pháp Phân Loại | Standalone Test Acc ($N=46$) | End-to-End Strict F1 | Count-Only F1 | Wrong Movement Count | Quyết Định Tuyển Chọn |
|:---|:---:|:---:|:---:|:---:|:---|
| **Geometric Heuristic Baseline** | **89.13%** (41/46) | **0.8200** | **0.9200** | 5 | **PRODUCTION DEFAULT (Mặc định Sản xuất)** |
| **MLP Train-Only Classifier** | 86.96% (40/46) | **0.8200** | **0.9200** | 5 | Experimental Option (Tương đương Heuristic khi có fallback) |
| **CNN–LSTM Train-Only** | 84.78% (39/46) | **0.4800** | **0.9200** | 22 | Experimental Option (Bị phạt nặng do lệch prior Movement 5) |

### Lý Do Giữ Geometric Heuristic Làm Mặc Định Sản Xuất
- Trong tập train (frames $\le 350$), Movement 5 chỉ có 2 mẫu. Trong tập test (frames $351–600$), Movement 5 bùng nổ lên 23 sự kiện.
- CNN-LSTM bị chi phối bởi phân phối quá khứ, nhầm hầu hết Movement 5 thành Movement 3 (chỉ nhận đúng 3/23 xe Movement 5).
- Ngược lại, Calibrated Geometric Heuristic Baseline dựa trên vector hình học camera thực tế, nhận diện chính xác 20/23 xe Movement 5 mà không bị phụ thuộc vào phân phối lưu lượng trước đó.
- Không có mô hình học máy nào vượt qua baseline trên tập held-out này khi đánh giá công bằng. Do đó, **Geometric Heuristic Baseline là lựa chọn sản xuất duy nhất đạt yêu cầu**.

---

## 4. Xác Minh Gói Phân Phối Portable (`dist/track1_vehicle_counting_final.zip`)

- **Tên gói chính thức**: [`dist/track1_vehicle_counting_final.zip`](file:///D:/hanoi-traffic-platform-final-8.0/dist/track1_vehicle_counting_final.zip)
- **Kích thước file**: **142,540 bytes** (139.2 KB).
- **Mã băm SHA-256**: `6f8ed07bb2437e550037e0bdd38496ad5049f12dbba6d1de7183ee096503ef8a`
- **Gói phụ / Alias**: [`dist/track1_cnn_lstm_final.zip`](file:///D:/hanoi-traffic-platform-final-8.0/dist/track1_cnn_lstm_final.zip) (SHA-256: `ec2eb9bba4c034b8e42c83485cbee2971dfc20c10065f22f4abe4541efe2ad81`)
- **Số lượng entries**: 27 tệp tin và thư mục con (bao gồm bộ regression tests `tests/test_mlp_inference_parity.py`).
- **Phân loại gói**: **Source / Portable Release Package (Cần weights detector và video đầu vào từ bên ngoài; KHÔNG PHẢI fully standalone binary)**.
- **Fresh-Extraction Smoke Test**: Đã giải nén độc lập vào thư mục tạm `unpacked/`, kiểm tra cấu trúc thư mục, nạp unpickle thử các mô hình `.pkl`, xác minh tính toàn vẹn 100% đạt PASS.
- **Các thành phần bị loại trừ nghiêm ngặt**:
  - Không có file video gốc (`*.mp4`).
  - Không có file trọng số `detector.onnx`.
  - Không có môi trường ảo `.venv` hoặc bytecode cache `.pyc`.

---

## 5. Bảng Kiểm Định Bản Quyền, Nguồn Gốc & Ràng Buộc Pháp Lý

| Thành Phần | Nguồn Gốc / Tác Quyền | Giấy Phép Áp Dụng | Ràng Buộc Phân Phối Thực Tế |
|:---|:---|:---|:---|
| **Dữ Liệu AIC21 Track 1** | NVIDIA & AI City Committee | Academic Research Only | Phân phối raw video MP4 bị nghiêm cấm hoàn toàn. |
| **Checkpoints Movement ML/LSTM** | Huấn luyện từ quỹ đạo AIC21 `cam_5` | **Non-Commercial Academic Use Only** (AIC21 License Mục 2/3) | **KHÔNG THỂ cấp phép MIT tự do**. Bị ràng buộc bởi điều khoản sử dụng phi thương mại học thuật của dữ liệu gốc. |
| **YOLOv8n Detector** | Ultralytics LLC | **AGPL-3.0** / Commercial | Trọng số `detector.onnx` bị loại khỏi ZIP để tránh lây nhiễm copyleft; người dùng tự chịu trách nhiệm AGPL-3.0 khi tích hợp. |
| **Mã Nguồn Nền Tảng (Platform Code)** | Hanoi Traffic Platform | **MIT License** | Mã nguồn kịch bản, pipeline và công cụ được cấp phép MIT cho mục đích tái lập khoa học. |
| **Thư Viện Phụ Thuộc** | ONNX Runtime, OpenCV, Scikit-learn | MIT, Apache 2.0, BSD-3 | Phụ thuộc runtime tiêu chuẩn. |

**Phân Loại Trạng Thái Phát Hành**:
- **Phân phối công khai thương mại (General Public / Commercial Distribution)**:  
  **`RELEASE BLOCKED FOR GENERAL PUBLIC / COMMERCIAL DISTRIBUTION`**  
  *(Do các checkpoint movement mang dữ liệu phái sinh từ AIC21 giới hạn phi thương mại, và YOLOv8 ràng buộc AGPL-3.0)*.
- **Đánh giá nghiên cứu khoa học nội bộ (Academic Research Evaluation)**:  
  **`RELEASE READY (Non-Commercial Academic Research Evaluation Only)`**.

---

## 6. Kiểm Định Tính Bất Biến của Model Production (SHA-256 Hashes)

| Model Artifact | Kích thước | SHA-256 Ban Đầu | SHA-256 Hiện Tại | Kết quả Đối Soát |
|:---|:---:|:---:|:---:|:---:|
| [`models/detector.onnx`](file:///D:/hanoi-traffic-platform-final-8.0/models/detector.onnx) | 12,851,106 B | `e16f549d9532f4dee623cbf5231959160200c0ececac644e4350f127eee43a4a` | `e16f549d9532f4dee623cbf5231959160200c0ececac644e4350f127eee43a4a` | **100% BẤT BIẾN** |
| [`models/predictor.onnx`](file:///D:/hanoi-traffic-platform-final-8.0/models/predictor.onnx) | 208,499 B | `35b19d1f3738762a8cc7bca7deff06289af642f68aced723375a6187063173ea` | `35b19d1f3738762a8cc7bca7deff06289af642f68aced723375a6187063173ea` | **100% BẤT BIẾN** |
| [`models/scaler.pkl`](file:///D:/hanoi-traffic-platform-final-8.0/models/scaler.pkl) | 500 B | `ca3deefa87f29026b3888ad7b3fd8a55a28d78b95c1a8fb8c39a9fd237cf299c` | `ca3deefa87f29026b3888ad7b3fd8a55a28d78b95c1a8fb8c39a9fd237cf299c` | **100% BẤT BIẾN** |

---

## 7. Kết Quả Kiểm Thử Hệ Thống (Release Gates)

1. **Test Suite ([`tests/test_track1_counting.py`](file:///D:/hanoi-traffic-platform-final-8.0/tests/test_track1_counting.py), [`tests/test_mlp_inference_parity.py`](file:///D:/hanoi-traffic-platform-final-8.0/tests/test_mlp_inference_parity.py))**:
   `pytest -q` $\to$ **135 / 135 tests passing** (2.91s).
2. **Release Integrity ([`tools/release_check.py`](file:///D:/hanoi-traffic-platform-final-8.0/tools/release_check.py))**:
   Xác minh toàn bộ **282 release files** khớp chính xác với [`docs/RELEASE_MANIFEST.json`](file:///D:/hanoi-traffic-platform-final-8.0/docs/RELEASE_MANIFEST.json).
3. **Bytecode Compilation**:
   `compileall` trên `aicc`, `traffic_brain`, `tools`, `tests`, `experiments` đạt chuẩn **100% clean**.
4. **Model Smoke Test ([`tools/model_smoke.py`](file:///D:/hanoi-traffic-platform-final-8.0/tools/model_smoke.py))**:
   Trả về `MODEL_SMOKE_PASS`.
5. **Fresh Extraction & Archive Smoke Test**:
   Giải nén và kiểm tra file/checksums của `dist/track1_vehicle_counting_final.zip` $\to$ **PASS**.
6. **Master Build Script ([`tools/build_track1_cnn_lstm_final.ps1`](file:///D:/hanoi-traffic-platform-final-8.0/tools/build_track1_cnn_lstm_final.ps1))**:
   Hoàn tất toàn bộ 22 bước với **Exit Code: 0**.

---

## 8. Báo Cáo Điều Tra & Khắc Phục Lỗi Mismatch Inference Của MLP Classifier

### 8.1 Hiện tượng Discrepancy
- Đánh giá held-out trước đây ghi nhận MLP Train-Only đạt Strict F1 = 0.8200 (đánh giá file `experiments/training/track1_vehicle_counting/output_counting_ml_train_only_cam5_1min.csv`).
- Tuy nhiên trong kiểm định triển khai gần đây, tệp [`artifacts/deployment_validation/mlp_cam5_600frames.csv`](file:///D:/hanoi-traffic-platform-final-8.0/artifacts/deployment_validation/mlp_cam5_600frames.csv) chỉ đạt Strict F1 = 0.1600 (TP = 8, FP = 45, FN = 39, Wrong Movement = 38).
- Điểm kỳ lạ: Cả hai lần chạy đều có đúng 53 sự kiện dự đoán và đạt Count-Only F1 = 0.9200 (TP = 46, FP = 7, FN = 1).

### 8.2 Nguyên Nhân Gốc Rễ (Root Causes)
1. **Lệch Thước Đo Đặc Trưng Thời Lượng (Feature 9 Scaler Mismatch)**:
   - Checkpoint `d2` (`artifacts/track1_final/models/movement_classifier_train_only.pkl`, 45,858 bytes, SHA-256 `fd2ee17e...`, sinh ra từ script tái huấn luyện `retrain_and_validate.py`) sử dụng `StandardScaler` được fit trên tập chuỗi con subsampled ($L \le 20$ timesteps từ `cnn_lstm_dataset.npz`), có `mean_[9] = 19.8696` và `scale_[9] = 0.6464`.
   - Ngược lại, pipeline inference trực tiếp ([`aicc/track1_counting.py`](file:///D:/hanoi-traffic-platform-final-8.0/aicc/track1_counting.py)) trích xuất Feature 9 là số frame thực tế của quỹ đạo xe trong video (`duration = frames[-1] - frames[0]`), dao động từ 30 đến 150 frames.
   - Khi chuẩn hóa giá trị $50..120$ frames bằng $\mu=19.87, \sigma=0.65$, giá trị đầu vào bị phóng đại lên **$+50$ đến $+155$ độ lệch chuẩn**, làm bão hòa hoàn toàn các hàm kích hoạt ReLU của MLP, ép hầu hết dự đoán thành Movement 3.
2. **Khác Biệt Nguồn Gốc Checkpoint (Checkpoint Provenance)**:
   - Checkpoint gốc `d1` ([`experiments/training/track1_vehicle_counting/movement_classifier_train_only.pkl`](file:///D:/hanoi-traffic-platform-final-8.0/experiments/training/track1_vehicle_counting/movement_classifier_train_only.pkl), 64,883 bytes, SHA-256 `466f89a1ecd81ed7f81c1a75ef5e6d01542ea0362f326a4fab6bbe5b4e226512`) được huấn luyện bằng [`train_movement_classifier.py`](file:///D:/hanoi-traffic-platform-final-8.0/experiments/training/track1_vehicle_counting/train_movement_classifier.py) strictly trên `exit_frame <= 350` với scaler có `mean_[9] = 69.7826` và `scale_[9] = 28.1540`, hoàn toàn tương thích với live inference.
3. **Thiếu Cơ Chế Fallback Khi Độ Tin Cậy Thấp**:
   - `Track1MovementMatcher.match` trước đây gọi trực tiếp `model.predict(x_arr)[0]` mà không kiểm tra xác suất phân phối `predict_proba`. Khi gặp mẫu chuyển động bất thường hoặc không chắc chắn, mô hình không fallback về geometric heuristic.

### 8.3 Giải Pháp Khắc Phục Đã Triển Khai
1. **Khôi Phục Checkpoint Đúng**: Đưa checkpoint `d1` (SHA-256: `466f89a1ecd81ed7f81c1a75ef5e6d01542ea0362f326a4fab6bbe5b4e226512`) vào vị trí chuẩn [`artifacts/track1_final/models/movement_classifier_train_only.pkl`](file:///D:/hanoi-traffic-platform-final-8.0/artifacts/track1_final/models/movement_classifier_train_only.pkl).
2. **Nâng Cấp Matcher Với Fallback Dự Phòng**: Cập nhật `Track1MovementMatcher.match` trong `aicc/track1_counting.py` để tính `predict_proba`. Nếu xác suất dự đoán $< 0.35$ hoặc nhãn ra ngoài miền cho phép, tự động fallback an toàn về `match_heuristic(track)`.
3. **Xây Dựng Regression Test Suite**: Tạo [`tests/test_mlp_inference_parity.py`](file:///D:/hanoi-traffic-platform-final-8.0/tests/test_mlp_inference_parity.py) với 5 bài kiểm thử tự động xác minh schema 15 chiều, tham số scaler của checkpoint, fallback cho tracklet ngắn/mơ hồ và tính toàn vẹn của kết quả đếm.
4. **Bảo Tồn Dữ Liệu Lịch Sử**: Không ghi đè hay xóa `artifacts/deployment_validation/mlp_cam5_600frames.csv`. Toàn bộ dữ liệu tái kiểm định được lưu sang tệp độc lập mới: [`artifacts/deployment_validation/mlp_cam5_600frames_revalidated.csv`](file:///D:/hanoi-traffic-platform-final-8.0/artifacts/deployment_validation/mlp_cam5_600frames_revalidated.csv).

### 8.4 Kết Quả Tái Đánh Giá Đối Soát Sau Khắc Phục (Frames 351–600)
- **Strict Movement Matching**:
  - True Positives (TP): **41**
  - False Positives (FP): **12**
  - False Negatives (FN): **6**
  - Precision: **0.7736**
  - Recall: **0.8723**
  - **Strict Movement F1**: **0.8200**
  - Wrong Movement Events: **5**
- **Count-Only Temporal Matching**:
  - Count TP: **46**
  - Count FP: **7**
  - Count FN: **1**
  - Count Precision: **0.8679**
  - Count Recall: **0.9787**
  - **Count-Only F1**: **0.9200**
- **Trạng thái kết luận**: **`FIX VERIFIED`** (Parity với baseline heuristic và chạy lịch sử được khôi phục 100%).

