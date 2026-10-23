# AI City Challenge Track 1 Vehicle Counting Experiment (`experiments/training/track1_vehicle_counting/`)

Thực nghiệm xây dựng và đánh giá baseline cho bài toán AI City Challenge Track 1 (Vehicle Counting by Movement).

## 1. Mô hình sử dụng:
- **Tái sử dụng**: `models/detector.onnx` (YOLOv8n trained on COCO).
  - COCO class 2 (`car`) -> Track 1 Class 1 (`Car`).
  - COCO class 7 (`truck`) và 5 (`bus`) -> Track 1 Class 2 (`Truck`).
- **Không sử dụng**: `models/predictor.onnx` và `models/scaler.pkl` (đây là mô hình phân loại mức độ tắc nghẽn Thông thoáng / Đông đúc của đề án AICC, không có vai trò trong bài toán đếm xe theo hướng rẽ).

## 2. Dữ liệu thực nghiệm:
- Video: `AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4`.
- ROI: `AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt`.
- Ground truth tham chiếu: `AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv` (chỉ có trong 60 giây đầu / 600 khung hình).
- Các video còn lại không có ground truth công khai đi kèm trong bộ dữ liệu ban tổ chức.

## 3. Lệnh chạy thực nghiệm:
```powershell
.\.venv\Scripts\python tools/run_track1_counting.py --video AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4 --roi AIC21_Track1_Vehicle_Counting/ROIs/cam_5.txt --detector models/detector.onnx --max-frames 600 --ground-truth AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv --output-csv experiments/training/track1_vehicle_counting/output_counting_cam5_1min.csv --metrics-output experiments/training/track1_vehicle_counting/evaluation_metrics.json
```
