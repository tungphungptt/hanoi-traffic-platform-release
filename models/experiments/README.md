# Experimental Model Checkpoints (`models/experiments/`)

Thư mục lưu trữ các trọng số mô hình thử nghiệm, checkpoints huấn luyện dở dang và mô hình ứng viên.

## Phân định với Mô hình Production:
- **Production Models** nằm trực tiếp tại `models/`:
  - `models/detector.onnx`
  - `models/predictor.onnx`
  - `models/scaler.pkl`
  - `models/manifest.json`
- **Experimental Models** nằm tại `models/experiments/<experiment_id>/`:
  - Checkpoint PyTorch (`.pt`, `.pth`), TensorFlow/Keras (`.h5`), hoặc ONNX xuất khẩu thử nghiệm.
  - Chỉ khi một mô hình thử nghiệm vượt qua bài kiểm tra đánh giá (A/B evaluation hoặc benchmark test), mô hình đó mới được đóng gói và thăng cấp (promote) thành mô hình Production tại `models/`.
- File trọng số trong thư mục này được `.gitignore` loại trừ, không đưa lên Git repository.
