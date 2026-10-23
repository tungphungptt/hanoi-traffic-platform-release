# Hanoi Traffic Platform - Full-stack + AI Release

Download the ZIP archive: hanoi-traffic-platform-fullstack-ai_FINAL_20261024_002623.zip

## Contents
- FastAPI backend and static dashboard
- Python AI/AICC pipeline and traffic analytics
- Available ONNX model artifacts and model scaler
- Configuration, tests, and documentation

## Quick start
1. Download and extract the ZIP.
2. Use Python 3.12 on Windows.
3. Open PowerShell in the extracted project directory.
4. Run:
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\deploy_demo.ps1" -Port 8000

## Important limitations
- Model smoke tests verify technical inference on synthetic tensors, not real-world Hanoi accuracy.
- Forecasting is not confirmed ready until a compatible trained forecaster and valid observations are provided.
- Dataset and source videos are excluded from this archive.
- Check third-party licenses before redistributing models or using them commercially.
