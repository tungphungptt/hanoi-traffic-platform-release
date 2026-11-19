# Third-party notices

- **YOLOX-s detector** (`models/detector.onnx`): Megvii YOLOX, COCO-pretrained weights, licensed
  Apache-2.0 (https://github.com/Megvii-BaseDetection/YOLOX).
- **AI City Challenge 2021 Track 1** (`models/forecast_library/` AIC records): derived numeric series
  only; the dataset is licensed for non-commercial academic research.
- **Hanoi traffic-camera broadcast recordings** (`models/forecast_library/` Hanoi records, Hanoi demo):
  derived numeric series only, from recordings of the VOV Giao thông FM90 / Hà Nội LIVE programme
  shared in the "Hanoi Flow" Google Drive folder. No video is redistributed. Research/demo use.
- **Leaflet 1.9.4** (BSD-2-Clause), loaded from unpkg with Subresource Integrity on the map page.
- **OpenStreetMap**: map tiles from tile.openstreetmap.org (usage policy: https://operations.osmfoundation.org/policies/tiles/)
  (Humanitarian style tiles from OpenStreetMap France as fallback), the Hanoi boundary (`static/hanoi_boundary.geojson`, relation 1903516), and KCN Thuận Thành III outline (`static/industrial_park_thuan_thanh_iii.geojson`, way 1467190501), © OpenStreetMap contributors, ODbL.
- **OSRM** public demo server (router.project-osrm.org) for route alternatives: best-effort, no SLA; run your own for production.
- **Google Fonts** DM Sans and Manrope (SIL Open Font License).
- **Open-Meteo** weather data, CC BY 4.0, attributed on the map page.
- **DejaVu Sans** (`traffic_brain/fonts/`, Bitstream Vera / public-domain-style licence, see `DejaVu-LICENSE.txt`): embedded in PDF reports so Vietnamese text renders.
- Python packages in `requirements.txt` keep their own licences (FastAPI MIT, Uvicorn BSD-3, openpyxl MIT, ReportLab BSD,
  NumPy BSD-3, ONNX Runtime MIT, OpenCV Apache-2.0).

- **YOLOX** (optional, not shipped): Megvii, Apache-2.0, https://github.com/Megvii-BaseDetection/YOLOX.
  `traffic_brain/detector.py` can load its official ONNX exports through `DETECTOR_MODEL`; see `models/README.md`.
