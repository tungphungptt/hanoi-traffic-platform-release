# Third-party notices

- **YOLOv8n detector** (`models/detector.onnx`): model architecture and COCO-pretrained weights by
  Ultralytics, licensed AGPL-3.0 (https://github.com/ultralytics/ultralytics). Distributing a service
  built on it has AGPL obligations; check before commercial or public deployment.
- **AI City Challenge 2021 Track 1** (`models/forecast_library/` AIC records): derived numeric series
  only; the dataset is licensed for non-commercial academic research.
- **Hanoi traffic-camera broadcast recordings** (`models/forecast_library/` Hanoi records, Hanoi demo):
  derived numeric series only, from recordings of the VOV Giao thông FM90 / Hà Nội LIVE programme
  shared in the "Hanoi Flow" Google Drive folder. No video is redistributed. Research/demo use.
- **Leaflet 1.9.4** (BSD-2-Clause), loaded from unpkg with Subresource Integrity on the map page.
- **OpenStreetMap**: map tiles from tile.openstreetmap.org (usage policy: https://operations.osmfoundation.org/policies/tiles/)
  (Humanitarian style tiles from OpenStreetMap France as fallback) and the Hanoi boundary (`static/hanoi_boundary.geojson`, relation 1903516), © OpenStreetMap contributors, ODbL.
- **Open-Meteo** weather data, CC BY 4.0, attributed on the map page.
- Python packages in `requirements.txt` keep their own licences (FastAPI MIT, Uvicorn BSD-3,
  NumPy BSD-3, ONNX Runtime MIT, OpenCV Apache-2.0).
