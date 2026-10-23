# Observation storage base

The folder is an empty base; no fabricated observations are included. Once a feature extraction pipeline is implemented and calibrated, store one row per camera and fixed time window.

Suggested UTF-8 CSV fields (align units and feature semantics to the model's actual training pipeline):

```csv
timestamp_utc,camera_id,window_seconds,car_count,motorcycle_count,bus_count,truck_count,pcu_equivalent,mean_speed_kmh,speed_cv_ratio,density_areal_occupancy,ci,source_id,calibration_version,quality_flag
```

Requirements:

- timestamps must be timezone-aware and windows must be aligned consistently;
- counts/PCU must have a documented counting rule (unique tracks vs frame counts);
- `mean_speed_kmh` needs calibrated ground-plane geometry; never copy pixel/s values into this column;
- `density_areal_occupancy` must derive from a configured road ROI/area and a documented method;
- preserve camera/source/calibration versions and quality flags;
- missing values remain missing/flagged until handled by the training pipeline; do not silently fill a training-time default in data collection.
