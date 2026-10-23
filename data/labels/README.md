# Ground-truth labels base

No synthetic future labels are provided. To train and evaluate a multi-horizon congestion model, join each observation window to independently measured target values at +5, +10 and +15 minutes for the same camera/road segment.

Suggested UTF-8 CSV fields:

```csv
window_timestamp_utc,camera_id,target_timestamp_5m,ci_5m,level_5m,target_timestamp_10m,ci_10m,level_10m,target_timestamp_15m,ci_15m,level_15m,label_source,review_status
```

Use labels from a documented and consistent ground-truth definition (for example, the proposal's calibrated congestion index and four levels). Split train/validation/test by time and preferably camera/site to reduce leakage. Report MAE/RMSE for CI and Macro-F1 at every horizon; separately report detector mAP, tracking IDF1, vehicle-count error, latency and RAM where those parts are evaluated. The KPI targets in the proposal are targets to test, not results achieved by this demo.
