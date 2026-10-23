# AICC model bundle

Put these three files here, matching `manifest.json`:

- `detector.onnx`
- `predictor.onnx`
- `scaler.pkl`

The final source ZIP intentionally excludes binary model artifacts. The Windows bootstrap copies trusted local files from `D:\AICC\models` into this folder at launch. The upstream integration is an adapted inference module, not a vendored clone of the full repository. Do not push large or untrusted `.pkl`/model files to a public repository. The `.pkl` scaler is deserialized by Python and should only come from a source you trust.
