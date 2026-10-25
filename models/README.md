# AICC model bundle

The active AICC video-inference bundle uses these files, matching `manifest.json`:

- `detector.onnx`
- `predictor.onnx`
- `scaler.pkl`

The separately versioned Hanoi forecast candidate is stored alongside the active
bundle:

- `hanoi_predictor.onnx`
- `hanoi_scaler.pkl`
- `hanoi_manifest.json`

This candidate has a different inference contract (30 steps, nine features, and
three forecast horizons). The source videos were sampled at one row per second,
so the forecast horizons are 5, 10, and 15 seconds. It is not loaded by the
current AICC video-inference pipeline; do not replace the active bundle until
the runtime is migrated and validated against this contract.

The Hanoi artifacts are an explicit, first-party model-bundle exception to the
general rule that model binaries are excluded from source releases. The scaler
is a Python pickle and must only be loaded from this trusted repository source.
Other large or untrusted `.pkl`/model files must not be committed.

The upstream integration is an adapted inference module, not a vendored clone
of the full repository.
