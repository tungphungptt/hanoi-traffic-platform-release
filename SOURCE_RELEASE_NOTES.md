# Hanoi Traffic Platform - Full-stack + AI package

This archive includes the FastAPI backend, static dashboard, Python traffic
analytics, AICC pipeline source, available model artifacts, configurations,
tests, and documentation.

## Models

The package is intended to include existing model files from the project,
including detector.onnx, predictor.onnx, scaler.pkl, and their manifest when
those files exist in the source checkout.

The car-only AICC classifier distinguishes the bundled traffic-state classes.
Its tensor smoke test is not an accuracy evaluation on real Hanoi CCTV.

A compatible, trained Hanoi multi-horizon forecaster and verified live CCTV
telemetry must be supplied/configured separately if they are not present in
the source checkout. Dashboard health does not prove that live forecasting
is operational.

## Excluded

Raw datasets, source videos, runtime camera feeds, large generated training
arrays, generated video-frame/label samples, local virtual environments,
caches, secret environment files, and previous dist archives are excluded.

## Before public redistribution

Check the licenses and provenance of every model weight, dataset-derived
artifact, dependency, and third-party component. Do not publish model weights
unless redistribution is permitted. No project-wide open-source license is
asserted by this note.
