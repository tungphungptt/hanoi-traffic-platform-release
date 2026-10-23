# Final 8.0 — Improvement and acceptance matrix

| Area | Final 8.0 change | Acceptance evidence | Remaining external dependency |
|---|---|---|---|
| Windows Python bootstrap | One function executes Python, captures `$LASTEXITCODE` immediately, parses a full major/minor line, rebuilds only project-owned `.venv` if invalid, verifies final interpreter | Static regression test, release checker, UTF-8 BOM check | Actual execution on target Windows host remains required |
| Model artifact integrity | Copy all four AICC files as one bundle; never mix incomplete source weights and manifest | SHA-256 printed by bootstrap; model smoke runs if bundle is complete | Upstream model provenance and licensing must be independently verified |
| Runtime model compatibility | Added bounded CPU smoke inference for detector, scaler and predictor before the server starts | `tools/model_smoke.py` and regression tests; executes real ONNX if weights exist | Does not establish video accuracy or traffic generalization |
| End-to-end video check | Added operator-invoked CLI for inference on a chosen local video, duration-limited, exports summary JSON | CLI static/unit tests; run manually on actual clip for acceptance | Requires local video, OpenCV codecs and complete AICC artifact bundle |
| Release consistency | Required-file, version, manifest, BOM and contract gates plus a per-file SHA-256 inventory | `tools/release_check.py`, `docs/RELEASE_MANIFEST.json`, pytest, compileall, Node syntax check | Runtime-owned manifests/camera data are intentionally excluded from source checksums so operator configuration changes do not break deployment |
| Forecasting integrity | Keeps AICC two-class classifier separate from multi-horizon forecaster contract; no prediction if measured features are insufficient | Existing feature/provider/routing test suite | A trained and independently evaluated Hanoi forecaster is not bundled |
| Live traffic honesty | Map/tile, route reference durations and camera references retain source/telemetry limits | Existing map/telemetry/routing API tests | Requires authorized CCTV/RTSP or traffic API access |
| Safe packaging | Excludes `.env`, cache, bytecode, user footage and model weights | ZIP file-list audit + CRC test | Operator must configure local model/traffic providers |

## Non-negotiable acceptance gates

1. On Windows 10/11, run `deploy_demo.ps1` from a clean extraction and record the detected interpreter.
2. With weights present, observe `MODEL_SMOKE_PASS` before Uvicorn starts.
3. Run `tools/verify_video.py` against a short known-good clip and inspect its summary/results.
4. Do not label trend extrapolation as a trained prediction.
5. Do not recommend a road as “less congested” unless current traffic observations cover candidate alternatives with compatible timestamps and freshness.
6. Before external/public deployment: configure auth/roles, HTTPS, persistent storage, audit/retention, rate limits, and authorized traffic/CCTV source.

This matrix is a software release checklist, not a certification of real-world traffic prediction accuracy.
