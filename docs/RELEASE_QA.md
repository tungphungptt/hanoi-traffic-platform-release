# Release QA â€” Hanoi Traffic Platform Final 8.0.0

## Verified in build environment

| Check | Result | Evidence |
|---|---|---|
| Release integrity | PASS | `python tools/release_check.py` including per-file SHA-256 inventory verification |
| Python syntax | PASS | `python -m compileall -q app aicc traffic_brain tests tools` |
| Automated tests | Per-commit CI | Use the GitHub Actions result for the exact commit as the authoritative test result. |
| Front-end syntax | PASS | `node --check static/app.js` |
| Manifest/config JSON | PASS | Six named JSON contracts parsed; camera IDs unique |
| PowerShell script bytes | PASS â€” static only | Exactly one UTF-8 BOM, required header and robust Python-version-check tokens verified |
| ZIP integrity | PASS | All archive entries CRC-checked and tests rerun from a clean extraction |
| API smoke | PASS | Uvicorn served health, home, dashboard, camera list, map config, provider status, model manifest, AICC status, telemetry status and fallback street image: all HTTP 200 |
| Real artifact smoke in builder | NOT RUN | Model weights are intentionally excluded from release package and unavailable in this Linux build environment |

## Final 8.0 additions

- Replaced the fragile Python-version expression with `Get-PythonMinor`, which captures the native executable's exit code immediately and parses a complete supported version line.
- Recreates only the project's `.venv` if the interpreter cannot be verified as Python 3.11/3.12. It does not change global Python installations or project files outside `.venv`.
- Deployment runs `tools/model_smoke.py` if and only if all four AICC artifacts exist. The smoke runs real ONNX sessions on bounded synthetic tensors and stops if a present bundle is broken or incompatible; when the artifacts are missing, it logs a warning and continues without AICC video inference.
- Added `tools/verify_video.py` to run the real AICC pipeline on an operator-selected local clip, with file-size and 20-minute duration limits, progress messages and JSON diagnostics.
- Added a SHA-256 inventory for immutable release source. Mutable operator-controlled manifests/camera config and local feed index are intentionally excluded from that inventory but validated by the release checker.

## Limitations and acceptance still needed

- Build environment is Linux, not Windows PowerShell. The deployment script is statically checked; this process could not natively execute Windows PowerShell on the target OS. The first clean run on the user's Windows host remains an acceptance gate.
- No ONNX weights/video are bundled. The actual `tools/model_smoke.py` and `tools/verify_video.py` must be run on a machine with the trusted four-file AICC bundle and a short known-good clip. Smoke success is not an accuracy benchmark.
- AICC v1.0-car-only is a two-class sequence classifier with pixel/second motion features, not the Hanoi-specific +5/+10/+15 forecaster. The multi-horizon manifest is a contract; no trained Hanoi forecaster is bundled.
- No authorized Hanoi CCTV/RTSP feed is bundled. OSM is a real basemap; OSRM public routing provides reference road-network durations rather than live traffic ETA.

## Release status

**Hardened local release candidate.** Suitable for controlled local testing after bootstrap succeeds on Windows and, when weights are present, the AICC model smoke passes. It is not certified for traffic-control decisions or public production without verified data ingestion, a trained/evaluated Hanoi forecaster, access controls, persistent storage, TLS, quotas, operational monitoring and an authorized traffic provider.

