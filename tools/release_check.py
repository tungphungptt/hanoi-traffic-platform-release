from __future__ import annotations

import json
import sys
from pathlib import Path
try:
    from release_hashing import canonical_release_bytes
except ModuleNotFoundError:
    from tools.release_hashing import canonical_release_bytes

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "app/main.py",
    "aicc/pipeline.py",
    "traffic_brain/video_review.py",
    "traffic_brain/routing.py",
    "traffic_brain/providers.py",
    "traffic_brain/feature_engine.py",
    "static/index.html",
    "static/app.js",
    "static/styles.css",
    "models/manifest.json",
    "artifacts/current/manifest.json",
    "deploy_demo.ps1",
    "requirements.txt",
    "requirements-dev.txt",
    "tools/model_smoke.py",
    "tools/verify_video.py",
    "tools/release_manifest.py",
    "docs/V8_IMPROVEMENT_MATRIX.md",
    "docs/RELEASE_MANIFEST.json",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        fail("Missing release files: " + ", ".join(missing))

    version = (ROOT / "VERSION").read_text(encoding="utf-8-sig").strip()
    if version != "8.0.0":
        fail(f"VERSION must be 8.0.0, found {version!r}")

    app_source = (ROOT / "app/main.py").read_text(encoding="utf-8-sig")
    html = (ROOT / "static/index.html").read_text(encoding="utf-8-sig")
    if 'version="8.0.0"' not in app_source:
        fail("FastAPI metadata version does not match 8.0.0")
    if "8.0.0" not in html:
        fail("UI version label does not contain 8.0.0")

    # Windows PowerShell 5.1 requires the UTF-8 BOM to interpret non-ASCII source
    # reliably. Exactly one BOM must be present, not zero or a duplicated prefix.
    ps = (ROOT / "deploy_demo.ps1").read_bytes()
    bom = b"\xef\xbb\xbf"
    if not ps.startswith(bom):
        fail("deploy_demo.ps1 must start with one UTF-8 BOM for Windows PowerShell 5.1")
    if ps.startswith(bom + bom):
        fail("deploy_demo.ps1 has duplicate UTF-8 BOMs (previous release parser bug)")
    try:
        ps_text = ps.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        fail(f"deploy_demo.ps1 is not valid UTF-8: {exc}")
    if ps_text.startswith("\ufeff"):
        fail("deploy_demo.ps1 still contains a second leading BOM after decoding")
    if not ps_text.lstrip().startswith("[CmdletBinding()]"):
        fail("deploy_demo.ps1 must start with [CmdletBinding()] and its param block")
    for token in ("pip check", "tools/release_check.py", "pytest -q", "DEMO READY", "$BundleFiles", "if ($SourceComplete)", "function Get-PythonMinor", "$pythonExitCode = $LASTEXITCODE", "tools/model_smoke.py"):
        if token not in ps_text:
            fail(f"Deployment script is missing safety gate: {token}")

    try:
        aicc = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8-sig"))
        forecast = json.loads((ROOT / "artifacts/current/manifest.json").read_text(encoding="utf-8-sig"))
        cameras = json.loads((ROOT / "data/cameras.json").read_text(encoding="utf-8-sig"))
        json_paths = [
            ROOT / "models/manifest.json",
            ROOT / "artifacts/current/manifest.json",
            ROOT / "data/cameras.json",
            ROOT / "data/camera_calibrations/camera-calibration.schema.json",
            ROOT / "data/labels/forecast-target.schema.json",
            ROOT / "data/observations/observation.schema.json",
            ROOT / "data/manifests/dataset_manifest.schema.json",
            ROOT / "data/manifests/hanoi_videos_manifest.json",
        ]
        for json_path in json_paths:
            # Parse named release-owned contracts only; never crawl user data, .venv, or generated job output.
            json.loads(json_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Could not parse release JSON config: {exc}")

    for key in ("version", "classes", "sequence_length", "features", "models"):
        if key not in aicc:
            fail(f"AICC manifest is missing {key}")
    if not isinstance(aicc.get("sequence_length"), int) or aicc["sequence_length"] < 1:
        fail("AICC sequence_length must be a positive integer")
    models = aicc["models"]
    for key in ("detector", "predictor", "scaler"):
        if not isinstance(models.get(key), str):
            fail(f"AICC models.{key} path is missing")
    horizons = forecast.get("forecaster_config", {}).get("horizons_minutes")
    if horizons != [5, 10, 15]:
        fail("Forecast manifest must declare ordered horizons [5, 10, 15]")
    camera_rows = cameras.get("cameras") if isinstance(cameras, dict) else None
    if not isinstance(camera_rows, list) or not camera_rows:
        fail("data/cameras.json must contain a non-empty 'cameras' array")
    camera_ids = [row.get("id") for row in camera_rows if isinstance(row, dict)]
    if len(camera_ids) != len(camera_rows) or any(not isinstance(cid, str) or not cid for cid in camera_ids):
        fail("Each configured camera row must have a non-empty string id")
    if len(set(camera_ids)) != len(camera_ids):
        fail("Camera IDs must be unique")

    try:
        inventory = json.loads((ROOT / "docs/RELEASE_MANIFEST.json").read_text(encoding="utf-8-sig"))
        entries = inventory.get("files")
        if inventory.get("project_version") != version or not isinstance(entries, list):
            fail("Release file manifest version or entries are invalid")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                fail("Release file manifest has an invalid entry")
            target = (ROOT / entry["path"]).resolve()
            try:
                target.relative_to(ROOT.resolve())
            except ValueError:
                fail("Release file manifest path escapes the project root")
            if not target.is_file():
                fail("Release file listed in manifest is missing: " + entry["path"])
            import hashlib
            data = canonical_release_bytes(target, target.read_bytes())
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry.get("sha256") or len(data) != entry.get("bytes"):
                fail("Release file checksum/size mismatch: " + entry["path"])
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Could not verify release file manifest: {exc}")

    print("PASS: release files are present")
    print("PASS: version strings are aligned to 8.0.0")
    print("PASS: deploy_demo.ps1 has exactly one UTF-8 BOM and safety gates")
    print(f"PASS: release SHA-256 inventory verified ({len(entries)} files)")
    print("PASS: AICC and multi-horizon manifests parse and meet required contract")
    print(f"PASS: all {len(json_paths)} shipped JSON files parse; {len(camera_rows)} configured camera slots have unique IDs")
    print("NOTE: This static checker does not replace Windows PowerShell execution, real model inference, or licensed CCTV integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
