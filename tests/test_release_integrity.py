from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent():
    version = (ROOT / "VERSION").read_text(encoding="utf-8-sig").strip()
    app_source = (ROOT / "app/main.py").read_text(encoding="utf-8-sig")
    page = (ROOT / "static/index.html").read_text(encoding="utf-8-sig")
    assert version == "8.0.0"
    assert 'version="8.0.0"' in app_source
    assert "8.0.0" in page


def test_deploy_script_has_exactly_one_utf8_bom_and_clean_header():
    data = (ROOT / "deploy_demo.ps1").read_bytes()
    bom = b"\xef\xbb\xbf"
    assert data.startswith(bom)
    assert not data.startswith(bom + bom)
    text = data.decode("utf-8-sig")
    assert not text.startswith("\ufeff")
    assert text.lstrip().startswith("[CmdletBinding()]")
    for token in ("pip check", "tools/release_check.py", "pytest -q", "$BundleFiles", "if ($SourceComplete)"):
        assert token in text


def test_deploy_python_version_gate_captures_exit_code_and_parses_full_version():
    text = (ROOT / "deploy_demo.ps1").read_text(encoding="utf-8-sig")
    assert "function Get-PythonMinor" in text
    assert "$pythonExitCode = $LASTEXITCODE" in text
    assert r"3\.(?:11|12)" in text
    assert "$SupportedPythonVersions -contains $PythonMinor" in text
    assert "$PythonMinor.Trim() -notin" not in text


def test_model_smoke_is_in_bootstrap_and_video_verifier_is_packaged():
    script = (ROOT / "deploy_demo.ps1").read_text(encoding="utf-8-sig")
    assert "tools/model_smoke.py" in script
    smoke = (ROOT / "tools/model_smoke.py").read_text(encoding="utf-8-sig")
    for required in ("detector", "predictor", "scaler", "CPUExecutionProvider", "MODEL_SMOKE_PASS"):
        assert required in smoke
    assert (ROOT / "tools/verify_video.py").is_file()


def test_aicc_manifest_and_forecaster_contract_remain_separate():
    aicc = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8-sig"))
    forecast = json.loads((ROOT / "artifacts/current/manifest.json").read_text(encoding="utf-8-sig"))
    assert aicc["sequence_length"] == 5
    assert aicc["features"] == ["car_count", "avg_speed", "speed_cv"]
    assert forecast["forecaster_config"]["horizons_minutes"] == [5, 10, 15]


def test_release_file_manifest_is_versioned_and_non_empty():
    payload = json.loads((ROOT / "docs/RELEASE_MANIFEST.json").read_text(encoding="utf-8-sig"))
    assert payload["project_version"] == "8.0.0"
    assert payload["files_count"] >= 40
    assert any(row["path"] == "deploy_demo.ps1" for row in payload["files"])


def test_local_model_weights_and_video_are_excluded_from_version_control_by_default():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8-sig")
    assert "/models/*.onnx" in gitignore
    assert "/models/*.pkl" in gitignore
    assert "/data/*" in gitignore
    assert "!/data/cameras.json" in gitignore


def test_local_bootstrap_actually_sets_write_mode_before_launch():
    # A setting hidden at the end of a comment is not a PowerShell assignment.
    script = (ROOT / "deploy_demo.ps1").read_text(encoding="utf-8-sig")
    launch = script.index("$Proc = Start-Process")
    for name, value in (("PUBLIC_DEMO_MODE", "false"), ("AICC_VIDEO_ENABLED", "true")):
        assignment = re.search(rf"(?m)^\s*\$env:{name}\s*=\s*'{value}'\s*$", script)
        assert assignment, f"Local bootstrap must execute the {name} assignment, not comment it out"
        assert assignment.start() < launch
