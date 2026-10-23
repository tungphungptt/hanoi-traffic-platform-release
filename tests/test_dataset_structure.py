"""Unit and regression tests for Hanoi Traffic Platform 8.0 dataset and experiment structure."""
from __future__ import annotations

import json
from pathlib import Path
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_manifest_schema_and_catalog_are_valid():
    """Verify that dataset_manifest.schema.json and hanoi_videos_manifest.json parse and match contract."""
    schema_path = ROOT / "data/manifests/dataset_manifest.schema.json"
    manifest_path = ROOT / "data/manifests/hanoi_videos_manifest.json"

    assert schema_path.is_file(), "Schema file must exist"
    assert manifest_path.is_file(), "Catalog manifest file must exist"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert schema.get("title") == "HanoiTrafficVideoDatasetManifest"
    assert manifest.get("manifest_version") == "hanoi-video-manifest/1.0"
    assert manifest.get("storage_root") == "data/raw/hanoi_traffic"
    assert isinstance(manifest.get("videos"), list)
    assert len(manifest["videos"]) >= 1

    # Check baseline video record
    base = manifest["videos"][0]
    assert base["video_id"] == "HN-BASE-001"
    assert base["relative_path"] == "data/input.mp4"
    if not (ROOT / base["relative_path"]).is_file():
        pytest.skip("Baseline input.mp4 intentionally excluded from dataset-free release")
    assert base["fps"] == 10.0
    assert base["resolution"] == {"width": 1280, "height": 960}
    assert base["duration_seconds"] == 300.0
    assert base["frame_count"] == 3000
    assert base["label_status"] in {"unlabeled", "auto_labeled", "verified_ground_truth"}
    assert base["split"] in {"train", "val", "test", "unassigned"}


def test_standardized_directories_and_readmes_exist():
    """Verify all prepared directories for data, experiments, models, and artifacts exist with documentation."""
    expected_dirs = [
        "data/raw/hanoi_traffic",
        "data/staging",
        "data/processed",
        "data/annotations",
        "data/manifests",
        "experiments",
        "experiments/auto_labeling",
        "experiments/training",
        "experiments/route_forecasting",
        "models/experiments",
        "artifacts/reports",
        "artifacts/evaluation",
        "artifacts/audit",
    ]
    for d in expected_dirs:
        dir_path = ROOT / d
        assert dir_path.is_dir(), f"Directory missing: {d}"
        readme = dir_path / "README.md"
        assert readme.is_file(), f"README.md missing in {d}"


def test_experiment_config_templates_parse_as_valid_yaml():
    """Verify that YAML configuration templates for experiments are valid and contain experiment_id and seed."""
    templates = [
        "experiments/auto_labeling/config.example.yaml",
        "experiments/training/config.example.yaml",
        "experiments/route_forecasting/config.example.yaml",
    ]
    for rel_path in templates:
        file_path = ROOT / rel_path
        assert file_path.is_file(), f"Template missing: {rel_path}"
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{rel_path} must be a valid dictionary"
        assert "experiment_id" in data, f"{rel_path} must define experiment_id"
        assert "seed" in data, f"{rel_path} must define random seed for reproducibility"
        assert data["seed"] == 42


def test_gitignore_contains_required_video_and_checkpoint_rules():
    """Verify that .gitignore properly excludes raw videos, checkpoints, archives, and temporary data."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    # Required baseline rules
    assert "/models/*.onnx" in gitignore
    assert "/models/*.pkl" in gitignore
    assert "/data/*" in gitignore
    assert "!/data/cameras.json" in gitignore

    # Video and archive rules
    assert "*.mp4" in gitignore
    assert "*.zip" in gitignore
    assert "*.tar.gz" in gitignore or "*.tar" in gitignore

    # Model checkpoint rules
    assert "*.pt" in gitignore
    assert "*.pth" in gitignore
    assert "/models/experiments/*" in gitignore

    # Data structure preservation
    assert "!/data/raw/" in gitignore
    assert "!/data/staging/" in gitignore
    assert "!/data/processed/" in gitignore
    assert "!/data/annotations/" in gitignore
    assert "!/data/manifests/" in gitignore


def test_clip_level_split_prevents_data_leakage():
    """Ensure that video manifest split assignment enforces clip-level independence."""
    manifest_path = ROOT / "data/manifests/hanoi_videos_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    seen_videos = set()
    for item in manifest["videos"]:
        vid = item["video_id"]
        assert vid not in seen_videos, f"Duplicate video_id: {vid}"
        seen_videos.add(vid)
        assert item["split"] in {"train", "val", "test", "unassigned"}
