#!/usr/bin/env python3
"""Create a deterministic SHA-256 inventory of release files (not ZIP hash)."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", "AIC21_Track1_Vehicle_Counting"}
EXCLUDED_NAMES = {".env"}
MUTABLE_RUNTIME_PATHS = {
    "models/manifest.json",
    "data/cameras.json",
    "artifacts/current/manifest.json",
    "data/camera_feeds/index.json",
    "data/manifests/hanoi_videos_manifest.json",
}
EXCLUDED_SUFFIXES = {".pyc", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v", ".mts", ".m2ts", ".npz", ".npy", ".parquet", ".arrow", ".feather", ".h5", ".hdf5", ".tfrecord", ".jsonl", ".ndjson"}

def files_to_hash():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if rel.as_posix() in MUTABLE_RUNTIME_PATHS:
            continue
        # Camera-feed indexes and operator-added data are runtime state, not release source.
        if rel.parts[:2] in {
            ("data", "camera_feeds"),
            ("data", "video_reviews"),
            ("data", "diagnostics"),
            ("data", "uploads"),
            ("data", "raw"),
            ("data", "staging"),
            ("data", "processed"),
            ("data", "annotations"),
            ("models", "experiments"),
            ("artifacts", "reports"),
            ("artifacts", "evaluation"),
            ("artifacts", "audit"),
        } and path.name not in {"README.md", ".gitkeep"}:
            continue
        if len(rel.parts) >= 3 and rel.parts[0] == "experiments" and rel.parts[2] in {"runs", "checkpoints", "outputs", "logs"}:
            continue
        if rel.parts[:2] == ("data", "video_reviews") or rel.parts[:2] == ("data", "diagnostics"):
            continue
        if rel.as_posix() == "docs/RELEASE_MANIFEST.json":
            continue  # avoid a self-referential checksum entry
        yield path, rel.as_posix()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "RELEASE_MANIFEST.json")
    args = parser.parse_args()
    rows = []
    for path, rel in files_to_hash():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": digest})
    payload = {"manifest_version": "hanoi-release-file-manifest/1.0", "project_version": (ROOT / "VERSION").read_text(encoding="utf-8-sig").strip(), "files_count": len(rows), "files": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: wrote {len(rows)} file checksums to {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
