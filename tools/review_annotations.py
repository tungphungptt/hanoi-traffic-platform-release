#!/usr/bin/env python3
"""CLI utility to review, filter, and curate pseudo-labels into a verified training subset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicc.auto_labeling import filter_and_curate_pseudo_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/auto_labeling/cam_5_sample/auto_labels_manifest.json",
        help="Path to auto_labels_manifest.json to review.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/auto_labeling/cam_5_sample/verified_subset",
        help="Directory to export the curated/verified subset.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum detection confidence threshold for inclusion (e.g. 0.60).",
    )
    parser.add_argument(
        "--no-require-roi",
        action="store_true",
        help="If set, allow boxes outside the camera ROI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.manifest.is_file():
        print(f"[ERROR] Manifest file not found: {args.manifest}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("AI City Track 1 - Pseudo-Label Review & Curation Gate")
    print("=" * 60)
    print(f"Source manifest:   {args.manifest}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Min confidence:    {args.min_confidence}")
    print(f"Require inside ROI:{not args.no_require_roi}")

    curated = filter_and_curate_pseudo_labels(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        require_roi=not args.no_require_roi,
    )

    prov = curated["curation_provenance"]
    stats = curated["curated_statistics"]

    print("-" * 60)
    print(f"Status:            {prov['curated_subset_status']}")
    print(f"Original boxes:    {prov['original_pseudo_boxes']}")
    print(f"Curated boxes:     {stats['total_curated_boxes']}")
    print(f"  - Cars:          {stats['curated_cars']}")
    print(f"  - Trucks:        {stats['curated_trucks']}")
    print(f"Acceptance ratio:  {prov['acceptance_ratio'] * 100:.2f}%")
    print(f"Curated manifest:  {args.output_dir / 'verified_subset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
