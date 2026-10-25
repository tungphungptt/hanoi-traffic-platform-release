#!/usr/bin/env python3
"""Evaluate an already-produced multi-horizon forecast CSV against ground truth.

CSV columns required:
  y_true_5, y_pred_5, y_true_10, y_pred_10, y_true_15, y_pred_15
All targets are congestion index (CI) on [0, 1]. This evaluator does not train a model
and does not treat demo/dashboard metrics as ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

HORIZONS = (5, 10, 15)


def _read_pairs(path: Path) -> dict[int, list[tuple[float, float]]]:
    pairs: dict[int, list[tuple[float, float]]] = {h: [] for h in HORIZONS}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {f"{kind}_{h}" for h in HORIZONS for kind in ("y_true", "y_pred")}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV thiếu cột: " + ", ".join(sorted(missing)))
        for row_number, row in enumerate(reader, start=2):
            for h in HORIZONS:
                try:
                    truth, pred = float(row[f"y_true_{h}"]), float(row[f"y_pred_{h}"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Dòng {row_number}, chân trời {h}: giá trị không phải số") from exc
                if not math.isfinite(truth) or not math.isfinite(pred) or not (0 <= truth <= 1 and 0 <= pred <= 1):
                    raise ValueError(f"Dòng {row_number}, chân trời {h}: CI phải hữu hạn trong [0,1]")
                pairs[h].append((truth, pred))
    if not pairs[5] or any(len(pairs[h]) != len(pairs[5]) for h in HORIZONS):
        raise ValueError("CSV không có đủ mẫu đồng bộ cho ba chân trời")
    return pairs


def _metrics(pairs: Iterable[tuple[float, float]]) -> dict[str, float | int]:
    rows = list(pairs)
    n = len(rows)
    mae = sum(abs(y - p) for y, p in rows) / n
    rmse = math.sqrt(sum((y - p) ** 2 for y, p in rows) / n)
    return {"n": n, "mae": round(mae, 6), "rmse": round(rmse, 6)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV có truth/prediction cho +5/+10/+15 phút")
    parser.add_argument("--output", type=Path, help="Tùy chọn: lưu JSON metrics vào file")
    args = parser.parse_args()
    if not args.csv_path.is_file():
        parser.error(f"Không tìm thấy file: {args.csv_path}")
    try:
        pairs = _read_pairs(args.csv_path)
    except ValueError as exc:
        parser.error(str(exc))
    report = {
        "evaluator_version": "hanoi-ci-evaluator/1.0",
        "input": args.csv_path.name,
        "warning": "Metrics chỉ có ý nghĩa nếu y_true là ground truth độc lập; không đánh giá bằng dashboard fixture hoặc label giả lập.",
        "horizons": {f"t_plus_{h}_minutes": _metrics(pairs[h]) for h in HORIZONS},
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
