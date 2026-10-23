#!/usr/bin/env python3
"""Generates an event-by-event reconciliation table between Ground Truth and Predictions."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]


def build_reconciliation(gt_csv: Path, pred_csv: Path, tol_frames: int = 15) -> Dict[str, Any]:
    with open(gt_csv, "r", encoding="utf-8-sig") as f:
        gt_rows = list(csv.DictReader(f))
    with open(pred_csv, "r", encoding="utf-8-sig") as f:
        pred_rows = list(csv.DictReader(f))

    # Parse typed records
    gt_list = []
    for idx, r in enumerate(gt_rows):
        gt_list.append({
            "gt_id": idx + 1,
            "video_clip_id": int(r["video_clip_id"]),
            "frame_id": int(r["frame_id"]),
            "movement_id": int(r["movement_id"]),
            "vehicle_class_id": int(r["vehicle_class_id"]),
            "matched_pred_id": None,
            "error_type": "FALSE_NEGATIVE",
            "frame_diff": None,
            "pred_movement": None,
            "pred_class": None,
        })

    pred_list = []
    for idx, r in enumerate(pred_rows):
        pred_list.append({
            "pred_id": idx + 1,
            "video_clip_id": int(r["video_clip_id"]),
            "frame_id": int(r["frame_id"]),
            "movement_id": int(r["movement_id"]),
            "vehicle_class_id": int(r["vehicle_class_id"]),
            "matched_gt_id": None,
            "assigned_status": "FALSE_POSITIVE",
        })

    # Pass 1: Optimal Bipartite Match for Exact Matches (same movement & class, frame tol <= 15)
    cost_exact = np.full((len(gt_list), len(pred_list)), 1e6, dtype=np.float32)
    for g_idx, g in enumerate(gt_list):
        for p_idx, p in enumerate(pred_list):
            if p["movement_id"] == g["movement_id"] and p["vehicle_class_id"] == g["vehicle_class_id"]:
                diff = abs(p["frame_id"] - g["frame_id"])
                if diff <= tol_frames:
                    cost_exact[g_idx, p_idx] = diff

    row_ind, col_ind = linear_sum_assignment(cost_exact)
    for r, c in zip(row_ind, col_ind):
        if cost_exact[r, c] <= tol_frames:
            g = gt_list[r]
            p = pred_list[c]
            g["matched_pred_id"] = p["pred_id"]
            g["error_type"] = "EXACT_MATCH"
            g["frame_diff"] = p["frame_id"] - g["frame_id"]
            g["pred_movement"] = p["movement_id"]
            g["pred_class"] = p["vehicle_class_id"]
            p["matched_gt_id"] = g["gt_id"]
            p["assigned_status"] = "EXACT_MATCH"

    # Pass 2: Optimal Bipartite Match for Wrong Movement (same class, different movement, frame tol <= 15)
    unmatched_gt_indices = [idx for idx, g in enumerate(gt_list) if g["matched_pred_id"] is None]
    unmatched_pred_indices = [idx for idx, p in enumerate(pred_list) if p["matched_gt_id"] is None]

    if unmatched_gt_indices and unmatched_pred_indices:
        cost_wm = np.full((len(unmatched_gt_indices), len(unmatched_pred_indices)), 1e6, dtype=np.float32)
        for i, g_idx in enumerate(unmatched_gt_indices):
            g = gt_list[g_idx]
            for j, p_idx in enumerate(unmatched_pred_indices):
                p = pred_list[p_idx]
                if p["vehicle_class_id"] == g["vehicle_class_id"] and p["movement_id"] != g["movement_id"]:
                    diff = abs(p["frame_id"] - g["frame_id"])
                    if diff <= tol_frames:
                        cost_wm[i, j] = diff

        row_wm, col_wm = linear_sum_assignment(cost_wm)
        for r, c in zip(row_wm, col_wm):
            if cost_wm[r, c] <= tol_frames:
                g = gt_list[unmatched_gt_indices[r]]
                p = pred_list[unmatched_pred_indices[c]]
                g["matched_pred_id"] = p["pred_id"]
                g["error_type"] = "WRONG_MOVEMENT"
                g["frame_diff"] = p["frame_id"] - g["frame_id"]
                g["pred_movement"] = p["movement_id"]
                g["pred_class"] = p["vehicle_class_id"]
                p["matched_gt_id"] = g["gt_id"]
                p["assigned_status"] = "WRONG_MOVEMENT"

    # Pass 3: Match Wrong Class on remaining (same movement, different class, frame tol <= 15)
    unmatched_gt_indices = [idx for idx, g in enumerate(gt_list) if g["matched_pred_id"] is None]
    unmatched_pred_indices = [idx for idx, p in enumerate(pred_list) if p["matched_gt_id"] is None]

    if unmatched_gt_indices and unmatched_pred_indices:
        cost_wc = np.full((len(unmatched_gt_indices), len(unmatched_pred_indices)), 1e6, dtype=np.float32)
        for i, g_idx in enumerate(unmatched_gt_indices):
            g = gt_list[g_idx]
            for j, p_idx in enumerate(unmatched_pred_indices):
                p = pred_list[p_idx]
                if p["movement_id"] == g["movement_id"] and p["vehicle_class_id"] != g["vehicle_class_id"]:
                    diff = abs(p["frame_id"] - g["frame_id"])
                    if diff <= tol_frames:
                        cost_wc[i, j] = diff

        row_wc, col_wc = linear_sum_assignment(cost_wc)
        for r, c in zip(row_wc, col_wc):
            if cost_wc[r, c] <= tol_frames:
                g = gt_list[unmatched_gt_indices[r]]
                p = pred_list[unmatched_pred_indices[c]]
                g["matched_pred_id"] = p["pred_id"]
                g["error_type"] = "WRONG_CLASS"
                g["frame_diff"] = p["frame_id"] - g["frame_id"]
                g["pred_movement"] = p["movement_id"]
                g["pred_class"] = p["vehicle_class_id"]
                p["matched_gt_id"] = g["gt_id"]
                p["assigned_status"] = "WRONG_CLASS"

    # Pass 4: Detect duplicate counts among remaining predictions (occurring within 4 frames of another prediction of same movement)
    for idx, p in enumerate(pred_list):
        if p["assigned_status"] == "FALSE_POSITIVE":
            for other_idx, other_p in enumerate(pred_list):
                if idx != other_idx and abs(p["frame_id"] - other_p["frame_id"]) <= 4 and p["movement_id"] == other_p["movement_id"]:
                    p["assigned_status"] = "DUPLICATE_COUNT"
                    break

    # Summary statistics with full mathematical invariants
    exact_count = sum(1 for g in gt_list if g["error_type"] == "EXACT_MATCH")
    wm_count = sum(1 for g in gt_list if g["error_type"] == "WRONG_MOVEMENT")
    wc_count = sum(1 for g in gt_list if g["error_type"] == "WRONG_CLASS")
    fn_count = sum(1 for g in gt_list if g["error_type"] == "FALSE_NEGATIVE")
    dup_count = sum(1 for p in pred_list if p["assigned_status"] == "DUPLICATE_COUNT")
    spurious_fp_count = sum(1 for p in pred_list if p["assigned_status"] == "FALSE_POSITIVE")

    # Invariants verification
    assert exact_count + wm_count + wc_count + fn_count == len(gt_list), "GT reconciliation invariant failed"
    assert exact_count + wm_count + wc_count + dup_count + spurious_fp_count == len(pred_list), "Pred reconciliation invariant failed"

    stats = {
        "total_gt": len(gt_list),
        "total_pred": len(pred_list),
        "exact_match_tp": exact_count,
        "wrong_movement": wm_count,
        "wrong_class": wc_count,
        "false_negative_missed": fn_count,
        "duplicate_counts": dup_count,
        "false_positive_spurious": spurious_fp_count,
        "total_temporal_movement_fp": len(pred_list) - exact_count,
        "total_temporal_movement_fn": len(gt_list) - exact_count,
    }
    return {
        "stats": stats,
        "gt_reconciliation": gt_list,
        "pred_reconciliation": pred_list,
    }


def export_markdown_reconciliation(data: Dict[str, Any], output_path: Path) -> None:
    stats = data["stats"]
    gt_list = data["gt_reconciliation"]
    pred_list = data["pred_reconciliation"]

    lines = [
        "# Bảng Đối Chiếu Sự Kiện Ground Truth & Dự Đoán (Event-Level Reconciliation Table)",
        "",
        "## 1. Tổng Kết Phân Loại Lỗi & Bất Biến Số Học (Reconciliation Invariants)",
        f"- **Tổng số Ground Truth**: {stats['total_gt']}",
        f"- **Tổng số Dự đoán**: {stats['total_pred']}",
        f"- **Khớp chính xác (Exact Match / TP)**: {stats['exact_match_tp']} ({stats['exact_match_tp']/stats['total_gt']*100:.1f}%)",
        f"- **Sai hướng rẽ (Wrong Movement)**: {stats['wrong_movement']} ({stats['wrong_movement']/stats['total_gt']*100:.1f}%)",
        f"- **Sai loại xe (Wrong Class)**: {stats['wrong_class']} ({stats['wrong_class']/stats['total_gt']*100:.1f}%)",
        f"- **Bỏ sót hoàn toàn (Pure False Negative)**: {stats['false_negative_missed']} ({stats['false_negative_missed']/stats['total_gt']*100:.1f}%)",
        f"- **Dự đoán thừa ảo (Spurious False Positive)**: {stats['false_positive_spurious']}",
        f"- **Đếm lặp/trùng (Duplicate Count)**: {stats['duplicate_counts']}",
        "",
        "### Phương trình khớp số học (Exact Invariants):",
        f"- `Total_GT = {stats['total_gt']}` = `TP_Exact ({stats['exact_match_tp']})` + `Wrong_Movement ({stats['wrong_movement']})` + `Wrong_Class ({stats['wrong_class']})` + `Pure_FN ({stats['false_negative_missed']})`",
        f"- `Total_Pred = {stats['total_pred']}` = `TP_Exact ({stats['exact_match_tp']})` + `Wrong_Movement ({stats['wrong_movement']})` + `Wrong_Class ({stats['wrong_class']})` + `Duplicate_Pred ({stats['duplicate_counts']})` + `Spurious_FP ({stats['false_positive_spurious']})`",
        f"- `Temporal_Movement_FN = {stats['total_temporal_movement_fn']}` = `Wrong_Movement ({stats['wrong_movement']})` + `Wrong_Class ({stats['wrong_class']})` + `Pure_FN ({stats['false_negative_missed']})`",
        f"- `Temporal_Movement_FP = {stats['total_temporal_movement_fp']}` = `Wrong_Movement ({stats['wrong_movement']})` + `Duplicate_Pred ({stats['duplicate_counts']})` + `Spurious_FP ({stats['false_positive_spurious']})`",
        "",
        "## 2. Bảng Đối Chiếu 96 Sự Kiện Ground Truth",
        "| GT ID | Frame GT | Time GT (s) | Mov GT | Cls GT | Pred ID | Frame Pred | Mov Pred | Cls Pred | Sai lệch Frame | Phân loại Lỗi |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for g in gt_list:
        f_diff_str = f"{g['frame_diff']:+d}" if g['frame_diff'] is not None else "-"
        p_id_str = str(g['matched_pred_id']) if g['matched_pred_id'] else "-"
        p_mov_str = str(g['pred_movement']) if g['pred_movement'] else "-"
        p_cls_str = "Car" if g['pred_class'] == 1 else ("Truck" if g['pred_class'] == 2 else "-")
        g_cls_str = "Car" if g['vehicle_class_id'] == 1 else "Truck"
        time_s = f"{g['frame_id'] / 10.0:.1f}"

        # Frame pred
        pred_obj = next((p for p in pred_list if p['pred_id'] == g['matched_pred_id']), None)
        f_pred_str = str(pred_obj['frame_id']) if pred_obj else "-"

        lines.append(f"| {g['gt_id']} | {g['frame_id']} | {time_s} | {g['movement_id']} | {g_cls_str} | {p_id_str} | {f_pred_str} | {p_mov_str} | {p_cls_str} | {f_diff_str} | {g['error_type']} |")

    lines.append("")
    lines.append("## 3. Danh Sách Các Dự Đoán Thừa / Đếm Lặp (Unmatched Predictions)")
    lines.append("| Pred ID | Frame Pred | Time (s) | Mov Pred | Cls Pred | Trạng thái |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :--- |")
    for p in pred_list:
        if p["assigned_status"] in {"FALSE_POSITIVE", "DUPLICATE_COUNT"}:
            cls_str = "Car" if p['vehicle_class_id'] == 1 else "Truck"
            lines.append(f"| {p['pred_id']} | {p['frame_id']} | {p['frame_id']/10.0:.1f} | {p['movement_id']} | {cls_str} | {p['assigned_status']} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported reconciliation Markdown table to: {output_path}")


if __name__ == "__main__":
    gt = ROOT / "AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv"
    pred = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/training/track1_vehicle_counting/output_counting_cam5_1min.csv"
    out_md = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "experiments/training/track1_vehicle_counting/RECONCILIATION_TABLE.md"
    res = build_reconciliation(gt, pred)
    export_markdown_reconciliation(res, out_md)
