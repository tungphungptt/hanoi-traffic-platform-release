#!/usr/bin/env python3
"""Check the visual congestion index (CI) against people's judgement.

1. ``sample``: pick frames from videos (spread over time and over CI levels), save them with the
   CI the system measured, and write one self-contained HTML page for labelling. Open the page in
   any browser, label each frame Thông thoáng / Đông đúc / Tắc, then save the labels (.json).
   The page never shows the system's CI, so labels are independent.
2. ``score``: compare one or more label files with the measured CI: agreement per level,
   Cohen's kappa, agreement between labellers.

    python tools/label_frames.py sample video1.mp4 video2.mp4 --per-video 60 --out labels/
    python tools/label_frames.py score labels/frames.json labels/labels_an.json labels/labels_binh.json
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from traffic_brain.detector import Detector  # noqa: E402
from traffic_brain.traffic_levels import level_for_ci  # noqa: E402
from traffic_brain.video_analysis import analyze_video  # noqa: E402

LEVELS = ["free", "busy", "gridlock"]
LEVEL_VI = {"free": "Thông thoáng", "busy": "Đông đúc", "gridlock": "Tắc nghẽn"}


def sample(videos: list[Path], per_video: int, out: Path, seed: int) -> None:
    rng = random.Random(seed)
    detector = Detector()
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for video in videos:
        analysis = analyze_video(video, detector, max_duration_seconds=4 * 3600)
        rows = [r for r in analysis["timeline"] if r["scene"] >= 0 or analysis["mode"] == "single_camera"]
        # Stratify by measured level so congested frames are not drowned by free-flow ones.
        by_level = {k: [r for r in rows if level_for_ci(r["raw_ci"])[0] == k] for k in LEVELS}
        chosen = []
        for k in LEVELS:
            pool = by_level[k]
            chosen += rng.sample(pool, min(len(pool), per_video // 3))
        cap = cv2.VideoCapture(str(video))
        for row in chosen:
            cap.set(cv2.CAP_PROP_POS_MSEC, row["timestamp_seconds"] * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames.append({"id": f"{video.stem}@{row['timestamp_seconds']:.0f}", "video": video.name,
                           "t": row["timestamp_seconds"], "ci": round(row["raw_ci"], 4),
                           "camera": row.get("scene"), "jpg": base64.b64encode(jpg.tobytes()).decode()})
        cap.release()
        print(f"{video.name}: {len(chosen)} frames")
    rng.shuffle(frames)
    (out / "frames.json").write_text(json.dumps([{k: v for k, v in f.items() if k != "jpg"} for f in frames],
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    page = PAGE.replace("__FRAMES__", json.dumps([{"id": f["id"], "jpg": f["jpg"]} for f in frames]))
    (out / "label.html").write_text(page, encoding="utf-8")
    print(f"{len(frames)} frames -> {out / 'label.html'} (open in a browser) and {out / 'frames.json'}")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(k) / n) * (b.count(k) / n) for k in LEVELS)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def score(frames_file: Path, label_files: list[Path]) -> dict:
    frames = {f["id"]: f for f in json.loads(frames_file.read_text(encoding="utf-8"))}
    labellers = {p.stem: json.loads(p.read_text(encoding="utf-8")).get("labels", {}) for p in label_files}
    report: dict = {"frames": len(frames), "labellers": {}}
    for name, labels in labellers.items():
        ids = [i for i in labels if i in frames and labels[i] in LEVELS]
        human = [labels[i] for i in ids]
        system = [level_for_ci(frames[i]["ci"])[0] for i in ids]
        matrix = {h: {s: sum(1 for x, y in zip(human, system) if x == h and y == s) for s in LEVELS} for h in LEVELS}
        ci_by_level = {h: round(float(np.mean([frames[i]["ci"] for i in ids if labels[i] == h])), 3)
                       for h in LEVELS if any(labels[i] == h for i in ids)}
        report["labellers"][name] = {
            "labelled": len(ids),
            "agreement": round(sum(x == y for x, y in zip(human, system)) / len(ids), 3) if ids else None,
            "kappa": round(cohen_kappa(human, system), 3) if ids else None,
            "confusion_human_rows_system_cols": matrix,
            "mean_ci_by_human_label": ci_by_level,
        }
    names = list(labellers)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = [k for k in labellers[a] if k in labellers[b] and labellers[a][k] in LEVELS and labellers[b][k] in LEVELS]
            pairs[f"{a}~{b}"] = {"frames": len(ids), "kappa": round(cohen_kappa([labellers[a][k] for k in ids],
                                                                                 [labellers[b][k] for k in ids]), 3)}
    report["between_labellers"] = pairs
    return report


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Gán nhãn ùn tắc</title>
<style>body{font:15px system-ui,sans-serif;margin:0;background:#faf7f0;color:#332b26}main{max-width:1000px;margin:auto;padding:20px}
img{width:100%;border-radius:10px;display:block}.bar{display:flex;gap:10px;margin:14px 0;flex-wrap:wrap}
button{font:inherit;padding:12px 18px;border-radius:10px;border:1px solid #ded0bd;background:#fff;cursor:pointer}
.free{border-color:#78a984}.busy{border-color:#d3a052}.gridlock{border-color:#a94d43}button.on{background:#b52226;color:#fff}
h1{color:#b52226;font-size:22px}small{color:#817467}</style></head><body><main>
<h1>Gán nhãn mức ùn tắc</h1><p><small>Nhìn ảnh và chọn mức bạn thấy (phím 1 / 2 / 3, ← → để chuyển ảnh). Nhãn lưu trong trình duyệt; bấm "Lưu nhãn" để tải file .json.</small></p>
<p>Người gán: <input id="who" placeholder="tên bạn"> · <b id="pos"></b> · đã gán <b id="done"></b></p>
<img id="img" alt="khung hình">
<div class="bar"><button class="free" data-l="free">1 · Thông thoáng</button><button class="busy" data-l="busy">2 · Đông đúc</button>
<button class="gridlock" data-l="gridlock">3 · Tắc nghẽn</button><button id="prev">←</button><button id="next">→</button><button id="save">Lưu nhãn</button></div>
</main><script>
const F=__FRAMES__; const KEY='hanoiflow-labels'; let L=JSON.parse(localStorage.getItem(KEY)||'{}'); let i=0;
const $=id=>document.getElementById(id);
function show(){ $('img').src='data:image/jpeg;base64,'+F[i].jpg; $('pos').textContent=(i+1)+' / '+F.length;
 $('done').textContent=Object.keys(L).length; document.querySelectorAll('[data-l]').forEach(b=>b.classList.toggle('on',L[F[i].id]===b.dataset.l)); }
function set(l){ L[F[i].id]=l; localStorage.setItem(KEY,JSON.stringify(L)); if(i<F.length-1)i++; show(); }
document.querySelectorAll('[data-l]').forEach(b=>b.onclick=()=>set(b.dataset.l));
$('prev').onclick=()=>{if(i>0)i--;show()}; $('next').onclick=()=>{if(i<F.length-1)i++;show()};
document.onkeydown=e=>{ if(e.target.tagName==='INPUT')return; if(e.key==='1')set('free'); if(e.key==='2')set('busy'); if(e.key==='3')set('gridlock');
 if(e.key==='ArrowLeft')$('prev').click(); if(e.key==='ArrowRight')$('next').click(); };
$('save').onclick=()=>{ const who=($('who').value||'nguoi_gan').replace(/[^\\w-]+/g,'_');
 const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify({labeller:who,labels:L})],{type:'application/json'}));
 a.download='labels_'+who+'.json'; a.click(); };
show();
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("videos", nargs="+", type=Path)
    s.add_argument("--per-video", type=int, default=60)
    s.add_argument("--out", type=Path, default=ROOT / "labels")
    s.add_argument("--seed", type=int, default=13)
    c = sub.add_parser("score")
    c.add_argument("frames", type=Path)
    c.add_argument("labels", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "sample":
        sample(args.videos, args.per_video, args.out, args.seed)
    else:
        print(json.dumps(score(args.frames, args.labels), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
