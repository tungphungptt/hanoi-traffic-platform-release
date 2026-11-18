"""Excel and PDF reports from a hanoiFlow forecast session (the same JSON the page exports).

    build_xlsx(session)  -> bytes   workbook: overview, forecasts, measurements, moments, charts
    build_pdf(session)   -> bytes   A4 report: summary, CI chart, accuracy, moments, evidence frames

Both only read the session; nothing is recomputed. A session may hold several cameras
(a broadcast video); ``camera_id`` limits the report to one of them.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

FONT_DIR = Path(__file__).resolve().parent / "fonts"
BRAND = "B52226"
LEVEL_FILL = {"free": "EAF3E9", "busy": "FBF2E3", "gridlock": "F8E6E3"}
LEVEL_INK = {"free": "578263", "busy": "AD7B39", "gridlock": "A14D43"}
DEFAULT_LABELS = {"free": "Thông thoáng", "busy": "Đông đúc", "gridlock": "Tắc nghẽn nghiêm trọng"}
MOMENT_KINDS = (("hits", "Cảnh báo đúng"), ("watch_hits", "Theo dõi đúng"),
                ("misses", "Bỏ sót"), ("false_alarms", "Báo nhầm"))
MAX_ROWS = 20000


class ReportError(ValueError):
    """The session cannot be turned into a report (shown to the person as is)."""


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


def fmt_time(seconds: Any) -> str:
    s = _num(seconds)
    if s is None:
        return ""
    s = int(round(s))
    h, rem = divmod(s, 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}" if h else f"{rem // 60:02d}:{rem % 60:02d}"


def _text(value: Any, limit: int = 300) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value if value is not None else ""))[:limit]


def _created(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value)).astimezone(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    except ValueError:
        return _text(value, 40)


def prepare(session: Any, camera_id: Any = None) -> tuple[dict, list[dict]]:
    """Validate the session and return (meta, cameras with a usable forecast)."""
    if not isinstance(session, Mapping) or not str(session.get("contract", "")).startswith("hanoiflow-forecast-session/"):
        raise ReportError("Không phải kết quả hanoiFlow.")
    cameras = session.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        mf = session.get("minute_forecast")
        cameras = [{"id": 0, "label": "Camera", "intervals": None, "minute_forecast": mf}] if isinstance(mf, Mapping) else []
    chosen = []
    for cam in cameras:
        mf = cam.get("minute_forecast") if isinstance(cam, Mapping) else None
        if not isinstance(mf, Mapping) or not mf.get("available", True) or not isinstance(mf.get("anchors"), list):
            continue
        if camera_id is not None and cam.get("id") != camera_id:
            continue
        chosen.append(dict(cam))
    if not chosen:
        raise ReportError("Không có dự báo theo phút để xuất.")
    video = session.get("video") if isinstance(session.get("video"), Mapping) else {}
    meta = {"filename": _text(session.get("filename") or "video", 120),
            "duration": _num(video.get("duration_seconds")),
            "created": _created(session.get("created_at")),
            "evidence": [e for e in (session.get("evidence") or []) if isinstance(e, Mapping)][:60]}
    return meta, chosen


def _levels(mf: Mapping) -> tuple[float, float, dict]:
    th = mf.get("thresholds") if isinstance(mf.get("thresholds"), Mapping) else {}
    labels = {**DEFAULT_LABELS, **{k: _text(v, 60) for k, v in (mf.get("state_labels") or {}).items() if k in DEFAULT_LABELS}}
    return _num(th.get("busy")) or 0.30, _num(th.get("gridlock")) or 0.55, labels


def level_of(ci: Any, busy: float, gridlock: float) -> str | None:
    v = _num(ci)
    if v is None:
        return None
    return "gridlock" if v >= gridlock else "busy" if v >= busy else "free"


def _rows(items: Any) -> list[Mapping]:
    return [x for x in (items if isinstance(items, list) else [])[:MAX_ROWS] if isinstance(x, Mapping)]


def _flat_forecasts(cam: Mapping) -> list[dict]:
    out = []
    for anchor in _rows(cam["minute_forecast"].get("anchors")):
        for h in _rows(anchor.get("horizons")):
            out.append({"t": _num(anchor.get("t")), "observed_ci": _num(anchor.get("observed_ci")), **{
                k: h.get(k) for k in ("horizon_minutes", "target_t", "ci", "lower", "upper", "p_gridlock", "warning",
                                      "watch", "actual_ci", "persistence_ci", "trend_ci", "level_key", "actual_level_key")}})
    return out


# ------------------------------------------------------------------ Excel
def build_xlsx(session: Any, camera_id: Any = None) -> bytes:
    from openpyxl import Workbook
    from openpyxl.chart import LineChart, Reference
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    meta, cameras = prepare(session, camera_id)
    head_fill = PatternFill("solid", fgColor=BRAND)
    head_font = Font(bold=True, color="FFFFFF")
    wb = Workbook()

    def header(ws, row: int, titles: Sequence[str]) -> None:
        for i, title in enumerate(titles, 1):
            c = ws.cell(row=row, column=i, value=title)
            c.fill, c.font = head_fill, head_font
            c.alignment = Alignment(wrap_text=True, vertical="center")

    def widths(ws, sizes: Sequence[float]) -> None:
        for i, w in enumerate(sizes, 1):
            ws.column_dimensions[get_column_letter(i)].width = w

    # Overview
    ws = wb.active
    ws.title = "Tổng quan"
    ws["A1"], ws["A1"].font = "Báo cáo dự báo ùn tắc hanoiFlow", Font(bold=True, size=16, color=BRAND)
    info = [("Tệp video", meta["filename"]), ("Thời lượng", fmt_time(meta["duration"]) or "—"),
            ("Số camera", len(cameras)), ("Kết quả tạo lúc", meta["created"] or "—"),
            ("Xuất báo cáo lúc", datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC"))]
    for i, (k, v) in enumerate(info, 3):
        ws.cell(row=i, column=1, value=k).font = Font(bold=True)
        ws.cell(row=i, column=2, value=v)
    row = len(info) + 5
    ws.cell(row=row, column=1, value="Độ chính xác từng camera (so với giữ nguyên mức hiện tại)").font = Font(bold=True, size=12)
    row += 1
    header(ws, row, ["Camera", "Chân trời (phút)", "Số lần kiểm chứng", "Sai số mô hình (MAE)", "Sai số giữ nguyên",
                     "Tốt hơn giữ nguyên", "Bắt đúng lần tắc", "Báo nhầm"])
    for cam in cameras:
        for m in _rows(cam["minute_forecast"].get("metrics")):
            row += 1
            onset = m.get("onset_model") if isinstance(m.get("onset_model"), Mapping) else {}
            vals = [_text(cam.get("label"), 60), m.get("horizon_minutes"), m.get("evaluated"), _num(m.get("mae_model")),
                    _num(m.get("mae_persistence")), _num(m.get("skill_vs_persistence")),
                    onset.get("true_positive"), onset.get("false_positive")]
            for i, v in enumerate(vals, 1):
                ws.cell(row=row, column=i, value=v)
            ws.cell(row=row, column=4).number_format = ws.cell(row=row, column=5).number_format = "0.000"
            ws.cell(row=row, column=6).number_format = "0.0%"
    row += 2
    mf0 = cameras[0]["minute_forecast"]
    busy, gridlock, labels = _levels(mf0)
    notes = [f"CI là chỉ số ùn tắc từ 0 đến 1: dưới {busy:.2f} {labels['free'].lower()}, từ {busy:.2f} {labels['busy'].lower()}, "
             f"từ {gridlock:.2f} {labels['gridlock'].lower()}.",
             *[_text(x, 400) for x in (mf0.get("limitations") or [])[:6]]]
    ws.cell(row=row, column=1, value="Cần biết khi đọc số liệu").font = Font(bold=True, size=12)
    for note in notes:
        row += 1
        ws.cell(row=row, column=1, value=note).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.row_dimensions[row].height = 32
    widths(ws, [26, 16, 18, 20, 18, 18, 16, 12])

    # Forecasts
    ws = wb.create_sheet("Dự báo")
    header(ws, 1, ["Camera", "Thời điểm dự báo", "CI lúc dự báo", "Chân trời (phút)", "Thời điểm đích", "CI dự báo", "CI thấp",
                   "CI cao", "Xác suất tắc", "Cảnh báo", "CI thực tế", "CI giữ nguyên", "CI xu hướng", "Mức dự báo", "Mức thực tế"])
    for cam in cameras:
        b, g, lab = _levels(cam["minute_forecast"])
        for f in _flat_forecasts(cam):
            pred, actual = level_of(f["ci"], b, g), level_of(f["actual_ci"], b, g)
            ws.append([_text(cam.get("label"), 60), fmt_time(f["t"]), f["observed_ci"], f["horizon_minutes"], fmt_time(f["target_t"]),
                       _num(f["ci"]), _num(f["lower"]), _num(f["upper"]), _num(f["p_gridlock"]),
                       "Có" if f["warning"] else "Không", _num(f["actual_ci"]), _num(f["persistence_ci"]), _num(f["trend_ci"]),
                       lab.get(pred, ""), lab.get(actual, "")])
            r = ws.max_row
            for col in (3, 6, 7, 8, 11, 12, 13):
                ws.cell(row=r, column=col).number_format = "0.000"
            ws.cell(row=r, column=9).number_format = "0.0%"
            if pred:
                ws.cell(row=r, column=14).fill = PatternFill("solid", fgColor=LEVEL_FILL[pred])
            if actual:
                ws.cell(row=r, column=15).fill = PatternFill("solid", fgColor=LEVEL_FILL[actual])
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    widths(ws, [22, 14, 12, 12, 14, 11, 10, 10, 12, 10, 11, 13, 13, 22, 22])

    # Measurements and charts
    ws = wb.create_sheet("Số liệu đo")
    header(ws, 1, ["Camera", "Thời điểm", "Giây", "CI đo được (trung bình 60 giây)", "Mức"])
    spans = []
    for cam in cameras:
        b, g, lab = _levels(cam["minute_forecast"])
        start = ws.max_row + 1
        for o in _rows(cam["minute_forecast"].get("observed")):
            ci = _num(o.get("ci"))
            lv = level_of(ci, b, g)
            ws.append([_text(cam.get("label"), 60), fmt_time(o.get("t")), _num(o.get("t")), ci, lab.get(lv, "")])
            ws.cell(row=ws.max_row, column=4).number_format = "0.000"
            if lv:
                ws.cell(row=ws.max_row, column=5).fill = PatternFill("solid", fgColor=LEVEL_FILL[lv])
        spans.append((cam, start, ws.max_row))
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    widths(ws, [22, 12, 10, 30, 24])
    chart_ws = wb.create_sheet("Biểu đồ")
    top = 1
    for cam, a, z in spans:
        if z - a < 1:
            continue
        chart = LineChart()
        chart.title = f"CI theo thời gian: {_text(cam.get('label'), 60)}"
        chart.y_axis.title, chart.x_axis.title = "CI (0 đến 1)", "Thời điểm trong video"
        chart.y_axis.scaling.min, chart.y_axis.scaling.max = 0, 1
        chart.height, chart.width = 8.5, 24
        chart.add_data(Reference(ws, min_col=4, min_row=a, max_row=z), titles_from_data=False)
        chart.set_categories(Reference(ws, min_col=2, min_row=a, max_row=z))
        chart.legend = None
        chart.x_axis.delete = chart.y_axis.delete = False
        chart.x_axis.tickLblSkip = max(1, (z - a) // 12)
        s = chart.series[0]
        s.graphicalProperties.line.solidFill = "332B26"
        s.graphicalProperties.line.width = 22000
        s.smooth = False
        chart_ws.add_chart(chart, f"A{top}")
        top += 19

    # Moments
    ws = wb.create_sheet("Khoảnh khắc")
    header(ws, 1, ["Camera", "Loại", "Thời điểm dự báo", "Chân trời (phút)", "Thời điểm đích", "CI lúc dự báo", "CI dự báo", "CI thực tế", "Xác suất tắc"])
    for cam in cameras:
        hl = cam["minute_forecast"].get("highlights") if isinstance(cam["minute_forecast"].get("highlights"), Mapping) else {}
        for key, name in MOMENT_KINDS:
            for m in _rows(hl.get(key)):
                ws.append([_text(cam.get("label"), 60), name, fmt_time(m.get("t")), m.get("horizon_minutes"), fmt_time(m.get("target_t")),
                           _num(m.get("observed_ci")), _num(m.get("predicted_ci")), _num(m.get("actual_ci")), _num(m.get("p_gridlock"))])
                r = ws.max_row
                for col in (6, 7, 8):
                    ws.cell(row=r, column=col).number_format = "0.000"
                ws.cell(row=r, column=9).number_format = "0.0%"
    if ws.max_row == 1:
        ws.append(["Không có khoảnh khắc nào được ghi nhận."])
    ws.freeze_panes = "A2"
    widths(ws, [22, 16, 16, 14, 14, 13, 11, 11, 12])

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# ------------------------------------------------------------------ PDF
def _fonts() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if "DV" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")
    return "DV", "DV-B"


def _chart(cam: Mapping, width: float, height: float):
    """Vector chart: three level bands, measured CI, and the +5 minute forecast drawn at its target time."""
    from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
    from reportlab.lib import colors
    mf = cam["minute_forecast"]
    busy, gridlock, labels = _levels(mf)
    obs = [(t, c) for o in _rows(mf.get("observed")) if (t := _num(o.get("t"))) is not None and (c := _num(o.get("ci"))) is not None]
    pred = []
    for f in _flat_forecasts(cam):
        if f["horizon_minutes"] == 5 and f["target_t"] is not None and _num(f["ci"]) is not None:
            pred.append((float(f["target_t"]), float(f["ci"]), bool(f["warning"])))
    times = [t for t, _ in obs] + [t for t, _, _ in pred]
    d = Drawing(width, height)
    if not times:
        return d
    padl, padb, padt, padr = 34, 22, 8, 8
    t0, t1 = min(times), max(max(times), min(times) + 1)
    w, h = width - padl - padr, height - padb - padt
    X = lambda t: padl + (t - t0) / (t1 - t0) * w
    Y = lambda v: padb + max(0.0, min(1.0, v)) * h
    for lo, hi, fill, key in ((0, busy, "#eaf3e9", "free"), (busy, gridlock, "#fbf2e3", "busy"), (gridlock, 1, "#f8e6e3", "gridlock")):
        d.add(Rect(padl, Y(lo), w, Y(hi) - Y(lo), fillColor=colors.HexColor(fill), strokeColor=None))
        d.add(String(padl + 4, Y(hi) - 11, labels[key], fontName="DV", fontSize=7, fillColor=colors.HexColor("#" + LEVEL_INK[key])))
    for v in (0, busy, gridlock, 1):
        d.add(String(padl - 4, Y(v) - 2.5, f"{v:.2f}".rstrip("0").rstrip("."), fontName="DV", fontSize=7, textAnchor="end", fillColor=colors.HexColor("#817467")))
    ticks = 6
    for i in range(ticks + 1):
        t = t0 + (t1 - t0) * i / ticks
        d.add(Line(X(t), padb, X(t), padb - 3, strokeColor=colors.HexColor("#817467"), strokeWidth=0.5))
        d.add(String(X(t), 9, fmt_time(t), fontName="DV", fontSize=7, textAnchor="middle", fillColor=colors.HexColor("#817467")))
    if len(pred) > 1:
        pts = [c for t, v, _ in sorted(pred) for c in (X(t), Y(v))]
        d.add(PolyLine(pts, strokeColor=colors.HexColor("#597e9b"), strokeWidth=1.2, strokeDashArray=[3, 2]))
    for t, v, warn in pred:
        if warn:
            d.add(Circle(X(t), Y(v), 2.6, fillColor=colors.HexColor("#a94d43"), strokeColor=None))
    run: list[float] = []
    prev = None
    for t, c in obs:
        if prev is not None and t - prev > 45 and len(run) >= 4:
            d.add(PolyLine(run, strokeColor=colors.HexColor("#332b26"), strokeWidth=1.3))
            run = []
        run += [X(t), Y(c)]
        prev = t
    if len(run) >= 4:
        d.add(PolyLine(run, strokeColor=colors.HexColor("#332b26"), strokeWidth=1.3))
    return d


def build_pdf(session: Any, camera_id: Any = None, evidence_dir: Path | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
    from xml.sax.saxutils import escape

    meta, cameras = prepare(session, camera_id)
    base, bold = _fonts()
    red, ink, muted, line = colors.HexColor("#" + BRAND), colors.HexColor("#332b26"), colors.HexColor("#817467"), colors.HexColor("#ece2d4")
    st = {
        "title": ParagraphStyle("t", fontName=bold, fontSize=20, leading=25, textColor=red, spaceAfter=4),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=13, leading=17, textColor=ink, spaceBefore=12, spaceAfter=5),
        "h3": ParagraphStyle("h3", fontName=bold, fontSize=10.5, leading=14, textColor=ink, spaceBefore=8, spaceAfter=3),
        "p": ParagraphStyle("p", fontName=base, fontSize=9, leading=13.5, textColor=ink, alignment=TA_LEFT),
        "small": ParagraphStyle("s", fontName=base, fontSize=8, leading=11.5, textColor=muted),
        "cell": ParagraphStyle("c", fontName=base, fontSize=8.5, leading=11.5, textColor=ink),
        "cellb": ParagraphStyle("cb", fontName=bold, fontSize=8.5, leading=11.5, textColor=colors.white),
    }
    P = lambda s, k="p": Paragraph(escape(str(s)), st[k])

    def table(rows, col_widths, zebra=True):
        data = [[Paragraph(escape(str(c)), st["cellb"]) for c in rows[0]]] + [[Paragraph(escape(str(c)), st["cell"]) for c in r] for r in rows[1:]]
        t = Table(data, colWidths=col_widths, repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), red), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 1), (-1, -1), 0.4, line),
                 ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
        if zebra:
            style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fffaf0")) for i in range(2, len(data), 2)]
        t.setStyle(TableStyle(style))
        return t

    story: list = [P("Báo cáo dự báo ùn tắc", "title"), P("hanoiFlow", "small"), Spacer(1, 8)]
    mf0 = cameras[0]["minute_forecast"]
    busy, gridlock, labels = _levels(mf0)
    info = [["Tệp video", meta["filename"]], ["Thời lượng", fmt_time(meta["duration"]) or "—"],
            ["Camera trong báo cáo", ", ".join(_text(c.get("label"), 60) for c in cameras)],
            ["Kết quả tạo lúc", meta["created"] or "—"],
            ["Xuất báo cáo lúc", datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")]]
    it = Table([[P(a, "small"), P(b)] for a, b in info], colWidths=[40 * mm, 130 * mm])
    it.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story += [it, Spacer(1, 6)]
    story.append(P(f"CI là chỉ số ùn tắc từ 0 đến 1: dưới {busy:.2f} là {labels['free'].lower()}, từ {busy:.2f} là {labels['busy'].lower()}, "
                   f"từ {gridlock:.2f} là {labels['gridlock'].lower()}. Dự báo ở mỗi thời điểm chỉ dùng phần video đã qua; "
                   "phần video sau đó dùng để kiểm chứng.", "p"))

    for cam in cameras:
        mf = cam["minute_forecast"]
        story.append(PageBreak() if cam is not cameras[0] else Spacer(1, 4))
        title = _text(cam.get("label"), 80) if len(cameras) > 1 else "Diễn biến và dự báo"
        story.append(P(title, "h2"))
        if len(cameras) > 1 and _num(cam.get("on_air_seconds")) is not None:
            story.append(P(f"Camera lên sóng {fmt_time(cam['on_air_seconds'])} trong video.", "small"))
        story += [P("Đường đen là CI đo được; đường xanh là dự báo +5 phút vẽ tại thời điểm đích; chấm đỏ là lúc hệ thống phát cảnh báo tắc.", "small"),
                  Spacer(1, 3), _chart(cam, 172 * mm, 62 * mm)]
        rows = [["Chân trời", "Số lần kiểm chứng", "Sai số mô hình", "Sai số giữ nguyên", "Tốt hơn giữ nguyên", "Bắt đúng lần tắc"]]
        for m in _rows(mf.get("metrics")):
            ev, skill = m.get("evaluated"), _num(m.get("skill_vs_persistence"))
            onset = m.get("onset_model") if isinstance(m.get("onset_model"), Mapping) else {}
            tp, fn = onset.get("true_positive"), onset.get("false_negative")
            rows.append([f"+{m.get('horizon_minutes')} phút", ev if ev else "—",
                         f"{_num(m.get('mae_model')):.3f}" if ev and _num(m.get("mae_model")) is not None else "—",
                         f"{_num(m.get('mae_persistence')):.3f}" if ev and _num(m.get("mae_persistence")) is not None else "—",
                         f"{skill * 100:+.1f}%" if ev and skill is not None else "—",
                         f"{tp} trên {tp + fn}" if isinstance(tp, int) and isinstance(fn, int) and tp + fn else "—"])
        story += [P("Độ chính xác của dự báo", "h3"), table(rows, [24 * mm, 30 * mm, 28 * mm, 30 * mm, 32 * mm, 28 * mm])]
        hl = mf.get("highlights") if isinstance(mf.get("highlights"), Mapping) else {}
        moments = [["Loại", "Lúc dự báo", "Chân trời", "CI dự báo", "CI thực tế", "Xác suất tắc"]]
        for key, name in MOMENT_KINDS:
            for m in _rows(hl.get(key))[:6]:
                moments.append([name, fmt_time(m.get("t")), f"+{m.get('horizon_minutes')} phút",
                                f"{_num(m.get('predicted_ci')):.2f}" if _num(m.get("predicted_ci")) is not None else "—",
                                f"{_num(m.get('actual_ci')):.2f}" if _num(m.get("actual_ci")) is not None else "—",
                                f"{_num(m.get('p_gridlock')) * 100:.0f}%" if _num(m.get("p_gridlock")) is not None else "—"])
        story.append(P("Khoảnh khắc đáng chú ý", "h3"))
        story.append(table(moments, [34 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm, 32 * mm]) if len(moments) > 1
                     else P("Không có khoảnh khắc nào được ghi nhận.", "small"))
        frames = [e for e in meta["evidence"] if e.get("camera") == cam.get("id")][:6]
        cells = []
        for e in frames:
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/evidence/(\d{2}\.jpg)", str(e.get("url", "")))
            path = evidence_dir / m.group(1) / m.group(2) if (m and evidence_dir) else None
            if path and path.is_file():
                img = Image(str(path), width=54 * mm, height=30.4 * mm)
                cap = Paragraph(escape(f"{fmt_time(e.get('t'))}, CI {(_num(e.get('ci')) or 0):.2f}. {_text(e.get('reason_label'), 60)}"), st["small"])
                cells.append([img, cap])
        if cells:
            story.append(P("Ảnh minh chứng (người đi bộ đã được làm mờ)", "h3"))
            grid = [[Table([[c[0]], [c[1]]], colWidths=[56 * mm]) for c in cells[i:i + 3]]
                    for i in range(0, len(cells), 3)]
            gt = Table([r + [""] * (3 - len(r)) for r in grid], colWidths=[57 * mm] * 3)
            gt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            story.append(gt)

    notes = [_text(x, 500) for x in (mf0.get("limitations") or [])[:6]]
    if notes:
        story.append(KeepTogether([P("Giới hạn cần biết", "h2")] + [P("• " + n, "p") for n in notes]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(base, 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 10 * mm, "hanoiFlow: chỉ báo tham khảo từ dữ liệu hiện có")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Trang {doc.page}")
        canvas.setFillColor(red)
        canvas.rect(0, A4[1] - 5 * mm, A4[0], 5 * mm, stroke=0, fill=1)
        canvas.restoreState()

    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title="Báo cáo dự báo ùn tắc hanoiFlow", author="hanoiFlow").build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
