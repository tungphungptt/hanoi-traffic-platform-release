import io
import json
from pathlib import Path

import pytest

from traffic_brain import report

DEMO = Path(__file__).resolve().parents[1] / "artifacts" / "demo_hanoi_tdh_htm.hanoiflow.json"


@pytest.fixture(scope="module")
def session():
    return json.loads(DEMO.read_text(encoding="utf-8"))


def test_xlsx_has_the_sheets_rows_and_charts(session):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(io.BytesIO(report.build_xlsx(session)))
    assert wb.sheetnames == ["Tổng quan", "Dự báo", "Số liệu đo", "Biểu đồ", "Khoảnh khắc"]
    forecasts = wb["Dự báo"]
    expected = sum(len(a["horizons"]) for c in session["cameras"] for a in c["minute_forecast"]["anchors"])
    assert forecasts.max_row == expected + 1
    assert forecasts["A1"].value == "Camera" and forecasts["N2"].value in ("Thông thoáng", "Đông đúc", "Tắc nghẽn nghiêm trọng", "")
    assert len(wb["Biểu đồ"]._charts) == len(session["cameras"])


def test_xlsx_can_be_limited_to_one_camera(session):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(io.BytesIO(report.build_xlsx(session, camera_id=1)))
    labels = {row[0].value for row in wb["Dự báo"].iter_rows(min_row=2)}
    assert labels == {session["cameras"][1]["label"]}


def test_pdf_is_a_real_pdf_with_the_vietnamese_font(session):
    pytest.importorskip("reportlab")
    data = report.build_pdf(session)
    assert data.startswith(b"%PDF") and b"DejaVuSans" in data


def test_bad_sessions_are_refused_with_a_message():
    for bad in (None, {}, {"contract": "x"}, {"contract": "hanoiflow-forecast-session/1.1", "cameras": []}):
        with pytest.raises(report.ReportError):
            report.prepare(bad)
    with pytest.raises(report.ReportError):
        report.prepare(json.loads(DEMO.read_text(encoding="utf-8")), camera_id=99)


def test_nan_and_text_values_do_not_break_the_report(session):
    messy = json.loads(json.dumps(session))
    mf = messy["cameras"][0]["minute_forecast"]
    mf["anchors"][0]["horizons"][0]["ci"] = float("nan")
    mf["anchors"][0]["observed_ci"] = "n/a"
    messy["filename"] = "tên \x00 lạ.mp4"
    assert report.build_pdf(messy).startswith(b"%PDF")
    assert report.build_xlsx(messy)[:2] == b"PK"


def test_time_format():
    assert report.fmt_time(75) == "01:15" and report.fmt_time(3725) == "1:02:05" and report.fmt_time(None) == ""
