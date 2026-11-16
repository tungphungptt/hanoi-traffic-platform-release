import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import label_frames  # noqa: E402


def test_scoring_against_people_and_between_people(tmp_path):
    frames = [{"id": "a", "ci": 0.1}, {"id": "b", "ci": 0.4}, {"id": "c", "ci": 0.7}, {"id": "d", "ci": 0.2}]
    (tmp_path / "frames.json").write_text(json.dumps(frames))
    (tmp_path / "an.json").write_text(json.dumps({"labels": {"a": "free", "b": "busy", "c": "gridlock", "d": "busy"}}))
    (tmp_path / "binh.json").write_text(json.dumps({"labels": {"a": "free", "b": "busy", "c": "gridlock", "d": "busy"}}))
    report = label_frames.score(tmp_path / "frames.json", [tmp_path / "an.json", tmp_path / "binh.json"])
    an = report["labellers"]["an"]
    assert an["labelled"] == 4 and an["agreement"] == 0.75
    assert an["confusion_human_rows_system_cols"]["busy"]["free"] == 1
    assert report["between_labellers"]["an~binh"]["kappa"] == 1.0
    assert label_frames.cohen_kappa(["free", "busy"], ["busy", "free"]) < 0
