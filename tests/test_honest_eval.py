import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("honest_eval", Path(__file__).resolve().parents[1] / "tools" / "honest_eval.py")
he = importlib.util.module_from_spec(spec)
spec.loader.exec_module(he)


def _rows(src, h, spec_):
    return [{"src": src, "h": h, "now": 0.3, "p": p, "act": 0.7 if hit else 0.3, "pred": 0.0} for p, hit in spec_]


def test_threshold_is_chosen_without_the_scored_video():
    # Video A looks perfectly separable at p=0.5 on its own; video B would pick a different cut-off.
    a = _rows("A", 5, [(0.9, True)] * 5 + [(0.1, False)] * 5)
    b = _rows("B", 5, [(0.5, True)] * 5 + [(0.45, False)] * 5)
    train_for_a = [r for r in a + b if r["src"] != "A"]
    chosen = he._pick(train_for_a, "warning")
    assert chosen is not None and chosen <= 0.45  # B has onsets from p=0.5 up, A's 0.9/0.1 split is never consulted
    result = he.nested(a + b)["+5"]
    assert result["onsets"] == 10 and result["onset_videos"] == 2
    # Scoring A with B's threshold (~0.5) still catches its five onsets; B scored with A's threshold (~0.1..0.9) is different.
    assert result["warning"]["caught"] <= result["warning"]["onsets"]


def test_a_fold_without_any_usable_threshold_counts_its_onsets_as_missed():
    only = _rows("A", 5, [(0.9, True)] * 3 + [(0.1, False)] * 3)
    other = _rows("B", 5, [(0.2, False)] * 4)  # no onset anywhere else: no threshold can be learned for A
    result = he.nested(only + other)["+5"]
    assert result["warning"]["caught"] == 0 and result["warning"]["onsets"] == 3
    assert result["warning"]["folds_without_threshold"] >= 1
