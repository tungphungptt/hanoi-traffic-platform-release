"""Shared CI presentation scale; does not alter model scores or input features."""
from __future__ import annotations

from typing import Mapping

BUSY_CI = 0.30
SEVERE_CI = 0.55
STATE_LABELS = {
    "free": "Thông thoáng",
    "busy": "Đông đúc",
    "gridlock": "Tắc nghẽn nghiêm trọng",
}
# Keep the existing manifest fields: heavy is a compatibility alias for gridlock,
# not a fourth state. Both starts now coincide at the severe boundary.
DEFAULT_THRESHOLDS = {"busy": BUSY_CI, "heavy": SEVERE_CI, "gridlock": SEVERE_CI}


def validate_thresholds(thresholds: Mapping[str, float] | None = None) -> dict[str, float]:
    """Prevent an operator manifest from reintroducing a different CI scale."""
    message = "Ngưỡng CI phải là busy=0.30, heavy=gridlock=0.55 (3 mức)"
    if thresholds is not None and not isinstance(thresholds, Mapping):
        raise ValueError(message)
    for name, expected in DEFAULT_THRESHOLDS.items():
        value = (thresholds or {}).get(name, expected)
        try:
            matches = not isinstance(value, bool) and float(value) == expected
        except (TypeError, ValueError, OverflowError):
            matches = False
        if not matches:
            raise ValueError(message)
    return dict(DEFAULT_THRESHOLDS)


def level_for_ci(ci: float) -> tuple[str, str]:
    """Classify continuous CI: [0,.30), [.30,.55), [.55,1]. Return key, label."""
    key = "free" if ci < BUSY_CI else "busy" if ci < SEVERE_CI else "gridlock"
    return key, STATE_LABELS[key]
