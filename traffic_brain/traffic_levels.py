"""Three congestion levels on the visual CI scale (0..1)."""
from __future__ import annotations

BUSY_CI = 0.30
SEVERE_CI = 0.55
STATE_LABELS = {
    "free": "Thông thoáng",
    "busy": "Đông đúc",
    "gridlock": "Ùn tắc",
}


def level_for_ci(ci: float) -> tuple[str, str]:
    """[0, .30) free, [.30, .55) busy, [.55, 1] gridlock. Returns (key, label)."""
    key = "free" if ci < BUSY_CI else "busy" if ci < SEVERE_CI else "gridlock"
    return key, STATE_LABELS[key]
