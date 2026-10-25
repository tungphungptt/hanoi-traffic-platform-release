"""Frame preprocessing configured by the repository's config.yaml."""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _settings() -> tuple[tuple[int, int], bool]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config: Any = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must be a YAML mapping")
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("config.yaml must define preprocessing")
    resize = preprocessing.get("resize")
    normalize = preprocessing.get("normalize")
    if not isinstance(resize, list) or len(resize) != 2 or any(type(value) is not int or value <= 0 for value in resize):
        raise ValueError("preprocessing.resize must contain two positive integers")
    if not isinstance(normalize, bool):
        raise ValueError("preprocessing.normalize must be a boolean")
    return (resize[0], resize[1]), normalize


(RESIZE, NORMALIZE) = _settings()


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        raise ValueError("frame must be a non-empty image array")
    height, width = RESIZE
    normalize = NORMALIZE
    processed = cv2.resize(frame, (width, height))
    return processed.astype(np.float32) / 255.0 if normalize else processed
