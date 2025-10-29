from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_image(path: str | Path, mode: str = "color") -> np.ndarray:
    """
    Load image as float32 array in [0, 1].

    mode: "color" (RGB) or "gray".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if mode == "gray":
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load grayscale image: {path}")
        image = image.astype(np.float32) / 255.0
        return image

    if mode == "color":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load color image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return image

    raise ValueError(f"Unsupported mode '{mode}'. Use 'color' or 'gray'.")


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save float image (0-1 or 0-255) to disk using OpenCV."""
    path = Path(path)
    ensure_dir(path.parent)

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0 if array.max() <= 1.0 else 255.0)
        if array.max() <= 1.0:
            array = (array * 255.0 + 0.5).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(path), array)
    if not success:
        raise IOError(f"Failed to save image to {path}")
    return path


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB or grayscale image to single-channel grayscale."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError("Unsupported image shape for to_gray.")


def upsample_to_size(image: np.ndarray, size: Tuple[int, int], method: str = "bilinear") -> np.ndarray:
    """Upsample image to (height, width)."""
    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }.get(method)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation method '{method}'.")
    height, width = size
    return cv2.resize(image, (width, height), interpolation=interpolation)


def save_parameters(path: str | Path, params: dict) -> Path:
    """Persist parameter dictionary as JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    return path
