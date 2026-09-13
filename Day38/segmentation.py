from __future__ import annotations

import cv2
import numpy as np

METHODS = ["Binary", "Adaptive", "Otsu"]


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected grayscale/BGR/BGRA image")


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def binary_threshold(image, threshold: int = 127) -> np.ndarray:
    gray = _gray(image)
    _, result = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    return result


def adaptive_threshold(
    image,
    block_size: int = 11,
    c_value: int = 2,
) -> np.ndarray:
    gray = _gray(image)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        _odd(block_size),
        int(c_value),
    )


def otsu_threshold(image) -> np.ndarray:
    gray = _gray(image)
    _, result = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return result


def segment_image(
    image,
    method: str,
    *,
    threshold: int = 127,
    block_size: int = 11,
    c_value: int = 2,
) -> np.ndarray:
    method = method.strip().title()
    if method == "Binary":
        return binary_threshold(image, threshold)
    if method == "Adaptive":
        return adaptive_threshold(image, block_size, c_value)
    if method == "Otsu":
        return otsu_threshold(image)
    raise ValueError(f"Unknown method: {method}")
