from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import colorsys
import math

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
MAX_SIDE = 960
TRAIL_LENGTH = 20
MIN_DIRECTION_DISPLACEMENT_PX = 6.0

# COCO classes requested by the project.
VEHICLE_CLASS_IDS: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_LOCK = Lock()


def load_model(weights: str = DEFAULT_MODEL) -> YOLO:
    """Load and cache a YOLO model."""
    with _MODEL_LOCK:
        if weights not in _MODEL_CACHE:
            _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


def reset_tracker(model: YOLO) -> None:
    """Reset Ultralytics predictor/tracker state before a new video."""
    # Ultralytics creates its predictor lazily. Clearing it ensures a new
    # uploaded/sample video cannot inherit tracker state from an earlier run.
    model.predictor = None


def resize_max_side(frame: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        out = frame
    else:
        out = cv2.resize(
            frame,
            (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    # YUV420/H.264 encoders prefer even dimensions.
    h2, w2 = out.shape[:2]
    if h2 % 2 or w2 % 2:
        out = out[: h2 - (h2 % 2), : w2 - (w2 % 2)]
    return out


def track_color(track_id: int) -> tuple[int, int, int]:
    """Stable BGR color derived only from a track ID."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class TrackedVehicle:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class MotionHistory:
    """Keeps centroid history for stable direction estimation."""

    def __init__(self, length: int = TRAIL_LENGTH):
        self.length = max(2, int(length))
        self._history: dict[int, deque[tuple[int, int]]] = {}

    def update(self, track_id: int, centroid: tuple[int, int]) -> None:
        history = self._history.setdefault(track_id, deque(maxlen=self.length))
        history.append(centroid)

    def trail(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._history.get(track_id, ()))

    def direction_vector(self, track_id: int) -> tuple[float, float] | None:
        history = self._history.get(track_id)
        if history is None or len(history) < 2:
            return None

        x0, y0 = history[0]
        x1, y1 = history[-1]
        dx, dy = x1 - x0, y1 - y0
        distance = math.hypot(dx, dy)
        if distance < MIN_DIRECTION_DISPLACEMENT_PX:
            return None
        return dx / distance, dy / distance

    def prune(self, active_ids: set[int]) -> None:
        for track_id in list(self._history):
            if track_id not in active_ids:
                del self._history[track_id]


def extract_vehicles(result) -> list[TrackedVehicle]:
    vehicles: list[TrackedVehicle] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return vehicles

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int)

    for box, conf, cls_id, track_id in zip(xyxy, confs, classes, ids):
        if int(cls_id) not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = map(int, box)
        vehicles.append(
            TrackedVehicle(
                track_id=int(track_id),
                class_id=int(cls_id),
                class_name=VEHICLE_CLASS_IDS[int(cls_id)],
                confidence=float(conf),
                box=(x1, y1, x2, y2),
            )
        )
    vehicles.sort(key=lambda item: item.track_id)
    return vehicles


def draw_trail(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if len(points) < 2:
        return
    for idx in range(1, len(points)):
        alpha = idx / max(1, len(points) - 1)
        faded = tuple(int(channel * (0.35 + 0.65 * alpha)) for channel in color)
        cv2.line(frame, points[idx - 1], points[idx], faded, max(1, thickness), cv2.LINE_AA)


def text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (15, 15, 15) if brightness > 155 else (245, 245, 245)
