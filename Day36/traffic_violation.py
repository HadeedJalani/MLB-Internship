from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import json
import math
import shutil
import tempfile
import time
from typing import Callable

import cv2
import imageio.v2 as imageio
import numpy as np

from tracker import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    DEFAULT_TRACKER,
    VEHICLE_CLASS_IDS,
    MotionHistory,
    TrackedVehicle,
    bgr_to_rgb,
    draw_trail,
    extract_vehicles,
    load_model,
    reset_tracker,
    resize_max_side,
    text_color_for,
    track_color,
)

WRONG_WAY = "wrong_way"
RESTRICTED_ZONE = "restricted_zone"

COMPASS: dict[str, tuple[float, float]] = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up-left": (-0.70710678, -0.70710678),
    "up-right": (0.70710678, -0.70710678),
    "down-left": (-0.70710678, 0.70710678),
    "down-right": (0.70710678, 0.70710678),
}

COMPASS_ARROWS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "up-left": "↖",
    "up-right": "↗",
    "down-left": "↙",
    "down-right": "↘",
}

DEFAULT_ANGLE_THRESHOLD_DEG = 130.0
DEFAULT_CONFIRM_WINDOW = 14
DEFAULT_CONFIRM_RATIO = 0.8


@dataclass
class RoadDirection:
    name: str = "down"
    angle_threshold_deg: float = DEFAULT_ANGLE_THRESHOLD_DEG
    confirm_window: int = DEFAULT_CONFIRM_WINDOW
    confirm_ratio: float = DEFAULT_CONFIRM_RATIO

    @property
    def vector(self) -> tuple[float, float]:
        return COMPASS[self.name]

    @property
    def arrow(self) -> str:
        return COMPASS_ARROWS[self.name]

    def angle_to(self, vector: tuple[float, float]) -> float:
        dx, dy = self.vector
        vx, vy = vector
        dot = max(-1.0, min(1.0, dx * vx + dy * vy))
        return math.degrees(math.acos(dot))


@dataclass(frozen=True)
class RegionBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def normalized(self) -> "RegionBox":
        return RegionBox(
            max(0.0, min(self.x_min, self.x_max)),
            min(1.0, max(self.x_min, self.x_max)),
            max(0.0, min(self.y_min, self.y_max)),
            min(1.0, max(self.y_min, self.y_max)),
        )

    def contains(self, point: tuple[int, int], frame_shape: tuple[int, ...]) -> bool:
        h, w = frame_shape[:2]
        x, y = point
        zone = self.normalized()
        return (
            zone.x_min * w <= x <= zone.x_max * w
            and zone.y_min * h <= y <= zone.y_max * h
        )

    def pixel_rect(self, frame_shape: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]]:
        h, w = frame_shape[:2]
        zone = self.normalized()
        return (
            int(zone.x_min * w),
            int(zone.y_min * h),
        ), (
            int(zone.x_max * w),
            int(zone.y_max * h),
        )


@dataclass(frozen=True)
class ViolationEvent:
    track_id: int
    class_name: str
    violation_type: str
    frame_idx: int
    time_s: float


@dataclass
class AnalysisResult:
    output_path: Path
    fps: float
    n_frames: int
    total_vehicles: int
    total_violations: int
    wrong_way_count: int
    restricted_zone_count: int
    vehicle_counts_by_class: dict[str, int]
    events: list[ViolationEvent]
    elapsed_s: float


class ViolationState:
    def __init__(
        self,
        direction: RoadDirection,
        zone: RegionBox | None = None,
        direction_roi: RegionBox | None = None,
    ):
        self.direction = direction
        self.zone = zone
        self.direction_roi = direction_roi

        self.vehicle_class_by_id: dict[int, str] = {}
        self.events: list[ViolationEvent] = []

        self.wrong_way_ids: set[int] = set()
        self._wrong_history: dict[int, deque[bool]] = {}
        self._zone_prev: dict[int, bool] = {}

    @property
    def total_vehicles(self) -> int:
        return len(self.vehicle_class_by_id)

    @property
    def total_violations(self) -> int:
        return len(self.events)

    @property
    def wrong_way_count(self) -> int:
        return sum(event.violation_type == WRONG_WAY for event in self.events)

    @property
    def restricted_zone_count(self) -> int:
        return sum(event.violation_type == RESTRICTED_ZONE for event in self.events)

    @property
    def vehicle_counts_by_class(self) -> dict[str, int]:
        return dict(Counter(self.vehicle_class_by_id.values()))

    def update(
        self,
        vehicles: list[TrackedVehicle],
        motion: MotionHistory,
        frame_shape: tuple[int, ...],
        frame_idx: int,
        fps: float,
    ) -> set[int]:
        just_violated: set[int] = set()

        for vehicle in vehicles:
            track_id = vehicle.track_id
            centroid = vehicle.centroid
            self.vehicle_class_by_id[track_id] = vehicle.class_name

            # Rule 1: wrong-way. Only a sufficiently sustained disagreement
            # with the configured road direction becomes a violation.
            if track_id not in self.wrong_way_ids:
                inside_direction_roi = (
                    self.direction_roi is None
                    or self.direction_roi.contains(centroid, frame_shape)
                )
                vector = motion.direction_vector(track_id) if inside_direction_roi else None
                is_wrong_now = (
                    vector is not None
                    and self.direction.angle_to(vector) >= self.direction.angle_threshold_deg
                )

                history = self._wrong_history.setdefault(
                    track_id, deque(maxlen=max(3, int(self.direction.confirm_window)))
                )
                history.append(is_wrong_now)

                if (
                    len(history) == history.maxlen
                    and sum(history) / len(history) >= self.direction.confirm_ratio
                ):
                    self.wrong_way_ids.add(track_id)
                    self.events.append(
                        ViolationEvent(
                            track_id=track_id,
                            class_name=vehicle.class_name,
                            violation_type=WRONG_WAY,
                            frame_idx=frame_idx,
                            time_s=frame_idx / fps if fps else 0.0,
                        )
                    )
                    just_violated.add(track_id)

            # Rule 2: restricted-zone entry (edge-triggered: only the
            # outside -> inside transition counts). A vehicle's very first
            # observation defaults "previously inside" to False, so a
            # vehicle that is already inside the zone the moment it's first
            # tracked counts as an entry immediately, not just later re-entries.
            if self.zone is not None:
                inside = self.zone.contains(centroid, frame_shape)
                was_inside = self._zone_prev.get(track_id, False)
                if inside and not was_inside:
                    self.events.append(
                        ViolationEvent(
                            track_id=track_id,
                            class_name=vehicle.class_name,
                            violation_type=RESTRICTED_ZONE,
                            frame_idx=frame_idx,
                            time_s=frame_idx / fps if fps else 0.0,
                        )
                    )
                    just_violated.add(track_id)
                self._zone_prev[track_id] = inside

        return just_violated


def draw_normal_direction(frame: np.ndarray, direction: RoadDirection) -> None:
    h, w = frame.shape[:2]
    cx, cy = min(72, w // 6), min(85, h // 5)
    dx, dy = direction.vector
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (170, 125), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)
    cv2.arrowedLine(
        frame,
        (cx, cy),
        (int(cx + dx * 43), int(cy + dy * 43)),
        (255, 255, 255),
        4,
        cv2.LINE_AA,
        tipLength=0.28,
    )
    cv2.putText(
        frame,
        f"NORMAL {direction.arrow}",
        (10, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_zone(frame: np.ndarray, zone: RegionBox) -> None:
    (x1, y1), (x2, y2) = zone.pixel_rect(frame.shape)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), -1)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (x1 + 5, max(y1 + 20, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )


def draw_variant1_vehicle(
    frame: np.ndarray,
    vehicle: TrackedVehicle,
    motion: MotionHistory,
    wrong_way: bool,
) -> None:
    color = (45, 45, 235) if wrong_way else track_color(vehicle.track_id)
    x1, y1, x2, y2 = vehicle.box
    thickness = 4 if wrong_way else 2

    draw_trail(frame, motion.trail(vehicle.track_id), color, thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    label = f"ID {vehicle.track_id} | {vehicle.class_name}"
    if wrong_way:
        label += " | WRONG WAY"

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    )
    label_top = max(0, y1 - th - baseline - 6)
    cv2.rectangle(
        frame,
        (x1, label_top),
        (x1 + tw + 10, y1),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(th + 2, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text_color_for(color),
        1,
        cv2.LINE_AA,
    )

    vector = motion.direction_vector(vehicle.track_id)
    if vector is not None:
        cx, cy = vehicle.centroid
        cv2.arrowedLine(
            frame,
            (cx, cy),
            (int(cx + vector[0] * 36), int(cy + vector[1] * 36)),
            color,
            3 if wrong_way else 2,
            cv2.LINE_AA,
            tipLength=0.35,
        )


def draw_variant1_badges(frame: np.ndarray, state: ViolationState, timestamp_s: float) -> None:
    x, y = 12, 135
    lines = [
        f"VEHICLES {state.total_vehicles}",
        f"VIOLATIONS {state.total_violations}",
        f"WRONG-WAY {state.wrong_way_count}",
        f"ZONE {state.restricted_zone_count}",
        f"TIME {timestamp_s:05.1f}s",
    ]

    widths = []
    for line in lines:
        (tw, th), _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        widths.append(tw)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 8, y - 24),
        (x + max(widths) + 18, y + 24 * len(lines) + 8),
        (15, 15, 15),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, dst=frame)

    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def render_variant1(
    frame: np.ndarray,
    vehicles: list[TrackedVehicle],
    motion: MotionHistory,
    state: ViolationState,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    canvas = frame.copy()
    if state.zone is not None:
        draw_zone(canvas, state.zone)
    draw_normal_direction(canvas, state.direction)

    for vehicle in vehicles:
        draw_variant1_vehicle(
            canvas,
            vehicle,
            motion,
            vehicle.track_id in state.wrong_way_ids,
        )

    draw_variant1_badges(
        canvas,
        state,
        frame_idx / fps if fps else 0.0,
    )
    return canvas


def _ensure_even_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h % 2:
        frame = frame[:-1]
    if w % 2:
        frame = frame[:, :-1]
    return frame


def _make_writer(path: Path, fps: float, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path),
        fps=float(fps or 25.0),
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


def load_config(config_path: str | Path) -> tuple[RoadDirection, RegionBox | None, RegionBox | None]:
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    direction_cfg = data.get("direction", {})
    direction = RoadDirection(
        name=direction_cfg.get("name", "down"),
        angle_threshold_deg=float(direction_cfg.get("angle_threshold_deg", DEFAULT_ANGLE_THRESHOLD_DEG)),
        confirm_window=int(direction_cfg.get("confirm_window", DEFAULT_CONFIRM_WINDOW)),
        confirm_ratio=float(direction_cfg.get("confirm_ratio", DEFAULT_CONFIRM_RATIO)),
    )

    def read_zone(key: str) -> RegionBox | None:
        value = data.get(key)
        if not value:
            return None
        return RegionBox(
            float(value["x_min"]),
            float(value["x_max"]),
            float(value["y_min"]),
            float(value["y_max"]),
        ).normalized()

    return direction, read_zone("restricted_zone"), read_zone("direction_roi")


def _process_frame(
    model,
    frame: np.ndarray,
    motion: MotionHistory,
    state: ViolationState,
    frame_idx: int,
    fps: float,
    conf: float,
    iou: float,
    variant: str,
):
    result = model.track(
        frame,
        persist=True,
        tracker=DEFAULT_TRACKER,
        conf=float(conf),
        iou=float(iou),
        classes=list(VEHICLE_CLASS_IDS),
        verbose=False,
    )[0]

    vehicles = extract_vehicles(result)
    active_ids = {vehicle.track_id for vehicle in vehicles}
    for vehicle in vehicles:
        motion.update(vehicle.track_id, vehicle.centroid)
    state.update(vehicles, motion, frame.shape, frame_idx, fps)
    motion.prune(active_ids)

    if variant == "wrong_way":
        annotated = render_variant1(frame, vehicles, motion, state, frame_idx, fps)
    else:
        from analytics import ViolationHistory, draw_analytics_frame
        # This function is only called by process_video, which creates the
        # analytics history and passes it via the closure.
        annotated = None
    return vehicles, annotated


def process_video(
    model,
    input_path: str | Path,
    output_path: str | Path,
    direction: RoadDirection,
    restricted_zone: RegionBox | None,
    direction_roi: RegionBox | None,
    *,
    variant: str = "wrong_way",
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    max_frames: int | None = 300,
    progress_cb: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    """Run YOLO + ByteTrack + traffic rules and create one processed MP4."""

    if variant not in {"wrong_way", "analytics"}:
        raise ValueError("variant must be 'wrong_way' or 'analytics'")

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_for_progress = min(source_frames, max_frames) if max_frames else source_frames
    if total_for_progress <= 0:
        total_for_progress = 1

    state = ViolationState(direction, restricted_zone, direction_roi)
    motion = MotionHistory()
    start = time.perf_counter()
    frame_idx = 0

    reset_tracker(model)

    writer = None
    analytics_history = None
    if variant == "analytics":
        from analytics import ViolationHistory
        analytics_history = ViolationHistory()

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_max_side(frame)
            vehicles = []
            result = model.track(
                frame,
                persist=True,
                tracker=DEFAULT_TRACKER,
                conf=float(conf),
                iou=float(iou),
                classes=list(VEHICLE_CLASS_IDS),
                verbose=False,
            )[0]

            vehicles = extract_vehicles(result)
            active_ids = {vehicle.track_id for vehicle in vehicles}
            for vehicle in vehicles:
                motion.update(vehicle.track_id, vehicle.centroid)

            state.update(
                vehicles,
                motion,
                frame.shape,
                frame_idx,
                source_fps,
            )
            motion.prune(active_ids)

            if variant == "wrong_way":
                annotated = render_variant1(
                    frame,
                    vehicles,
                    motion,
                    state,
                    frame_idx,
                    source_fps,
                )
            else:
                analytics_history.append(
                    frame_idx / source_fps if source_fps else 0.0,
                    state,
                )
                from analytics import draw_analytics_frame
                annotated = draw_analytics_frame(
                    frame,
                    vehicles,
                    state,
                    analytics_history,
                    frame_idx,
                    source_fps,
                )

            annotated = _ensure_even_frame(annotated)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = _make_writer(output_path, source_fps, w, h)

            writer.append_data(bgr_to_rgb(annotated))
            frame_idx += 1

            if progress_cb:
                progress_cb(frame_idx, total_for_progress)

    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if frame_idx == 0:
        raise RuntimeError("No decodable frames were found in the input video.")

    elapsed = time.perf_counter() - start
    return AnalysisResult(
        output_path=output_path,
        fps=source_fps,
        n_frames=frame_idx,
        total_vehicles=state.total_vehicles,
        total_violations=state.total_violations,
        wrong_way_count=state.wrong_way_count,
        restricted_zone_count=state.restricted_zone_count,
        vehicle_counts_by_class=state.vehicle_counts_by_class,
        events=list(state.events),
        elapsed_s=elapsed,
    )


def default_demo_config() -> tuple[RoadDirection, RegionBox, RegionBox]:
    return (
        RoadDirection(name="down"),
        RegionBox(0.72, 0.96, 0.42, 0.92),
        RegionBox(0.05, 0.95, 0.10, 0.95),
    )
