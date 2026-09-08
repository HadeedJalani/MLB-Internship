from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from traffic_violation import RESTRICTED_ZONE, WRONG_WAY, RegionBox, ViolationState
from tracker import TrackedVehicle, track_color

STATUS_LOW_MAX = 2
STATUS_HIGH_MIN = 6
PANEL_FRACTION = 0.36


def traffic_status(total_violations: int) -> str:
    if total_violations <= STATUS_LOW_MAX:
        return "NORMAL"
    if total_violations >= STATUS_HIGH_MIN:
        return "HIGH VIOLATIONS"
    return "CAUTION"


@dataclass(frozen=True)
class HistoryPoint:
    time_s: float
    total_violations: int
    wrong_way: int
    restricted_zone: int


@dataclass
class ViolationHistory:
    points: list[HistoryPoint] = field(default_factory=list)
    max_points: int = 90

    def append(self, time_s: float, state: ViolationState) -> None:
        self.points.append(
            HistoryPoint(
                time_s=time_s,
                total_violations=state.total_violations,
                wrong_way=state.wrong_way_count,
                restricted_zone=state.restricted_zone_count,
            )
        )


def _fit_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _draw_minimal_video(
    frame: np.ndarray,
    vehicles: list[TrackedVehicle],
    state: ViolationState,
) -> None:
    radius = max(4, min(frame.shape[:2]) // 150)

    for vehicle in vehicles:
        color = (
            (45, 45, 235)
            if vehicle.track_id in state.wrong_way_ids
            else track_color(vehicle.track_id)
        )
        cv2.circle(frame, vehicle.centroid, radius + 1, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(frame, vehicle.centroid, radius, color, -1, cv2.LINE_AA)


def _draw_zone_outline(frame: np.ndarray, zone: RegionBox | None) -> None:
    if zone is None:
        return
    (x1, y1), (x2, y2) = zone.pixel_rect(frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2, cv2.LINE_AA)


def _draw_sparkline(
    panel: np.ndarray,
    points: list[HistoryPoint],
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    cv2.rectangle(panel, (x, y), (x + w, y + h), (45, 45, 45), -1)
    if len(points) < 2:
        return

    max_value = max(point.total_violations for point in points) or 1
    line_points = []
    for idx, point in enumerate(points):
        px = x + int(idx * w / max(1, len(points) - 1))
        py = y + h - int(point.total_violations * h / max_value)
        line_points.append((px, py))

    cv2.polylines(
        panel,
        [np.asarray(line_points, dtype=np.int32)],
        False,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_event_table(
    panel: np.ndarray,
    state: ViolationState,
    x: int,
    y: int,
    max_rows: int,
) -> int:
    cv2.putText(
        panel,
        "RECENT VIOLATIONS",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (165, 165, 165),
        1,
        cv2.LINE_AA,
    )
    y += 20

    events = state.events[-max_rows:][::-1]
    if not events:
        cv2.putText(
            panel,
            "No events recorded",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (165, 165, 165),
            1,
            cv2.LINE_AA,
        )
        return y + 20

    for event in events:
        if y > panel.shape[0] - 38:
            break
        kind = "WRONG-WAY" if event.violation_type == WRONG_WAY else "ZONE"
        label = _fit_text(
            f"#{event.track_id:<3} {event.class_name:<10} {kind:<9} {event.time_s:5.1f}s",
            34,
        )
        color = (65, 65, 235) if event.violation_type == WRONG_WAY else (0, 165, 255)
        cv2.putText(
            panel,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 20

    return y


def draw_analytics_frame(
    frame: np.ndarray,
    vehicles: list[TrackedVehicle],
    state: ViolationState,
    history: ViolationHistory,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Return a video + right-side analytics dashboard composition."""
    video = frame.copy()
    _draw_zone_outline(video, state.zone)
    _draw_minimal_video(video, vehicles, state)

    h, w = video.shape[:2]
    panel_w = max(300, int(w * PANEL_FRACTION))
    panel_w += panel_w % 2

    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = video

    panel = np.full((h, panel_w, 3), (24, 24, 24), dtype=np.uint8)
    pad = max(14, panel_w // 18)
    y = 24

    cv2.putText(
        panel,
        "TRAFFIC VIOLATION",
        (pad, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    y += 24
    cv2.putText(
        panel,
        "ANALYTICS DASHBOARD",
        (pad, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (155, 155, 155),
        1,
        cv2.LINE_AA,
    )

    y += 46
    cv2.putText(
        panel,
        str(state.total_vehicles),
        (pad, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (245, 245, 245),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "TOTAL VEHICLES",
        (pad, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    y += 60
    cv2.putText(
        panel,
        str(state.total_violations),
        (pad, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (245, 245, 245),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "TOTAL VIOLATIONS",
        (pad, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    y += 50
    status = traffic_status(state.total_violations)
    status_color = {
        "NORMAL": (70, 190, 80),
        "CAUTION": (0, 190, 255),
        "HIGH VIOLATIONS": (50, 50, 235),
    }[status]
    cv2.rectangle(
        panel,
        (pad, y),
        (panel_w - pad, y + 28),
        status_color,
        -1,
    )
    cv2.putText(
        panel,
        status,
        (pad + 8, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )

    y += 52
    metrics = [
        ("Wrong-way", state.wrong_way_count, (60, 60, 235)),
        ("Restricted zone", state.restricted_zone_count, (0, 165, 255)),
    ]
    for label, value, color in metrics:
        cv2.putText(
            panel,
            f"{label}: {value}",
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 24

    y += 6
    cv2.putText(
        panel,
        "VEHICLE TYPES",
        (pad, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    y += 22

    type_counts = state.vehicle_counts_by_class
    type_text = (
        f"Cars {type_counts.get('car', 0)}   "
        f"Trucks {type_counts.get('truck', 0)}"
    )
    type_text2 = (
        f"Buses {type_counts.get('bus', 0)}   "
        f"Motos {type_counts.get('motorcycle', 0)}"
    )
    for line in (type_text, type_text2):
        cv2.putText(
            panel,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 21

    y += 8
    cv2.putText(
        panel,
        "VIOLATIONS OVER TIME",
        (pad, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    y += 14
    _draw_sparkline(
        panel,
        history.points[-history.max_points:],
        pad,
        y,
        panel_w - 2 * pad,
        64,
    )

    y += 88
    _draw_event_table(panel, state, pad, y, max_rows=6)

    timestamp = frame_idx / fps if fps else 0.0
    stamp = f"FRAME {frame_idx}   |   {timestamp:05.1f}s"
    cv2.putText(
        panel,
        stamp,
        (pad, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (130, 130, 130),
        1,
        cv2.LINE_AA,
    )

    canvas[:, w:] = panel
    return canvas
