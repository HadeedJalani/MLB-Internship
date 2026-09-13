from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import colorsys
import time

import cv2
import imageio.v2 as imageio
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
MAX_SIDE = 1280
TRAIL_LENGTH = 18
PERSON_CLASS_ID = 0

_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_LOCK = Lock()


@dataclass(frozen=True)
class PersonTrack:
    track_id: int
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass(frozen=True)
class CountingLine:
    orientation: str = "horizontal"
    position: float = 0.5

    def __post_init__(self):
        if self.orientation not in {"horizontal", "vertical"}:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        if not 0.0 <= self.position <= 1.0:
            raise ValueError("position must be between 0 and 1")

    def coordinate(self, shape: tuple[int, ...]) -> int:
        h, w = shape[:2]
        return int(h * self.position) if self.orientation == "horizontal" else int(w * self.position)

    def side(self, point: tuple[int, int], shape: tuple[int, ...]) -> int:
        x, y = point
        coord = self.coordinate(shape)
        value = y if self.orientation == "horizontal" else x
        if value < coord:
            return -1
        if value > coord:
            return 1
        return 0


@dataclass(frozen=True)
class ROI:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def normalized(self) -> "ROI":
        return ROI(
            max(0.0, min(self.x_min, self.x_max)),
            min(1.0, max(self.x_min, self.x_max)),
            max(0.0, min(self.y_min, self.y_max)),
            min(1.0, max(self.y_min, self.y_max)),
        )

    def contains(self, point: tuple[int, int], shape: tuple[int, ...]) -> bool:
        h, w = shape[:2]
        x, y = point
        roi = self.normalized()
        return (
            roi.x_min * w <= x <= roi.x_max * w
            and roi.y_min * h <= y <= roi.y_max * h
        )


@dataclass
class PeopleCountResult:
    out_path: Path
    n_frames: int
    fps: float
    elapsed_s: float
    people_per_frame: list[int]
    peak_count: int
    current_count: int
    total_people_seen: int
    line_crossings: int
    entries: int
    exits: int
    roi_count: int


class CountingState:
    """Persistent state for one video."""

    def __init__(self, line: CountingLine | None = None, roi: ROI | None = None):
        self.line = line
        self.roi = roi
        self.previous_side: dict[int, int] = {}
        self.seen_ids: set[int] = set()
        self.current_count = 0
        self.peak_count = 0
        self.roi_count = 0
        self.line_crossings = 0
        self.entries = 0
        self.exits = 0

    @property
    def total_people_seen(self) -> int:
        return len(self.seen_ids)

    def update(self, people: list[PersonTrack], shape: tuple[int, ...]) -> None:
        active_ids = {p.track_id for p in people}
        self.seen_ids.update(active_ids)
        self.current_count = len(active_ids)
        self.peak_count = max(self.peak_count, self.current_count)

        if self.roi is not None:
            self.roi_count = sum(
                self.roi.contains(p.centroid, shape) for p in people
            )
        else:
            self.roi_count = 0

        if self.line is None:
            return

        for person in people:
            current = self.line.side(person.centroid, shape)
            if current == 0:
                continue

            previous = self.previous_side.get(person.track_id)
            if previous is not None and previous != 0 and previous != current:
                self.line_crossings += 1
                if previous == -1 and current == 1:
                    self.entries += 1
                elif previous == 1 and current == -1:
                    self.exits += 1

            self.previous_side[person.track_id] = current


def load_model(model_name: str = DEFAULT_MODEL) -> YOLO:
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = YOLO(model_name)
    return _MODEL_CACHE[model_name]


def reset_tracker(model: YOLO) -> None:
    model.predictor = None


def resize_max_side(frame: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    h, w = frame.shape[:2]
    if h % 2:
        frame = frame[:-1]
    if w % 2:
        frame = frame[:, :-1]
    return frame


def track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (10, 10, 10) if brightness > 155 else (245, 245, 245)


def _extract_people(result) -> list[PersonTrack]:
    people: list[PersonTrack] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return people

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int)

    for box, confidence, class_id, track_id in zip(xyxy, confs, classes, ids):
        if int(class_id) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = map(int, box)
        people.append(
            PersonTrack(
                track_id=int(track_id),
                confidence=float(confidence),
                box=(x1, y1, x2, y2),
            )
        )

    people.sort(key=lambda p: p.track_id)
    return people


def _extract_people_from_detection(result) -> list[PersonTrack]:
    people: list[PersonTrack] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return people

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for index, (box, confidence, class_id) in enumerate(zip(xyxy, confs, classes), start=1):
        if int(class_id) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = map(int, box)
        people.append(
            PersonTrack(index, float(confidence), (x1, y1, x2, y2))
        )
    return people


def _draw_line(frame: np.ndarray, line: CountingLine | None) -> None:
    if line is None:
        return
    h, w = frame.shape[:2]
    color = (0, 215, 255)
    if line.orientation == "horizontal":
        y = line.coordinate(frame.shape)
        cv2.line(frame, (0, y), (w, y), color, 3, cv2.LINE_AA)
    else:
        x = line.coordinate(frame.shape)
        cv2.line(frame, (x, 0), (x, h), color, 3, cv2.LINE_AA)

    cv2.putText(
        frame,
        "COUNTING LINE",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_roi(frame: np.ndarray, roi: ROI | None) -> None:
    if roi is None:
        return
    roi = roi.normalized()
    h, w = frame.shape[:2]
    x1, x2 = int(roi.x_min * w), int(roi.x_max * w)
    y1, y2 = int(roi.y_min * h), int(roi.y_max * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 215, 80), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "ROI",
        (x1 + 6, max(20, y1 + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (70, 215, 80),
        2,
        cv2.LINE_AA,
    )


def _draw_person(
    frame: np.ndarray,
    person: PersonTrack,
    trail: deque[tuple[int, int]],
) -> None:
    x1, y1, x2, y2 = person.box
    color = track_color(person.track_id)
    thickness = max(2, min(frame.shape[:2]) // 420)

    trail_points = list(trail)
    for i in range(1, len(trail_points)):
        cv2.line(
            frame,
            trail_points[i - 1],
            trail_points[i],
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    label = f"ID {person.track_id} | {person.confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    )
    top = max(0, y1 - th - baseline - 5)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 10, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(th + 3, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        _text_color(color),
        1,
        cv2.LINE_AA,
    )


def _draw_dashboard(frame: np.ndarray, state: CountingState) -> None:
    lines = [
        f"PEOPLE NOW: {state.current_count}",
        f"PEAK: {state.peak_count}",
        f"TOTAL UNIQUE SEEN: {state.total_people_seen}",
        f"LINE CROSSES: {state.line_crossings}",
        f"ENTRIES: {state.entries}",
        f"EXITS: {state.exits}",
    ]
    if state.roi is not None:
        lines.append(f"ROI PEOPLE: {state.roi_count}")

    x, y = 12, 52
    panel_h = 25 * len(lines) + 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 38), (x + 305, y - 38 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, dst=frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def _process_video_frame(
    model: YOLO,
    frame: np.ndarray,
    state: CountingState,
    trails: dict[int, deque[tuple[int, int]]],
    conf: float,
    iou: float,
) -> tuple[np.ndarray, list[PersonTrack]]:
    result = model.track(
        frame,
        persist=True,
        tracker=DEFAULT_TRACKER,
        conf=conf,
        iou=iou,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]

    people = _extract_people(result)

    for person in people:
        history = trails.setdefault(
            person.track_id,
            deque(maxlen=TRAIL_LENGTH),
        )
        history.append(person.centroid)

    state.update(people, frame.shape)

    output = frame.copy()
    _draw_line(output, state.line)
    _draw_roi(output, state.roi)

    for person in people:
        _draw_person(output, person, trails[person.track_id])

    _draw_dashboard(output, state)
    return output, people


def process_video(
    model: YOLO,
    in_path: str | Path,
    out_path: str | Path,
    *,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    line: CountingLine | None = None,
    roi: ROI | None = None,
    progress_cb=None,
    max_frames: int | None = None,
) -> PeopleCountResult:
    input_path = Path(in_path)
    output_path = Path(out_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_for_progress = (
        min(source_frames, max_frames)
        if max_frames is not None and source_frames > 0
        else source_frames
    )
    if total_for_progress <= 0:
        total_for_progress = 1

    state = CountingState(line=line, roi=roi)
    trails: dict[int, deque[tuple[int, int]]] = {}
    people_per_frame: list[int] = []
    frame_index = 0
    writer = None
    start = time.perf_counter()

    reset_tracker(model)

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_max_side(frame)
            annotated, people = _process_video_frame(
                model,
                frame,
                state,
                trails,
                conf,
                iou,
            )

            people_per_frame.append(len(people))

            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                h, w = annotated.shape[:2]
                writer = imageio.get_writer(
                    str(output_path),
                    fps=fps,
                    codec="libx264",
                    quality=7,
                    macro_block_size=None,
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                )

            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            frame_index += 1

            if progress_cb:
                progress_cb(frame_index, total_for_progress)

    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if frame_index == 0:
        raise RuntimeError("No readable frames found in the input video.")

    return PeopleCountResult(
        out_path=output_path,
        n_frames=frame_index,
        fps=fps,
        elapsed_s=time.perf_counter() - start,
        people_per_frame=people_per_frame,
        peak_count=state.peak_count,
        current_count=state.current_count,
        total_people_seen=state.total_people_seen,
        line_crossings=state.line_crossings,
        entries=state.entries,
        exits=state.exits,
        roi_count=state.roi_count,
    )


def process_image(
    model: YOLO,
    image_path: str | Path,
    out_path: str | Path,
    *,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    roi: ROI | None = None,
) -> tuple[np.ndarray, int]:
    input_path = Path(image_path)
    output_path = Path(out_path)
    frame = cv2.imread(str(input_path))

    if frame is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    frame = resize_max_side(frame)
    result = model.predict(
        frame,
        conf=conf,
        iou=iou,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]

    people = _extract_people_from_detection(result)
    state = CountingState(roi=roi)
    state.update(people, frame.shape)

    annotated = frame.copy()
    _draw_roi(annotated, roi)
    for person in people:
        _draw_person(
            annotated,
            person,
            deque([person.centroid], maxlen=1),
        )
    _draw_dashboard(annotated, state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save image: {output_path}")

    return annotated, len(people)
