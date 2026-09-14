from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
EVENT_COLUMNS = [
    "track_id", "event", "frame", "timestamp_s",
    "entry_time_s", "exit_time_s", "duration_s", "status",
]


@dataclass
class VideoConfig:
    conf: float = 0.40
    iou: float = 0.50
    imgsz: int = 640
    frame_skip: int = 1
    tracker: str = "ByteTrack"
    roi_x1: float = 0.20
    roi_y1: float = 0.20
    roi_x2: float = 0.80
    roi_y2: float = 0.85
    show_trails: bool = True


def load_model() -> YOLO:
    return YOLO(MODEL_NAME)


def build_roi(width: int, height: int, config: VideoConfig) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(config.roi_x1 * width), width - 1))
    y1 = max(0, min(int(config.roi_y1 * height), height - 1))
    x2 = max(0, min(int(config.roi_x2 * width), width - 1))
    y2 = max(0, min(int(config.roi_y2 * height), height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def point_inside_roi(x: int, y: int, roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        frame, "MONITORING ROI", (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )


def draw_text(frame: np.ndarray, text: str, x: int, y: int, scale: float = 0.55) -> None:
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA,
    )


def normalize_event_dataframe(events: list[list]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame({
            "track_id": pd.Series(dtype="int64"),
            "event": pd.Series(dtype="string"),
            "frame": pd.Series(dtype="int64"),
            "timestamp_s": pd.Series(dtype="float64"),
            "entry_time_s": pd.Series(dtype="float64"),
            "exit_time_s": pd.Series(dtype="float64"),
            "duration_s": pd.Series(dtype="float64"),
            "status": pd.Series(dtype="string"),
        })

    df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").fillna(-1).astype("int64")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").fillna(-1).astype("int64")
    for col in ["timestamp_s", "entry_time_s", "exit_time_s", "duration_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["event", "status"]:
        df[col] = df[col].astype("string")
    return df


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    config: VideoConfig,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path, dict]:

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_model()
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("OpenCV could not open the input video.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    roi = build_roi(width, height, config)

    output_video = output_dir / f"{video_path.stem}_processed.mp4"
    events_csv = output_dir / "events.csv"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not create the output video.")

    tracker_config = "botsort.yaml" if config.tracker == "BoT-SORT" else "bytetrack.yaml"

    track_state: dict[int, dict] = {}
    unique_ids: set[int] = set()
    trails = defaultdict(lambda: deque(maxlen=25))
    events: list[list] = []

    frame_no = 0
    inferred_frames = 0
    peak_roi = 0
    roi_sum = 0
    roi_samples = 0
    start = perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_no % max(1, config.frame_skip) == 0:
                results = model.track(
                    source=frame,
                    conf=config.conf,
                    iou=config.iou,
                    imgsz=config.imgsz,
                    classes=[0],
                    tracker=tracker_config,
                    persist=True,
                    verbose=False,
                )

                current_roi_ids: set[int] = set()

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes

                    if boxes.id is not None:
                        ids = boxes.id.int().cpu().tolist()
                        coords = boxes.xyxy.cpu().tolist()

                        for track_id, box in zip(ids, coords):
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            inside = point_inside_roi(cx, cy, roi)

                            unique_ids.add(track_id)

                            if inside:
                                current_roi_ids.add(track_id)

                            trails[track_id].append((cx, cy))

                            if config.show_trails:
                                pts = list(trails[track_id])
                                for a, b in zip(pts, pts[1:]):
                                    cv2.line(frame, a, b, (255, 255, 255), 2)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            draw_text(frame, f"ID {track_id}", x1, max(20, y1 - 8))

                            timestamp = frame_no / source_fps
                            previous = track_state.get(track_id)

                            if previous is None:
                                track_state[track_id] = {
                                    "inside": inside,
                                    "entry": timestamp if inside else None,
                                }
                                if inside:
                                    events.append([
                                        track_id, "ENTRY", frame_no, timestamp,
                                        timestamp, np.nan, np.nan, "active",
                                    ])
                            else:
                                was_inside = previous["inside"]

                                if inside and not was_inside:
                                    track_state[track_id] = {
                                        "inside": True,
                                        "entry": timestamp,
                                    }
                                    events.append([
                                        track_id, "ENTRY", frame_no, timestamp,
                                        timestamp, np.nan, np.nan, "active",
                                    ])

                                elif not inside and was_inside:
                                    entry = previous["entry"] or timestamp
                                    duration = max(0.0, timestamp - entry)
                                    events.append([
                                        track_id, "EXIT", frame_no, timestamp,
                                        entry, timestamp, duration, "completed",
                                    ])
                                    track_state[track_id] = {
                                        "inside": False,
                                        "entry": entry,
                                    }

                count = len(current_roi_ids)
                peak_roi = max(peak_roi, count)
                roi_sum += count
                roi_samples += 1
                inferred_frames += 1

            active_count = sum(
                1 for state in track_state.values()
                if state["inside"]
            )

            elapsed = max(perf_counter() - start, 1e-6)
            live_fps = frame_no / elapsed

            draw_roi(frame, roi)
            draw_text(frame, f"Current ROI Count: {active_count}", 15, 30, 0.65)
            draw_text(frame, f"Unique IDs: {len(unique_ids)}", 15, 60, 0.65)
            draw_text(frame, f"FPS: {live_fps:.2f}", 15, 90, 0.65)
            draw_text(frame, f"Frame: {frame_no}", 15, 120, 0.65)

            writer.write(frame)
            frame_no += 1

            if progress_callback and total_frames:
                progress_callback(
                    min(frame_no / total_frames, 1.0),
                    f"Processing {frame_no}/{total_frames}",
                )

    finally:
        capture.release()
        writer.release()

    end_time = frame_no / source_fps

    for track_id, state in track_state.items():
        if state["inside"]:
            entry = state["entry"] or end_time
            duration = max(0.0, end_time - entry)
            events.append([
                track_id, "EXIT", frame_no, end_time,
                entry, end_time, duration, "completed_at_video_end",
            ])

    df = normalize_event_dataframe(events)
    df.to_csv(events_csv, index=False)

    durations = pd.to_numeric(
        df.loc[df["event"] == "EXIT", "duration_s"],
        errors="coerce",
    ).dropna()

    runtime = max(perf_counter() - start, 1e-6)

    summary = {
        "total_objects": len(unique_ids),
        "total_entries": int((df["event"] == "ENTRY").sum()),
        "total_exits": int((df["event"] == "EXIT").sum()),
        "maximum_objects_in_roi": int(peak_roi),
        "average_fps": float(frame_no / runtime),
        "processing_time_s": float(runtime),
        "video_duration_s": float(end_time),
        "frames_total": int(frame_no),
        "frames_inferred": int(inferred_frames),
        "average_objects_in_roi": float(roi_sum / roi_samples) if roi_samples else 0.0,
        "average_dwell_time_s": float(durations.mean()) if not durations.empty else 0.0,
    }

    return output_video, events_csv, summary
