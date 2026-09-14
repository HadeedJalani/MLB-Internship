from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"

EVENT_COLUMNS = [
    "track_id",
    "event",
    "roi",
    "frame",
    "timestamp_s",
    "entry_time_s",
    "exit_time_s",
    "duration_s",
    "status",
]


# ---------------------------------------------------------------------
# Model and data helpers
# ---------------------------------------------------------------------

def load_model() -> YOLO:
    """
    Load the YOLO model.

    Ultralytics downloads yolov8n.pt automatically on first use
    if the model is not already available locally.
    """
    return YOLO(MODEL_NAME)


def point_inside_roi(
    x: int,
    y: int,
    roi: tuple[int, int, int, int] | None,
) -> bool:
    """
    Check whether a point is inside a rectangular ROI.

    ROI format:
        (x1, y1, x2, y2)
    """
    if roi is None:
        return True

    x1, y1, x2, y2 = roi

    return x1 <= x <= x2 and y1 <= y <= y2


def normalize_events(events: list[list]) -> pd.DataFrame:
    """
    Convert raw event records into a dataframe with stable dtypes.

    This avoids mixed integer/float/object columns that can cause
    Streamlit or PyArrow serialization errors.
    """
    if not events:
        return pd.DataFrame(
            {
                "track_id": pd.Series(dtype="int64"),
                "event": pd.Series(dtype="string"),
                "roi": pd.Series(dtype="string"),
                "frame": pd.Series(dtype="int64"),
                "timestamp_s": pd.Series(dtype="float64"),
                "entry_time_s": pd.Series(dtype="float64"),
                "exit_time_s": pd.Series(dtype="float64"),
                "duration_s": pd.Series(dtype="float64"),
                "status": pd.Series(dtype="string"),
            }
        )

    dataframe = pd.DataFrame(
        events,
        columns=EVENT_COLUMNS,
    )

    for column in ["track_id", "frame"]:
        dataframe[column] = (
            pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )
            .fillna(-1)
            .astype("int64")
        )

    for column in [
        "timestamp_s",
        "entry_time_s",
        "exit_time_s",
        "duration_s",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).astype("float64")

    for column in ["event", "roi", "status"]:
        dataframe[column] = dataframe[column].astype("string")

    return dataframe


def draw_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """
    Draw the configured ROI on a frame.
    """
    if roi is None:
        return frame

    x1, y1, x2, y2 = roi

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "MONITORING ROI",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_label(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
) -> None:
    """
    Draw a readable label on a frame.
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------

def process_image(
    image: Image.Image,
    conf: float = 0.45,
    iou: float = 0.50,
    imgsz: int = 512,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Process one image.

    Only class 0, person, is detected.
    """
    model = load_model()

    rgb_array = np.array(image.convert("RGB"))
    bgr_frame = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2BGR,
    )

    results = model.predict(
        source=bgr_frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=[0],
        verbose=False,
    )

    output_frame = bgr_frame.copy()
    roi_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            coordinates = box.xyxy[0].tolist()

            x1, y1, x2, y2 = [
                int(value)
                for value in coordinates
            ]

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            inside = point_inside_roi(
                center_x,
                center_y,
                roi,
            )

            if inside:
                roi_count += 1

            if roi is None or inside:
                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                )

                draw_label(
                    output_frame,
                    "person",
                    (x1, max(20, y1 - 8)),
                )

    draw_roi(output_frame, roi)

    output_rgb = cv2.cvtColor(
        output_frame,
        cv2.COLOR_BGR2RGB,
    )

    return output_rgb, {
        "roi_count": roi_count,
    }


# ---------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------

def process_video(
    video_path: str | Path,
    conf: float = 0.45,
    iou: float = 0.50,
    imgsz: int = 512,
    frame_skip: int = 1,
    tracker_name: str = "ByteTrack",
    show_trails: bool = True,
    roi: tuple[int, int, int, int] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path, dict]:
    """
    Process a video with YOLO tracking.

    Features:
    - Person-only detection
    - Persistent IDs
    - ByteTrack or BoT-SORT
    - ROI entry and exit events
    - Dwell-time calculation
    - Tracking trails
    - Processed MP4 output
    - CSV event report
    - Frame skipping
    """

    video_path = Path(video_path)

    model = load_model()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Unable to open the input video.")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
    )

    output_directory = video_path.parent / "outputs"
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video_path = (
        output_directory / "day39_processed_video.mp4"
    )

    output_csv_path = (
        output_directory / "day39_event_report.csv"
    )

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not video_writer.isOpened():
        capture.release()
        raise ValueError(
            "Unable to create the output video. "
            "Check the output directory and codec support."
        )

    if tracker_name == "BoT-SORT":
        tracker_config = "botsort.yaml"
    else:
        tracker_config = "bytetrack.yaml"

    # Stores whether a track is currently inside the ROI.
    active_tracks: dict[int, dict] = {}

    # Stores the IDs that have appeared at least once.
    unique_track_ids: set[int] = set()

    # Stores recent center points for trail drawing.
    track_trails = defaultdict(
        lambda: deque(maxlen=30)
    )

    events: list[list] = []

    frame_number = 0
    sampled_frames = 0
    peak_roi_count = 0
    total_roi_count = 0

    processing_start = perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            should_process = (
                frame_number % max(1, frame_skip) == 0
            )

            if should_process:
                results = model.track(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=[0],
                    tracker=tracker_config,
                    persist=True,
                    verbose=False,
                )

                current_inside_ids: set[int] = set()

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes

                    if boxes.id is not None:
                        track_ids = (
                            boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )

                        coordinates = (
                            boxes.xyxy
                            .cpu()
                            .tolist()
                        )

                        for track_id, box in zip(
                            track_ids,
                            coordinates,
                        ):
                            x1, y1, x2, y2 = [
                                int(value)
                                for value in box
                            ]

                            center_x = (
                                x1 + x2
                            ) // 2

                            center_y = (
                                y1 + y2
                            ) // 2

                            inside_roi = point_inside_roi(
                                center_x,
                                center_y,
                                roi,
                            )

                            unique_track_ids.add(track_id)

                            if inside_roi:
                                current_inside_ids.add(
                                    track_id
                                )

                            if show_trails:
                                track_trails[
                                    track_id
                                ].append(
                                    (
                                        center_x,
                                        center_y,
                                    )
                                )

                                trail_points = list(
                                    track_trails[
                                        track_id
                                    ]
                                )

                                for point_a, point_b in zip(
                                    trail_points,
                                    trail_points[1:],
                                ):
                                    cv2.line(
                                        frame,
                                        point_a,
                                        point_b,
                                        (255, 255, 255),
                                        2,
                                    )

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (255, 255, 255),
                                2,
                            )

                            draw_label(
                                frame,
                                f"ID {track_id}",
                                (
                                    x1,
                                    max(20, y1 - 8),
                                ),
                            )

                            timestamp_s = (
                                frame_number / fps
                            )

                            previous_state = active_tracks.get(
                                track_id,
                                {
                                    "inside": False,
                                    "entry_time_s": None,
                                },
                            )

                            was_inside = previous_state[
                                "inside"
                            ]

                            if (
                                inside_roi
                                and not was_inside
                            ):
                                active_tracks[track_id] = {
                                    "inside": True,
                                    "entry_time_s": timestamp_s,
                                }

                                events.append(
                                    [
                                        track_id,
                                        "ENTRY",
                                        "ROI",
                                        frame_number,
                                        timestamp_s,
                                        timestamp_s,
                                        np.nan,
                                        np.nan,
                                        "active",
                                    ]
                                )

                            elif (
                                not inside_roi
                                and was_inside
                            ):
                                entry_time_s = previous_state[
                                    "entry_time_s"
                                ]

                                if entry_time_s is None:
                                    entry_time_s = timestamp_s

                                duration_s = max(
                                    0.0,
                                    timestamp_s - entry_time_s,
                                )

                                events.append(
                                    [
                                        track_id,
                                        "EXIT",
                                        "ROI",
                                        frame_number,
                                        timestamp_s,
                                        entry_time_s,
                                        timestamp_s,
                                        duration_s,
                                        "completed",
                                    ]
                                )

                                active_tracks[track_id] = {
                                    "inside": False,
                                    "entry_time_s": entry_time_s,
                                }

                current_roi_count = len(
                    current_inside_ids
                )

                peak_roi_count = max(
                    peak_roi_count,
                    current_roi_count,
                )

                total_roi_count += current_roi_count
                sampled_frames += 1

            active_count = sum(
                1
                for state in active_tracks.values()
                if state.get("inside", False)
            )

            draw_roi(frame, roi)

            draw_label(
                frame,
                f"Active ROI: {active_count}",
                (15, 30),
            )

            draw_label(
                frame,
                f"Frame: {frame_number}",
                (15, 60),
            )

            video_writer.write(frame)

            frame_number += 1

            if progress_callback and total_frames > 0:
                progress = frame_number / total_frames

                progress_callback(
                    progress,
                    (
                        f"Processed {frame_number} "
                        f"of {total_frames} frames"
                    ),
                )

    finally:
        capture.release()
        video_writer.release()

    # Close any sessions that were still active at the end.
    final_timestamp_s = (
        frame_number / fps
        if fps > 0
        else 0.0
    )

    for track_id, state in active_tracks.items():
        if state.get("inside", False):
            entry_time_s = state.get(
                "entry_time_s",
                final_timestamp_s,
            )

            duration_s = max(
                0.0,
                final_timestamp_s - entry_time_s,
            )

            events.append(
                [
                    track_id,
                    "EXIT",
                    "ROI",
                    frame_number,
                    final_timestamp_s,
                    entry_time_s,
                    final_timestamp_s,
                    duration_s,
                    "completed_at_video_end",
                ]
            )

    event_dataframe = normalize_events(events)

    event_dataframe.to_csv(
        output_csv_path,
        index=False,
    )

    completed_durations = pd.to_numeric(
        event_dataframe.loc[
            event_dataframe["event"] == "EXIT",
            "duration_s",
        ],
        errors="coerce",
    ).dropna()

    elapsed_seconds = max(
        perf_counter() - processing_start,
        1e-6,
    )

    average_dwell_s = (
        float(completed_durations.mean())
        if not completed_durations.empty
        else 0.0
    )

    statistics = {
        "peak_roi": int(peak_roi_count),
        "unique_ids": int(len(unique_track_ids)),
        "entries": int(
            (event_dataframe["event"] == "ENTRY").sum()
        ),
        "exits": int(
            (event_dataframe["event"] == "EXIT").sum()
        ),
        "avg_dwell_s": average_dwell_s,
        "avg_fps": float(
            frame_number / elapsed_seconds
        ),
        "frames_processed": int(frame_number),
        "sampled_frames": int(sampled_frames),
        "average_roi_count": (
            float(total_roi_count / sampled_frames)
            if sampled_frames > 0
            else 0.0
        ),
    }

    return (
        output_video_path,
        output_csv_path,
        statistics,
    )