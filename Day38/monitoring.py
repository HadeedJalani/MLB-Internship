from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
PERSON_CLASS_ID = 0
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

EVENT_COLUMNS = [
    "timestamp_s", "event", "track_id", "roi", "entry_time_s",
    "exit_time_s", "duration_s", "frame", "status",
]

def events_to_dataframe(events: list[dict]) -> pd.DataFrame:
    """Return an Arrow-safe event table with consistent column dtypes."""
    df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if df.empty:
        return pd.DataFrame({
            "timestamp_s": pd.Series(dtype="float64"),
            "event": pd.Series(dtype="string"),
            "track_id": pd.Series(dtype="int64"),
            "roi": pd.Series(dtype="string"),
            "entry_time_s": pd.Series(dtype="float64"),
            "exit_time_s": pd.Series(dtype="float64"),
            "duration_s": pd.Series(dtype="float64"),
            "frame": pd.Series(dtype="int64"),
            "status": pd.Series(dtype="string"),
        })
    for col in ["timestamp_s", "entry_time_s", "exit_time_s", "duration_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["track_id", "frame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int64")
    for col in ["event", "roi", "status"]:
        df[col] = df[col].fillna("").astype("string")
    return df



@dataclass(frozen=True)
class ROI:
    name: str
    points: list[list[int]]

    def contour(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.int32)

    def contains(self, point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self.contour(), point, False) >= 0


def read_first_frame(video_path: str | Path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def validate_rois(rois, width: int, height: int) -> list[ROI]:
    if not isinstance(rois, list) or not rois:
        raise ValueError("ROI configuration must be a non-empty list.")

    cleaned = []
    for item in rois:
        if not isinstance(item, dict):
            raise ValueError("Each ROI must be an object.")
        name = str(item.get("name", "")).strip()
        points = item.get("points")

        if not name:
            raise ValueError("Each ROI requires a name.")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"ROI '{name}' requires at least 3 points.")

        safe_points = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"ROI '{name}' has an invalid point.")
            x = max(0, min(width - 1, int(point[0])))
            y = max(0, min(height - 1, int(point[1])))
            safe_points.append([x, y])

        cleaned.append(ROI(name, safe_points))

    return cleaned


def _write_h264(avi_path: Path, mp4_path: Path, fps: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y",
        "-i", str(avi_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-r", str(max(1, round(fps))),
        str(mp4_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("FFmpeg conversion failed:\n" + result.stderr[-1500:])


def process_video(
    video_path: str | Path,
    rois: list[dict],
    *,
    model_path: str = DEFAULT_MODEL,
    conf: float = 0.35,
    process_every: int = 1,
    min_stable_frames: int = 3,
    output_dir: str | Path | None = None,
    progress_cb=None,
) -> dict:
    """Process one video from start to finish, writing MP4 + CSV."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    rois_checked = validate_rois(rois, width, height)

    out_dir = Path(output_dir) if output_dir else Path(
        tempfile.mkdtemp(prefix="day38_monitor_")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    raw_avi = out_dir / f"{stem}_security_monitor.avi"
    mp4_path = out_dir / f"{stem}_security_monitor.mp4"
    csv_path = out_dir / f"{stem}_events.csv"

    writer = cv2.VideoWriter(
        str(raw_avi),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create video writer.")

    model = YOLO(model_path)

    # Per (track ID, ROI) state.
    states: dict[tuple[int, str], dict] = {}
    sessions: dict[tuple[int, str], dict] = {}
    active_keys: set[tuple[int, str]] = set()

    events: list[dict] = []
    unique_track_ids: set[int] = set()
    entries = 0
    exits = 0
    max_active = 0
    frame_index = 0
    stride = max(1, int(process_every))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % stride != 0:
                writer.write(frame)
                frame_index += 1
                continue

            results = model.track(
                frame,
                persist=True,
                tracker=DEFAULT_TRACKER,
                classes=[PERSON_CLASS_ID],
                conf=float(conf),
                verbose=False,
            )
            result = results[0]
            annotated = frame.copy()

            for roi in rois_checked:
                pts = roi.contour()
                cv2.polylines(
                    annotated,
                    [pts],
                    True,
                    (0, 215, 255),
                    3,
                    cv2.LINE_AA,
                )
                x, y = pts[0]
                cv2.putText(
                    annotated,
                    roi.name,
                    (int(x), max(22, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids_tensor = result.boxes.id
                confs = result.boxes.conf.cpu().numpy()
                track_ids = (
                    ids_tensor.cpu().numpy().astype(int)
                    if ids_tensor is not None
                    else np.array([], dtype=int)
                )

                for box, track_id, det_conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    track_id = int(track_id)
                    unique_track_ids.add(track_id)

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (70, 220, 90),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        annotated,
                        (cx, cy),
                        5,
                        (45, 80, 230),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated,
                        f"Person #{track_id}  {float(det_conf):.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (70, 220, 90),
                        2,
                        cv2.LINE_AA,
                    )

                    for roi in rois_checked:
                        key = (track_id, roi.name)
                        inside = roi.contains((cx, cy))
                        state = states.setdefault(
                            key,
                            {
                                "inside": False,
                                "inside_streak": 0,
                                "outside_streak": 0,
                            },
                        )

                        if inside:
                            state["inside_streak"] += 1
                            state["outside_streak"] = 0
                        else:
                            state["outside_streak"] += 1
                            state["inside_streak"] = 0

                        if (
                            not state["inside"]
                            and state["inside_streak"] >= min_stable_frames
                        ):
                            state["inside"] = True
                            active_keys.add(key)
                            entries += 1

                            timestamp = frame_index / fps
                            sessions[key] = {
                                "entry_time_s": timestamp,
                                "entry_frame": frame_index,
                            }

                            events.append(
                                {
                                    "timestamp_s": round(timestamp, 2),
                                    "event": "ENTRY",
                                    "track_id": track_id,
                                    "roi": roi.name,
                                    "entry_time_s": round(timestamp, 2),
                                    "exit_time_s": "",
                                    "duration_s": "",
                                    "frame": frame_index,
                                    "status": "active",
                                }
                            )

                        elif (
                            state["inside"]
                            and state["outside_streak"] >= min_stable_frames
                        ):
                            state["inside"] = False
                            active_keys.discard(key)
                            exits += 1

                            timestamp = frame_index / fps
                            session = sessions.pop(key, None)
                            entry_frame = (
                                session["entry_frame"]
                                if session
                                else frame_index
                            )
                            entry_time = (
                                session["entry_time_s"]
                                if session
                                else ""
                            )
                            duration = max(
                                0.0,
                                (frame_index - entry_frame) / fps,
                            )

                            events.append(
                                {
                                    "timestamp_s": round(timestamp, 2),
                                    "event": "EXIT",
                                    "track_id": track_id,
                                    "roi": roi.name,
                                    "entry_time_s": entry_time,
                                    "exit_time_s": round(timestamp, 2),
                                    "duration_s": round(duration, 2),
                                    "frame": frame_index,
                                    "status": "inactive",
                                }
                            )

                        if state["inside"]:
                            active_keys.add(key)
                        else:
                            active_keys.discard(key)

            active_count = len(active_keys)
            max_active = max(max_active, active_count)

            cv2.rectangle(
                annotated,
                (10, 10),
                (400, 112),
                (12, 12, 12),
                -1,
            )
            cv2.putText(
                annotated,
                "INTELLIGENT SECURITY MONITOR",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Active in ROI: {active_count}",
                (20, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Entries: {entries} | Exits: {exits} | IDs: {len(unique_track_ids)}",
                (20, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

            writer.write(annotated)
            frame_index += 1

            if progress_cb:
                progress_cb(
                    frame_index,
                    max(1, total_frames),
                )
    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        raise RuntimeError("Video had no readable frames.")

    _write_h264(raw_avi, mp4_path, fps)

    event_df = events_to_dataframe(events)
    event_df.to_csv(csv_path, index=False)

    return {
        "video_path": str(mp4_path),
        "csv_path": str(csv_path),
        "events": event_df,
        "unique_people": len(unique_track_ids),
        "entries": entries,
        "exits": exits,
        "max_active": max_active,
        "processed_frames": frame_index,
        "fps": fps,
    }


def process_video_folder(
    input_dir: str | Path,
    rois: list[dict],
    *,
    output_dir: str | Path = "outputs/security_monitoring",
    model_path: str = DEFAULT_MODEL,
    conf: float = 0.35,
    process_every: int = 1,
    min_stable_frames: int = 3,
    progress_cb=None,
) -> list[dict]:
    """Batch-process ALL video files in a folder."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise FileNotFoundError(f"No supported videos found in {input_dir}")

    results = []
    for index, video in enumerate(videos, start=1):
        def child_progress(done, total):
            if progress_cb:
                progress_cb(index - 1 + done / max(total, 1), len(videos))

        result = process_video(
            video,
            rois,
            model_path=model_path,
            conf=conf,
            process_every=process_every,
            min_stable_frames=min_stable_frames,
            output_dir=output_dir,
            progress_cb=child_progress,
        )
        result["input_video"] = str(video)
        results.append(result)
        if progress_cb:
            progress_cb(index, len(videos))

    return results
