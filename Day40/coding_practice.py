from __future__ import annotations

from time import perf_counter

import cv2
from ultralytics import YOLO

VIDEO_PATH = "sample_videos/video_01.mp4"
MODEL = YOLO("yolov8n.pt")
CONFIDENCE = 0.40


def roi_for(width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(width * 0.20),
        int(height * 0.20),
        int(width * 0.80),
        int(height * 0.85),
    )


def inside_roi(x: int, y: int, roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise ValueError(f"Could not open {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi = roi_for(width, height)

frame_count = 0
unique_ids = set()
previous_state = {}
entries = 0
exits = 0
start = perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = MODEL.track(
        frame,
        conf=CONFIDENCE,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )

    current_ids = set()

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            coords = boxes.xyxy.cpu().tolist()

            for track_id, box in zip(ids, coords):
                x1, y1, x2, y2 = [int(v) for v in box]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                inside = inside_roi(
                    cx,
                    cy,
                    roi,
                )

                unique_ids.add(track_id)

                if inside:
                    current_ids.add(track_id)

                was_inside = previous_state.get(
                    track_id,
                    False,
                )

                if inside and not was_inside:
                    entries += 1

                elif not inside and was_inside:
                    exits += 1

                previous_state[track_id] = inside

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

    cv2.rectangle(
        frame,
        (roi[0], roi[1]),
        (roi[2], roi[3]),
        (255, 255, 255),
        2,
    )

    elapsed = max(
        perf_counter() - start,
        1e-6,
    )

    fps = frame_count / elapsed

    overlays = [
        f"Current ROI Count: {len(current_ids)}",
        f"Unique Objects: {len(unique_ids)}",
        f"Entries: {entries}",
        f"Exits: {exits}",
        f"FPS: {fps:.2f}",
    ]

    for i, text in enumerate(overlays):
        cv2.putText(
            frame,
            text,
            (20, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow(
        "Day 40 Coding Practice",
        frame,
    )

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

runtime = max(
    perf_counter() - start,
    1e-6,
)

print("\nDay 40 Coding Practice Summary")
print(f"Unique Objects: {len(unique_ids)}")
print(f"Entries: {entries}")
print(f"Exits: {exits}")
print(f"Average FPS: {frame_count / runtime:.2f}")
