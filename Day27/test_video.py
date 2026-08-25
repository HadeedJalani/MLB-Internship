from pathlib import Path

import cv2
from ultralytics import YOLO


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = "yolo11n.pt"

INPUT_DIR = Path("sample_videos")
OUTPUT_DIR = Path("outputs/videos")

CONFIDENCE_THRESHOLD = 0.25


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

video_files = sorted(
    [
        p
        for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    ]
)

if not video_files:
    print(f"No videos found in: {INPUT_DIR}")
    print("Add at least 2 short videos and run again.")
    raise SystemExit(1)

print(f"\nFound {len(video_files)} video(s).")
print("=" * 60)


# ---------------------------------------------------------
# Process videos
# ---------------------------------------------------------

for video_path in video_files:

    print(f"\nProcessing: {video_path.name}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print("  ERROR: Could not open video.")
        continue

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {frame_count}")

    output_path = OUTPUT_DIR / f"{video_path.stem}_detected.mp4"

    # MP4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("  ERROR: Could not create output video.")
        capture.release()
        continue

    frame_number = 0
    total_detections = 0

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        # YOLO inference
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        # Draw YOLO annotations
        annotated_frame = result.plot()

        writer.write(annotated_frame)

        # Count detections
        if result.boxes is not None:
            detections = len(result.boxes)
            total_detections += detections

        # Progress
        if frame_number % 30 == 0:

            if frame_count > 0:
                progress = (frame_number / frame_count) * 100
                print(
                    f"  Progress: {progress:.1f}% "
                    f"({frame_number}/{frame_count})"
                )
            else:
                print(f"  Processed frames: {frame_number}")

    capture.release()
    writer.release()

    print(f"  Total detection boxes : {total_detections}")
    print(f"  Saved output          : {output_path}")


# ---------------------------------------------------------
# Complete
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("YOLO video detection testing completed.")
print(f"Outputs saved in: {OUTPUT_DIR}")