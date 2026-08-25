import cv2
import numpy as np
from ultralytics import YOLO


MODEL_NAME = "yolo11n.pt"


def load_model():
    """
    Load the lightweight pretrained YOLO model.
    """
    return YOLO(MODEL_NAME)


def detect_objects(model, image, confidence=0.25):
    """
    Run YOLO object detection on a BGR OpenCV image.

    Returns:
        annotated_image
        detections
    """

    if image is None:
        raise ValueError("Input image is empty.")

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
        device="cpu",
    )

    result = results[0]

    annotated_image = result.plot(
        conf=True,
        labels=True,
        boxes=True,
    )

    detections = []

    if result.boxes is not None:

        names = result.names

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": confidence_score,
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                }
            )

    return annotated_image, detections


def process_video(
    model,
    input_path,
    output_path,
    confidence=0.25,
):
    """
    Process a video frame-by-frame using YOLO.
    """

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Could not open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create output video.")

    total_frames = 0
    total_detections = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=confidence,
            verbose=False,
            device="cpu",
        )

        result = results[0]

        annotated = result.plot(
            conf=True,
            labels=True,
            boxes=True,
        )

        writer.write(annotated)

        total_frames += 1

        if result.boxes is not None:
            total_detections += len(result.boxes)

    cap.release()
    writer.release()

    return {
        "frames": total_frames,
        "detections": total_detections,
        "output_path": output_path,
    }