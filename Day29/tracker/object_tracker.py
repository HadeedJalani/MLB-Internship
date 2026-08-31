import os
import time
from collections import Counter

import cv2
from ultralytics import YOLO


class ObjectTracker:
    """
    YOLO-based Multi-Object Tracking System.

    Features:
    - YOLO11 object detection
    - ByteTrack / BoT-SORT support
    - Persistent tracking IDs
    - Per-class detection statistics
    - Unique object counting
    - Confidence scores
    - Custom bounding box visualization
    - Annotated video generation
    """

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.25,
        tracker="ByteTrack",
    ):

        self.model_path = model_path
        self.confidence = confidence

        tracker_mapping = {
            "ByteTrack": "bytetrack.yaml",
            "BoT-SORT": "botsort.yaml",
        }

        self.tracker_config = tracker_mapping.get(
            tracker,
            "bytetrack.yaml",
        )

        print(f"Loading YOLO model: {model_path}")

        self.model = YOLO(model_path)

    @staticmethod
    def get_color(track_id):
        """
        Generate a consistent color for every tracking ID.
        """

        if track_id is None:
            return (0, 255, 255)

        return (
            int((track_id * 37) % 255),
            int((track_id * 17) % 255),
            int((track_id * 29) % 255),
        )

    def process_video(
        self,
        input_path,
        output_path,
        progress_callback=None,
    ):
        """
        Run YOLO detection + multi-object tracking on a video.

        Returns complete tracking statistics.
        """

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {input_path}"
            )

        # -------------------------------------------------
        # Video properties
        # -------------------------------------------------

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0:
            fps = 25.0

        # -------------------------------------------------
        # Output setup
        # -------------------------------------------------

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():

            cap.release()

            raise RuntimeError(
                "Could not create output video."
            )

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        unique_ids = set()

        # Classes detected across ALL frames
        detected_classes = set()

        # Number of detections per class
        class_detection_counter = Counter()

        # Track ID -> Class mapping
        tracked_objects = {}

        # Total detections
        total_detections = 0

        frame_count = 0

        start_time = time.perf_counter()

        # -------------------------------------------------
        # Process frames
        # -------------------------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break

            # -------------------------------------------------
            # YOLO Tracking
            # -------------------------------------------------

            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.confidence,
                verbose=False,
            )

            result = results[0]

            annotated_frame = frame.copy()

            # -------------------------------------------------
            # Process ALL detections
            # -------------------------------------------------

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                class_ids = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                # Tracking IDs may temporarily be unavailable
                if result.boxes.id is not None:

                    track_ids = (
                        result.boxes.id
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                else:

                    track_ids = [
                        None
                    ] * len(boxes)

                # -------------------------------------------------
                # Iterate through every detection
                # -------------------------------------------------

                for (
                    box,
                    track_id,
                    confidence,
                    class_id,
                ) in zip(
                    boxes,
                    track_ids,
                    confidences,
                    class_ids,
                ):

                    x1, y1, x2, y2 = box

                    class_name = self.model.names[
                        class_id
                    ]

                    # ---------------------------------------------
                    # Detection statistics
                    # ---------------------------------------------

                    detected_classes.add(
                        class_name
                    )

                    class_detection_counter[
                        class_name
                    ] += 1

                    total_detections += 1

                    # ---------------------------------------------
                    # Tracking statistics
                    # ---------------------------------------------

                    if track_id is not None:

                        unique_ids.add(
                            int(track_id)
                        )

                        tracked_objects[
                            int(track_id)
                        ] = class_name

                    # ---------------------------------------------
                    # Visualization
                    # ---------------------------------------------

                    color = self.get_color(
                        track_id
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    # Label

                    if track_id is not None:

                        label = (
                            f"{class_name} | "
                            f"ID: {track_id} | "
                            f"{confidence:.0%}"
                        )

                    else:

                        label = (
                            f"{class_name} | "
                            f"{confidence:.0%}"
                        )

                    (
                        text_width,
                        text_height,
                    ), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        2,
                    )

                    label_y = max(
                        y1 - 8,
                        text_height + 12,
                    )

                    # Label background

                    cv2.rectangle(
                        annotated_frame,
                        (
                            x1,
                            label_y
                            - text_height
                            - 10,
                        ),
                        (
                            x1
                            + text_width
                            + 10,
                            label_y + 5,
                        ),
                        color,
                        -1,
                    )

                    cv2.putText(
                        annotated_frame,
                        label,
                        (
                            x1 + 5,
                            label_y - 4,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # -------------------------------------------------
            # Statistics overlay
            # -------------------------------------------------

            overlay_text = (
                f"Tracked Objects: "
                f"{len(unique_ids)}"
            )

            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (350, 55),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                annotated_frame,
                overlay_text,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # -------------------------------------------------
            # Write frame
            # -------------------------------------------------

            writer.write(
                annotated_frame
            )

            frame_count += 1

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                progress_callback is not None
                and total_frames > 0
            ):

                progress = (
                    frame_count
                    / total_frames
                )

                progress_callback(
                    min(progress, 1.0)
                )

        # -------------------------------------------------
        # Cleanup
        # -------------------------------------------------

        cap.release()
        writer.release()

        processing_time = (
            time.perf_counter()
            - start_time
        )

        # -------------------------------------------------
        # Per-class unique tracked objects
        # -------------------------------------------------

        unique_objects_per_class = Counter()

        for track_id, class_name in (
            tracked_objects.items()
        ):

            unique_objects_per_class[
                class_name
            ] += 1

        # -------------------------------------------------
        # Return statistics
        # -------------------------------------------------

        return {
            "output_path": output_path,

            "total_frames": frame_count,

            "fps": fps,

            "processing_time": processing_time,

            # Unique tracking IDs
            "unique_objects": len(unique_ids),

            # Classes detected anywhere in video
            "unique_classes": len(
                detected_classes
            ),

            # IDs
            "tracking_ids": sorted(
                list(unique_ids)
            ),

            # Class names
            "classes_detected": sorted(
                list(detected_classes)
            ),

            # Total detections across frames
            "total_detections": (
                total_detections
            ),

            # Detection frequency
            "class_detection_counts": dict(
                class_detection_counter
            ),

            # Unique tracked objects per class
            "unique_objects_per_class": dict(
                unique_objects_per_class
            ),
        }