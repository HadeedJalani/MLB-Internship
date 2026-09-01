import os
import time
from collections import Counter

import cv2
from ultralytics import YOLO


class VehicleCounter:
    """
    YOLO-based Smart Vehicle Counting System.

    Features:
    - YOLO vehicle detection
    - ByteTrack / BoT-SORT tracking
    - Persistent tracking IDs
    - Counting line crossing detection
    - Duplicate prevention
    - Per-class counting
    - Direction-wise counting
    - Annotated output generation
    """

    VEHICLE_CLASSES = {
        "car",
        "bus",
        "truck",
        "motorcycle",
    }

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.25,
        tracker="ByteTrack",
        line_position=0.5,
    ):
        """
        Parameters
        ----------
        model_path : str
            YOLO model path.

        confidence : float
            Detection confidence threshold.

        tracker : str
            ByteTrack or BoT-SORT.

        line_position : float
            Vertical position of horizontal counting line.

            0.0 = top of video
            0.5 = center
            1.0 = bottom of video
        """

        self.model_path = model_path
        self.confidence = confidence
        self.line_position = line_position

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

    # ---------------------------------------------------------
    # Utility Functions
    # ---------------------------------------------------------

    @staticmethod
    def get_color(track_id):
        """
        Generate deterministic color for each tracking ID.
        """

        return (
            int((track_id * 37) % 255),
            int((track_id * 17) % 255),
            int((track_id * 29) % 255),
        )

    @staticmethod
    def get_center(box):
        """
        Calculate center of bounding box.
        """

        x1, y1, x2, y2 = box

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return center_x, center_y

    @staticmethod
    def check_line_crossing(
        previous_y,
        current_y,
        line_y,
    ):
        """
        Detect whether an object crossed the line.

        Returns:
            "down" if moving downward
            "up" if moving upward
            None if no crossing
        """

        # Moving from above line to below line
        if (
            previous_y < line_y
            and current_y >= line_y
        ):
            return "down"

        # Moving from below line to above line
        if (
            previous_y > line_y
            and current_y <= line_y
        ):
            return "up"

        return None

    # ---------------------------------------------------------
    # Main Video Processing
    # ---------------------------------------------------------

    def process_video(
        self,
        input_path,
        output_path,
        progress_callback=None,
    ):
        """
        Detect, track and count vehicles crossing
        a virtual horizontal line.
        """

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {input_path}"
            )

        # -----------------------------------------------------
        # Video properties
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Counting line position
        # -----------------------------------------------------

        line_y = int(
            height * self.line_position
        )

        print(
            f"Video resolution: {width}x{height}"
        )

        print(
            f"Counting line position: {line_y}"
        )

        # -----------------------------------------------------
        # Output setup
        # -----------------------------------------------------

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:

            os.makedirs(
                output_directory,
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

        # -----------------------------------------------------
        # Tracking state
        # -----------------------------------------------------

        previous_positions = {}

        counted_ids = set()

        vehicle_counts = Counter()

        up_counts = Counter()

        down_counts = Counter()

        detected_classes = set()

        frame_count = 0

        start_time = time.perf_counter()

        # -----------------------------------------------------
        # Process frames
        # -----------------------------------------------------

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
            # Draw counting line
            # -------------------------------------------------

            cv2.line(
                annotated_frame,
                (0, line_y),
                (width, line_y),
                (0, 255, 255),
                3,
            )

            cv2.putText(
                annotated_frame,
                "COUNTING LINE",
                (20, max(30, line_y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # -------------------------------------------------
            # Extract detections
            # -------------------------------------------------

            if (
                result.boxes is not None
                and len(result.boxes) > 0
                and result.boxes.id is not None
            ):

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                track_ids = (
                    result.boxes.id
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

                # -------------------------------------------------
                # Process each detection
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

                    class_name = self.model.names[
                        class_id
                    ]

                    # ---------------------------------------------
                    # Only process target vehicles
                    # ---------------------------------------------

                    if (
                        class_name
                        not in self.VEHICLE_CLASSES
                    ):
                        continue

                    detected_classes.add(
                        class_name
                    )

                    track_id = int(track_id)

                    x1, y1, x2, y2 = box

                    # ---------------------------------------------
                    # Get center
                    # ---------------------------------------------

                    center_x, center_y = (
                        self.get_center(box)
                    )

                    # ---------------------------------------------
                    # Get previous position
                    # ---------------------------------------------

                    previous_position = (
                        previous_positions.get(
                            track_id
                        )
                    )

                    # ---------------------------------------------
                    # Check crossing
                    # ---------------------------------------------

                    if previous_position is not None:

                        previous_y = (
                            previous_position[1]
                        )

                        direction = (
                            self.check_line_crossing(
                                previous_y,
                                center_y,
                                line_y,
                            )
                        )

                        # -----------------------------------------
                        # Count only once
                        # -----------------------------------------

                        if (
                            direction is not None
                            and track_id
                            not in counted_ids
                        ):

                            counted_ids.add(
                                track_id
                            )

                            vehicle_counts[
                                class_name
                            ] += 1

                            if direction == "up":

                                up_counts[
                                    class_name
                                ] += 1

                            elif direction == "down":

                                down_counts[
                                    class_name
                                ] += 1

                            print(
                                f"\nCOUNTED → "
                                f"{class_name.upper()} "
                                f"| ID {track_id} "
                                f"| Direction: "
                                f"{direction.upper()}"
                            )

                    # ---------------------------------------------
                    # Update position
                    # ---------------------------------------------

                    previous_positions[
                        track_id
                    ] = (
                        center_x,
                        center_y,
                    )

                    # ---------------------------------------------
                    # Draw object
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

                    cv2.circle(
                        annotated_frame,
                        (
                            center_x,
                            center_y,
                        ),
                        5,
                        color,
                        -1,
                    )

                    label = (
                        f"{class_name} | "
                        f"ID:{track_id} | "
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
                        text_height + 15,
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
            # Statistics Panel
            # -------------------------------------------------

            total_count = sum(
                vehicle_counts.values()
            )

            panel_height = 220

            overlay = annotated_frame.copy()

            cv2.rectangle(
                overlay,
                (10, 10),
                (370, panel_height),
                (15, 15, 15),
                -1,
            )

            cv2.addWeighted(
                overlay,
                0.85,
                annotated_frame,
                0.15,
                0,
                annotated_frame,
            )

            cv2.putText(
                annotated_frame,
                "SMART VEHICLE COUNT",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                f"TOTAL VEHICLES: {total_count}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            vehicle_order = [
                "car",
                "bus",
                "truck",
                "motorcycle",
            ]

            y_offset = 110

            for vehicle in vehicle_order:

                count = vehicle_counts[
                    vehicle
                ]

                cv2.putText(
                    annotated_frame,
                    f"{vehicle.title()}: {count}",
                    (25, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )

                y_offset += 25

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

        # -----------------------------------------------------
        # Cleanup
        # -----------------------------------------------------

        cap.release()

        writer.release()

        processing_time = (
            time.perf_counter()
            - start_time
        )

        total_count = sum(
            vehicle_counts.values()
        )

        # -----------------------------------------------------
        # Return statistics
        # -----------------------------------------------------

        return {
            "output_path": output_path,
            "total_frames": frame_count,
            "fps": fps,
            "processing_time": processing_time,
            "total_vehicles": total_count,
            "vehicle_counts": dict(
                vehicle_counts
            ),
            "unique_tracking_ids": len(
                counted_ids
            ),
            "counted_ids": sorted(
                list(counted_ids)
            ),
            "classes_detected": sorted(
                list(detected_classes)
            ),
            "up_counts": dict(
                up_counts
            ),
            "down_counts": dict(
                down_counts
            ),
            "counting_line_y": line_y,
        }