# ==========================================================
# MLB Summer Internship - Day 20
# Real-Time Webcam Processing
# ==========================================================

import cv2
import time

from video_operations import (
    convert_to_grayscale,
    apply_gaussian_blur,
    apply_canny,
)


# ==========================================================
# Configuration
# ==========================================================

CAMERA_INDEX = 0

BLUR_KERNEL = 15
CANNY_LOWER = 50
CANNY_UPPER = 150


# ==========================================================
# FPS Counter
# ==========================================================

class FPSCounter:

    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def update(self):

        self.frame_count += 1

        elapsed = (
            time.time()
            - self.start_time
        )

        if elapsed > 0:

            self.fps = (
                self.frame_count
                / elapsed
            )

        return self.fps


# ==========================================================
# Add Information Overlay
# ==========================================================

def add_overlay(
    frame,
    fps,
    mode
):
    """
    Add FPS and processing information
    to the webcam frame.
    """

    result = frame.copy()

    # ------------------------------------------------------
    # Background panel
    # ------------------------------------------------------

    cv2.rectangle(
        result,
        (10, 10),
        (330, 90),
        (0, 0, 0),
        -1
    )

    # ------------------------------------------------------
    # FPS
    # ------------------------------------------------------

    cv2.putText(
        result,
        f"FPS: {fps:.1f}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # ------------------------------------------------------
    # Processing mode
    # ------------------------------------------------------

    cv2.putText(
        result,
        f"Mode: {mode}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return result


# ==========================================================
# Process Webcam
# ==========================================================

def main():

    print("=" * 65)
    print("Day 20 - Real-Time Webcam Processing")
    print("=" * 65)

    print()
    print("Starting webcam...")
    print()
    print("Controls:")
    print("  G = Grayscale")
    print("  B = Gaussian Blur")
    print("  C = Canny Edge Detection")
    print("  O = Original")
    print("  Q = Quit")
    print()

    # ------------------------------------------------------
    # Open webcam
    # ------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "ERROR: Unable to access webcam."
        )

        return

    # ------------------------------------------------------
    # Set webcam resolution
    # ------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    fps_counter = FPSCounter()

    processing_mode = "canny"

    # ------------------------------------------------------
    # Main webcam loop
    # ------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Unable to read webcam frame."
            )

            break

        # --------------------------------------------------
        # FPS
        # --------------------------------------------------

        current_fps = (
            fps_counter.update()
        )

        # --------------------------------------------------
        # Processing
        # --------------------------------------------------

        grayscale = convert_to_grayscale(
            frame
        )

        if processing_mode == "original":

            processed = frame.copy()

        elif processing_mode == "grayscale":

            processed = grayscale

        elif processing_mode == "blur":

            processed = apply_gaussian_blur(
                frame,
                BLUR_KERNEL
            )

        elif processing_mode == "canny":

            blurred = apply_gaussian_blur(
                grayscale,
                BLUR_KERNEL
            )

            processed = apply_canny(
                blurred,
                CANNY_LOWER,
                CANNY_UPPER
            )

        else:

            processed = frame.copy()

        # --------------------------------------------------
        # Convert grayscale result for display
        # --------------------------------------------------

        if len(processed.shape) == 2:

            processed_display = cv2.cvtColor(
                processed,
                cv2.COLOR_GRAY2BGR
            )

        else:

            processed_display = processed

        # --------------------------------------------------
        # Overlay
        # --------------------------------------------------

        processed_display = add_overlay(
            processed_display,
            current_fps,
            processing_mode.upper()
        )

        # --------------------------------------------------
        # Display
        # --------------------------------------------------

        cv2.imshow(
            "Day 20 - Webcam Processing",
            processed_display
        )

        # --------------------------------------------------
        # Keyboard controls
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == ord("g"):

            processing_mode = "grayscale"

        elif key == ord("b"):

            processing_mode = "blur"

        elif key == ord("c"):

            processing_mode = "canny"

        elif key == ord("o"):

            processing_mode = "original"

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()

    print()
    print(
        "Webcam processing stopped."
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()