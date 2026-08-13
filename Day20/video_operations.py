# ==========================================================
# MLB Summer Internship - Day 20
# Video Processing Operations
# ==========================================================

import cv2
import os


# ==========================================================
# Video Information
# ==========================================================

def get_video_info(video_path):
    """
    Extract basic video properties.

    Returns:
        dictionary containing:
        - FPS
        - width
        - height
        - total frames
        - duration
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps > 0:
        duration = total_frames / fps
    else:
        duration = 0

    capture.release()

    return {
        "fps": round(float(fps), 2),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": round(duration, 2),
    }


# ==========================================================
# Frame Processing
# ==========================================================

def convert_to_grayscale(frame):
    """
    Convert a video frame to grayscale.
    """

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


def apply_gaussian_blur(
    frame,
    kernel_size=5
):
    """
    Apply Gaussian blur.

    Kernel size must be a positive odd number.
    """

    if kernel_size < 3:
        kernel_size = 3

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(
        frame,
        (kernel_size, kernel_size),
        0
    )


def apply_canny(
    frame,
    lower_threshold=50,
    upper_threshold=150
):
    """
    Apply Canny edge detection.
    """

    return cv2.Canny(
        frame,
        lower_threshold,
        upper_threshold
    )


# ==========================================================
# Complete Frame Pipeline
# ==========================================================

def process_frame(
    frame,
    mode="canny",
    blur_kernel=5,
    canny_lower=50,
    canny_upper=150
):
    """
    Process one video frame.

    Supported modes:

        original
        grayscale
        blur
        canny
    """

    if frame is None:
        return None

    if mode == "original":

        return frame.copy()

    grayscale = convert_to_grayscale(
        frame
    )

    if mode == "grayscale":

        return grayscale

    blurred = apply_gaussian_blur(
        grayscale,
        blur_kernel
    )

    if mode == "blur":

        return blurred

    if mode == "canny":

        edges = apply_canny(
            blurred,
            canny_lower,
            canny_upper
        )

        return edges

    raise ValueError(
        f"Unsupported processing mode: {mode}"
    )


# ==========================================================
# Prepare Frame For Video Writer
# ==========================================================

def prepare_for_video_writer(
    frame,
    width,
    height
):
    """
    Convert processed frames into BGR format
    so they can safely be written to a video.
    """

    if frame is None:
        return None

    frame = cv2.resize(
        frame,
        (width, height)
    )

    if len(frame.shape) == 2:

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_GRAY2BGR
        )

    return frame


# ==========================================================
# Process Recorded Video
# ==========================================================

def process_video(
    input_path,
    output_path,
    mode="canny",
    blur_kernel=5,
    canny_lower=50,
    canny_upper=150,
    progress_callback=None
):
    """
    Process a recorded video frame by frame.

    Returns:
        video information and processing statistics.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    capture = cv2.VideoCapture(
        input_path
    )

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {input_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30.0

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # ------------------------------------------------------
    # Video Writer
    # ------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()

        raise ValueError(
            f"Unable to create output video: {output_path}"
        )

    processed_frames = 0

    # ------------------------------------------------------
    # Frame Processing Loop
    # ------------------------------------------------------

    while True:

        success, frame = capture.read()

        if not success:
            break

        processed = process_frame(
            frame,
            mode=mode,
            blur_kernel=blur_kernel,
            canny_lower=canny_lower,
            canny_upper=canny_upper
        )

        processed = prepare_for_video_writer(
            processed,
            width,
            height
        )

        writer.write(
            processed
        )

        processed_frames += 1

        # --------------------------------------------------
        # Optional progress callback
        # --------------------------------------------------

        if progress_callback:

            if total_frames > 0:

                progress = (
                    processed_frames
                    / total_frames
                )

                progress_callback(
                    progress
                )

    capture.release()
    writer.release()

    return {
        "fps": round(float(fps), 2),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "output_path": output_path,
    }


# ==========================================================
# Frame Generator
# ==========================================================

def frame_generator(
    video_path
):
    """
    Generate video frames one at a time.

    Useful for Streamlit previews.
    """

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {video_path}"
        )

    while True:

        success, frame = capture.read()

        if not success:
            break

        yield frame

    capture.release()