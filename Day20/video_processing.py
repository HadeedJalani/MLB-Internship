# ==========================================================
# MLB Summer Internship - Day 20
# Recorded Video Processing
# ==========================================================

import os
import cv2

from video_operations import (
    get_video_info,
    process_video,
)


# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = "input_videos"
OUTPUT_FOLDER = "output_videos"

DEFAULT_MODE = "canny"

BLUR_KERNEL = 5
CANNY_LOWER = 50
CANNY_UPPER = 150


# ==========================================================
# Progress Display
# ==========================================================

def show_progress(progress):
    """
    Display processing progress in the terminal.
    """

    percentage = int(progress * 100)

    print(
        f"\rProcessing video: {percentage}%",
        end=""
    )


# ==========================================================
# Process One Video
# ==========================================================

def process_one_video(filename):
    """
    Process a single video from the input folder.
    """

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    base_name = os.path.splitext(
        filename
    )[0]

    output_filename = (
        f"{base_name}_processed.mp4"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )

    print()
    print("=" * 65)
    print(f"Processing: {filename}")
    print("=" * 65)

    # ------------------------------------------------------
    # Video Information
    # ------------------------------------------------------

    info = get_video_info(
        input_path
    )

    print(
        f"FPS          : {info['fps']}"
    )

    print(
        f"Resolution   : "
        f"{info['width']} x {info['height']}"
    )

    print(
        f"Total Frames : "
        f"{info['total_frames']}"
    )

    print(
        f"Duration     : "
        f"{info['duration']} seconds"
    )

    print(
        f"Processing   : "
        f"{DEFAULT_MODE.upper()}"
    )

    print()

    # ------------------------------------------------------
    # Process Video
    # ------------------------------------------------------

    result = process_video(
        input_path=input_path,
        output_path=output_path,
        mode=DEFAULT_MODE,
        blur_kernel=BLUR_KERNEL,
        canny_lower=CANNY_LOWER,
        canny_upper=CANNY_UPPER,
        progress_callback=show_progress
    )

    print()

    print(
        f"Processed Frames: "
        f"{result['processed_frames']}"
    )

    print(
        f"Output saved to: "
        f"{result['output_path']}"
    )


# ==========================================================
# Main Program
# ==========================================================

def main():

    print("=" * 65)
    print("Day 20 - Recorded Video Processing")
    print("=" * 65)

    # ------------------------------------------------------
    # Create folders
    # ------------------------------------------------------

    os.makedirs(
        INPUT_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Find videos
    # ------------------------------------------------------

    video_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    )

    video_files = [
        file
        for file in os.listdir(
            INPUT_FOLDER
        )
        if file.lower().endswith(
            video_extensions
        )
    ]

    # ------------------------------------------------------
    # No videos
    # ------------------------------------------------------

    if not video_files:

        print()
        print(
            "No videos found."
        )

        print(
            f"Place your videos inside: "
            f"{INPUT_FOLDER}"
        )

        print()

        return

    # ------------------------------------------------------
    # Display number of videos
    # ------------------------------------------------------

    print()
    print(
        f"Videos found: "
        f"{len(video_files)}"
    )

    # ------------------------------------------------------
    # Process videos
    # ------------------------------------------------------

    for filename in sorted(
        video_files
    ):

        try:

            process_one_video(
                filename
            )

        except Exception as error:

            print()
            print(
                f"ERROR processing "
                f"{filename}:"
            )

            print(error)

    # ------------------------------------------------------
    # Completion
    # ------------------------------------------------------

    print()
    print("=" * 65)
    print(
        "Video processing completed successfully."
    )
    print("=" * 65)

    print()
    print(
        f"Processed videos saved in: "
        f"{OUTPUT_FOLDER}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()