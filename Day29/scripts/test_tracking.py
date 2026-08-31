import os
import sys


# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.append(PROJECT_ROOT)


from tracker.object_tracker import ObjectTracker


INPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "sample_inputs",
    "videos",
)

OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "tracked_videos",
)


def main():

    print("\n" + "=" * 70)
    print("SMART OBJECT TRACKING SYSTEM")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Find videos
    # -----------------------------------------------------

    video_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    )

    videos = [

        file

        for file in os.listdir(
            INPUT_DIRECTORY
        )

        if file.lower().endswith(
            video_extensions
        )

    ]

    if not videos:

        print(
            "\nNo videos found."
        )

        print(
            f"Add videos to:\n"
            f"{INPUT_DIRECTORY}"
        )

        return

    print(
        f"\nFound {len(videos)} video(s)."
    )

    # -----------------------------------------------------
    # Initialize tracker
    # -----------------------------------------------------

    tracker = ObjectTracker(
        model_path="yolo11n.pt",

        # Lower threshold improves detection
        confidence=0.25,

        tracker="ByteTrack",
    )

    # -----------------------------------------------------
    # Process videos
    # -----------------------------------------------------

    for index, video_name in enumerate(
        videos,
        start=1,
    ):

        input_path = os.path.join(
            INPUT_DIRECTORY,
            video_name,
        )

        output_name = (
            f"{os.path.splitext(video_name)[0]}"
            f"_tracked.mp4"
        )

        output_path = os.path.join(
            OUTPUT_DIRECTORY,
            output_name,
        )

        print("\n" + "-" * 70)

        print(
            f"Processing Video {index}/{len(videos)}"
        )

        print(
            f"File: {video_name}"
        )

        print("-" * 70)

        try:

            def progress_callback(progress):

                percentage = progress * 100

                print(
                    f"\rProgress: "
                    f"{percentage:6.2f}%",
                    end="",
                    flush=True,
                )

            stats = tracker.process_video(
                input_path=input_path,
                output_path=output_path,
                progress_callback=progress_callback,
            )

            print("\n")

            print("=" * 70)
            print("TRACKING COMPLETED")
            print("=" * 70)

            print(
                f"Frames Processed : "
                f"{stats['total_frames']}"
            )

            print(
                f"Unique Objects   : "
                f"{stats['unique_objects']}"
            )

            print(
                f"Classes Detected : "
                f"{', '.join(stats['classes_detected'])}"
            )

            print(
                f"Total Detections : "
                f"{stats['total_detections']}"
            )

            print(
                f"Processing Time  : "
                f"{stats['processing_time']:.2f} sec"
            )

            print(
                f"Saved Output     : "
                f"{stats['output_path']}"
            )

            # -------------------------------------------------
            # Per-class statistics
            # -------------------------------------------------

            print("\nPER-CLASS TRACKING SUMMARY")

            print("-" * 70)

            if (
                stats[
                    "unique_objects_per_class"
                ]
            ):

                for (
                    class_name,
                    count,
                ) in sorted(
                    stats[
                        "unique_objects_per_class"
                    ].items()
                ):

                    print(
                        f"{class_name:<20}"
                        f"Unique IDs: {count}"
                    )

            # -------------------------------------------------
            # Detection frequency
            # -------------------------------------------------

            print(
                "\nDETECTION FREQUENCY"
            )

            print("-" * 70)

            for (
                class_name,
                count,
            ) in sorted(
                stats[
                    "class_detection_counts"
                ].items(),
                key=lambda item: item[1],
                reverse=True,
            ):

                print(
                    f"{class_name:<20}"
                    f"Detections: {count}"
                )

        except Exception as error:

            print(
                f"\nError processing "
                f"{video_name}"
            )

            print(error)

    print("\n" + "=" * 70)

    print(
        "ALL VIDEO TRACKING TESTS COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()