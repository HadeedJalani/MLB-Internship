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


from counter.vehicle_counter import VehicleCounter


INPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "sample_inputs",
    "videos",
)

OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "counted_videos",
)


def main():

    print("\n" + "=" * 70)

    print(
        "SMART VEHICLE COUNTING SYSTEM"
    )

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
            "\nNo traffic videos found."
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
    # Initialize vehicle counter
    # -----------------------------------------------------

    counter = VehicleCounter(
        model_path="yolo11n.pt",
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
            f"_counted.mp4"
        )

        output_path = os.path.join(
            OUTPUT_DIRECTORY,
            output_name,
        )

        print("\n" + "-" * 70)

        print(
            f"Processing Video "
            f"{index}/{len(videos)}"
        )

        print(
            f"Video: {video_name}"
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

            stats = counter.process_video(
                input_path=input_path,
                output_path=output_path,
                progress_callback=progress_callback,
            )

            print("\n")

            print("=" * 70)
            print("VEHICLE COUNTING COMPLETED")
            print("=" * 70)

            print(
                f"Frames Processed : "
                f"{stats['total_frames']}"
            )

            print(
                f"Total Vehicles   : "
                f"{stats['total_vehicles']}"
            )

            print(
                f"Tracking IDs     : "
                f"{stats['unique_tracking_ids']}"
            )

            print(
                f"Classes Detected : "
                f"{', '.join(stats['classes_detected'])}"
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
            # Vehicle counts
            # -------------------------------------------------

            print("\nVEHICLE COUNT SUMMARY")

            print("-" * 70)

            for vehicle in [
                "car",
                "bus",
                "truck",
                "motorcycle",
            ]:

                count = (
                    stats["vehicle_counts"]
                    .get(vehicle, 0)
                )

                print(
                    f"{vehicle.title():<15}: "
                    f"{count}"
                )

            # -------------------------------------------------
            # Direction counts
            # -------------------------------------------------

            print(
                "\nDIRECTION-WISE COUNT"
            )

            print("-" * 70)

            print("UPWARD MOVEMENT:")

            for vehicle, count in (
                stats["up_counts"].items()
            ):

                print(
                    f"  {vehicle:<15}: "
                    f"{count}"
                )

            print("\nDOWNWARD MOVEMENT:")

            for vehicle, count in (
                stats["down_counts"].items()
            ):

                print(
                    f"  {vehicle:<15}: "
                    f"{count}"
                )

        except Exception as error:

            print(
                f"\nError processing "
                f"{video_name}"
            )

            print(error)

    print("\n" + "=" * 70)

    print(
        "ALL VEHICLE COUNTING TESTS COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()