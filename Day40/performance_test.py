from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from video_analytics import VideoConfig, process_video


def run_test(video_path: Path, output_dir: Path, image_size: int, frame_skip: int) -> dict:
    test_dir = output_dir / f"benchmark_{image_size}_skip_{frame_skip}"
    test_dir.mkdir(parents=True, exist_ok=True)

    config = VideoConfig(
        conf=0.40,
        iou=0.50,
        imgsz=image_size,
        frame_skip=frame_skip,
        tracker="ByteTrack",
        show_trails=False,
    )

    start = perf_counter()

    _, _, summary = process_video(
        video_path,
        test_dir,
        config,
    )

    wall_time = perf_counter() - start

    return {
        "video": video_path.name,
        "image_size": image_size,
        "frame_skip": frame_skip,
        "average_fps": summary["average_fps"],
        "processing_time_s": wall_time,
        "total_objects": summary["total_objects"],
        "entries": summary["total_entries"],
        "exits": summary["total_exits"],
        "max_roi": summary["maximum_objects_in_roi"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 40 performance benchmark."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = [
        (640, 1),
        (480, 1),
        (640, 2),
        (480, 2),
    ]

    results = []

    for image_size, frame_skip in configurations:
        print(
            f"\nRunning image_size={image_size}, "
            f"frame_skip={frame_skip}"
        )

        result = run_test(
            video_path,
            output_dir,
            image_size,
            frame_skip,
        )

        results.append(result)

        print(
            f"FPS={result['average_fps']:.2f} | "
            f"time={result['processing_time_s']:.2f}s"
        )

    df = pd.DataFrame(results).sort_values(
        "average_fps",
        ascending=False,
    )

    comparison = output_dir / "performance_comparison.csv"
    df.to_csv(comparison, index=False)

    print("\nPerformance Comparison")
    print(df.to_string(index=False))

    best = df.iloc[0]

    print("\nBest configuration:")
    print(f"Image size: {int(best['image_size'])}")
    print(f"Frame skip: {int(best['frame_skip'])}")
    print(f"Average FPS: {best['average_fps']:.2f}")
    print(f"Processing time: {best['processing_time_s']:.2f}s")
    print(f"Saved to: {comparison}")


if __name__ == "__main__":
    main()
