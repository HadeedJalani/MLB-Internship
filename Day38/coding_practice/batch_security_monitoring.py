from __future__ import annotations

import argparse
import json

from monitoring import process_video_folder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process ALL videos in a folder with YOLO + ByteTrack + ROI events."
    )
    parser.add_argument("--input-dir", default="sample_videos")
    parser.add_argument("--output-dir", default="outputs/security_monitoring")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--process-every", type=int, default=1)
    parser.add_argument("--min-stable", type=int, default=3)
    parser.add_argument(
        "--rois",
        default='[{"name":"Main Entrance","points":[[80,80],[1180,80],[1180,620],[80,620]]}]',
        help="JSON list of polygon ROIs.",
    )
    args = parser.parse_args()

    rois = json.loads(args.rois)
    results = process_video_folder(
        args.input_dir,
        rois,
        output_dir=args.output_dir,
        conf=args.conf,
        process_every=args.process_every,
        min_stable_frames=args.min_stable,
    )

    print("\nBatch processing completed.")
    for result in results:
        print(
            f"{result['input_video']} -> "
            f"unique={result['unique_people']}, "
            f"entries={result['entries']}, "
            f"exits={result['exits']}, "
            f"max_active={result['max_active']}"
        )
        print(f"  video: {result['video_path']}")
        print(f"  csv:   {result['csv_path']}")


if __name__ == "__main__":
    main()
