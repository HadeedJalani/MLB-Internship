from __future__ import annotations

import argparse
from pathlib import Path

from people_counter import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    CountingLine,
    ROI,
    load_model,
    process_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 + ByteTrack people counting on one video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="outputs/people_counted.mp4")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument(
        "--line-orientation",
        choices=["horizontal", "vertical"],
        default="horizontal",
    )
    parser.add_argument("--line-position", type=float, default=0.50)
    parser.add_argument("--no-line", action="store_true")
    parser.add_argument("--roi", action="store_true")
    args = parser.parse_args()

    model = load_model()

    line = None
    if not args.no_line:
        line = CountingLine(args.line_orientation, args.line_position)

    roi = ROI(0.10, 0.90, 0.10, 0.90) if args.roi else None

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = process_video(
        model=model,
        in_path=args.video,
        out_path=output_path,
        conf=args.conf,
        iou=args.iou,
        line=line,
        roi=roi,
    )

    print(f"Frames processed: {result.n_frames}")
    print(f"Peak people: {result.peak_count}")
    print(f"Total unique people seen: {result.total_people_seen}")
    print(f"Line crossings: {result.line_crossings}")
    print(f"Entries: {result.entries}")
    print(f"Exits: {result.exits}")
    print(f"Saved: {result.out_path}")


if __name__ == "__main__":
    main()
