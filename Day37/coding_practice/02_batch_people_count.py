from __future__ import annotations

import argparse
from pathlib import Path

from people_counter import CountingLine, load_model, process_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process all people videos in a folder."
    )
    parser.add_argument("--input-dir", default="sample_videos")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--line-position", type=float, default=0.50)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )

    if not videos:
        print(f"No videos found in {input_dir}")
        return

    model = load_model()
    line = CountingLine("horizontal", args.line_position)

    for video in videos:
        output = output_dir / f"{video.stem}_people_counted.mp4"
        print(f"\nProcessing {video.name}")

        result = process_video(
            model=model,
            in_path=video,
            out_path=output,
            line=line,
        )

        print(f"  peak={result.peak_count}")
        print(f"  unique_seen={result.total_people_seen}")
        print(f"  crossings={result.line_crossings}")
        print(f"  entries={result.entries}")
        print(f"  exits={result.exits}")
        print(f"  saved={result.out_path}")


if __name__ == "__main__":
    main()
