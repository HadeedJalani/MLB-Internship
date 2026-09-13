from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from segmentation import METHODS, segment_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Binary, Adaptive and Otsu segmentation to every image in a folder."
    )
    parser.add_argument("--input-dir", default="sample_input_images")
    parser.add_argument("--output-dir", default="outputs/segmentation")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--block-size", type=int, default=11)
    parser.add_argument("--c", type=int, default=2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in {input_dir}")
        return

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        for method in METHODS:
            result = segment_image(
                image,
                method,
                threshold=args.threshold,
                block_size=args.block_size,
                c_value=args.c,
            )
            output_path = output_dir / f"{image_path.stem}_{method.lower()}.png"
            cv2.imwrite(str(output_path), result)
            print(f"{image_path.name} | {method} -> {output_path}")


if __name__ == "__main__":
    main()
