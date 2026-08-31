"""
Run inference on a folder of test images and save annotated results with
per-detection confidence scores printed to the console.

Usage:
    python scripts/test_inference.py --source sample_inputs/images --weights models/best.pt
"""

import argparse
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a folder of images.")
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained weights (best.pt)")
    parser.add_argument("--source", required=True, help="Path to an image, video, or folder of images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--out", default="outputs/predictions", help="Where to save annotated results")
    args = parser.parse_args()

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=os.path.dirname(args.out) or ".",
        name=os.path.basename(args.out),
        exist_ok=True,
    )

    total_detections = 0
    for r in results:
        fname = os.path.basename(r.path)
        boxes = r.boxes
        total_detections += len(boxes)
        print(f"\n{fname}: {len(boxes)} detections")
        for box in boxes:
            class_name = model.names[int(box.cls)]
            print(f"   {class_name}: {float(box.conf):.2f}")

    print(f"\nTotal: {total_detections} detections across {len(results)} files.")
    print(f"Annotated results saved under: {args.out}")


if __name__ == "__main__":
    main()