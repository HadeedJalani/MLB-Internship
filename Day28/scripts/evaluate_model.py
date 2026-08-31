"""
Re-run evaluation on the trained model against the dataset's validation split.
Usage:
    python scripts/evaluate_model.py --data path/to/data.yaml --weights models/best.pt
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained weights (best.pt)")
    parser.add_argument("--data", required=True, help="Path to the dataset's data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate on")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split, plots=True)

    print("\n=== Evaluation Results ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("\nPlots (confusion matrix, PR curve, etc.) saved under the run directory printed above.")


if __name__ == "__main__":
    main()