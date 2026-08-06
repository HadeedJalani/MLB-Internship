# ==========================================================
# MLB Summer Internship - Day 15
# Object Detection Project using YOLOv8
# ==========================================================

import os
import shutil
from collections import Counter

from ultralytics import YOLO

print("=" * 70)
print("OBJECT DETECTION USING YOLOv8")
print("=" * 70)

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "yolov8n.pt"

IMAGE_FOLDER = "dataset/test/images"

OUTPUT_FOLDER = "output_images"

CONFIDENCE = 0.25

IMAGE_SIZE = 640

# ==========================================================
# Check Dataset
# ==========================================================

if not os.path.exists(IMAGE_FOLDER):

    raise FileNotFoundError(
        f"\nFolder not found:\n{IMAGE_FOLDER}"
    )

print("✓ Dataset Found")

# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ==========================================================
# Load Model
# ==========================================================

print("\nLoading YOLOv8 Model...")

model = YOLO(MODEL_PATH)

print("✓ Model Loaded Successfully")

# ==========================================================
# Run Inference
# ==========================================================

print("\nRunning Object Detection...\n")

results = model.predict(

    source=IMAGE_FOLDER,

    conf=CONFIDENCE,

    imgsz=IMAGE_SIZE,

    save=True,

    project=".",

    name="temp_predictions",

    exist_ok=True,

    verbose=False

)

print("✓ Detection Completed")

# ==========================================================
# Copy Output Images
# ==========================================================

prediction_folder = os.path.join(
        "runs",
    "detect",
    "temp_predictions"
)

for file in os.listdir(prediction_folder):

    src = os.path.join(
        prediction_folder,
        file
    )

    dst = os.path.join(
        OUTPUT_FOLDER,
        file
    )

    shutil.copy(src, dst)

print("✓ Output Images Saved")

# ==========================================================
# Detection Results
# ==========================================================

total_images = len(results)

total_objects = 0

object_counter = Counter()

print("\n" + "=" * 70)
print("DETECTION RESULTS")
print("=" * 70)

for result in results:

    image_name = os.path.basename(result.path)

    print(f"\nImage : {image_name}")

    print("-" * 60)

    if len(result.boxes) == 0:

        print("No objects detected.")

        continue

    for i, box in enumerate(result.boxes, start=1):

        cls = int(box.cls)

        class_name = model.names[cls]

        confidence = float(box.conf)

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        total_objects += 1

        object_counter[class_name] += 1

        print(f"Object {i}")

        print(f"Class      : {class_name}")

        print(f"Confidence : {confidence:.2f}")

        print(
            f"Bounding Box : "
            f"({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})"
        )

        print()

# ==========================================================
# Project Summary
# ==========================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Images Processed : {total_images}")

print(f"Objects Detected : {total_objects}")

print("\nDetected Objects")

print("-" * 30)

for name, count in sorted(object_counter.items()):

    print(f"{name:<20}{count}")

# ==========================================================
# Output Information
# ==========================================================

print("\nAnnotated images saved in:")

print(OUTPUT_FOLDER)

# ==========================================================
# Clean Temporary Folder
# ==========================================================

if os.path.exists(prediction_folder):
    shutil.rmtree(prediction_folder)

print("\nTemporary files removed.")

print("\n" + "=" * 70)
print("OBJECT DETECTION PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)