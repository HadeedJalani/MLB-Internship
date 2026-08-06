# ==========================================================
# MLB Summer Internship - Day 15
# YOLO Practice
# Object Detection using Pretrained YOLOv8
# ==========================================================

from ultralytics import YOLO
import os

print("=" * 70)
print("YOLO Object Detection Practice")
print("=" * 70)

# ==========================================================
# Load Model
# ==========================================================

print("\nLoading YOLOv8 Nano Model...")

model = YOLO("yolov8n.pt")

print("✓ Model Loaded Successfully")

# ==========================================================
# Input Folder
# ==========================================================

IMAGE_FOLDER = "dataset/test/images"

if not os.path.exists(IMAGE_FOLDER):

    raise FileNotFoundError(
        f"\nFolder not found:\n{IMAGE_FOLDER}"
    )

print(f"\nInput Folder : {IMAGE_FOLDER}")

# ==========================================================
# Perform Detection
# ==========================================================

print("\nRunning Object Detection...")

results = model.predict(

    source=IMAGE_FOLDER,

    save=True,

    conf=0.25,

    imgsz=640,

    show=False

)

print("✓ Detection Completed")

# ==========================================================
# Display Results
# ==========================================================

print("\n" + "=" * 70)
print("DETECTION RESULTS")
print("=" * 70)

for result in results:

    print(f"\nImage : {os.path.basename(result.path)}")
    print("-" * 50)

    if len(result.boxes) == 0:

        print("No objects detected.")
        continue

    for i, box in enumerate(result.boxes, start=1):

        cls = int(box.cls)

        confidence = float(box.conf)

        class_name = model.names[cls]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f"Object {i}")
        print(f"Class      : {class_name}")
        print(f"Confidence : {confidence:.2f}")
        print(
            f"Bounding Box : "
            f"({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})"
        )
        print()

# ==========================================================
# Output Information
# ==========================================================

print("=" * 70)

print("Prediction images saved successfully.")

print("Output Folder:")

print("runs/detect/predict")

print("=" * 70)

print("YOLO Practice Completed Successfully!")

print("=" * 70)