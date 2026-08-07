# ==========================================================
# MLB Summer Internship - Day 16
# Drawing Shapes and Adding Text using OpenCV
# ==========================================================

import cv2
import os
from datetime import datetime

print("=" * 70)
print("Drawing Shapes using OpenCV")
print("=" * 70)

# ==========================================================
# Paths
# ==========================================================

IMAGE_PATH = "input_images/landscape.jpg"
OUTPUT_FOLDER = "output_images"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"\nImage not found:\n{IMAGE_PATH}")

# ==========================================================
# Read Image
# ==========================================================

image = cv2.imread(IMAGE_PATH)

canvas = image.copy()

height, width = canvas.shape[:2]

# ==========================================================
# Draw Rectangle
# ==========================================================

cv2.rectangle(

    canvas,

    (50, 50),

    (300, 220),

    (0, 255, 0),

    3

)

# ==========================================================
# Draw Circle
# ==========================================================

cv2.circle(

    canvas,

    (width // 2, height // 2),

    100,

    (255, 0, 0),

    4

)

# ==========================================================
# Draw Line
# ==========================================================

cv2.line(

    canvas,

    (0, 0),

    (width, height),

    (0, 0, 255),

    3

)

import numpy as np

# ==========================================================
# Draw Polygon
# ==========================================================

h, w = canvas.shape[:2]

points = np.array([
    [int(w*0.70), int(h*0.15)],
    [int(w*0.90), int(h*0.30)],
    [int(w*0.82), int(h*0.55)],
    [int(w*0.60), int(h*0.50)],
    [int(w*0.55), int(h*0.25)]
], dtype=np.int32)

points = points.reshape((-1,1,2))

cv2.polylines(
    canvas,
    [points],
    True,
    (0,255,255),
    3
)
# ==========================================================
# Add Custom Text
# ==========================================================

today = datetime.now().strftime("%d-%m-%Y")

cv2.putText(

    canvas,

    "Hadeed Jalani",

    (40, height - 70),

    cv2.FONT_HERSHEY_SIMPLEX,

    1,

    (255, 255, 255),

    2

)

cv2.putText(

    canvas,

    f"Date : {today}",

    (40, height - 30),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.8,

    (0, 255, 255),

    2

)

# ==========================================================
# Save
# ==========================================================

draw_folder = os.path.join(OUTPUT_FOLDER, "drawings")

os.makedirs(draw_folder, exist_ok=True)

output_path = os.path.join(

    draw_folder,

    "drawing_shapes.jpg"

)

cv2.imwrite(

    output_path,

    canvas

)

# ==========================================================
# Display
# ==========================================================

cv2.imshow(

    "Drawing Shapes",

    canvas

)

cv2.waitKey(0)

cv2.destroyAllWindows()

# ==========================================================
# Finish
# ==========================================================

print("\nOutput saved to:")

print(output_path)

print("\n" + "=" * 70)
print("Drawing Shapes Completed Successfully!")
print("=" * 70)