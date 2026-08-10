# ==========================================================
# MLB Summer Internship - Day 17
# Image Transformations using OpenCV
# ==========================================================

import cv2
import os
import numpy as np


# ==========================================================
# Configuration
# ==========================================================

IMAGE_PATH = "input_images/document_01.jpg"
OUTPUT_FOLDER = "output_images/transformations"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================================
# Load Image
# ==========================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}\n"
        "Please place a document image inside input_images."
    )

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("OpenCV could not read the image.")

height, width = image.shape[:2]

print("=" * 70)
print("Day 17 - Image Transformations")
print("=" * 70)
print(f"Input Image : {IMAGE_PATH}")
print(f"Image Size  : {width} x {height}")


# ==========================================================
# 1. Translation
# ==========================================================

print("\n[1] Applying Translation...")

translation_matrix = np.float32([
    [1, 0, 100],
    [0, 1, 50]
])

translated = cv2.warpAffine(
    image,
    translation_matrix,
    (width, height)
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "translation.jpg"),
    translated
)

print("✓ Translation completed")


# ==========================================================
# 2. Rotation
# ==========================================================

print("\n[2] Applying Rotation...")

center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    30,
    1.0
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "rotation_30_degree.jpg"),
    rotated
)

print("✓ Rotation completed")


# ==========================================================
# 3. Scaling
# ==========================================================

print("\n[3] Applying Scaling...")

scaled_up = cv2.resize(
    image,
    None,
    fx=1.5,
    fy=1.5,
    interpolation=cv2.INTER_LINEAR
)

scaled_down = cv2.resize(
    image,
    None,
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_AREA
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "scaled_up.jpg"),
    scaled_up
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "scaled_down.jpg"),
    scaled_down
)

print("✓ Scaling completed")


# ==========================================================
# 4. Affine Transformation
# ==========================================================

print("\n[4] Applying Affine Transformation...")

source_points = np.float32([
    [0, 0],
    [width - 1, 0],
    [0, height - 1]
])

destination_points = np.float32([
    [0, 0],
    [width - 1, 50],
    [50, height - 1]
])

affine_matrix = cv2.getAffineTransform(
    source_points,
    destination_points
)

affine = cv2.warpAffine(
    image,
    affine_matrix,
    (width, height)
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "affine_transformation.jpg"),
    affine
)

print("✓ Affine transformation completed")


# ==========================================================
# 5. Perspective Transformation
# ==========================================================

print("\n[5] Applying Perspective Transformation...")

# Source points represent the four corners
# of the document/image.
perspective_source = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

perspective_destination = np.float32([
    [40, 40],
    [width - 40, 20],
    [width - 20, height - 40],
    [20, height - 20]
])

perspective_matrix = cv2.getPerspectiveTransform(
    perspective_source,
    perspective_destination
)

perspective = cv2.warpPerspective(
    image,
    perspective_matrix,
    (width, height)
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "perspective_transformation.jpg"),
    perspective
)

print("✓ Perspective transformation completed")


# ==========================================================
# Display Results
# ==========================================================

cv2.imshow("Original", image)
cv2.imshow("Translated", translated)
cv2.imshow("Rotated", rotated)
cv2.imshow("Affine Transformation", affine)
cv2.imshow("Perspective Transformation", perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ==========================================================
# Completion
# ==========================================================

print("\n" + "=" * 70)
print("All image transformations completed successfully.")
print(f"Results saved to: {OUTPUT_FOLDER}")
print("=" * 70)