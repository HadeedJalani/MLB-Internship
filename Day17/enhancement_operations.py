# ==========================================================
# MLB Summer Internship - Day 17
# Image Enhancement using OpenCV
# ==========================================================

import cv2
import os
import numpy as np


# ==========================================================
# Configuration
# ==========================================================

IMAGE_PATH = "input_images/document_01.jpg"
OUTPUT_FOLDER = "output_images/enhancement"

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

print("=" * 70)
print("Day 17 - Image Enhancement")
print("=" * 70)
print(f"Input Image : {IMAGE_PATH}")


# ==========================================================
# 1. Grayscale Conversion
# ==========================================================

print("\n[1] Converting image to grayscale...")

grayscale = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "grayscale.jpg"),
    grayscale
)

print("✓ Grayscale conversion completed")


# ==========================================================
# 2. Brightness Adjustment
# ==========================================================

print("\n[2] Adjusting brightness...")

brightness_increased = cv2.convertScaleAbs(
    image,
    alpha=1.0,
    beta=50
)

brightness_decreased = cv2.convertScaleAbs(
    image,
    alpha=1.0,
    beta=-50
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "brightness_increased.jpg"),
    brightness_increased
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "brightness_decreased.jpg"),
    brightness_decreased
)

print("✓ Brightness adjustment completed")


# ==========================================================
# 3. Contrast Adjustment
# ==========================================================

print("\n[3] Adjusting contrast...")

contrast_increased = cv2.convertScaleAbs(
    image,
    alpha=1.5,
    beta=0
)

contrast_decreased = cv2.convertScaleAbs(
    image,
    alpha=0.7,
    beta=0
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "contrast_increased.jpg"),
    contrast_increased
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "contrast_decreased.jpg"),
    contrast_decreased
)

print("✓ Contrast adjustment completed")


# ==========================================================
# 4. Gaussian Blur
# ==========================================================

print("\n[4] Applying Gaussian Blur...")

gaussian_blur = cv2.GaussianBlur(
    image,
    (5, 5),
    0
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "gaussian_blur.jpg"),
    gaussian_blur
)

print("✓ Gaussian Blur completed")


# ==========================================================
# 5. Median Blur
# ==========================================================

print("\n[5] Applying Median Blur...")

median_blur = cv2.medianBlur(
    image,
    5
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "median_blur.jpg"),
    median_blur
)

print("✓ Median Blur completed")


# ==========================================================
# 6. Bilateral Filter
# ==========================================================

print("\n[6] Applying Bilateral Filter...")

bilateral = cv2.bilateralFilter(
    image,
    9,
    75,
    75
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "bilateral_filter.jpg"),
    bilateral
)

print("✓ Bilateral filtering completed")


# ==========================================================
# 7. Image Sharpening
# ==========================================================

print("\n[7] Sharpening image...")

sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharpened = cv2.filter2D(
    image,
    -1,
    sharpen_kernel
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "sharpened.jpg"),
    sharpened
)

print("✓ Sharpening completed")


# ==========================================================
# Display Results
# ==========================================================

cv2.imshow("Original", image)
cv2.imshow("Grayscale", grayscale)
cv2.imshow("Brightness Increased", brightness_increased)
cv2.imshow("Contrast Increased", contrast_increased)
cv2.imshow("Gaussian Blur", gaussian_blur)
cv2.imshow("Median Blur", median_blur)
cv2.imshow("Bilateral Filter", bilateral)
cv2.imshow("Sharpened", sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ==========================================================
# Completion
# ==========================================================

print("\n" + "=" * 70)
print("All image enhancement operations completed successfully.")
print(f"Results saved to: {OUTPUT_FOLDER}")
print("=" * 70)