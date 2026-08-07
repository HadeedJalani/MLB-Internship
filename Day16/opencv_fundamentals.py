# ==========================================================
# MLB Summer Internship
# Day 16 - OpenCV Fundamentals
# ==========================================================

import cv2
import os

print("=" * 70)
print("OpenCV Fundamentals")
print("=" * 70)

# ==========================================================
# Input Image
# ==========================================================

IMAGE_PATH = "input_images/landscape.jpg"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"\nImage not found:\n{IMAGE_PATH}")

# ==========================================================
# Read Image
# ==========================================================

image = cv2.imread(IMAGE_PATH)

print("\nImage Loaded Successfully")

# ==========================================================
# Image Properties
# ==========================================================

height, width, channels = image.shape

print("\nImage Properties")
print("-" * 50)

print(f"Height      : {height} pixels")
print(f"Width       : {width} pixels")
print(f"Channels    : {channels}")

file_size = os.path.getsize(IMAGE_PATH) / 1024

print(f"File Size   : {file_size:.2f} KB")

# ==========================================================
# Convert BGR → RGB
# ==========================================================

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==========================================================
# Convert to Grayscale
# ==========================================================

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==========================================================
# Create Output Folder
# ==========================================================

OUTPUT_FOLDER = os.path.join(
    "output_images",
    "grayscale"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================================
# Save Images
# ==========================================================

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "original.jpg"),
    image
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "grayscale.jpg"),
    gray_image
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "rgb_image.jpg"),
    cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
)

print("\nImages Saved Successfully")

print(f"\nSaved to: {OUTPUT_FOLDER}")

# ==========================================================
# Display Images
# ==========================================================

cv2.imshow("Original Image", image)
cv2.imshow("Grayscale Image", gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nOpenCV Fundamentals Completed Successfully!")

print("=" * 70)