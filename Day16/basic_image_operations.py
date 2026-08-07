# ==========================================================
# MLB Summer Internship - Day 16
# Basic Image Operations using OpenCV
# ==========================================================

import cv2
import os

print("=" * 70)
print("Basic Image Operations using OpenCV")
print("=" * 70)

# ==========================================================
# Paths
# ==========================================================

IMAGE_PATH = "input_images/landscape.jpg"

OUTPUT_FOLDER = "output_images"

RESIZE_FOLDER = os.path.join(OUTPUT_FOLDER, "resized")
CROP_FOLDER = os.path.join(OUTPUT_FOLDER, "cropped")
ROTATE_FOLDER = os.path.join(OUTPUT_FOLDER, "rotated")
FLIP_FOLDER = os.path.join(OUTPUT_FOLDER, "flipped")

os.makedirs(RESIZE_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)
os.makedirs(ROTATE_FOLDER, exist_ok=True)
os.makedirs(FLIP_FOLDER, exist_ok=True)

# ==========================================================
# Read Image
# ==========================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"\nImage not found:\n{IMAGE_PATH}")

image = cv2.imread(IMAGE_PATH)

height, width = image.shape[:2]

print(f"\nImage Size : {width} x {height}")

# ==========================================================
# Resize Images
# ==========================================================

print("\nResizing Images...")

resize_800 = cv2.resize(image, (800, 600))
resize_400 = cv2.resize(image, (400, 400))
resize_half = cv2.resize(image, (width // 2, height // 2))

cv2.imwrite(
    os.path.join(RESIZE_FOLDER, "resized_800x600.jpg"),
    resize_800
)

cv2.imwrite(
    os.path.join(RESIZE_FOLDER, "resized_400x400.jpg"),
    resize_400
)

cv2.imwrite(
    os.path.join(RESIZE_FOLDER, "resized_half.jpg"),
    resize_half
)

print("✓ Resized images saved")

# ==========================================================
# Crop Images
# ==========================================================

print("\nCropping Images...")

center_crop = image[
    height//4 : height*3//4,
    width//4 : width*3//4
]

top_left_crop = image[
    0:height//2,
    0:width//2
]

bottom_right_crop = image[
    height//2:height,
    width//2:width
]

cv2.imwrite(
    os.path.join(CROP_FOLDER, "crop_center.jpg"),
    center_crop
)

cv2.imwrite(
    os.path.join(CROP_FOLDER, "crop_top_left.jpg"),
    top_left_crop
)

cv2.imwrite(
    os.path.join(CROP_FOLDER, "crop_bottom_right.jpg"),
    bottom_right_crop
)

print("✓ Cropped images saved")

# ==========================================================
# Rotate Images
# ==========================================================

print("\nRotating Images...")

rotate90 = cv2.rotate(
    image,
    cv2.ROTATE_90_CLOCKWISE
)

rotate180 = cv2.rotate(
    image,
    cv2.ROTATE_180
)

rotate270 = cv2.rotate(
    image,
    cv2.ROTATE_90_COUNTERCLOCKWISE
)

cv2.imwrite(
    os.path.join(ROTATE_FOLDER, "rotate_90.jpg"),
    rotate90
)

cv2.imwrite(
    os.path.join(ROTATE_FOLDER, "rotate_180.jpg"),
    rotate180
)

cv2.imwrite(
    os.path.join(ROTATE_FOLDER, "rotate_270.jpg"),
    rotate270
)

print("✓ Rotated images saved")

# ==========================================================
# Flip Images
# ==========================================================

print("\nFlipping Images...")

flip_horizontal = cv2.flip(image, 1)

flip_vertical = cv2.flip(image, 0)

flip_both = cv2.flip(image, -1)

cv2.imwrite(
    os.path.join(FLIP_FOLDER, "flip_horizontal.jpg"),
    flip_horizontal
)

cv2.imwrite(
    os.path.join(FLIP_FOLDER, "flip_vertical.jpg"),
    flip_vertical
)

cv2.imwrite(
    os.path.join(FLIP_FOLDER, "flip_both.jpg"),
    flip_both
)

print("✓ Flipped images saved")

# ==========================================================
# Display Results
# ==========================================================

cv2.imshow("Original", image)
cv2.imshow("Resize", resize_800)
cv2.imshow("Center Crop", center_crop)
cv2.imshow("Rotate 90", rotate90)
cv2.imshow("Flip Horizontal", flip_horizontal)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ==========================================================
# Finish
# ==========================================================

print("\n" + "=" * 70)
print("All Basic Image Operations Completed Successfully!")
print("Outputs saved inside:")
print("output_images/")
print("   ├── resized/")
print("   ├── cropped/")
print("   ├── rotated/")
print("   └── flipped/")
print("=" * 70)