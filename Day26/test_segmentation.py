import os
import cv2

from segmentation import (
    read_image,
    binary_threshold,
    adaptive_threshold,
    otsu_threshold,
    foreground_segmentation,
    segment_foreground,
)


INPUT_IMAGE = "sample_images/test.jpg"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    print("\n===== DAY 26 SEGMENTATION TEST =====\n")

    if not os.path.exists(INPUT_IMAGE):
        print(f"ERROR: Image not found:")
        print(INPUT_IMAGE)
        print("\nPut an image at:")
        print("Day-26/sample_images/test.jpg")
        return

    image = read_image(INPUT_IMAGE)

    print("Image loaded successfully.")
    print("Image shape:", image.shape)

    # Binary
    binary = binary_threshold(image)

    binary_path = os.path.join(
        OUTPUT_DIR,
        "binary_test.jpg"
    )

    cv2.imwrite(binary_path, binary)

    print("Binary threshold saved:", binary_path)

    # Adaptive
    adaptive = adaptive_threshold(image)

    adaptive_path = os.path.join(
        OUTPUT_DIR,
        "adaptive_test.jpg"
    )

    cv2.imwrite(adaptive_path, adaptive)

    print("Adaptive threshold saved:", adaptive_path)

    # Otsu
    otsu = otsu_threshold(image)

    otsu_path = os.path.join(
        OUTPUT_DIR,
        "otsu_test.jpg"
    )

    cv2.imwrite(otsu_path, otsu)

    print("Otsu threshold saved:", otsu_path)

    # Foreground
    mask, segmented = segment_foreground(image)

    mask_path = os.path.join(
        OUTPUT_DIR,
        "foreground_mask.jpg"
    )

    segmented_path = os.path.join(
        OUTPUT_DIR,
        "foreground_segmented.jpg"
    )

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(segmented_path, segmented)

    print("Foreground mask saved:", mask_path)
    print("Foreground segmentation saved:", segmented_path)

    print("\n===== TEST COMPLETE =====")


if __name__ == "__main__":
    main()