# ==========================================================
# MLB Summer Internship - Day 18
# Edge Detection using OpenCV
# ==========================================================

import cv2
import os

# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
EDGE_FOLDER = os.path.join(OUTPUT_FOLDER, "edges")

os.makedirs(EDGE_FOLDER, exist_ok=True)


# ==========================================================
# Edge Detection
# ==========================================================

def process_image(filename):

    input_path = os.path.join(INPUT_FOLDER, filename)

    image = cv2.imread(input_path)

    if image is None:
        print(f"Unable to read: {filename}")
        return

    print(f"Processing: {filename}")

    # ------------------------------------------------------
    # Grayscale
    # ------------------------------------------------------

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------------------------------
    # Gaussian Blur
    # ------------------------------------------------------

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    # ------------------------------------------------------
    # Sobel Edge Detection
    # ------------------------------------------------------

    sobel_x = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel = cv2.addWeighted(
        sobel_x,
        0.5,
        sobel_y,
        0.5,
        0
    )

    # ------------------------------------------------------
    # Laplacian Edge Detection
    # ------------------------------------------------------

    laplacian = cv2.Laplacian(
        blurred,
        cv2.CV_64F
    )

    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    # ------------------------------------------------------
    # Canny Edge Detection
    # ------------------------------------------------------

    canny = cv2.Canny(
        blurred,
        50,
        150
    )

    # ------------------------------------------------------
    # File names
    # ------------------------------------------------------

    base_name = os.path.splitext(filename)[0]

    grayscale_path = os.path.join(
        EDGE_FOLDER,
        f"{base_name}_grayscale.jpg"
    )

    blurred_path = os.path.join(
        EDGE_FOLDER,
        f"{base_name}_blurred.jpg"
    )

    sobel_path = os.path.join(
        EDGE_FOLDER,
        f"{base_name}_sobel.jpg"
    )

    laplacian_path = os.path.join(
        EDGE_FOLDER,
        f"{base_name}_laplacian.jpg"
    )

    canny_path = os.path.join(
        EDGE_FOLDER,
        f"{base_name}_canny.jpg"
    )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    cv2.imwrite(
        grayscale_path,
        grayscale
    )

    cv2.imwrite(
        blurred_path,
        blurred
    )

    cv2.imwrite(
        sobel_path,
        sobel
    )

    cv2.imwrite(
        laplacian_path,
        laplacian
    )

    cv2.imwrite(
        canny_path,
        canny
    )

    print("  ✓ Grayscale saved")
    print("  ✓ Gaussian blur saved")
    print("  ✓ Sobel edges saved")
    print("  ✓ Laplacian edges saved")
    print("  ✓ Canny edges saved")


# ==========================================================
# Main Program
# ==========================================================

def main():

    print("=" * 70)
    print("Day 18 - OpenCV Edge Detection")
    print("=" * 70)

    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_FOLDER}"
        )

    image_files = [
        file
        for file in os.listdir(INPUT_FOLDER)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if not image_files:
        raise FileNotFoundError(
            "No images found inside input_images."
        )

    print(
        f"\nImages found: {len(image_files)}"
    )

    for filename in sorted(image_files):
        process_image(filename)

    print("\n" + "=" * 70)
    print("Edge detection completed successfully.")
    print(f"Outputs saved to: {EDGE_FOLDER}")
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()