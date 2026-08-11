# ==========================================================
# MLB Summer Internship - Day 18
# Morphological Operations using OpenCV
# ==========================================================

import cv2
import os

# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
MORPHOLOGY_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "morphology"
)

os.makedirs(MORPHOLOGY_FOLDER, exist_ok=True)


# ==========================================================
# Morphological Processing
# ==========================================================

def process_image(filename):

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

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
    # Binary Threshold
    # ------------------------------------------------------

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ------------------------------------------------------
    # Kernel
    # ------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    # ------------------------------------------------------
    # Erosion
    # ------------------------------------------------------

    erosion = cv2.erode(
        binary,
        kernel,
        iterations=1
    )

    # ------------------------------------------------------
    # Dilation
    # ------------------------------------------------------

    dilation = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    # ------------------------------------------------------
    # Opening
    # ------------------------------------------------------

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # ------------------------------------------------------
    # Closing
    # ------------------------------------------------------

    closing = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ------------------------------------------------------
    # Morphological Gradient
    # ------------------------------------------------------

    gradient = cv2.morphologyEx(
        binary,
        cv2.MORPH_GRADIENT,
        kernel
    )

    # ------------------------------------------------------
    # Top Hat
    # ------------------------------------------------------

    top_hat = cv2.morphologyEx(
        binary,
        cv2.MORPH_TOPHAT,
        kernel
    )

    # ------------------------------------------------------
    # Black Hat
    # ------------------------------------------------------

    black_hat = cv2.morphologyEx(
        binary,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # ------------------------------------------------------
    # File naming
    # ------------------------------------------------------

    base_name = os.path.splitext(
        filename
    )[0]

    # ------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------

    outputs = {
        "binary": binary,
        "erosion": erosion,
        "dilation": dilation,
        "opening": opening,
        "closing": closing,
        "gradient": gradient,
        "top_hat": top_hat,
        "black_hat": black_hat
    }

    for operation_name, result in outputs.items():

        output_path = os.path.join(
            MORPHOLOGY_FOLDER,
            f"{base_name}_{operation_name}.jpg"
        )

        cv2.imwrite(
            output_path,
            result
        )

        print(
            f"  ✓ {operation_name.replace('_', ' ').title()} saved"
        )


# ==========================================================
# Main Program
# ==========================================================

def main():

    print("=" * 70)
    print("Day 18 - OpenCV Morphological Operations")
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
    print("Morphological processing completed successfully.")
    print(
        f"Outputs saved to: {MORPHOLOGY_FOLDER}"
    )
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()