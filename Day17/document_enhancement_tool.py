# ==========================================================
# MLB Summer Internship - Day 17
# Document Image Enhancement Tool
# ==========================================================

import cv2
import os
import numpy as np


# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"

PERSPECTIVE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "perspective_corrected"
)

COMPARISON_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "comparisons"
)

ENHANCED_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "enhanced"
)

os.makedirs(PERSPECTIVE_FOLDER, exist_ok=True)
os.makedirs(COMPARISON_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)


# ==========================================================
# Utility Functions
# ==========================================================

def resize_for_comparison(image, width=500):
    """
    Resize an image while maintaining its aspect ratio.
    """

    height, original_width = image.shape[:2]

    ratio = width / original_width
    new_height = int(height * ratio)

    return cv2.resize(
        image,
        (width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ==========================================================
# Perspective Correction
# ==========================================================

def order_points(points):
    """
    Arrange four points in the following order:

    top-left
    top-right
    bottom-right
    bottom-left
    """

    points = np.array(points, dtype=np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]

    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]

    return ordered


def four_point_transform(image, points):
    """
    Apply perspective transformation using
    four document corner points.
    """

    rect = order_points(points)

    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(
        top_right - top_left
    )

    width_bottom = np.linalg.norm(
        bottom_right - bottom_left
    )

    max_width = max(
        int(width_top),
        int(width_bottom)
    )

    height_right = np.linalg.norm(
        bottom_right - top_right
    )

    height_left = np.linalg.norm(
        bottom_left - top_left
    )

    max_height = max(
        int(height_right),
        int(height_left)
    )

    if max_width <= 0 or max_height <= 0:
        return image.copy()

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return corrected


# ==========================================================
# Automatic Document Detection
# ==========================================================

def detect_document(image):
    """
    Attempt to locate the largest four-sided contour
    that can represent a document.
    """

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small contours
        if area < image_area * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) == 4:

            return approximation.reshape(4, 2)

    return None


# ==========================================================
# Grayscale Conversion
# ==========================================================

def convert_to_grayscale(image):
    """
    Convert BGR image to grayscale.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# ==========================================================
# Noise Reduction
# ==========================================================

def reduce_noise(image):
    """
    Reduce noise while preserving important edges.
    """

    return cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )


# ==========================================================
# Brightness and Contrast
# ==========================================================

def enhance_brightness_contrast(
    image,
    brightness=10,
    contrast=1.2
):
    """
    Adjust brightness and contrast.

    alpha = contrast
    beta  = brightness
    """

    return cv2.convertScaleAbs(
        image,
        alpha=contrast,
        beta=brightness
    )


# ==========================================================
# Sharpening
# ==========================================================

def sharpen_image(image):
    """
    Enhance document edges and text using
    a sharpening kernel.
    """

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


# ==========================================================
# Complete Enhancement Pipeline
# ==========================================================

def enhance_document(image):
    """
    Complete document enhancement pipeline.

    Steps:
    1. Perspective correction
    2. Grayscale conversion
    3. Noise reduction
    4. Brightness and contrast enhancement
    5. Sharpening
    """

    # ------------------------------------------------------
    # Step 1: Perspective Correction
    # ------------------------------------------------------

    document_points = detect_document(image)

    if document_points is not None:

        corrected = four_point_transform(
            image,
            document_points
        )

        perspective_status = "Detected"

    else:

        corrected = image.copy()

        perspective_status = (
            "Not detected - original geometry retained"
        )

    # ------------------------------------------------------
    # Step 2: Grayscale
    # ------------------------------------------------------

    grayscale = convert_to_grayscale(
        corrected
    )

    # ------------------------------------------------------
    # Step 3: Noise Reduction
    # ------------------------------------------------------

    denoised = reduce_noise(
        grayscale
    )

    # ------------------------------------------------------
    # Step 4: Brightness and Contrast
    # ------------------------------------------------------

    enhanced = enhance_brightness_contrast(
        denoised,
        brightness=10,
        contrast=1.2
    )

    # ------------------------------------------------------
    # Step 5: Sharpening
    # ------------------------------------------------------

    final_image = sharpen_image(
        enhanced
    )

    return (
        corrected,
        final_image,
        perspective_status
    )


# ==========================================================
# Create Before / After Comparison
# ==========================================================

def create_comparison(original, enhanced):
    """
    Create a side-by-side comparison image.
    """

    original_preview = resize_for_comparison(
        original
    )

    enhanced_preview = resize_for_comparison(
        enhanced
    )

    # Make both images the same height
    target_height = min(
        original_preview.shape[0],
        enhanced_preview.shape[0]
    )

    original_preview = cv2.resize(
        original_preview,
        (
            original_preview.shape[1],
            target_height
        )
    )

    enhanced_preview = cv2.resize(
        enhanced_preview,
        (
            enhanced_preview.shape[1],
            target_height
        )
    )

    # Convert grayscale enhanced image to BGR
    # so both images can be combined.
    if len(enhanced_preview.shape) == 2:

        enhanced_preview = cv2.cvtColor(
            enhanced_preview,
            cv2.COLOR_GRAY2BGR
        )

    # Add labels
    cv2.putText(
        original_preview,
        "Original",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        enhanced_preview,
        "Enhanced",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 150, 0),
        2,
        cv2.LINE_AA
    )

    comparison = np.hstack(
        [
            original_preview,
            enhanced_preview
        ]
    )

    return comparison


# ==========================================================
# Process One Document
# ==========================================================

def process_document(filename):

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    if not os.path.exists(input_path):

        print(
            f"Skipping missing file: {filename}"
        )

        return

    image = cv2.imread(
        input_path
    )

    if image is None:

        print(
            f"Unable to read: {filename}"
        )

        return

    print(
        f"\nProcessing: {filename}"
    )

    corrected, enhanced, status = enhance_document(
        image
    )

    base_name = os.path.splitext(
        filename
    )[0]

    # ------------------------------------------------------
    # Output Paths
    # ------------------------------------------------------

    corrected_path = os.path.join(
        PERSPECTIVE_FOLDER,
        f"{base_name}_perspective_corrected.jpg"
    )

    enhanced_path = os.path.join(
        ENHANCED_FOLDER,
        f"{base_name}_enhanced.jpg"
    )

    comparison_path = os.path.join(
        COMPARISON_FOLDER,
        f"{base_name}_comparison.jpg"
    )

    # ------------------------------------------------------
    # Save Perspective Corrected Image
    # ------------------------------------------------------

    cv2.imwrite(
        corrected_path,
        corrected
    )

    # ------------------------------------------------------
    # Save Final Enhanced Image
    # ------------------------------------------------------

    cv2.imwrite(
        enhanced_path,
        enhanced
    )

    # ------------------------------------------------------
    # Create and Save Comparison
    # ------------------------------------------------------

    comparison = create_comparison(
        image,
        enhanced
    )

    cv2.imwrite(
        comparison_path,
        comparison
    )

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(
        f"Perspective correction: {status}"
    )

    print(
        f"✓ Corrected : {corrected_path}"
    )

    print(
        f"✓ Enhanced  : {enhanced_path}"
    )

    print(
        f"✓ Comparison: {comparison_path}"
    )


# ==========================================================
# Main Program
# ==========================================================

def main():

    print("=" * 70)
    print(
        "Day 17 - Document Image Enhancement Tool"
    )
    print("=" * 70)

    if not os.path.exists(INPUT_FOLDER):

        raise FileNotFoundError(
            f"Input folder not found: {INPUT_FOLDER}"
        )

    image_files = [
        file
        for file in os.listdir(INPUT_FOLDER)
        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    if not image_files:

        raise FileNotFoundError(
            "No document images were found inside input_images."
        )

    print(
        f"\nDocuments found: {len(image_files)}"
    )

    # Process all documents
    for filename in sorted(image_files):

        process_document(
            filename
        )

    print("\n" + "=" * 70)
    print(
        "Document enhancement completed successfully."
    )
    print("=" * 70)

    print("\nOutput folders:")

    print(
        f"Perspective corrected : "
        f"{PERSPECTIVE_FOLDER}"
    )

    print(
        f"Enhanced images       : "
        f"{ENHANCED_FOLDER}"
    )

    print(
        f"Comparisons            : "
        f"{COMPARISON_FOLDER}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()