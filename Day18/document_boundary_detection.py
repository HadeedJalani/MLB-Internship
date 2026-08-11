# ==========================================================
# MLB Summer Internship - Day 18
# Document Boundary Detection Tool
# ==========================================================

import cv2
import os
import numpy as np

# ==========================================================
# Configuration
# ==========================================================

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"

BOUNDARY_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "boundaries"
)

COMPARISON_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "comparisons"
)

os.makedirs(BOUNDARY_FOLDER, exist_ok=True)
os.makedirs(COMPARISON_FOLDER, exist_ok=True)


# ==========================================================
# Document Boundary Detection
# ==========================================================

def detect_document_boundary(image):
    """
    Detect the largest four-sided contour
    that is likely to represent a document.
    """

    # ------------------------------------------------------
    # Convert to grayscale
    # ------------------------------------------------------

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------------------------------
    # Reduce noise
    # ------------------------------------------------------

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    # ------------------------------------------------------
    # Canny edge detection
    # ------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # ------------------------------------------------------
    # Morphological closing
    # ------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    cleaned_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # ------------------------------------------------------
    # Find contours
    # ------------------------------------------------------

    contours, _ = cv2.findContours(
        cleaned_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = (
        image.shape[0] *
        image.shape[1]
    )

    # ------------------------------------------------------
    # Find largest suitable four-sided contour
    # ------------------------------------------------------

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

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

            return (
                approximation.reshape(4, 2),
                edges,
                cleaned_edges
            )

    return (
        None,
        edges,
        cleaned_edges
    )


# ==========================================================
# Process One Image
# ==========================================================

def process_image(filename):

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

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

    # ------------------------------------------------------
    # Detect boundary
    # ------------------------------------------------------

    document_points, edges, cleaned_edges = (
        detect_document_boundary(image)
    )

    # ------------------------------------------------------
    # Draw boundary
    # ------------------------------------------------------

    boundary_image = image.copy()

    if document_points is not None:

        points = document_points.reshape(
            (-1, 1, 2)
        )

        cv2.polylines(
            boundary_image,
            [points],
            True,
            (0, 255, 0),
            4
        )

        print(
            "  ✓ Document boundary detected"
        )

    else:

        print(
            "  ! Document boundary not detected"
        )

    # ------------------------------------------------------
    # File names
    # ------------------------------------------------------

    base_name = os.path.splitext(
        filename
    )[0]

    edges_path = os.path.join(
        BOUNDARY_FOLDER,
        f"{base_name}_edges.jpg"
    )

    morphology_path = os.path.join(
        BOUNDARY_FOLDER,
        f"{base_name}_morphology.jpg"
    )

    boundary_path = os.path.join(
        BOUNDARY_FOLDER,
        f"{base_name}_boundary.jpg"
    )

    comparison_path = os.path.join(
        COMPARISON_FOLDER,
        f"{base_name}_comparison.jpg"
    )

    # ------------------------------------------------------
    # Save edge detection result
    # ------------------------------------------------------

    cv2.imwrite(
        edges_path,
        edges
    )

    # ------------------------------------------------------
    # Save morphological result
    # ------------------------------------------------------

    cv2.imwrite(
        morphology_path,
        cleaned_edges
    )

    # ------------------------------------------------------
    # Save boundary result
    # ------------------------------------------------------

    cv2.imwrite(
        boundary_path,
        boundary_image
    )

    # ------------------------------------------------------
    # Create comparison
    # ------------------------------------------------------

    original_preview = image.copy()
    boundary_preview = boundary_image.copy()

    target_width = 600

    original_height = int(
        original_preview.shape[0]
        * target_width
        / original_preview.shape[1]
    )

    boundary_height = int(
        boundary_preview.shape[0]
        * target_width
        / boundary_preview.shape[1]
    )

    original_preview = cv2.resize(
        original_preview,
        (
            target_width,
            original_height
        )
    )

    boundary_preview = cv2.resize(
        boundary_preview,
        (
            target_width,
            boundary_height
        )
    )

    comparison_height = min(
        original_preview.shape[0],
        boundary_preview.shape[0]
    )

    original_preview = cv2.resize(
        original_preview,
        (
            target_width,
            comparison_height
        )
    )

    boundary_preview = cv2.resize(
        boundary_preview,
        (
            target_width,
            comparison_height
        )
    )

    comparison = np.hstack(
        [
            original_preview,
            boundary_preview
        ]
    )

    cv2.imwrite(
        comparison_path,
        comparison
    )

    print(
        f"  ✓ Edges: {edges_path}"
    )

    print(
        f"  ✓ Morphology: {morphology_path}"
    )

    print(
        f"  ✓ Boundary: {boundary_path}"
    )

    print(
        f"  ✓ Comparison: {comparison_path}"
    )


# ==========================================================
# Main Program
# ==========================================================

def main():

    print("=" * 70)
    print("Day 18 - Document Boundary Detection Tool")
    print("=" * 70)

    if not os.path.exists(
        INPUT_FOLDER
    ):

        raise FileNotFoundError(
            f"Input folder not found: {INPUT_FOLDER}"
        )

    image_files = [
        file
        for file in os.listdir(
            INPUT_FOLDER
        )
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
            "No document images found "
            "inside input_images."
        )

    print(
        f"\nDocuments found: "
        f"{len(image_files)}"
    )

    for filename in sorted(
        image_files
    ):

        process_image(
            filename
        )

    print("\n" + "=" * 70)
    print(
        "Document boundary detection "
        "completed successfully."
    )
    print("=" * 70)

    print(
        f"\nBoundary outputs: "
        f"{BOUNDARY_FOLDER}"
    )

    print(
        f"Comparisons: "
        f"{COMPARISON_FOLDER}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()